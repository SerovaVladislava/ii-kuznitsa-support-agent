"""FastAPI-приложение: сборка маршрутов, фоновые каналы и мониторинг SLA."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import db, seed
from ..assistant import engine
from ..config import get_settings
from . import admin, public
from .auth import LoginRequired
from .templating import STATIC_DIR

log = logging.getLogger("app")


def create_app(start_background: bool = True) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        db.init_db()
        created = seed.seed_if_empty()
        if any(created.values()):
            log.info("Загружены демо-данные: %s", created)
        stop = threading.Event()
        if start_background:
            engine.start_sla_monitor(stop)
            from ..services import classifier
            classifier.warm_up()  # обучить классификатор обращений заранее, а не на первом сообщении
            if settings.email_enabled:
                from ..channels import email_channel
                email_channel.start(stop)
            if settings.telegram_bot_token:
                from ..channels import telegram_channel
                telegram_channel.start(stop)
        log.info("Режим ассистента: %s", "ИИ-режим" if settings.llm_enabled else "офлайн (без ИИ)")
        if settings.admin_password == "admin":
            log.warning("Пароль админки по умолчанию 'admin' — задайте ADMIN_PASSWORD в .env")
        yield
        stop.set()

    app = FastAPI(title="ИИ-консультант отдела продаж", lifespan=lifespan, docs_url="/api/docs", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=settings.widget_allowed_origins,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"], allow_credentials=False)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(LoginRequired)
    async def _login_redirect(request: Request, exc: LoginRequired) -> RedirectResponse:
        return RedirectResponse("/admin/login", status_code=303)

    app.include_router(public.router)
    app.include_router(admin.router)
    return app
