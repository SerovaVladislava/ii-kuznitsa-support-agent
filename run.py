"""Запуск: python run.py  → http://127.0.0.1:8000 (демо-сайт с виджетом) и /admin (панель менеджера)."""
from __future__ import annotations

import logging
import sys

import uvicorn

from app.config import get_settings
from app.web.main import create_app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()

if __name__ == "__main__":
    s = get_settings()
    print(f"\n  Демо-сайт с виджетом:  http://{s.host}:{s.port}/demo")
    print(f"  Панель менеджера:      http://{s.host}:{s.port}/admin  (пароль из ADMIN_PASSWORD)\n")
    uvicorn.run(app, host=s.host, port=s.port, log_level="info")
