"""Публичная часть: API чат-виджета, документы по ссылке, приём сообщений из внешних каналов."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from ..assistant import engine
from ..config import get_settings
from ..services import customers, dialogs, documents, settings_store
from .templating import STATIC_DIR, templates

router = APIRouter()


# ---------- Простейшее ограничение частоты запросов (защита от спама и перерасхода ИИ) ----------

class RateLimiter:
    def __init__(self, limit: int, window: float):
        self.limit, self.window = limit, window
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


message_limiter = RateLimiter(limit=20, window=60)
start_limiter = RateLimiter(limit=10, window=60)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def public_message(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": m["id"],
        "role": m["role"],
        "text": m["text"],
        "created_at": m["created_at"],
        "documents": [{"title": d["title"], "number": d["number"], "url": d["url"]}
                      for d in (m.get("meta") or {}).get("documents", [])],
    }


# ---------- Виджет ----------

@router.get("/", include_in_schema=False)
def index() -> RedirectResponse:
    return RedirectResponse("/demo")


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo(request: Request) -> HTMLResponse:
    from ..services import catalog
    cards = []
    seen: set[str] = set()
    for query, ask in DEMO_CARDS:  # карточки подбираются из текущего каталога (демо или загруженный фид)
        found = catalog.search_families(query, limit=1)
        if found and found[0]["relevance"] >= 0.4 and found[0]["family"] not in seen:
            seen.add(found[0]["family"])
            cards.append({**found[0], "ask": ask})
    return templates.TemplateResponse(request, "demo.html", {"company": settings_store.get("company"), "cards": cards})


DEMO_CARDS = [
    ("MDaemon", "Нужен MDaemon Standard"),
    ("Kaspersky Endpoint Security для бизнеса Стандартный", "Продлить Касперский Стандартный на 40 компьютеров"),
    ("Dr.Web Mail Security Suite", "Расскажите про Dr.Web для почты"),
    ("РЕД ОС", "Сколько стоит РЕД ОС на 10 рабочих мест?"),
    ("Р7-Офис для юридических лиц", "Нужен Р7-Офис на 15 пользователей"),
    ("CommuniGate Pro", "Есть российский почтовый сервер?"),
]


@router.get("/widget.js", include_in_schema=False)
def widget_js() -> FileResponse:
    return FileResponse(STATIC_DIR / "widget.js", media_type="application/javascript",
                        headers={"Cache-Control": "public, max-age=300"})


class StartIn(BaseModel):
    page_url: str = Field(default="", max_length=500)
    dialog_id: str = Field(default="", max_length=64)


@router.post("/api/chat/start")
def chat_start(data: StartIn, request: Request) -> dict[str, Any]:
    a = settings_store.get("assistant")
    dialog = dialogs.get_by_public_id(data.dialog_id) if data.dialog_id else None
    history: list[dict[str, Any]] = []
    if dialog:
        history = [public_message(m) for m in dialogs.list_messages(dialog["id"]) if m["role"] != "system"]
    else:
        if not start_limiter.allow(_client_ip(request)):
            raise HTTPException(429, "Слишком много запросов, попробуйте через минуту")
        dialog = dialogs.create("web", subject=data.page_url[:200])
    return {
        "dialog_id": dialog["public_id"],
        "bot_name": a["bot_name"],
        "greeting": a["greeting"],
        "privacy_notice": a["privacy_notice"],
        "status": dialog["status"],
        "messages": history,
    }


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


@router.post("/api/chat/{public_id}/messages")
def chat_send(public_id: str, data: MessageIn, request: Request) -> dict[str, Any]:
    dialog = dialogs.get_by_public_id(public_id)
    if not dialog:
        raise HTTPException(404, "Диалог не найден")
    if not message_limiter.allow(_client_ip(request)):
        raise HTTPException(429, "Слишком много сообщений, подождите минуту")
    replies = engine.handle_client_message(dialog["id"], data.text, {"ip": _client_ip(request)})
    status = dialogs.get(dialog["id"])["status"]
    return {"messages": [public_message(m) for m in replies], "status": status}


@router.get("/api/chat/{public_id}/messages")
def chat_poll(public_id: str, after: int = 0) -> dict[str, Any]:
    dialog = dialogs.get_by_public_id(public_id)
    if not dialog:
        raise HTTPException(404, "Диалог не найден")
    msgs = [public_message(m) for m in dialogs.list_messages(dialog["id"], after_id=after)
            if m["role"] in ("bot", "operator")]
    return {"messages": msgs, "status": dialog["status"]}


# ---------- Документы ----------

@router.get("/d/{token}.docx", include_in_schema=False)
def document_docx(token: str) -> Response:
    doc = documents.get_by_token(token)
    if not doc:
        raise HTTPException(404, "Документ не найден")
    name = f"{'KP' if doc['kind'] == 'quote' else 'Schet'}_{doc['number']}.docx".replace("КП-", "")
    return Response(documents.render_docx(doc),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/d/{token}", response_class=HTMLResponse, include_in_schema=False)
def document_html(token: str) -> HTMLResponse:
    doc = documents.get_by_token(token)
    if not doc:
        raise HTTPException(404, "Документ не найден")
    return HTMLResponse(documents.render_html(doc), headers={"X-Robots-Tag": "noindex"})


# ---------- Внешние каналы (телефония с распознаванием речи, мессенджеры, формы сайта) ----------

class InboundIn(BaseModel):
    channel: str = Field(default="api", max_length=20)
    external_id: str = Field(min_length=1, max_length=200, description="ID клиента в канале: номер, chat id, email")
    text: str = Field(min_length=1, max_length=4000)
    subject: str = Field(default="", max_length=300)
    name: str = Field(default="", max_length=200)
    phone: str = Field(default="", max_length=50)
    email: str = Field(default="", max_length=200)


@router.post("/api/channels/inbound")
def inbound(data: InboundIn, authorization: str = Header(default="")) -> dict[str, Any]:
    token = get_settings().integration_token
    if not token or authorization != f"Bearer {token}":
        raise HTTPException(401, "Неверный токен интеграции")
    dialog = dialogs.get_or_create_by_external(data.channel, data.external_id, data.subject)
    if (data.name or data.phone or data.email) and not dialog["customer_id"]:
        customer_id, _ = customers.upsert({"name": data.name, "phone": data.phone, "email": data.email})
        dialogs.set_customer(dialog["id"], customer_id)
    replies = engine.handle_client_message(dialog["id"], data.text, {"channel": data.channel})
    return {"dialog_id": dialog["id"], "status": dialogs.get(dialog["id"])["status"],
            "replies": [m["text"] for m in replies]}


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "assistant_mode": engine.assistant_mode()}
