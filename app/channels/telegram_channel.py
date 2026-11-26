"""Канал Telegram: бот для клиентов (long polling, без вебхука и публичного адреса)."""
from __future__ import annotations

import logging
import threading

import httpx

from ..assistant import engine
from ..config import get_settings
from ..services import customers, dialogs, settings_store
from ..services.notify import send_telegram

log = logging.getLogger("telegram")


def handle_update(update: dict) -> None:
    msg = update.get("message") or {}
    text = msg.get("text") or ""
    chat = msg.get("chat") or {}
    chat_id = str(chat.get("id", ""))
    if not chat_id or not text or chat.get("type") != "private":
        return
    if chat_id == get_settings().manager_telegram_chat_id:
        return  # чат уведомлений менеджеров — не клиент
    if text.startswith("/start"):
        send_telegram(chat_id, settings_store.get("assistant")["greeting"])
        return
    sender = msg.get("from") or {}
    dialog = dialogs.get_or_create_by_external("telegram", chat_id, "Telegram")
    if not dialog["customer_id"]:
        name = " ".join(filter(None, [sender.get("first_name"), sender.get("last_name")]))
        cid, _ = customers.upsert({"name": name})
        dialogs.set_customer(dialog["id"], cid)
    for r in engine.handle_client_message(dialog["id"], text, {"tg_user": sender.get("username", "")}):
        send_telegram(chat_id, r["text"])


def start(stop: threading.Event) -> threading.Thread:
    token = get_settings().telegram_bot_token
    api = f"https://api.telegram.org/bot{token}"

    def loop() -> None:
        offset = 0
        log.info("Telegram-канал запущен")
        with httpx.Client(timeout=40) as client:
            while not stop.is_set():
                try:
                    r = client.get(f"{api}/getUpdates", params={"offset": offset, "timeout": 30})
                    for upd in r.json().get("result", []):
                        offset = upd["update_id"] + 1
                        try:
                            handle_update(upd)
                        except Exception:  # noqa: BLE001
                            log.exception("Ошибка обработки сообщения Telegram")
                except (httpx.HTTPError, ValueError) as exc:
                    log.error("Telegram: %s", exc)
                    stop.wait(5)

    thread = threading.Thread(target=loop, name="telegram-channel", daemon=True)
    thread.start()
    return thread
