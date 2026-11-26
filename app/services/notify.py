"""Уведомления менеджерам (эскалации, просрочки SLA) и отправка писем."""
from __future__ import annotations

import logging
import smtplib
import ssl
import threading
from email.message import EmailMessage
from email.utils import make_msgid

import httpx

from .. import db
from ..config import get_settings

log = logging.getLogger("notify")


def send_email(to: str, subject: str, body: str, in_reply_to: str = "") -> bool:
    s = get_settings()
    if not s.smtp_enabled or not to:
        return False
    msg = EmailMessage()
    msg["From"] = s.email_user
    msg["To"] = to
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid()
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = in_reply_to
    msg.set_content(body)
    try:
        if s.email_smtp_port == 465:
            with smtplib.SMTP_SSL(s.email_smtp_host, s.email_smtp_port, context=ssl.create_default_context(),
                                  timeout=30) as smtp:
                smtp.login(s.email_user, s.email_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(s.email_smtp_host, s.email_smtp_port, timeout=30) as smtp:
                smtp.starttls(context=ssl.create_default_context())
                smtp.login(s.email_user, s.email_password)
                smtp.send_message(msg)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        log.error("SMTP: не удалось отправить письмо %s: %s", to, exc)
        return False


def send_telegram(chat_id: str, text: str) -> bool:
    s = get_settings()
    if not s.telegram_bot_token or not chat_id:
        return False
    try:
        r = httpx.post(f"https://api.telegram.org/bot{s.telegram_bot_token}/sendMessage",
                       json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True}, timeout=20)
        return r.status_code == 200
    except httpx.HTTPError as exc:
        log.error("Telegram: не удалось отправить сообщение: %s", exc)
        return False


def notify_managers(title: str, text: str, dialog_id: int | None = None) -> None:
    """Уведомить менеджеров во всех настроенных каналах. Выполняется в фоне, не задерживая ответ клиенту."""
    s = get_settings()
    link = f"{s.public_base_url}/admin/dialogs/{dialog_id}" if dialog_id else ""
    body = f"{title}\n\n{text}\n\n{link}".strip()
    db.log_event("manager_notified", dialog_id, title=title)
    log.warning("[Менеджерам] %s", body.replace("\n", " | "))

    def _send() -> None:
        if s.manager_telegram_chat_id:
            send_telegram(s.manager_telegram_chat_id, body)
        if s.manager_email:
            send_email(s.manager_email, title, body)

    if s.manager_telegram_chat_id or s.manager_email:
        threading.Thread(target=_send, daemon=True).start()
