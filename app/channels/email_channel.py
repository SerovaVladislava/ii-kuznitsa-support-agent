"""Канал «почта»: забирает новые письма по IMAP, отвечает через ассистента по SMTP.

Один отправитель = один открытый диалог. Если диалог у менеджера, ассистент молчит, а письмо
появляется в общей ленте; ответ менеджера из панели уходит письмом.
"""
from __future__ import annotations

import email
import imaplib
import logging
import re
import threading
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr

from ..assistant import engine
from ..config import get_settings
from ..services import customers, dialogs, settings_store
from ..services.notify import send_email

log = logging.getLogger("email")


def _decode(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except (UnicodeDecodeError, LookupError):
        return value


def _body(msg: Message) -> str:
    part = None
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/plain" and "attachment" not in str(p.get("Content-Disposition", "")):
                part = p
                break
        if part is None:
            for p in msg.walk():
                if p.get_content_type() == "text/html":
                    part = p
                    break
    else:
        part = msg
    if part is None:
        return ""
    payload = part.get_payload(decode=True) or b""
    text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    if part.get_content_type() == "text/html":
        text = re.sub(r"<[^>]+>", " ", text)
    return strip_quoted(text)


def strip_quoted(text: str) -> str:
    """Отрезать цитату предыдущей переписки и подпись."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(">") or re.match(r"^(-{2,}\s*$|-----Original Message|От:|From:|\d{1,2}\.\d{1,2}\.\d{2,4}.*пиш|On .* wrote:)", s):
            break
        lines.append(line)
    return "\n".join(lines).strip()[:4000]


def handle_incoming_email(sender: str, sender_name: str, subject: str, text: str,
                          message_id: str = "") -> dict:
    """Обработать письмо клиента: диалог «почта», ответ ассистента, отправка ответа по SMTP (если настроен)."""
    sender = sender.strip().lower()
    dialog = dialogs.get_or_create_by_external("email", sender, subject)
    if not dialog["customer_id"]:
        cid, _ = customers.upsert({"email": sender, "name": sender_name})
        dialogs.set_customer(dialog["id"], cid)
    replies = engine.handle_client_message(dialog["id"], strip_quoted(text), {"subject": subject})
    company = settings_store.get("company")
    sent = False
    for r in replies:
        sent = send_email(sender, subject if subject.lower().startswith("re:") else f"Re: {subject or 'Ваш запрос'}",
                          f"{r['text']}\n\n--\n{company['name']}\n{company.get('store_phone') or company['phone']}",
                          in_reply_to=message_id) or sent
    return {"dialog_id": dialog["id"], "replies": [r["text"] for r in replies], "sent": sent}


def process_message(raw: bytes) -> None:
    msg = email.message_from_bytes(raw)
    sender_name, sender = parseaddr(_decode(msg.get("From")))
    if not sender or sender.lower() == get_settings().email_user.lower():
        return
    text = _body(msg)
    if text:
        handle_incoming_email(sender, sender_name, _decode(msg.get("Subject")), text, msg.get("Message-ID", ""))


def poll_once() -> int:
    s = get_settings()
    count = 0
    with imaplib.IMAP4_SSL(s.email_imap_host, s.email_imap_port) as imap:
        imap.login(s.email_user, s.email_password)
        imap.select("INBOX")
        _, data = imap.search(None, "UNSEEN")
        for num in data[0].split():
            _, msg_data = imap.fetch(num, "(RFC822)")
            raw = msg_data[0][1] if msg_data and msg_data[0] else None
            if raw:
                try:
                    process_message(raw)
                    count += 1
                except Exception:  # noqa: BLE001
                    log.exception("Ошибка обработки письма")
    return count


def start(stop: threading.Event) -> threading.Thread:
    interval = max(15, get_settings().email_poll_seconds)

    def loop() -> None:
        log.info("Почтовый канал запущен, опрос каждые %s с", interval)
        while not stop.is_set():
            try:
                poll_once()
            except (imaplib.IMAP4.error, OSError) as exc:
                log.error("IMAP: %s", exc)
            stop.wait(interval)

    thread = threading.Thread(target=loop, name="email-channel", daemon=True)
    thread.start()
    return thread
