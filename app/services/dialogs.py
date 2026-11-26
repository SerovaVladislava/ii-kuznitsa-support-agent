"""Диалоги и сообщения во всех каналах: единая лента для бота и менеджеров."""
from __future__ import annotations

import uuid
from typing import Any

from .. import db

STATUSES = {
    "bot": "Ведёт ассистент",
    "waiting_operator": "Ждёт менеджера",
    "operator": "У менеджера",
    "closed": "Закрыт",
}
CHANNELS = {"web": "Сайт", "email": "Почта", "telegram": "Telegram", "api": "Интеграция", "phone": "Телефон"}
ESCALATION_REASONS = {
    "special_terms": "Особые условия / скидка",
    "client_request": "Клиент просит менеджера",
    "no_answer": "Нет ответа в базе знаний",
    "complaint": "Претензия / проблема",
    "large_deal": "Крупная сделка",
    "error": "Сбой ассистента",
    "other": "Другое",
}


def _hydrate(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row:
        row["state"] = db.loads(row["state"], {})
        if "classification" in row:
            row["classification"] = db.loads(row["classification"], {}) or {}
    return row


def create(channel: str = "web", external_id: str = "", subject: str = "",
           customer_id: int | None = None) -> dict[str, Any]:
    now = db.now_iso()
    dialog_id = db.execute(
        "INSERT INTO dialogs (public_id, channel, external_id, subject, customer_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, channel, external_id, subject, customer_id, now, now),
    )
    db.log_event("dialog_started", dialog_id, channel=channel)
    return get(dialog_id)


def get(dialog_id: int) -> dict[str, Any] | None:
    return _hydrate(db.query_one("SELECT * FROM dialogs WHERE id = ?", (dialog_id,)))


def get_by_public_id(public_id: str) -> dict[str, Any] | None:
    return _hydrate(db.query_one("SELECT * FROM dialogs WHERE public_id = ?", (public_id,)))


def find_open_by_external(channel: str, external_id: str) -> dict[str, Any] | None:
    return _hydrate(db.query_one(
        "SELECT * FROM dialogs WHERE channel = ? AND external_id = ? AND status != 'closed' "
        "ORDER BY id DESC LIMIT 1", (channel, external_id)))


def get_or_create_by_external(channel: str, external_id: str, subject: str = "") -> dict[str, Any]:
    return find_open_by_external(channel, external_id) or create(channel, external_id, subject)


def add_message(dialog_id: int, role: str, text: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    now = db.now_iso()
    msg_id = db.execute(
        "INSERT INTO messages (dialog_id, role, text, meta, created_at) VALUES (?, ?, ?, ?, ?)",
        (dialog_id, role, text, db.dumps(meta or {}), now),
    )
    if role == "client":
        db.execute("UPDATE dialogs SET updated_at = ?, last_client_msg_at = ?, "
                   "unread = unread + CASE WHEN status IN ('waiting_operator', 'operator') THEN 1 ELSE 0 END "
                   "WHERE id = ?", (now, now, dialog_id))
    else:
        db.execute("UPDATE dialogs SET updated_at = ? WHERE id = ?", (now, dialog_id))
    return {"id": msg_id, "dialog_id": dialog_id, "role": role, "text": text, "meta": meta or {}, "created_at": now}


def list_messages(dialog_id: int, after_id: int = 0) -> list[dict[str, Any]]:
    rows = db.query("SELECT * FROM messages WHERE dialog_id = ? AND id > ? ORDER BY id", (dialog_id, after_id))
    for r in rows:
        r["meta"] = db.loads(r["meta"], {})
    return rows


def set_status(dialog_id: int, status: str) -> None:
    assert status in STATUSES
    db.execute("UPDATE dialogs SET status = ?, updated_at = ? WHERE id = ?", (status, db.now_iso(), dialog_id))
    db.log_event("status_changed", dialog_id, status=status)


def mark_read(dialog_id: int) -> None:
    db.execute("UPDATE dialogs SET unread = 0 WHERE id = ?", (dialog_id,))


def escalate(dialog_id: int, reason: str, summary: str) -> bool:
    """Передать менеджеру. Возвращает True, если это новая эскалация (нужно уведомить)."""
    dialog = get(dialog_id)
    if dialog["status"] in ("waiting_operator", "operator"):
        db.execute("UPDATE dialogs SET escalation_summary = ? WHERE id = ?",
                   ((dialog["escalation_summary"] + "\n" + summary).strip()[:4000], dialog_id))
        return False
    reason = reason if reason in ESCALATION_REASONS else "other"
    db.execute(
        "UPDATE dialogs SET status = 'waiting_operator', escalation_reason = ?, escalation_summary = ?, "
        "escalated_at = ?, was_escalated = 1, escalation_notified_overdue = 0, unread = unread + 1, "
        "updated_at = ? WHERE id = ?",
        (reason, summary[:4000], db.now_iso(), db.now_iso(), dialog_id),
    )
    db.log_event("escalated", dialog_id, reason=reason, summary=summary[:500])
    return True


def set_customer(dialog_id: int, customer_id: int) -> None:
    db.execute("UPDATE dialogs SET customer_id = ? WHERE id = ?", (customer_id, dialog_id))


def save_state(dialog_id: int, state: dict[str, Any]) -> None:
    db.execute("UPDATE dialogs SET state = ? WHERE id = ?", (db.dumps(state), dialog_id))


def update_state(dialog_id: int, **changes: Any) -> dict[str, Any]:
    state = get(dialog_id)["state"]
    state.update(changes)
    save_state(dialog_id, state)
    return state


def load_llm_history(dialog_id: int) -> tuple[list[dict[str, Any]], int]:
    row = db.query_one("SELECT llm_history, llm_synced_msg_id FROM dialogs WHERE id = ?", (dialog_id,))
    return db.loads(row["llm_history"], []), row["llm_synced_msg_id"]


def save_llm_history(dialog_id: int, history: list[dict[str, Any]], synced_msg_id: int) -> None:
    db.execute("UPDATE dialogs SET llm_history = ?, llm_synced_msg_id = ? WHERE id = ?",
               (db.dumps(history), synced_msg_id, dialog_id))


def list_dialogs(status: str = "", channel: str = "", limit: int = 200) -> list[dict[str, Any]]:
    where, params = [], []
    if status == "active":
        where.append("d.status IN ('waiting_operator', 'operator')")
    elif status:
        where.append("d.status = ?")
        params.append(status)
    if channel:
        where.append("d.channel = ?")
        params.append(channel)
    sql = (
        "SELECT d.*, c.name AS customer_name, c.company AS customer_company, c.phone AS customer_phone, "
        "c.email AS customer_email, "
        "(SELECT text FROM messages m WHERE m.dialog_id = d.id ORDER BY m.id DESC LIMIT 1) AS last_text, "
        "(SELECT COUNT(*) FROM messages m WHERE m.dialog_id = d.id AND m.role = 'client') AS client_msgs "
        "FROM dialogs d LEFT JOIN customers c ON c.id = d.customer_id "
        + (f"WHERE {' AND '.join(where)} " if where else "")
        + "ORDER BY CASE d.status WHEN 'waiting_operator' THEN 0 WHEN 'operator' THEN 1 ELSE 2 END, "
          "d.updated_at DESC LIMIT ?"
    )
    return [_hydrate(r) for r in db.query(sql, (*params, limit))]
