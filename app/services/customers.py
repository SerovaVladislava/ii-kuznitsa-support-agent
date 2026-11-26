"""Клиенты (лиды): сохранение контактов и реквизитов, привязка к диалогам."""
from __future__ import annotations

from typing import Any

from .. import db
from .textutil import validate_inn

FIELDS = ["name", "phone", "email", "company", "inn", "kpp", "address"]


def get(customer_id: int | None) -> dict[str, Any] | None:
    if not customer_id:
        return None
    return db.query_one("SELECT * FROM customers WHERE id = ?", (customer_id,))


def find_existing(email: str = "", phone: str = "", inn: str = "") -> dict[str, Any] | None:
    for field, value in (("email", email), ("inn", inn), ("phone", phone)):
        if value:
            row = db.query_one(f"SELECT * FROM customers WHERE {field} = ? ORDER BY id DESC LIMIT 1", (value,))
            if row:
                return row
    return None


def upsert(data: dict[str, Any], customer_id: int | None = None) -> tuple[int, list[str]]:
    """Создать/дополнить карточку клиента. Пустые значения не затирают заполненные.

    Возвращает (id, предупреждения).
    """
    clean = {f: str(data.get(f) or "").strip() for f in FIELDS}
    clean["inn"] = "".join(c for c in clean["inn"] if c.isdigit())
    clean["kpp"] = "".join(c for c in clean["kpp"] if c.isdigit())
    warnings = []
    if clean["inn"] and not validate_inn(clean["inn"]):
        warnings.append(f"ИНН {clean['inn']} не прошёл проверку контрольной суммы — уточните у клиента")
    if clean["kpp"] and len(clean["kpp"]) != 9:
        warnings.append("КПП должен содержать 9 цифр")

    existing = get(customer_id) or find_existing(clean["email"], clean["phone"], clean["inn"])
    now = db.now_iso()
    if existing:
        merged = {f: clean[f] or existing[f] for f in FIELDS}
        db.execute(
            f"UPDATE customers SET {', '.join(f + '=?' for f in FIELDS)}, updated_at=? WHERE id=?",
            (*[merged[f] for f in FIELDS], now, existing["id"]),
        )
        return existing["id"], warnings
    new_id = db.execute(
        f"INSERT INTO customers ({', '.join(FIELDS)}, created_at, updated_at) VALUES ({', '.join('?' * 9)})",
        (*[clean[f] for f in FIELDS], now, now),
    )
    return new_id, warnings


def display_name(c: dict[str, Any] | None) -> str:
    if not c:
        return ""
    parts = [c.get("company"), c.get("name")]
    return " / ".join(p for p in parts if p) or c.get("email") or c.get("phone") or f"Клиент #{c['id']}"
