"""SQLite: схема и вспомогательные функции. Одна БД-файл, WAL-режим, соединение на операцию."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Каталог ПО: одна строка = один артикул (SKU) товарного фида.
-- family объединяет варианты одного продукта (редакции, сроки, число пользователей, новая/продление).
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    vendor TEXT NOT NULL DEFAULT '',
    family TEXT NOT NULL DEFAULT '',
    edition TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    unit TEXT NOT NULL DEFAULT 'лиц.',
    pricing TEXT NOT NULL DEFAULT 'per_unit',      -- per_unit: цена за польз./узел; package: цена за пакет
    price REAL NOT NULL DEFAULT 0,                  -- 0 = цена по запросу
    purchase_type TEXT NOT NULL DEFAULT '',         -- new | renewal | upgrade | ''
    license_type TEXT NOT NULL DEFAULT '',          -- subscription | perpetual | service | ''
    term_months INTEGER NOT NULL DEFAULT 0,         -- 0 = бессрочно / не применимо
    users_min INTEGER NOT NULL DEFAULT 0,
    users_max INTEGER NOT NULL DEFAULT 0,           -- 0 = без ограничения
    deployment TEXT NOT NULL DEFAULT '',            -- on-prem | cloud | hybrid | ''
    org_type TEXT NOT NULL DEFAULT '',              -- commercial | academic | government | nonprofit | home | ''
    available INTEGER NOT NULL DEFAULT 1,           -- доступен к заказу на витрине (available в фиде)
    platforms TEXT NOT NULL DEFAULT '',
    is_russian INTEGER NOT NULL DEFAULT 0,          -- в реестре российского ПО
    certificates TEXT NOT NULL DEFAULT '',
    vat_rate REAL,                                  -- NULL = ставка по умолчанию из настроек
    url TEXT NOT NULL DEFAULT '',
    delivery TEXT NOT NULL DEFAULT 'electronic',    -- electronic | box
    stock_qty INTEGER NOT NULL DEFAULT -1,          -- -1 = без ограничений (электронная поставка)
    lead_time_days INTEGER NOT NULL DEFAULT 1,      -- срок поставки, рабочих дней
    min_order_qty INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_products_family_nc ON products(family COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS ix_products_sku_nc ON products(sku COLLATE NOCASE);

CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
    sku, name, family, vendor, category, tags, description,
    content='products', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
    INSERT INTO products_fts(rowid, sku, name, family, vendor, category, tags, description)
    VALUES (new.id, new.sku, new.name, new.family, new.vendor, new.category, new.tags, new.description);
END;
CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
    INSERT INTO products_fts(products_fts, rowid, sku, name, family, vendor, category, tags, description)
    VALUES ('delete', old.id, old.sku, old.name, old.family, old.vendor, old.category, old.tags, old.description);
END;
CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
    INSERT INTO products_fts(products_fts, rowid, sku, name, family, vendor, category, tags, description)
    VALUES ('delete', old.id, old.sku, old.name, old.family, old.vendor, old.category, old.tags, old.description);
    INSERT INTO products_fts(rowid, sku, name, family, vendor, category, tags, description)
    VALUES (new.id, new.sku, new.name, new.family, new.vendor, new.category, new.tags, new.description);
END;

-- Примеры из обезличенных переписок службы поддержки: реальные формулировки и типовые ответы
CREATE TABLE IF NOT EXISTS support_examples (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_articles (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'manual',
    url TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    hits INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_candidates (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    dialog_id INTEGER,
    status TEXT NOT NULL DEFAULT 'new',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    phone TEXT NOT NULL DEFAULT '',
    email TEXT NOT NULL DEFAULT '',
    company TEXT NOT NULL DEFAULT '',
    inn TEXT NOT NULL DEFAULT '',
    kpp TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dialogs (
    id INTEGER PRIMARY KEY,
    public_id TEXT NOT NULL UNIQUE,
    channel TEXT NOT NULL DEFAULT 'web',
    external_id TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    customer_id INTEGER REFERENCES customers(id),
    status TEXT NOT NULL DEFAULT 'bot',
    escalation_reason TEXT NOT NULL DEFAULT '',
    escalation_summary TEXT NOT NULL DEFAULT '',
    escalated_at TEXT,
    escalation_notified_overdue INTEGER NOT NULL DEFAULT 0,
    operator_first_reply_at TEXT,
    was_escalated INTEGER NOT NULL DEFAULT 0,
    unread INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT '{}',
    llm_history TEXT NOT NULL DEFAULT '[]',
    llm_synced_msg_id INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_client_msg_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_dialogs_status ON dialogs(status);
CREATE INDEX IF NOT EXISTS ix_dialogs_external ON dialogs(channel, external_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    dialog_id INTEGER NOT NULL REFERENCES dialogs(id),
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_dialog ON messages(dialog_id, id);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    number TEXT NOT NULL,
    token TEXT NOT NULL UNIQUE,
    dialog_id INTEGER,
    customer_id INTEGER,
    quote_id INTEGER,
    buyer TEXT NOT NULL DEFAULT '{}',
    items TEXT NOT NULL,
    totals TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    valid_until TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_dialog ON documents(dialog_id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    dialog_id INTEGER,
    type TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(type, created_at);

-- Классификатор обращений: дополнительные размеченные примеры
-- (исправления менеджеров, загруженные файлы), справочник ключевых слов и сведения об обучении.
CREATE TABLE IF NOT EXISTS clf_training (
    id INTEGER PRIMARY KEY,
    head TEXT NOT NULL,                 -- request_type | it_topic
    text TEXT NOT NULL,
    category TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manager',
    dialog_id INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(head, text)
);
CREATE TABLE IF NOT EXISTS clf_keywords (
    id INTEGER PRIMARY KEY,
    head TEXT NOT NULL,
    category TEXT NOT NULL,
    keyword TEXT NOT NULL,
    strong INTEGER NOT NULL DEFAULT 0,
    UNIQUE(head, category, keyword)
);
CREATE TABLE IF NOT EXISTS clf_models (
    head TEXT PRIMARY KEY,
    info TEXT NOT NULL DEFAULT '{}',
    trained_at TEXT NOT NULL
);
"""

_init_lock = threading.Lock()
_initialized_for: str | None = None


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_settings().db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


# Колонки, добавленные после первых версий схемы: создаются в существующих базах автоматически
MIGRATIONS = {
    "support_examples": {"category": "TEXT NOT NULL DEFAULT ''", "page_url": "TEXT NOT NULL DEFAULT ''",
                         "external_id": "TEXT NOT NULL DEFAULT ''"},
    "products": {"org_type": "TEXT NOT NULL DEFAULT ''", "available": "INTEGER NOT NULL DEFAULT 1"},
    # Результат классификатора обращений: тип обращения и IT-тема (source: model | keyword | manager)
    "dialogs": {"request_type": "TEXT NOT NULL DEFAULT ''", "request_type_conf": "REAL NOT NULL DEFAULT 0",
                "request_type_source": "TEXT NOT NULL DEFAULT ''", "it_topic": "TEXT NOT NULL DEFAULT ''",
                "it_topic_conf": "REAL NOT NULL DEFAULT 0", "it_topic_source": "TEXT NOT NULL DEFAULT ''",
                "classification": "TEXT NOT NULL DEFAULT '{}'"},
    # Однозначное ключевое слово (бренд из облака слов): решает тему одно
    "clf_keywords": {"strong": "INTEGER NOT NULL DEFAULT 0"},
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, ddl in columns.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def init_db() -> None:
    global _initialized_for
    path = str(get_settings().db_path)
    with _init_lock:
        conn = _connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            _migrate(conn)
            conn.commit()
        finally:
            conn.close()
        _initialized_for = path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    if _initialized_for != str(get_settings().db_path):
        init_db()
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def query(sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def scalar(sql: str, params: tuple | dict = ()) -> Any:
    with connect() as conn:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None


def execute(sql: str, params: tuple | dict = ()) -> int:
    """Выполнить запрос, вернуть lastrowid."""
    with connect() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def log_event(event_type: str, dialog_id: int | None = None, **payload: Any) -> None:
    execute(
        "INSERT INTO events (dialog_id, type, payload, created_at) VALUES (?, ?, ?, ?)",
        (dialog_id, event_type, dumps(payload), now_iso()),
    )
