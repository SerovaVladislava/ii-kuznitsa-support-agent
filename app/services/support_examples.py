"""Обезличенные переписки службы поддержки: реальные формулировки клиентов и типовые ответы.

Используются ассистентом как примеры («как обычно спрашивают и что отвечает поддержка») и как
источник кандидатов в базу знаний. Формат файла заранее неизвестен, поэтому импорт гибкий:
- JSON: [{"messages": [{"role"/"author": ..., "text": ...}, ...]}, ...] или {"dialogs": [...]}
- CSV/XLSX: колонки dialog_id (chat_id, ид диалога), role/author (кто), text/message (текст)
- TXT: строки вида «Клиент: ...» / «Оператор: ...», диалоги разделены пустой строкой или «---»
- TXT выгрузки чатов магазина: «#103038 [Категория] [страница сайта] реплики клиента» — только вопросы
  клиентов с разметкой категории (без ответов): используются как реальные формулировки, статистика спроса
  и проверочный набор для оценки агента (scripts/evaluate_chats.py)
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Iterable

from .. import db
from .search import TextIndex
from .textutil import normalize

CLIENT_ROLES = re.compile(r"client|customer|user|клиент|покупател|посетител|гость|заказчик")
PII_RE = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}"), "[телефон]"),
    (re.compile(r"(?<!\d)\d{10}(?:\d{2})?(?!\d)"), "[ИНН]"),
]


def _load_docs() -> list[tuple[int, str]]:
    return [(r["id"], r["question"]) for r in db.query("SELECT id, question FROM support_examples")]


index = TextIndex(_load_docs, char_weight=0.5)


def scrub(text: str) -> str:
    """Дополнительное обезличивание: почта, телефоны, ИНН."""
    for rx, repl in PII_RE:
        text = rx.sub(repl, text)
    return text.strip()


def add(question: str, answer: str, source: str = "", category: str = "", page_url: str = "",
        external_id: str = "") -> None:
    if external_id and db.scalar("SELECT 1 FROM support_examples WHERE external_id = ? AND source = ?",
                                 (external_id, source)):
        return  # повторная загрузка того же файла не дублирует обращения
    db.execute("INSERT INTO support_examples (question, answer, source, category, page_url, external_id, created_at) "
               "VALUES (?, ?, ?, ?, ?, ?, ?)",
               (scrub(question)[:2000], scrub(answer)[:4000], source, category, page_url, external_id, db.now_iso()))
    index.invalidate()


CHAT_LINE = re.compile(r"^#(\d+)\s+\[([^\]]+)\]\s+\[([^\]]*)\]\s*(.+)$")


def parse_chat_export(text: str) -> list[dict[str, str]]:
    """Выгрузка чатов магазина: одна строка — одно обращение (номер, категория, страница, текст клиента)."""
    out = []
    for line in text.splitlines():
        m = CHAT_LINE.match(line.strip())
        if m:
            out.append({"external_id": m.group(1), "category": m.group(2).strip(),
                        "page_url": m.group(3).strip(), "question": m.group(4).strip()})
    return out


def categories() -> list[dict[str, Any]]:
    return db.query("SELECT category, COUNT(*) AS n FROM support_examples WHERE category != '' "
                    "GROUP BY category ORDER BY n DESC")


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    hits = index.search(query, top_k=top_k)
    if not hits:
        return []
    rows = {r["id"]: r for r in db.query(
        f"SELECT * FROM support_examples WHERE id IN ({','.join('?' * len(hits))})", tuple(i for i, _ in hits))}
    return [{**rows[i], "score": round(s, 3)} for i, s in hits if i in rows]


def count() -> int:
    return db.scalar("SELECT COUNT(*) FROM support_examples") or 0


def _pairs(dialog: list[tuple[bool, str]]) -> Iterable[tuple[str, str]]:
    """(клиент?, текст) -> пары «вопросы клиента подряд» -> «ответы поддержки подряд»."""
    q: list[str] = []
    a: list[str] = []
    for is_client, text in dialog:
        if not text.strip():
            continue
        if is_client:
            if a and q:
                yield " ".join(q), " ".join(a)
                q, a = [], []
            elif a:
                a = []
            q.append(text.strip())
        elif q:
            a.append(text.strip())
    if q and a:
        yield " ".join(q), " ".join(a)


def _from_json(data: Any) -> Iterable[list[tuple[bool, str]]]:
    dialogs = data.get("dialogs") or data.get("chats") or data.get("conversations") if isinstance(data, dict) else data
    for d in dialogs or []:
        msgs = d.get("messages") if isinstance(d, dict) else d
        out = []
        for m in msgs or []:
            role = str(m.get("role") or m.get("author") or m.get("from") or m.get("sender") or "")
            text = str(m.get("text") or m.get("message") or m.get("content") or "")
            out.append((bool(CLIENT_ROLES.search(normalize(role))), text))
        yield out


def _from_rows(rows: list[list[Any]]) -> Iterable[list[tuple[bool, str]]]:
    header = [normalize(str(h or "")) for h in rows[0]]

    def col(*names: str) -> int | None:
        for i, h in enumerate(header):
            if any(n in h for n in names):
                return i
        return None

    c_dialog = col("dialog", "chat", "диалог", "чат", "conversation", "ticket", "обращ")
    c_role = col("role", "author", "sender", "from", "автор", "отправ", "роль", "кто")
    c_text = col("text", "message", "сообщ", "текст", "content")
    if c_role is None or c_text is None:
        raise ValueError("Нужны колонки с автором сообщения (role/автор) и текстом (text/сообщение)")
    current, key = [], object()
    for row in rows[1:]:
        k = row[c_dialog] if c_dialog is not None and c_dialog < len(row) else None
        if current and k != key:
            yield current
            current = []
        key = k
        current.append((bool(CLIENT_ROLES.search(normalize(str(row[c_role] or "")))), str(row[c_text] or "")))
    if current:
        yield current


def _from_text(text: str) -> Iterable[list[tuple[bool, str]]]:
    for block in re.split(r"\n\s*(?:-{3,}|={3,})?\s*\n", text):
        out = []
        for line in block.splitlines():
            m = re.match(r"\s*([^:]{2,25}):\s*(.+)", line)
            if m:
                out.append((bool(CLIENT_ROLES.search(normalize(m.group(1)))), m.group(2)))
            elif out:
                out[-1] = (out[-1][0], out[-1][1] + " " + line.strip())
        if out:
            yield out


def import_file(filename: str, content: bytes) -> int:
    name = filename.lower()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1251")
    chats = parse_chat_export(text) if not name.endswith((".json", ".csv", ".xlsx", ".xlsm")) else []
    if chats:
        # Реальные обращения заменяют синтетические демо-примеры
        db.execute("DELETE FROM support_examples WHERE source = 'демо-примеры'")
        index.invalidate()
        for c in chats:
            add(c["question"], "", source=filename, category=c["category"], page_url=c["page_url"],
                external_id=c["external_id"])
        return len(chats)
    if name.endswith(".json"):
        dialogs = _from_json(json.loads(text))
    elif name.endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        ws = load_workbook(io.BytesIO(content), read_only=True, data_only=True).active
        dialogs = _from_rows([list(r) for r in ws.iter_rows(values_only=True)])
    elif name.endswith(".csv"):
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=";,\t")
        dialogs = _from_rows(list(csv.reader(io.StringIO(text), dialect=dialect)))
    else:
        dialogs = _from_text(text)
    n = 0
    for dialog in dialogs:
        for q, a in _pairs(dialog):
            if len(q) >= 5 and len(a) >= 10:
                add(q, a, source=filename)
                n += 1
    return n
