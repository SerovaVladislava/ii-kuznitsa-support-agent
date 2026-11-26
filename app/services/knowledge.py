"""База знаний: статьи «вопрос — ответ», поиск, учёт популярности и пополнение из ответов менеджеров."""
from __future__ import annotations

import csv
import io
from typing import Any

from .. import db
from .search import TextIndex


def _load_docs() -> list[tuple[int, str]]:
    rows = db.query("SELECT id, question, answer, tags FROM kb_articles WHERE active = 1")
    # Индексируем вопрос и теги: длинный текст ответа размывает релевантность коротких вопросов
    return [(r["id"], f"{r['question']} {r['tags']}") for r in rows]


index = TextIndex(_load_docs, char_weight=0.5)


def search(query: str, top_k: int = 3, dialog_id: int | None = None, track: bool = True) -> list[dict[str, Any]]:
    hits = index.search(query, top_k=top_k)
    if not hits:
        if track:
            db.log_event("kb_miss", dialog_id, query=query[:500])
        return []
    by_id = {r["id"]: r for r in db.query(
        f"SELECT id, question, answer, tags, url FROM kb_articles WHERE id IN ({','.join('?' * len(hits))})",
        tuple(i for i, _ in hits),
    )}
    results = []
    for art_id, score in hits:
        if art_id in by_id:
            results.append({**by_id[art_id], "score": round(score, 3)})
    return results


def register_hit(article_id: int, dialog_id: int | None, query: str, score: float) -> None:
    db.execute("UPDATE kb_articles SET hits = hits + 1 WHERE id = ?", (article_id,))
    db.log_event("kb_hit", dialog_id, article_id=article_id, query=query[:500], score=round(score, 3))


def register_miss(dialog_id: int | None, query: str) -> None:
    db.log_event("kb_miss", dialog_id, query=query[:500])


def list_articles(search_text: str = "") -> list[dict[str, Any]]:
    if search_text:
        ids = [i for i, _ in index.search(search_text, top_k=50)]
        if not ids:
            return []
        rows = {r["id"]: r for r in db.query(
            f"SELECT * FROM kb_articles WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))}
        return [rows[i] for i in ids if i in rows]
    return db.query("SELECT * FROM kb_articles ORDER BY active DESC, hits DESC, id DESC")


def get_article(article_id: int) -> dict[str, Any] | None:
    return db.query_one("SELECT * FROM kb_articles WHERE id = ?", (article_id,))


def save_article(question: str, answer: str, tags: str = "", source: str = "manual",
                 article_id: int | None = None, active: bool = True, url: str = "") -> int:
    now = db.now_iso()
    if article_id:
        db.execute(
            "UPDATE kb_articles SET question=?, answer=?, tags=?, url=?, active=?, updated_at=? WHERE id=?",
            (question.strip(), answer.strip(), tags.strip(), url.strip(), int(active), now, article_id),
        )
    else:
        article_id = db.execute(
            "INSERT INTO kb_articles (question, answer, tags, source, url, active, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (question.strip(), answer.strip(), tags.strip(), source, url.strip(), int(active), now, now),
        )
    index.invalidate()
    return article_id


def delete_article(article_id: int) -> None:
    db.execute("DELETE FROM kb_articles WHERE id = ?", (article_id,))
    index.invalidate()


def import_csv(content: bytes) -> int:
    """CSV с колонками question;answer;tags (разделитель ; или ,), кодировка UTF-8 или cp1251."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1251")
    dialect = csv.Sniffer().sniff(text[:2000], delimiters=";,\t")
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    count = 0
    for row in reader:
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        q = row.get("question") or row.get("вопрос")
        a = row.get("answer") or row.get("ответ")
        if q and a:
            save_article(q, a, row.get("tags") or row.get("теги") or "", source="import")
            count += 1
    return count


# --- Кандидаты: ответы менеджеров, которые стоит добавить в базу знаний ---

def add_candidate(question: str, answer: str, dialog_id: int | None) -> int | None:
    question, answer = question.strip(), answer.strip()
    if len(question) < 8 or len(answer) < 15:
        return None
    return db.execute(
        "INSERT INTO kb_candidates (question, answer, dialog_id, created_at) VALUES (?, ?, ?, ?)",
        (question[:2000], answer[:4000], dialog_id, db.now_iso()),
    )


def list_candidates(status: str = "new") -> list[dict[str, Any]]:
    return db.query("SELECT * FROM kb_candidates WHERE status = ? ORDER BY id DESC", (status,))


def resolve_candidate(candidate_id: int, approve: bool, question: str = "", answer: str = "",
                      tags: str = "") -> int | None:
    cand = db.query_one("SELECT * FROM kb_candidates WHERE id = ?", (candidate_id,))
    if not cand:
        return None
    article_id = None
    if approve:
        article_id = save_article(question or cand["question"], answer or cand["answer"], tags, source="operator")
    db.execute("UPDATE kb_candidates SET status = ? WHERE id = ?",
               ("approved" if approve else "rejected", candidate_id))
    return article_id
