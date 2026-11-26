"""Метрики для руководителя: нагрузка, доля автоматизации, сэкономленные часы, повторяющиеся вопросы,
пробелы базы знаний, КП/счета, лиды и «зависшие» сделки."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from .. import db
from . import settings_store


def _since(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).replace(microsecond=0).isoformat(sep=" ")


def dashboard(days: int = 30) -> dict[str, Any]:
    since = _since(days)
    a = settings_store.get("assistant")
    sla = int(a["operator_sla_minutes"])

    total = db.scalar("SELECT COUNT(*) FROM dialogs WHERE created_at >= ? "
                      "AND EXISTS (SELECT 1 FROM messages m WHERE m.dialog_id = dialogs.id AND m.role='client')",
                      (since,)) or 0
    by_channel = db.query(
        "SELECT channel, COUNT(*) AS n FROM dialogs WHERE created_at >= ? GROUP BY channel ORDER BY n DESC", (since,))
    bot_only = db.scalar(
        "SELECT COUNT(*) FROM dialogs d WHERE d.created_at >= ? AND d.was_escalated = 0 "
        "AND EXISTS (SELECT 1 FROM messages m WHERE m.dialog_id = d.id AND m.role = 'bot')", (since,)) or 0
    escalated = db.scalar("SELECT COUNT(*) FROM dialogs WHERE created_at >= ? AND was_escalated = 1", (since,)) or 0
    reasons = db.query(
        "SELECT escalation_reason AS reason, COUNT(*) AS n FROM dialogs WHERE created_at >= ? AND was_escalated = 1 "
        "GROUP BY escalation_reason ORDER BY n DESC", (since,))

    bot_stats = db.query_one(
        "SELECT COUNT(*) AS n, AVG(json_extract(payload, '$.ms')) AS avg_ms FROM events "
        "WHERE type = 'bot_reply' AND created_at >= ?", (since,))
    llm_share = db.scalar(
        "SELECT AVG(CASE WHEN json_extract(payload, '$.mode') = 'llm' THEN 1.0 ELSE 0 END) FROM events "
        "WHERE type = 'bot_reply' AND created_at >= ?", (since,))
    op_wait = db.scalar("SELECT AVG(json_extract(payload, '$.wait_seconds')) FROM events "
                        "WHERE type = 'operator_first_reply' AND created_at >= ?", (since,))
    waiting_now = db.scalar("SELECT COUNT(*) FROM dialogs WHERE status = 'waiting_operator'") or 0
    overdue_now = db.scalar("SELECT COUNT(*) FROM dialogs WHERE status = 'waiting_operator' AND escalated_at < ?",
                            (_since(0) if sla == 0 else (datetime.now() - timedelta(minutes=sla))
                             .replace(microsecond=0).isoformat(sep=" "),)) or 0

    kb_hits = db.scalar("SELECT COUNT(*) FROM events WHERE type = 'kb_hit' AND created_at >= ?", (since,)) or 0
    kb_misses = db.scalar("SELECT COUNT(*) FROM events WHERE type = 'kb_miss' AND created_at >= ?", (since,)) or 0
    top_questions = db.query(
        "SELECT a.id, a.question, COUNT(*) AS n FROM events e JOIN kb_articles a "
        "ON a.id = json_extract(e.payload, '$.article_id') WHERE e.type = 'kb_hit' AND e.created_at >= ? "
        "GROUP BY a.id ORDER BY n DESC LIMIT 10", (since,))
    unanswered = db.query(
        "SELECT json_extract(payload, '$.query') AS query, MAX(created_at) AS last_at, COUNT(*) AS n, "
        "MAX(dialog_id) AS dialog_id FROM events WHERE type = 'kb_miss' AND created_at >= ? "
        "GROUP BY lower(json_extract(payload, '$.query')) ORDER BY last_at DESC LIMIT 15", (since,))

    docs = {r["kind"]: r for r in db.query(
        "SELECT kind, COUNT(*) AS n, COALESCE(SUM(json_extract(totals, '$.total')), 0) AS amount "
        "FROM documents WHERE created_at >= ? GROUP BY kind", (since,))}
    paid = db.query_one(
        "SELECT COUNT(*) AS n, COALESCE(SUM(json_extract(totals, '$.total')), 0) AS amount FROM documents "
        "WHERE kind = 'invoice' AND status = 'paid' AND created_at >= ?", (since,))
    dialogs_with_docs = db.scalar(
        "SELECT COUNT(DISTINCT dialog_id) FROM documents WHERE created_at >= ? AND dialog_id IS NOT NULL",
        (since,)) or 0
    leads = db.scalar("SELECT COUNT(*) FROM customers WHERE created_at >= ?", (since,)) or 0

    quotes_n = docs.get("quote", {}).get("n", 0)
    invoices_n = docs.get("invoice", {}).get("n", 0)
    saved_minutes = bot_only * float(a["minutes_per_dialog"]) + (quotes_n + invoices_n) * float(a["minutes_per_document"])

    usage = db.query_one(
        "SELECT COALESCE(SUM(json_extract(meta, '$.usage.input_tokens')), 0) AS input_tokens, "
        "COALESCE(SUM(json_extract(meta, '$.usage.output_tokens')), 0) AS output_tokens, "
        "COALESCE(SUM(json_extract(meta, '$.usage.cache_read_input_tokens')), 0) AS cache_read "
        "FROM messages WHERE role = 'bot' AND created_at >= ?", (since,))

    # Решения менеджера: скорость относительно SLA
    dec = db.query_one(
        "SELECT COUNT(*) AS n, AVG(json_extract(payload, '$.seconds')) AS avg_s, "
        "SUM(CASE WHEN json_extract(payload, '$.seconds') <= ? THEN 1 ELSE 0 END) AS in_sla "
        "FROM events WHERE type = 'manager_decision' AND created_at >= ?", (sla * 60, since))

    # Экономика: стоимость ИИ по фактическим токенам и сравнение с работой сотрудников
    econ = settings_store.get("economics")
    usd = ((usage["input_tokens"] or 0) * float(econ["llm_input_usd_per_mtok"])
           + (usage["output_tokens"] or 0) * float(econ["llm_output_usd_per_mtok"])
           + (usage["cache_read"] or 0) * float(econ["llm_cache_read_usd_per_mtok"])) / 1_000_000
    llm_dialogs = db.scalar(
        "SELECT COUNT(DISTINCT dialog_id) FROM events WHERE type = 'bot_reply' AND created_at >= ? "
        "AND json_extract(payload, '$.mode') = 'llm'", (since,)) or 0
    per_dialog_rub = (usd / llm_dialogs * float(econ["usd_rub"])) if llm_dialogs else 0
    plan_month = int(econ["dialogs_per_month_plan"])
    human_month_rub = plan_month * float(a["minutes_per_dialog"]) / 60 * float(econ["operator_hour_cost_rub"])
    economics = {
        "llm_usd": usd,
        "llm_rub": usd * float(econ["usd_rub"]),
        "per_dialog_rub": per_dialog_rub,
        "plan_month_dialogs": plan_month,
        "plan_month_llm_rub": per_dialog_rub * plan_month,
        "plan_month_human_rub": human_month_rub,
        "plan_month_human_hours": plan_month * float(a["minutes_per_dialog"]) / 60,
    }

    return {
        "decisions": dec["n"] if dec else 0,
        "avg_decision_seconds": (dec["avg_s"] or 0) if dec else 0,
        "decisions_in_sla": (dec["in_sla"] or 0) / dec["n"] if dec and dec["n"] else 0,
        "economics": economics,
        "days": days,
        "dialogs_total": total,
        "by_channel": by_channel,
        "bot_only": bot_only,
        "bot_only_share": bot_only / total if total else 0,
        "escalated": escalated,
        "escalated_share": escalated / total if total else 0,
        "reasons": reasons,
        "bot_replies": bot_stats["n"] if bot_stats else 0,
        "avg_bot_seconds": (bot_stats["avg_ms"] or 0) / 1000 if bot_stats else 0,
        "llm_share": llm_share or 0,
        "avg_operator_wait_min": (op_wait or 0) / 60,
        "waiting_now": waiting_now,
        "overdue_now": overdue_now,
        "sla_minutes": sla,
        "kb_hits": kb_hits,
        "kb_misses": kb_misses,
        "kb_hit_rate": kb_hits / (kb_hits + kb_misses) if (kb_hits + kb_misses) else 0,
        "top_questions": top_questions,
        "unanswered": unanswered,
        "quotes": quotes_n,
        "quotes_amount": docs.get("quote", {}).get("amount", 0),
        "invoices": invoices_n,
        "invoices_amount": docs.get("invoice", {}).get("amount", 0),
        "paid": paid["n"] if paid else 0,
        "paid_amount": paid["amount"] if paid else 0,
        "conversion_to_docs": dialogs_with_docs / total if total else 0,
        "leads": leads,
        "saved_hours": saved_minutes / 60,
        "usage": usage,
        "daily": daily_series(min(days, 30)),
        "follow_ups": follow_ups(),
    }


def daily_series(days: int = 14) -> list[dict[str, Any]]:
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    rows = {r["day"]: r for r in db.query(
        "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS dialogs, SUM(was_escalated) AS escalated "
        "FROM dialogs WHERE created_at >= ? GROUP BY day", (since,))}
    out = []
    for i in range(days):
        day = (date.today() - timedelta(days=days - 1 - i)).isoformat()
        r = rows.get(day, {})
        out.append({"day": day, "dialogs": r.get("dialogs", 0) or 0, "escalated": r.get("escalated", 0) or 0})
    return out


def follow_ups() -> list[dict[str, Any]]:
    """Сделки, которые могут потеряться: КП без счёта > 2 дней, неоплаченные счета с истёкшим сроком."""
    border = _since(2)
    today = date.today().isoformat()
    rows = db.query(
        "SELECT d.*, c.name AS customer_name, c.company AS customer_company, c.phone AS customer_phone, "
        "c.email AS customer_email FROM documents d LEFT JOIN customers c ON c.id = d.customer_id "
        "WHERE (d.kind = 'quote' AND d.created_at < ? AND d.status IN ('draft', 'approved', 'sent') AND NOT EXISTS ("
        "   SELECT 1 FROM documents i WHERE i.kind = 'invoice' AND (i.quote_id = d.id OR "
        "   (i.dialog_id = d.dialog_id AND i.created_at > d.created_at)))) "
        "OR (d.kind = 'invoice' AND d.status IN ('draft', 'approved', 'sent') AND d.valid_until < ?) "
        "ORDER BY d.created_at DESC LIMIT 30", (border, today))
    for r in rows:
        r["totals"] = db.loads(r["totals"], {})
        r["problem"] = "КП без счёта" if r["kind"] == "quote" else "Счёт не оплачен в срок"
    return rows
