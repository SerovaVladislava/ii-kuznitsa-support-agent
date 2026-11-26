"""Оркестратор: принимает сообщения из всех каналов, выбирает ассистента (ИИ или офлайн),
передаёт диалоги менеджерам, доставляет ответы и следит за SLA."""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from .. import db
from ..config import get_settings
from ..services import classifier, customers, dialogs, knowledge, settings_store
from ..services.notify import notify_managers, send_email, send_telegram
from ..services.textutil import fmt_money
from .llm_agent import AgentReply, LLMAgent, LLMUnavailable
from .offline_agent import OfflineAgent
from .tools import ToolContext

log = logging.getLogger("engine")

_dialog_locks: defaultdict[int, threading.Lock] = defaultdict(threading.Lock)
_offline = OfflineAgent()
_llm_agent: LLMAgent | None = None
_llm_lock = threading.Lock()


def get_llm_agent() -> LLMAgent | None:
    global _llm_agent
    if not get_settings().llm_enabled:
        return None
    with _llm_lock:
        if _llm_agent is None:
            _llm_agent = LLMAgent()
        return _llm_agent


def set_llm_agent(agent: LLMAgent | None) -> None:
    """Для тестов и смены конфигурации."""
    global _llm_agent
    _llm_agent = agent


def assistant_mode() -> str:
    return "llm" if get_settings().llm_enabled else "offline"


# ---------------- Клиентские сообщения ----------------

def handle_client_message(dialog_id: int, text: str, meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Сохранить сообщение клиента и, если диалог ведёт ассистент, получить ответ.

    Возвращает новые сообщения для клиента (ответ ассистента), которые канал должен доставить.
    """
    text = (text or "").strip()[:4000]
    if not text:
        return []
    with _dialog_locks[dialog_id]:
        dialog = dialogs.get(dialog_id)
        if dialog["status"] == "closed":
            dialogs.set_status(dialog_id, "bot")
            dialog["status"] = "bot"
        client_msg = dialogs.add_message(dialog_id, "client", text, meta)
        try:
            classifier.classify_dialog(dialog_id)
        except Exception:  # noqa: BLE001 — классификация не должна мешать ответу клиенту
            log.exception("Сбой классификатора обращений")
        if dialog["status"] == "operator":
            return []  # диалог ведёт менеджер — ассистент молчит

        started = time.monotonic()
        ctx = ToolContext(dialog_id=dialog_id, channel=dialog["channel"])
        reply: AgentReply | None = None
        agent = get_llm_agent()
        if agent is not None:
            try:
                reply = agent.reply(dialog_id, client_msg["id"], text, ctx)
            except LLMUnavailable as exc:
                if exc.reset_history:
                    dialogs.save_llm_history(dialog_id, [], 0)
                db.log_event("llm_fallback", dialog_id, error=str(exc)[:300])
            except Exception:  # noqa: BLE001 — клиент не должен остаться без ответа
                log.exception("Сбой ИИ-ассистента, переключаюсь на офлайн-режим")
                db.log_event("llm_fallback", dialog_id, error="unexpected")
        if reply is None:
            try:
                reply = _offline.reply(dialog_id, text, ctx)
            except Exception:  # noqa: BLE001
                log.exception("Сбой офлайн-ассистента")
                ctx.escalation = ("error", f"Сбой ассистента. Сообщение клиента: {text[:500]}")
                reply = AgentReply(text="Секунду, подключаю менеджера.", mode="offline", escalation=ctx.escalation)

        for doc in reply.documents:
            if doc["url"] not in reply.text:
                reply.text += f"\n\n{doc['title']} № {doc['number']}: {doc['url']}"
        elapsed_ms = int((time.monotonic() - started) * 1000)
        bot_msg = dialogs.add_message(dialog_id, "bot", reply.text, {
            "mode": reply.mode,
            "ms": elapsed_ms,
            "kb_used": reply.kb_used,
            "tools": reply.tools_used,
            "documents": [{"number": d["number"], "url": d["url"], "title": d["title"], "id": d["id"]}
                          for d in reply.documents],
            "usage": reply.usage,
            "escalation": list(reply.escalation) if reply.escalation else None,
        })
        db.log_event("bot_reply", dialog_id, ms=elapsed_ms, mode=reply.mode)

        if reply.escalation:
            reason, summary = reply.escalation
            if dialogs.escalate(dialog_id, reason, summary):
                current = dialogs.get(dialog_id)
                cust = customers.display_name(customers.get(current["customer_id"]))
                kind = classifier.summary_line(current.get("classification"))
                notify_managers(f"Нужен менеджер: {dialogs.ESCALATION_REASONS.get(reason, reason)}",
                                f"Канал: {dialogs.CHANNELS.get(dialog['channel'], dialog['channel'])}\n"
                                f"Клиент: {cust or 'контакты не оставлены'}\n"
                                + (f"{kind}\n" if kind else "") + summary, dialog_id)
        return [bot_msg]


# ---------------- Действия менеджера ----------------

def deliver_to_client(dialog: dict[str, Any], text: str) -> bool:
    """Отправить сообщение клиенту в его канал. Веб-виджет забирает сообщения сам (polling)."""
    channel, target = dialog["channel"], dialog["external_id"]
    if channel == "email":
        company = settings_store.get("company")
        subject = dialog["subject"] or "Ваш запрос"
        subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        return send_email(target, subject, f"{text}\n\n--\n{company['name']}\n{company['phone']}")
    if channel == "telegram":
        return send_telegram(target, text)
    return True


def _pending_client_question(dialog_id: int) -> str:
    """Вопрос, на который отвечает менеджер: сообщение, из-за которого ассистент передал диалог,
    и всё, что клиент написал после него (но не раньше последнего ответа менеджера)."""
    msgs = dialogs.list_messages(dialog_id)
    last_operator = max((m["id"] for m in msgs if m["role"] == "operator"), default=0)
    recent = [m for m in msgs if m["id"] > last_operator]
    start = 0
    for i, m in enumerate(recent):
        if m["role"] == "bot" and m["meta"].get("escalation"):
            start = i - 1 if i and recent[i - 1]["role"] == "client" else i
    questions = [m["text"] for m in recent[start:] if m["role"] == "client"]
    if not questions:
        questions = [m["text"] for m in recent if m["role"] == "client"][-1:]
    return "\n".join(questions[-3:])


def operator_reply(dialog_id: int, text: str, operator: str = "Менеджер", feed_kb: bool = True) -> dict[str, Any]:
    text = text.strip()
    with _dialog_locks[dialog_id]:
        dialog = dialogs.get(dialog_id)
        question = _pending_client_question(dialog_id)
        msg = dialogs.add_message(dialog_id, "operator", text, {"operator": operator})
        if dialog["escalated_at"] and not dialog["operator_first_reply_at"]:
            wait = (datetime.now() - datetime.fromisoformat(dialog["escalated_at"])).total_seconds()
            db.execute("UPDATE dialogs SET operator_first_reply_at = ? WHERE id = ?", (db.now_iso(), dialog_id))
            db.log_event("operator_first_reply", dialog_id, wait_seconds=int(wait))
        if dialog["status"] in ("bot", "waiting_operator", "closed"):
            dialogs.set_status(dialog_id, "operator")
        dialogs.mark_read(dialog_id)
    deliver_to_client(dialog, text)
    if question and feed_kb:
        knowledge.add_candidate(question, text, dialog_id)
    return msg


def take_over(dialog_id: int) -> None:
    dialogs.set_status(dialog_id, "operator")
    dialogs.mark_read(dialog_id)


def return_to_bot(dialog_id: int) -> None:
    dialogs.set_status(dialog_id, "bot")
    dialogs.mark_read(dialog_id)


def close_dialog(dialog_id: int) -> None:
    dialog = dialogs.get(dialog_id)
    dialogs.set_status(dialog_id, "closed")
    dialogs.mark_read(dialog_id)
    db.log_event("dialog_closed", dialog_id, resolved_by="operator" if dialog["was_escalated"] else "bot")


def operator_draft(dialog_id: int) -> dict[str, Any]:
    """Черновик ответа для менеджера: ИИ (если доступен) или подборка из базы знаний и каталога."""
    agent = get_llm_agent()
    if agent is not None:
        try:
            text = agent.draft_for_operator(dialog_id)
            if text:
                return {"mode": "llm", "text": text}
        except LLMUnavailable:
            pass
        except Exception:  # noqa: BLE001
            log.exception("Сбой черновика ИИ")
    question = _pending_client_question(dialog_id) or next(
        (m["text"] for m in reversed(dialogs.list_messages(dialog_id)) if m["role"] == "client"), "")
    suggestions = [r for r in knowledge.search(question, top_k=3, track=False) if r["score"] >= 0.2]
    if suggestions:
        return {"mode": "offline", "text": suggestions[0]["answer"],
                "alternatives": [s["answer"] for s in suggestions[1:]]}
    return {"mode": "offline", "text": "", "alternatives": []}


# ---------------- Решения менеджера («Ждут решения») ----------------

def _seconds_since(ts: str | None) -> int:
    if not ts:
        return 0
    return max(0, int((datetime.now() - datetime.fromisoformat(ts)).total_seconds()))


def pending_decisions() -> dict[str, Any]:
    """Карточки для менеджера: диалоги, ждущие решения, и черновики КП/счетов на проверке."""
    from ..services import documents, pricing
    sla = int(settings_store.get("assistant")["operator_sla_minutes"]) * 60
    requests = []
    for d in dialogs.list_dialogs("waiting_operator", limit=100):
        state = d["state"]
        cart = state.get("cart") or []
        calc = pricing.calculate(cart, state.get("discount_percent"),
                                 state.get("approved_discount_percent", 0)) if cart else None
        waited = _seconds_since(d["escalated_at"])
        requests.append({**d, "calc": calc, "waited": waited, "overdue": waited > sla,
                         "requested_discount": state.get("requested_discount_percent"),
                         "focus": (state.get("focus") or {}).get("family", "")})
    drafts = []
    for doc in documents.list_documents(limit=200):
        if doc["status"] == "draft":
            waited = _seconds_since(doc["created_at"])
            drafts.append({**doc, "waited": waited, "overdue": waited > sla})
    return {"requests": requests, "drafts": drafts, "sla_seconds": sla}


def _log_decision(dialog_id: int | None, kind: str, since: str | None, **extra: Any) -> None:
    db.log_event("manager_decision", dialog_id, kind=kind, seconds=_seconds_since(since), **extra)


def approve_discount(dialog_id: int, percent: float, operator: str = "Менеджер") -> dict[str, Any]:
    """Менеджер согласовал скидку: она применяется в расчёте, КП пересоздаётся, диалог возвращается ассистенту."""
    from ..services import documents, pricing
    dialog = dialogs.get(dialog_id)
    dialogs.update_state(dialog_id, approved_discount_percent=percent, discount_percent=percent,
                         requested_discount_percent=None)
    text = f"Менеджер согласовал скидку {percent:g}%."
    cart = dialog["state"].get("cart") or []
    if cart:
        calc = pricing.calculate(cart, percent, percent)
        if calc["lines"]:
            for old in documents.list_documents(dialog_id):  # прежние черновики КП заменяются новым
                if old["kind"] == "quote" and old["status"] == "draft":
                    documents.set_status(old["id"], "cancelled")
            c = customers.get(dialog["customer_id"])
            buyer = {k: c[k] for k in customers.FIELDS} if c else {}
            doc = documents.create_quote(calc, buyer, dialog_id, dialog["customer_id"])
            documents.set_status(doc["id"], "approved")
            dialogs.update_state(dialog_id, last_quote=doc["number"])
            text += (f" Обновлённое коммерческое предложение № {doc['number']} на "
                     f"{fmt_money(calc['total'])} ₽: {doc['url']}")
    msg = operator_reply(dialog_id, text, operator, feed_kb=False)
    _log_decision(dialog_id, "discount_approved", dialog["escalated_at"], percent=percent)
    return_to_bot(dialog_id)
    return msg


def decline_request(dialog_id: int, text: str = "", operator: str = "Менеджер") -> dict[str, Any]:
    dialog = dialogs.get(dialog_id)
    text = text.strip() or ("К сожалению, не можем согласовать эти условия. Готовы предложить стандартные: "
                            f"скидка до {settings_store.get('pricing').get('max_agent_discount_percent', 5):g}% "
                            "и 100% предоплата.")
    msg = operator_reply(dialog_id, text, operator, feed_kb=False)
    dialogs.update_state(dialog_id, requested_discount_percent=None)
    _log_decision(dialog_id, "declined", dialog["escalated_at"])
    return_to_bot(dialog_id)
    return msg


def approve_document(doc_id: int, operator: str = "Менеджер") -> None:
    from ..services import documents
    doc = documents.get(doc_id)
    if not doc or doc["status"] != "draft":
        return
    documents.set_status(doc_id, "approved")
    _log_decision(doc["dialog_id"], "document_approved", doc["created_at"], document_id=doc_id)
    if doc["dialog_id"]:
        what = "Коммерческое предложение" if doc["kind"] == "quote" else "Счёт"
        text = f"{what} № {doc['number']} проверено менеджером — можно оплачивать: {doc['url']}"
        dialog = dialogs.get(doc["dialog_id"])
        dialogs.add_message(doc["dialog_id"], "operator", text, {"operator": operator, "auto": True})
        deliver_to_client(dialog, text)


# ---------------- SLA ----------------

def check_sla() -> int:
    """Напомнить менеджерам о диалогах, где клиент ждёт дольше SLA. Возвращает число просрочек."""
    sla = int(settings_store.get("assistant")["operator_sla_minutes"])
    border = (datetime.now() - timedelta(minutes=sla)).replace(microsecond=0).isoformat(sep=" ")
    overdue = db.query(
        "SELECT id, escalation_summary FROM dialogs WHERE status = 'waiting_operator' "
        "AND escalation_notified_overdue = 0 AND escalated_at < ?", (border,))
    for d in overdue:
        db.execute("UPDATE dialogs SET escalation_notified_overdue = 1 WHERE id = ?", (d["id"],))
        db.log_event("sla_overdue", d["id"])
        notify_managers(f"Клиент ждёт ответа дольше {sla} мин", d["escalation_summary"][:1000], d["id"])
    return len(overdue)


def start_sla_monitor(stop: threading.Event, interval: int = 60) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(interval):
            try:
                check_sla()
            except Exception:  # noqa: BLE001
                log.exception("Ошибка проверки SLA")

    thread = threading.Thread(target=loop, name="sla-monitor", daemon=True)
    thread.start()
    return thread
