from __future__ import annotations

import copy
from typing import Any

import anthropic
import httpx
from anthropic.types.beta import BetaMessage

from app import db
from app.assistant import engine
from app.assistant.llm_agent import LLMAgent
from app.assistant.offline_agent import extract_contacts, extract_params
from app.services import catalog, dialogs, documents, knowledge


def say(dialog_id: int, text: str) -> str:
    out = engine.handle_client_message(dialog_id, text)
    return out[0]["text"] if out else ""


# ---------------- Офлайн-режим: сценарии задачи ----------------

def test_extract_params_and_contacts():
    rows = catalog.family_rows("Kaspersky Endpoint Security для бизнеса Стандартный")
    p = extract_params("продлить на 40 компьютеров на год, стандартный", rows)
    assert p == {"users": 40, "term_months": 12, "purchase_type": "renewal", "edition": "Стандартный"}
    assert extract_params("хотим российское, сервер свой")["russian"] is True
    # Расшифровки звонков: числа словами и обратный порядок
    assert extract_params("почтовый сервер, человек на тридцать")["users"] == 30
    assert extract_params("сто двадцать пять ПК на два года") == {"users": 125, "term_months": 24}
    c = extract_contacts("ООО Альфа, ИНН 1234567894, тел. +7 916 123-45-67, a@b.ru")
    assert c["company"] == "ООО «Альфа»" and c["phone"] == "+79161234567" and c["email"] == "a@b.ru"


def test_scenario_client_knows_product():
    d = dialogs.create("web")
    reply = say(d["id"], "Нужен MDaemon Standard")
    assert "Сколько пользователей" in reply and "новая лицензия или продление" in reply
    assert "На какой срок" not in reply  # срок у продукта один — лишних вопросов нет

    reply = say(d["id"], "на 30 пользователей, новая лицензия")
    assert "MD-STD-N-1Y-50" in reply and "79 900,00" in reply

    reply = say(d["id"], "а есть российский аналог?")
    assert "CommuniGate Pro" in reply and "RuPost" in reply and "store.softline.ru" in reply

    assert "название организации" in say(d["id"], "сделайте КП")
    reply = say(d["id"], "ООО Ромашка, a@romashka.ru")
    assert "Черновик коммерческого предложения" in reply and "2 мин" in reply
    doc = documents.list_documents(d["id"])[0]
    assert doc["status"] == "draft" and doc["buyer"]["company"] == "ООО «Ромашка»"


def test_scenario_client_does_not_know_product():
    d = dialogs.create("web")
    reply = say(d["id"], "Не знаю, что нужно, но хотим защитить почту и файлы на 50 сотрудников")
    assert reply.count("•") in (2, 3) and "российское ПО" in reply  # уточняющие вопросы, не больше трёх

    reply = say(d["id"], "да, российское, сервер свой")
    assert "Вариант 1" in reply and "Вариант 2" in reply
    assert "Dr.Web Mail Security Suite" in reply or "Kaspersky Security для почтовых серверов" in reply
    assert "MDaemon" not in reply  # требование «российское ПО» соблюдено

    reply = say(d["id"], "второй вариант")
    assert "Итого" in reply and "без НДС" in reply
    assert dialogs.get(d["id"])["state"]["cart"]


def test_scenario_renewal_deferral_and_discount():
    d = dialogs.create("web")
    reply = say(d["id"], "Продлить Касперский Стандартный на 40 компьютеров на год")
    assert "KESB-STD-R-1Y-25-49" in reply and "84 240,00" in reply

    reply = say(d["id"], "можно с отсрочкой 30 дней?")
    assert "предоплат" in reply
    assert dialogs.get(d["id"])["status"] == "bot"  # правило — ответ сразу, без менеджера

    reply = say(d["id"], "дайте скидку 10%")
    assert "5%" in reply and "80 028,00" in reply
    dlg = dialogs.get(d["id"])
    assert dlg["status"] == "waiting_operator" and dlg["escalation_reason"] == "special_terms"
    assert dlg["state"]["requested_discount_percent"] == 10

    # Менеджер согласует 10%: КП пересоздаётся, клиент получает ссылку, диалог возвращается ассистенту
    engine.approve_discount(d["id"], 10)
    dlg = dialogs.get(d["id"])
    assert dlg["status"] == "bot"
    last = dialogs.list_messages(d["id"])[-1]
    assert last["role"] == "operator" and "согласовал скидку 10%" in last["text"]
    doc = documents.list_documents(d["id"])[0]
    assert doc["totals"]["discount_percent"] == 10 and doc["status"] == "approved"
    assert engine.pending_decisions()["drafts"] == []  # старые черновики КП этого диалога отменены
    assert db.scalar("SELECT COUNT(*) FROM events WHERE type = 'manager_decision'") == 1


def test_questions_answered_from_kb_with_sources():
    d = dialogs.create("web")
    assert "lk.softline.ru" in say(d["id"], "Как я получу ключ?")
    assert "149 НК РФ" in say(d["id"], "Почему антивирус без НДС?")
    assert "не предусмотрен" in say(d["id"], "Можно вернуть лицензию?")
    reply = say(d["id"], "Расскажите про Dr.Web Mail Security Suite")
    assert "ФСТЭК" in reply and "drweb-mail-security-suite" in reply
    assert dialogs.get(d["id"])["status"] == "bot"


def test_escalations_and_manager_decline():
    d = dialogs.create("web")
    say(d["id"], "Позовите менеджера")
    assert dialogs.get(d["id"])["escalation_reason"] == "client_request"
    engine.decline_request(d["id"])
    assert dialogs.get(d["id"])["status"] == "bot"
    assert "стандартные" in dialogs.list_messages(d["id"])[-1]["text"]

    d2 = dialogs.create("web")
    say(d2["id"], "Нужно участвовать в тендере по 44-ФЗ")
    assert dialogs.get(d2["id"])["escalation_reason"] == "special_terms"


def test_document_approval_notifies_client():
    d = dialogs.create("web")
    say(d["id"], "Нужен MDaemon Standard на 12 пользователей, новая лицензия")
    say(d["id"], "КП")
    say(d["id"], "без контактов")
    doc = documents.list_documents(d["id"])[0]
    assert engine.pending_decisions()["drafts"][0]["id"] == doc["id"]
    engine.approve_document(doc["id"])
    assert documents.get(doc["id"])["status"] == "approved"
    assert "проверено менеджером" in dialogs.list_messages(d["id"])[-1]["text"]
    assert engine.pending_decisions()["drafts"] == []


def test_operator_answer_becomes_kb_candidate():
    d = dialogs.create("web")
    say(d["id"], "Можно ли купить лицензию на юрлицо из Казахстана?")
    assert dialogs.get(d["id"])["status"] == "waiting_operator"
    engine.operator_reply(d["id"], "Да, для нерезидентов оформляем по договору, менеджер пришлёт реквизиты.")
    assert engine.handle_client_message(d["id"], "Спасибо") == []  # диалог у менеджера — ассистент молчит
    cand = knowledge.list_candidates()[0]
    assert "Казахстана" in cand["question"]


def test_sla_overdue_notification():
    d = dialogs.create("web")
    say(d["id"], "Позовите менеджера")
    db.execute("UPDATE dialogs SET escalated_at = '2000-01-01 00:00:00' WHERE id = ?", (d["id"],))
    assert engine.check_sla() == 1 and engine.check_sla() == 0
    assert engine.pending_decisions()["requests"][0]["overdue"]


# ---------------- ИИ-режим (имитация API языковой модели, без сети) ----------------

def msg(content: list[dict[str, Any]], stop_reason: str) -> BetaMessage:
    return BetaMessage.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50,
                  "cache_creation_input_tokens": 0},
    })


class FakeClient:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.beta = self
        self.messages = self

    def create(self, **params: Any) -> BetaMessage:
        self.calls.append(copy.deepcopy(params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def use_fake(responses: list[Any], monkeypatch) -> FakeClient:
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "llm_api_key", "test-key")
    fake = FakeClient(responses)
    engine.set_llm_agent(LLMAgent(client=fake))
    return fake


def test_llm_tool_loop_select_variant_and_history(monkeypatch):
    fake = use_fake([
        msg([{"type": "thinking", "thinking": "", "signature": "sig-1"},
             {"type": "tool_use", "id": "tu_1", "name": "select_variant",
              "input": {"product": "MDaemon Standard"}}], "tool_use"),
        msg([{"type": "text", "text": "Уточните, на сколько пользователей и новая лицензия или продление?"}],
            "end_turn"),
        msg([{"type": "text", "text": "Спасибо!"}], "end_turn"),
    ], monkeypatch)
    d = dialogs.create("web")
    assert "сколько пользователей" in say(d["id"], "нужен MDaemon Standard")

    first, second = fake.calls[0], fake.calls[1]
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "5%" in first["system"][0]["text"] and "предоплата" in first["system"][0]["text"]
    assert {t["name"] for t in first["tools"]} >= {"select_variant", "find_alternatives", "search_support_examples"}
    assert second["messages"][1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    result = second["messages"][2]["content"][0]
    assert result["tool_use_id"] == "tu_1" and '"missing"' in result["content"] and not result["is_error"]

    engine.operator_reply(d["id"], "Менеджер на связи.")
    engine.return_to_bot(d["id"])
    say(d["id"], "Спасибо")
    third = fake.calls[2]["messages"]
    assert third[:4] == second["messages"] + [third[3]]  # история только дополняется
    assert "Менеджер: Менеджер на связи." in " ".join(b["text"] for b in third[4]["content"])


def test_llm_discount_over_limit_is_blocked_in_quote(monkeypatch):
    fake = use_fake([
        msg([{"type": "tool_use", "id": "tu_1", "name": "create_quote",
              "input": {"items": [{"sku": "MD-STD-N-1Y-50", "quantity": 1}], "discount_percent": 15}}], "tool_use"),
        msg([{"type": "tool_use", "id": "tu_2", "name": "escalate_to_manager",
              "input": {"reason": "special_terms", "summary": "Просит 15%", "requested_discount_percent": 15}}],
            "tool_use"),
        msg([{"type": "text", "text": "Скидку 15% согласует менеджер."}], "end_turn"),
    ], monkeypatch)
    d = dialogs.create("web")
    say(d["id"], "Дайте 15% и КП")
    blocked = fake.calls[1]["messages"][-1]["content"][0]
    assert blocked["is_error"] and "согласования" in blocked["content"]
    assert documents.list_documents(d["id"]) == []
    dlg = dialogs.get(d["id"])
    assert dlg["status"] == "waiting_operator" and dlg["state"]["requested_discount_percent"] == 15


def test_llm_refusal_and_outage(monkeypatch):
    use_fake([msg([], "refusal")], monkeypatch)
    d = dialogs.create("web")
    assert "менеджер" in say(d["id"], "что-то странное").lower()
    assert dialogs.get(d["id"])["status"] == "waiting_operator"

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    use_fake([anthropic.APIConnectionError(request=req)], monkeypatch)
    d2 = dialogs.create("web")
    assert "lk.softline.ru" in say(d2["id"], "Как я получу ключ?")
    assert dialogs.list_messages(d2["id"])[-1]["meta"]["mode"] == "offline"
