"""Инструменты ассистента — единственный источник фактов (продукты, цены, условия) и действий.

Общие для ИИ-режима (языковая модель вызывает их через tool use) и офлайн-режима (вызываются правилами).
Каждый инструмент валидирует вход и возвращает JSON-совместимый dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .. import db
from ..services import (catalog, customers, dialogs, documents, knowledge, pricing, settings_store,
                        support_examples)
from ..services.notify import notify_managers
from ..services.textutil import fmt_money, validate_inn

KB_RELEVANCE = 0.25


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    dialog_id: int | None
    channel: str = "web"
    read_only: bool = False
    escalation: tuple[str, str] | None = None
    documents: list[dict[str, Any]] = field(default_factory=list)
    kb_used: list[int] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    @property
    def dialog(self) -> dict[str, Any] | None:
        return dialogs.get(self.dialog_id) if self.dialog_id else None

    @property
    def customer_id(self) -> int | None:
        d = self.dialog
        return d["customer_id"] if d else None

    @property
    def approved_discount(self) -> float:
        d = self.dialog
        return float(d["state"].get("approved_discount_percent", 0) or 0) if d else 0.0


# ---------- Схемы входных данных ----------

class Item(BaseModel):
    sku: str = Field(min_length=1, max_length=128)
    quantity: int = Field(ge=1, le=1_000_000)


class QueryInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)


class SearchProductsInput(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    russian_only: bool = False
    deployment: Optional[Literal["cloud", "on-prem"]] = None
    vendor: str = Field(default="", max_length=100)
    limit: int = Field(default=5, ge=1, le=10)


class ProductInput(BaseModel):
    product: str = Field(min_length=1, max_length=300, description="Название продукта (family) или артикул")


class SelectVariantInput(ProductInput):
    users: Optional[int] = Field(default=None, ge=1, le=1_000_000)
    term_months: Optional[int] = Field(default=None, ge=0, le=120)
    purchase_type: Optional[Literal["new", "renewal", "upgrade"]] = None
    edition: str = Field(default="", max_length=100)
    deployment: Optional[Literal["cloud", "on-prem"]] = None
    org_type: Optional[Literal["commercial", "academic", "government", "nonprofit", "home"]] = None


class AlternativesInput(ProductInput):
    russian_only: bool = True


class ItemsInput(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=100)
    discount_percent: Optional[float] = Field(default=None, ge=0, le=100)


class ContactInput(BaseModel):
    name: str = ""
    phone: str = ""
    email: str = ""
    company: str = ""
    inn: str = ""
    kpp: str = ""
    address: str = ""


class QuoteInput(ItemsInput):
    customer_comment: str = Field(default="", max_length=1000)


class InvoiceInput(BaseModel):
    quote_number: str = Field(default="", max_length=64)
    items: list[Item] = Field(default_factory=list, max_length=100)


class EscalateInput(BaseModel):
    reason: Literal["special_terms", "client_request", "no_answer", "complaint", "large_deal", "other"]
    summary: str = Field(min_length=5, max_length=3000)
    requested_discount_percent: Optional[float] = Field(default=None, ge=0, le=100)


# ---------- Реализация ----------

def search_knowledge_base(ctx: ToolContext, inp: QueryInput) -> dict[str, Any]:
    results = knowledge.search(inp.query, top_k=3, dialog_id=ctx.dialog_id, track=False)
    relevant = [r for r in results if r["score"] >= KB_RELEVANCE]
    if relevant:
        knowledge.register_hit(relevant[0]["id"], ctx.dialog_id, inp.query, relevant[0]["score"])
        ctx.kb_used.extend(r["id"] for r in relevant)
    else:
        knowledge.register_miss(ctx.dialog_id, inp.query)
    return {
        "results": [{"question": r["question"], "answer": r["answer"], "source": r.get("url", ""),
                     "relevance": r["score"]} for r in results],
        "note": "" if relevant else "Надёжного ответа в базе знаний нет. Не выдумывай: скажи, что уточнишь, и "
                                    "предложи передать вопрос менеджеру (escalate_to_manager, reason=no_answer).",
    }


def search_products(ctx: ToolContext, inp: SearchProductsInput) -> dict[str, Any]:
    filters = {"russian_only": inp.russian_only, "deployment": inp.deployment, "vendor": inp.vendor}
    found = catalog.search_families(inp.query, limit=inp.limit, filters=filters)
    return {
        "products": found,
        "note": ("" if found else "Ничего не найдено. Переформулируй запрос (синонимы, вендор, назначение) или "
                                  "уточни у клиента задачу."),
        "hint": "Чтобы предложить конкретный артикул, вызови select_variant для выбранного продукта.",
    }


def get_product_details(ctx: ToolContext, inp: ProductInput) -> dict[str, Any]:
    fam = catalog.resolve_family(inp.product)
    if not fam:
        raise ToolError(f"Продукт «{inp.product}» не найден. Используй search_products.")
    rows = catalog.family_rows(fam)
    return {"product": catalog.family_summary(fam, rows),
            "variants": [catalog.public_view(r) for r in rows[:40]],
            "variants_total": len(rows)}


def select_variant(ctx: ToolContext, inp: SelectVariantInput) -> dict[str, Any]:
    res = catalog.resolve_variant(inp.product, users=inp.users, term_months=inp.term_months,
                                  purchase_type=inp.purchase_type, edition=inp.edition or None,
                                  deployment=inp.deployment, org_type=inp.org_type)
    if not res["found"]:
        raise ToolError(res["note"])
    if ctx.dialog_id and not ctx.read_only:
        dialogs.update_state(ctx.dialog_id, focus={"family": res["family"], "users": inp.users,
                                                   "term_months": inp.term_months,
                                                   "purchase_type": inp.purchase_type, "edition": inp.edition,
                                                   "org_type": inp.org_type})
    return res


def find_alternatives(ctx: ToolContext, inp: AlternativesInput) -> dict[str, Any]:
    res = catalog.find_alternatives(inp.product, russian_only=inp.russian_only)
    if not res["base"]:
        raise ToolError(f"Продукт «{inp.product}» не найден")
    if not res["alternatives"]:
        res["note"] = "Аналогов в каталоге не найдено — честно скажи об этом и предложи консультацию менеджера."
    return res


def search_support_examples(ctx: ToolContext, inp: QueryInput) -> dict[str, Any]:
    found = support_examples.search(inp.query, top_k=5)
    examples = []
    for r in found:
        if r["score"] < 0.2:
            continue
        ex = {"client": r["question"], "relevance": r["score"]}
        if r.get("answer"):
            ex["support"] = r["answer"]
        if r.get("category"):
            ex["category"] = r["category"]
        if r.get("page_url"):
            ex["page"] = r["page_url"]
        examples.append(ex)
    return {"examples": examples,
            "note": "Реальные обращения клиентов магазина (категория — тип запроса по разметке поддержки; ответ, "
                    "если есть). Используй как ориентир: как обычно формулируют такие вопросы и что важно уточнить. "
                    "Цены, наличие и условия всё равно проверяй инструментами каталога и базы знаний."}


def calculate_offer(ctx: ToolContext, inp: ItemsInput) -> dict[str, Any]:
    items = [i.model_dump() for i in inp.items]
    calc = pricing.calculate(items, inp.discount_percent, ctx.approved_discount)
    if ctx.dialog_id and not ctx.read_only and calc["lines"]:
        dialogs.update_state(ctx.dialog_id, cart=pricing.merge_items(items),
                             discount_percent=calc["discount_percent"])
    if calc["requires_approval"]:
        calc["note"] = ("Скидка сверх лимита ассистента: применена максимально доступная. Предложи клиенту её "
                        "сейчас и, если он хочет больше, передай запрос менеджеру (escalate_to_manager, "
                        "reason=special_terms, requested_discount_percent).")
    return calc


def save_customer_contact(ctx: ToolContext, inp: ContactInput) -> dict[str, Any]:
    data = inp.model_dump()
    if not any(data.values()):
        raise ToolError("Нет данных для сохранения")
    customer_id, warnings = customers.upsert(data, ctx.customer_id)
    if ctx.dialog_id:
        dialogs.set_customer(ctx.dialog_id, customer_id)
        db.log_event("lead_captured", ctx.dialog_id, customer_id=customer_id)
    c = customers.get(customer_id)
    return {"saved": True, "customer": {k: c[k] for k in customers.FIELDS}, "warnings": warnings}


def _buyer(ctx: ToolContext) -> dict[str, Any]:
    c = customers.get(ctx.customer_id)
    return {k: c[k] for k in customers.FIELDS} if c else {}


def _notify_review(ctx: ToolContext, doc: dict[str, Any]) -> None:
    a = settings_store.get("assistant")
    total = doc["totals"]["total"]
    big = total >= float(a.get("large_deal_notify_from", 300000))
    notify_managers(
        f"{'Крупная сделка — ' if big else ''}{doc['title']} № {doc['number']} на проверку ({fmt_money(total)} ₽)",
        f"Клиент: {documents.buyer_line(doc['buyer']) or 'контакты не оставлены'}\n{doc['url']}", ctx.dialog_id)


def create_quote(ctx: ToolContext, inp: QuoteInput) -> dict[str, Any]:
    items = [i.model_dump() for i in inp.items]
    calc = pricing.calculate(items, inp.discount_percent, ctx.approved_discount)
    if not calc["lines"]:
        raise ToolError("Нет позиций с ценой: " + ", ".join(calc["not_found"] + [r["sku"] for r in calc["price_on_request"]]))
    if calc["requires_approval"]:
        raise ToolError(f"Скидка {calc['requested_discount_percent']:g}% требует согласования менеджера. Создай КП "
                        f"со скидкой не более {max(calc['agent_discount_limit_percent'], ctx.approved_discount):g}% "
                        f"или сначала получи согласование (escalate_to_manager).")
    buyer = _buyer(ctx)
    if inp.customer_comment:
        buyer["comment"] = inp.customer_comment
    doc = documents.create_quote(calc, buyer, ctx.dialog_id, ctx.customer_id)
    ctx.documents.append(doc)
    if ctx.dialog_id:
        dialogs.update_state(ctx.dialog_id, cart=pricing.merge_items(items), last_quote=doc["number"])
    _notify_review(ctx, doc)
    sla = settings_store.get("assistant")["operator_sla_minutes"]
    return {"number": doc["number"], "url": doc["url"], "status": "черновик на проверке у менеджера",
            "total": calc["total"], "discount_percent": calc["discount_percent"], "valid_until": doc["valid_until"],
            "price_on_request": calc["price_on_request"],
            "note": f"Передай клиенту ссылку и скажи, что это черновик: менеджер проверит его в течение {sla} мин., "
                    f"после проверки можно оплачивать (100% предоплата)."}


def create_invoice(ctx: ToolContext, inp: InvoiceInput) -> dict[str, Any]:
    buyer = _buyer(ctx)
    if not (buyer.get("company") or buyer.get("name")):
        raise ToolError("Нет реквизитов покупателя. Попроси название организации и ИНН (или ФИО для физлица) "
                        "и сохрани их через save_customer_contact.")
    if buyer.get("company") and not buyer.get("inn"):
        raise ToolError("Для счёта организации нужен ИНН. Попроси его у клиента.")
    if buyer.get("inn") and not validate_inn(buyer["inn"]):
        raise ToolError(f"ИНН {buyer['inn']} некорректен (не сходится контрольная сумма). Попроси клиента проверить.")
    quote = documents.get_by_number(inp.quote_number) if inp.quote_number else None
    if inp.quote_number and (not quote or quote["kind"] != "quote"):
        raise ToolError(f"КП № {inp.quote_number} не найдено")
    if quote and quote["valid_until"] >= date.today().isoformat():
        calc = {**quote["totals"], "lines": quote["items"], "quote_number": quote["number"]}
    else:
        items = [i.model_dump() for i in inp.items] or (quote["items"] if quote else [])
        if not items and ctx.dialog_id:
            items = ctx.dialog["state"].get("cart", [])
        if not items:
            raise ToolError("Не указаны позиции для счёта")
        calc = pricing.calculate([{"sku": i["sku"], "quantity": i["quantity"]} for i in items],
                                 ctx.dialog["state"].get("discount_percent") if ctx.dialog_id else None,
                                 ctx.approved_discount)
        if quote:
            calc["quote_number"] = quote["number"]
    doc = documents.create_invoice(calc, buyer, ctx.dialog_id, ctx.customer_id, quote["id"] if quote else None)
    ctx.documents.append(doc)
    _notify_review(ctx, doc)
    return {"number": doc["number"], "url": doc["url"], "total": doc["totals"]["total"],
            "pay_until": doc["valid_until"], "status": "черновик на проверке у менеджера",
            "note": "Передай ссылку на счёт. Оплата — 100% предоплата; лицензии отправляются на e-mail после оплаты."}


def escalate_to_manager(ctx: ToolContext, inp: EscalateInput) -> dict[str, Any]:
    ctx.escalation = (inp.reason, inp.summary)
    if ctx.dialog_id and inp.requested_discount_percent:
        dialogs.update_state(ctx.dialog_id, requested_discount_percent=inp.requested_discount_percent)
    a = settings_store.get("assistant")
    return {"escalated": True,
            "tell_client": a["handoff_message"].format(sla=a["operator_sla_minutes"], hours=a["working_hours"])}


# ---------- Реестр ----------

ITEMS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "sku": {"type": "string", "description": "Артикул из каталога (поле sku)"},
            "quantity": {"type": "integer", "minimum": 1,
                         "description": "Количество: число пользователей/устройств для цены «за 1», 1 — для пакета"},
        },
        "required": ["sku", "quantity"],
        "additionalProperties": False,
    },
}
PRODUCT_PROP = {"type": "string", "description": "Название продукта (поле family) или артикул"}
DISCOUNT_PROP = {"type": "number", "minimum": 0, "maximum": 100,
                 "description": "Скидка, %. До 5% можно дать самостоятельно; больше — только после согласования менеджером"}


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    model: type[BaseModel]
    fn: Callable[[ToolContext, Any], dict[str, Any]]
    side_effects: bool = False

    def api_definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


TOOLS: list[Tool] = [
    Tool("search_knowledge_base",
         "База знаний магазина: оформление заказа, оплата (только предоплата), доставка электронных лицензий и "
         "коробок, закрывающие документы и ЭДО, НДС и реестр российского ПО, возврат, продление, личный кабинет, "
         "контакты, режим работы. Используй для любого вопроса об условиях и правилах. Возвращает до 3 статей с "
         "релевантностью 0–1; ниже 0.25 — скорее всего не по теме.",
         {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
          "additionalProperties": False},
         QueryInput, search_knowledge_base),
    Tool("search_products",
         "Поиск продуктов в каталоге (товарном фиде) по названию, вендору, назначению или задаче "
         "(«защита почтового сервера», «офисный пакет»). Возвращает продукты с описанием, вендором, признаком "
         "российского ПО, развёртыванием, ценой «от», доступными редакциями, сроками, типами покупки и "
         "диапазонами пользователей.",
         {"type": "object", "properties": {
             "query": {"type": "string", "description": "Название, вендор или задача клиента"},
             "russian_only": {"type": "boolean", "description": "Только российское ПО из реестра"},
             "deployment": {"type": "string", "enum": ["cloud", "on-prem"]},
             "vendor": {"type": "string"},
             "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
          "required": ["query"], "additionalProperties": False},
         SearchProductsInput, search_products),
    Tool("get_product_details",
         "Полные сведения о продукте и таблица его вариантов (артикулы с редакцией, сроком, типом покупки, "
         "числом пользователей и ценой). Используй, чтобы ответить на вопрос о продукте или сравнить версии.",
         {"type": "object", "properties": {"product": PRODUCT_PROP}, "required": ["product"],
          "additionalProperties": False},
         ProductInput, get_product_details),
    Tool("select_variant",
         "Сверка версий: подобрать точный артикул продукта под требования клиента. Передай всё, что уже известно "
         "(число пользователей/устройств, срок в месяцах, новая лицензия или продление, редакция, развёртывание). "
         "Если вариант неоднозначен, вернёт missing — список параметров, которые нужно уточнить у клиента, с "
         "доступными значениями; спрашивай только про них.",
         {"type": "object", "properties": {
             "product": PRODUCT_PROP,
             "users": {"type": "integer", "minimum": 1, "description": "Число пользователей/устройств/узлов"},
             "term_months": {"type": "integer", "minimum": 0, "description": "Срок: 12, 24, 36; 0 — бессрочная"},
             "purchase_type": {"type": "string", "enum": ["new", "renewal", "upgrade"]},
             "edition": {"type": "string"},
             "deployment": {"type": "string", "enum": ["cloud", "on-prem"]},
             "org_type": {"type": "string", "enum": ["commercial", "academic", "government", "nonprofit", "home"],
                          "description": "Тип организации покупателя: цены и артикулы различаются"}},
          "required": ["product"], "additionalProperties": False},
         SelectVariantInput, select_variant),
    Tool("find_alternatives",
         "Найти аналоги продукта в каталоге, по умолчанию — российские (из реестра). Используй для вопросов "
         "«есть ли российский аналог?», «что ещё похожее есть?».",
         {"type": "object", "properties": {"product": PRODUCT_PROP,
                                           "russian_only": {"type": "boolean"}},
          "required": ["product"], "additionalProperties": False},
         AlternativesInput, find_alternatives),
    Tool("search_support_examples",
         "Похожие реальные обращения клиентов из обезличенных чатов поддержки (с категорией запроса и, если "
         "есть, ответом поддержки). Полезно для нетиповых вопросов: как их обычно формулируют и что уточнить.",
         {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
          "additionalProperties": False},
         QueryInput, search_support_examples),
    Tool("calculate_offer",
         "Рассчитать стоимость по артикулам: цены каталога, скидка, НДС по строкам (ПО из реестра — без НДС), "
         "сроки поставки. Вызывай перед тем, как называть итоговую сумму.",
         {"type": "object", "properties": {"items": ITEMS_SCHEMA, "discount_percent": DISCOUNT_PROP},
          "required": ["items"], "additionalProperties": False},
         ItemsInput, calculate_offer),
    Tool("save_customer_contact",
         "Сохранить контакты и реквизиты клиента (имя, телефон, email, организация, ИНН, КПП, адрес). Передавай "
         "только то, что клиент сообщил сам.",
         {"type": "object", "properties": {
             "name": {"type": "string"}, "phone": {"type": "string"}, "email": {"type": "string"},
             "company": {"type": "string", "description": "Организация с ОПФ, например ООО «Альфа»"},
             "inn": {"type": "string"}, "kpp": {"type": "string"}, "address": {"type": "string"}},
          "additionalProperties": False},
         ContactInput, save_customer_contact, side_effects=True),
    Tool("create_quote",
         "Сформировать черновик коммерческого предложения (КП) по шаблону компании, когда клиент согласился с "
         "выбором. КП уходит менеджеру на проверку. Возвращает номер и ссылку — передай её клиенту.",
         {"type": "object", "properties": {
             "items": ITEMS_SCHEMA, "discount_percent": DISCOUNT_PROP,
             "customer_comment": {"type": "string", "description": "Пожелания клиента для менеджера"}},
          "required": ["items"], "additionalProperties": False},
         QuoteInput, create_quote, side_effects=True),
    Tool("create_invoice",
         "Сформировать счёт на оплату (черновик на проверку менеджеру). Нужны реквизиты покупателя: организация "
         "и ИНН (или ФИО физлица), сохранённые через save_customer_contact. Можно по номеру КП.",
         {"type": "object", "properties": {"quote_number": {"type": "string"},
                                           "items": {**ITEMS_SCHEMA, "minItems": 0}},
          "additionalProperties": False},
         InvoiceInput, create_invoice, side_effects=True),
    Tool("escalate_to_manager",
         "Передать диалог менеджеру (он отвечает в течение пары минут). Используй: клиент просит человека; "
         "скидка больше допустимой; индивидуальные условия (отсрочка/постоплата, договор, тендер); претензия; "
         "нет сведений для ответа; цена по запросу. В summary — суть для менеджера: продукт, количество, сумма, "
         "что просит клиент, контакты.",
         {"type": "object", "properties": {
             "reason": {"type": "string", "enum": ["special_terms", "client_request", "no_answer", "complaint",
                                                    "large_deal", "other"]},
             "summary": {"type": "string"},
             "requested_discount_percent": {"type": "number", "minimum": 0, "maximum": 100}},
          "required": ["reason", "summary"], "additionalProperties": False},
         EscalateInput, escalate_to_manager, side_effects=True),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def api_tools(read_only: bool = False) -> list[dict[str, Any]]:
    return [t.api_definition() for t in TOOLS if not (read_only and t.side_effects)]


def run_tool(ctx: ToolContext, name: str, raw_input: Any) -> tuple[dict[str, Any], bool]:
    """Выполнить инструмент. Возвращает (результат, is_error)."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None or (ctx.read_only and tool.side_effects):
        return {"error": f"Инструмент {name} недоступен"}, True
    ctx.tools_used.append(name)
    try:
        inp = tool.model.model_validate(raw_input if isinstance(raw_input, dict) else {})
    except ValidationError as exc:
        return {"error": "Некорректные параметры", "details": exc.errors(include_url=False, include_context=False)}, True
    try:
        return tool.fn(ctx, inp), False
    except (ToolError, ValueError) as exc:
        return {"error": str(exc)}, True
