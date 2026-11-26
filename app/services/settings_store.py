"""Настройки бизнеса (реквизиты, ценовые правила, параметры ассистента) — хранятся в БД, правятся в админке."""
from __future__ import annotations

import copy
from typing import Any

from .. import db

DEFAULTS: dict[str, dict[str, Any]] = {
    # Реквизиты из шаблона КП. Банковские реквизиты и подписант заполняются в «Настройках».
    "company": {
        "name": "ПАО «Софтлайн»",
        "short_name": "Софтлайн",
        "inn": "",
        "kpp": "",
        "ogrn": "",
        "address": "119021, город Москва, ул. Льва Толстого, д. 5, стр. 1, этаж 3, помещ. 1, ком. №2, 2а (а-311)",
        "phone": "+7 495 232 00 23",
        "email": "info@softline.com",
        "website": "www.softline.ru | www.softline.com",
        "store_phone": "+7 (495) 232-00-60, 8 (800) 200-08-60",
        "store_email": "sales.s@softline.com",
        "signer_name": "",
        "signer_title": "",
        "bank_name": "",
        "bik": "",
        "account": "",
        "corr_account": "",
        "director": "",
        "accountant": "",
        "quote_salutation": "Уважаемый заказчик!",
        "quote_price_note": "Цена указана с учетом всех налогов, сборов, и других обязательных платежей.",
    },
    "pricing": {
        "vat_rate": 22,
        "prices_include_vat": True,
        # Скидка от суммы заказа: [{"min_total": 100000, "percent": 3}, ...]; по умолчанию не используется
        "volume_discounts": [],
        # Скидку до этого значения ассистент даёт сам; больше — только после согласования менеджером
        "max_agent_discount_percent": 5,
        "payment_terms": "100% предоплата",
        "quote_valid_days": 5,
        "invoice_valid_days": 3,
    },
    "assistant": {
        "bot_name": "Консультант магазина ПО",
        "greeting": "Здравствуйте! Я онлайн-консультант магазина. Помогу подобрать программу под задачу, "
                    "сверю редакции и варианты лицензий, отвечу на вопросы о продуктах и подготовлю "
                    "коммерческое предложение. Что ищете?",
        "working_hours": "обработка заказов — по рабочим дням 9:00–21:00 МСК",
        "operator_sla_minutes": 2,
        "handoff_message": "Передал вопрос менеджеру — он ответит здесь же, в чате, в течение {sla} мин. "
                           "({hours}). Пока ждёте, могу помочь с другими вопросами.",
        "privacy_notice": "Отправляя сообщение, вы соглашаетесь на обработку персональных данных.",
        # Порог уверенности поиска по базе знаний (0..1) для офлайн-режима
        "kb_min_score": 0.30,
        # Уведомлять менеджера о КП/счетах от этой суммы (горячий лид)
        "large_deal_notify_from": 300000,
        # Нормативы для расчёта экономии времени сотрудников
        "minutes_per_dialog": 22,
        "minutes_per_document": 15,
    },
    # Экономика: стоимость ИИ и сравнение с работой сотрудников (нормативы — укажите свои)
    "economics": {
        "llm_input_usd_per_mtok": 4.0,
        "llm_output_usd_per_mtok": 20.0,
        "llm_cache_read_usd_per_mtok": 0.2,
        "llm_cache_write_usd_per_mtok": 5.0,
        "usd_rub": 90.0,
        "operator_hour_cost_rub": 700.0,
        # Нагрузка из карточки задачи: 1000–2000 обращений в месяц
        "dialogs_per_month_plan": 1500,
    },
    # Классификатор обращений: алгоритм и порог «низкой уверенности»
    "classifier": {
        "algorithm": "hybrid",  # без PyTorch и скачанной нейросети работает как logreg
        "low_confidence": 0.8,
        # Учить на проверенной разметке спорных чатов (включить, когда все чаты размечены по единому правилу)
        "train_on_reviewed": False,
    },
}

_cache: dict[str, dict[str, Any]] = {}


def get(section: str) -> dict[str, Any]:
    if section in _cache:
        return copy.deepcopy(_cache[section])
    value = copy.deepcopy(DEFAULTS.get(section, {}))
    row = db.query_one("SELECT value FROM settings WHERE key = ?", (section,))
    if row:
        value.update(db.loads(row["value"], {}))
    _cache[section] = value
    return copy.deepcopy(value)


def update(section: str, values: dict[str, Any]) -> dict[str, Any]:
    current = get(section)
    current.update(values)
    db.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (section, db.dumps(current)),
    )
    _cache.pop(section, None)
    return current


def reset_cache() -> None:
    _cache.clear()
