"""Расчёт предложения по правилам магазина.

- Цены — только из каталога (фида). Ни клиент, ни языковая модель не могут назначить цену.
- Скидку до max_agent_discount_percent (по умолчанию 5%) ассистент даёт сам; больше — только
  после согласования менеджером (approved_discount_percent в диалоге).
- НДС считается по строкам: ПО из реестра российского ПО — без НДС (пп. 26 п. 2 ст. 149 НК РФ),
  остальное — по ставке из настроек (цены включают НДС).
- Позиции «цена по запросу» в расчёт не входят и помечаются для менеджера.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from . import catalog, settings_store
from .textutil import fmt_money, money


def merge_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, int] = {}
    order: list[str] = []
    for it in items:
        sku = str(it.get("sku", "")).strip()
        qty = int(it.get("quantity") or 0)
        if not sku or qty <= 0:
            continue
        key = sku.upper()
        if key not in merged:
            order.append(key)
            merged[key] = 0
        merged[key] += qty
    return [{"sku": k, "quantity": merged[k]} for k in order]


def _volume_discount(total: Decimal, tiers: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    current, next_tier = 0.0, None
    for tier in sorted(tiers or [], key=lambda t: t["min_total"]):
        if total >= Decimal(str(tier["min_total"])):
            current = float(tier["percent"])
        elif next_tier is None:
            next_tier = tier
    return current, next_tier


def calculate(items: list[dict[str, Any]], discount_percent: float | None = None,
              approved_discount_percent: float = 0.0, today: date | None = None) -> dict[str, Any]:
    rules = settings_store.get("pricing")
    agent_limit = float(rules.get("max_agent_discount_percent", 5))
    limit = max(agent_limit, float(approved_discount_percent or 0))
    default_vat = Decimal(str(rules.get("vat_rate", 22)))

    lines, not_found, on_request = [], [], []
    subtotal = Decimal("0")
    max_days, has_box = 0, False
    for it in merge_items(items):
        p = catalog.get_by_sku(it["sku"])
        if not p or not p["active"]:
            not_found.append(it["sku"])
            continue
        if p["price"] <= 0:
            on_request.append({"sku": p["sku"], "name": p["name"], "quantity": it["quantity"]})
            continue
        qty = it["quantity"]
        if p["pricing"] == "package":
            qty = max(1, qty)
        price = money(p["price"])
        amount = price * qty
        subtotal += amount
        vat_rate = Decimal(str(p["vat_rate"])) if p["vat_rate"] is not None else default_vat
        max_days = max(max_days, int(p["lead_time_days"] or 0))
        has_box = has_box or p["delivery"] == "box"
        lines.append({"sku": p["sku"], "name": p["name"], "unit": "пакет" if p["pricing"] == "package" else p["unit"],
                      "quantity": qty, "price": price, "vat_rate": vat_rate,
                      "availability": catalog.availability(p, qty, today)["summary"]})

    auto_percent, next_tier = _volume_discount(subtotal, rules.get("volume_discounts", []))
    requested = float(discount_percent) if discount_percent else 0.0
    wanted = max(auto_percent, requested, float(approved_discount_percent or 0))
    applied = min(wanted, limit)
    requires_approval = wanted > limit
    approval_note = ""
    if requires_approval:
        approval_note = (f"Запрошена скидка {wanted:g}%. Ассистент может дать не более {agent_limit:g}%"
                         + (f", менеджер согласовал {approved_discount_percent:g}%" if approved_discount_percent else "")
                         + ". Нужно согласование менеджера.")

    total = vat_total = Decimal("0")
    for ln in lines:
        ln["discount_percent"] = applied
        ln["price_final"] = money(ln["price"] * (Decimal("100") - Decimal(str(applied))) / 100)
        ln["amount"] = ln["price_final"] * ln["quantity"]
        rate = ln["vat_rate"]
        if rules.get("prices_include_vat", True):
            ln["vat_amount"] = money(ln["amount"] * rate / (100 + rate)) if rate else Decimal("0")
        else:
            ln["vat_amount"] = money(ln["amount"] * rate / 100)
            ln["amount"] += ln["vat_amount"]
        total += ln["amount"]
        vat_total += ln["vat_amount"]

    hint = ""
    if next_tier and lines:
        missing = Decimal(str(next_tier["min_total"])) - subtotal
        hint = f"До скидки {next_tier['percent']}% не хватает {fmt_money(missing)} ₽."

    def f(v: Decimal) -> float:
        return float(money(v))

    return {
        "lines": [{**ln, "price": f(ln["price"]), "price_final": f(ln["price_final"]), "amount": f(ln["amount"]),
                   "vat_amount": f(ln["vat_amount"]), "vat_rate": float(ln["vat_rate"])} for ln in lines],
        "not_found": not_found,
        "price_on_request": on_request,
        "subtotal": f(subtotal),
        "discount_percent": applied,
        "discount_amount": f(subtotal - sum((ln["price_final"] * ln["quantity"] for ln in lines), Decimal("0"))),
        "total": f(total),
        "vat_amount": f(vat_total),
        "vat_rate": float(default_vat),
        "prices_include_vat": bool(rules.get("prices_include_vat", True)),
        "items_count": len(lines),
        "next_discount_hint": hint,
        "requested_discount_percent": wanted or None,
        "agent_discount_limit_percent": agent_limit,
        "requires_approval": requires_approval,
        "approval_note": approval_note,
        "payment_terms": rules.get("payment_terms", "100% предоплата"),
        "delivery_terms": (("Коробочные версии — курьером, " if has_box else "Электронная поставка: лицензии и "
                            "ключи на e-mail, ") + (f"до {max_days} рабочих дней" if max_days > 1 else
                                                    "в течение рабочего дня") + " после оплаты"),
    }


def summary_text(calc: dict[str, Any]) -> str:
    out = []
    for i, ln in enumerate(calc["lines"], 1):
        disc = f" (скидка {ln['discount_percent']:g}%)" if ln["discount_percent"] else ""
        vat = "без НДС" if not ln["vat_rate"] else f"в т.ч. НДС {fmt_money(ln['vat_amount'])} ₽"
        out.append(f"{i}. {ln['name']} [{ln['sku']}] — {ln['quantity']} {ln['unit']} × "
                   f"{fmt_money(ln['price_final'])} ₽{disc} = {fmt_money(ln['amount'])} ₽ ({vat})")
    for r in calc["price_on_request"]:
        out.append(f"• {r['name']} [{r['sku']}] — цена по запросу, уточнит менеджер")
    if calc["not_found"]:
        out.append("Не найдены в каталоге: " + ", ".join(calc["not_found"]))
    if calc["lines"]:
        vat = f"в т.ч. НДС {fmt_money(calc['vat_amount'])} ₽" if calc["vat_amount"] else "без НДС"
        out.append(f"Итого: {fmt_money(calc['total'])} ₽ ({vat})")
        if calc["next_discount_hint"]:
            out.append(calc["next_discount_hint"])
    return "\n".join(out)
