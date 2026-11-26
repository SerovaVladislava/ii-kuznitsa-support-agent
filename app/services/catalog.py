"""Каталог ПО: поиск по товарному фиду (до сотен тысяч SKU), группировка вариантов продукта,
подбор точного артикула по требованиям клиента и поиск аналогов (в т.ч. российских).

Поиск: SQLite FTS5 (BM25) отбирает кандидатов за миллисекунды даже на 200 000 SKU, затем
кандидаты переранжируются по символьным n-граммам (устойчиво к опечаткам и транслиту).
"""
from __future__ import annotations

import re
import time
from datetime import date
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .. import db
from .search import prep_chars, stem, tokens
from .textutil import add_workdays, fmt_date, fmt_money, normalize

PRODUCT_FIELDS = ["sku", "name", "vendor", "family", "edition", "category", "description", "tags", "unit",
                  "pricing", "price", "purchase_type", "license_type", "term_months", "users_min", "users_max",
                  "deployment", "org_type", "platforms", "is_russian", "certificates", "vat_rate", "url", "delivery",
                  "stock_qty", "lead_time_days", "min_order_qty", "available", "active", "attributes"]

PURCHASE_LABELS = {"new": "новая лицензия", "renewal": "продление", "upgrade": "переход/обновление"}
LICENSE_LABELS = {"subscription": "подписка", "perpetual": "бессрочная", "service": "услуга"}
DEPLOY_LABELS = {"on-prem": "на своих серверах", "cloud": "облако", "hybrid": "облако или свои серверы"}
ORG_LABELS = {"commercial": "коммерческая организация", "academic": "образовательная организация",
              "government": "государственная организация", "nonprofit": "некоммерческая организация",
              "home": "для дома"}
DIM_LABELS = {"edition": "редакция", "purchase_type": "тип покупки", "term_months": "срок лицензии",
              "deployment": "развёртывание", "license_type": "тип лицензии", "org_type": "тип организации",
              "users": "количество пользователей/устройств"}

# Транслит и разговорные названия вендоров/продуктов -> как в фиде
SYNONYMS = {
    "касперский": "kaspersky", "касперского": "kaspersky", "каспер": "kaspersky", "kaspersky": "касперский",
    "доктор": "dr web", "drweb": "dr web", "др": "dr", "веб": "web",
    "мдемон": "mdaemon", "эмдемон": "mdaemon", "мдаемон": "mdaemon",
    "майкрософт": "microsoft", "мелкософт": "microsoft", "офис": "office", "ворд": "word",
    "яндекс": "yandex", "р7": "r7", "коммунигейт": "communigate", "рупост": "rupost",
    "кибербэкап": "кибер бэкап", "бэкап": "backup резервное", "антивирус": "антивирусная защита",
    "почту": "почта почтовый", "почты": "почта почтовый", "почтовый": "почта",
    "файлы": "файловый файлов", "сотрудников": "пользователей", "компьютеров": "узлов рабочих станций",
    "пк": "узлов рабочих станций", "компов": "узлов",
    "астра": "astra", "астру": "astra", "линукс": "linux", "линукса": "linux", "ред": "red", "редос": "red ос",
    "виндовс": "windows", "винда": "windows", "эксель": "excel", "фотошоп": "photoshop", "автокад": "autocad",
    "адоб": "adobe", "криптопро": "крипто про cryptopro", "крипто": "crypto", "1с": "1c", "веб": "web",
    "позитив": "positive", "кодбез": "код безопасности", "секрет": "secret", "ревит": "revit", "визио": "visio",
    "аутлук": "outlook", "тимс": "teams", "зум": "zoom",
}


def term_label(months: int) -> str:
    if not months:
        return "бессрочно"
    if months % 12 == 0:
        y = months // 12
        return f"{y} {'год' if y == 1 else 'года' if y < 5 else 'лет'}"
    return f"{months} мес."


def users_label(p: dict[str, Any]) -> str:
    lo, hi = int(p.get("users_min") or 0), int(p.get("users_max") or 0)
    if p.get("pricing") == "package" and hi:
        return f"до {hi}"
    if lo and hi:
        return f"{lo}–{hi}" if lo != hi else str(lo)
    if lo:
        return f"от {lo}"
    if hi:
        return f"до {hi}"
    return ""


def _tier_key(label: str) -> int:
    m = re.search(r"\d+", label)
    return int(m.group(0)) if m else 0


def vat_note(p: dict[str, Any], default_rate: float) -> str:
    rate = p["vat_rate"] if p.get("vat_rate") is not None else default_rate
    return "без НДС (ПО из реестра, пп. 26 п. 2 ст. 149 НК РФ)" if not rate else f"НДС {rate:g}% включён"


def get_by_sku(sku: str) -> dict[str, Any] | None:
    return db.query_one("SELECT * FROM products WHERE sku = ? COLLATE NOCASE", (sku.strip(),))


def get(product_id: int) -> dict[str, Any] | None:
    return db.query_one("SELECT * FROM products WHERE id = ?", (product_id,))


def public_view(p: dict[str, Any]) -> dict[str, Any]:
    """Что можно показать клиенту и модели — только данные фида."""
    from . import settings_store
    default_vat = float(settings_store.get("pricing").get("vat_rate", 22))
    view = {
        "sku": p["sku"],
        "name": p["name"],
        "vendor": p["vendor"],
        "family": p["family"],
        "edition": p["edition"],
        "category": p["category"],
        "purchase_type": PURCHASE_LABELS.get(p["purchase_type"], p["purchase_type"]),
        "license_type": LICENSE_LABELS.get(p["license_type"], p["license_type"]),
        "term": term_label(p["term_months"]) if p["license_type"] != "perpetual" or p["term_months"] else "бессрочно",
        "users": users_label(p),
        "pricing": "цена за пакет" if p["pricing"] == "package" else f"цена за 1 {p['unit']}",
        "price_rub": p["price"] if p["price"] > 0 else None,
        "price_note": "" if p["price"] > 0 else "цена по запросу",
        "in_stock": bool(p.get("available", 1)),
        "availability_note": "" if p.get("available", 1) else "сейчас не продаётся на витрине — наличие и цену уточнит менеджер",
        "organization": ORG_LABELS.get(p.get("org_type", ""), p.get("org_type", "")),
        "deployment": DEPLOY_LABELS.get(p["deployment"], p["deployment"]),
        "platforms": p["platforms"],
        "russian_software": bool(p["is_russian"]),
        "certificates": p["certificates"],
        "vat": vat_note(p, default_vat),
        "delivery": "электронная (ключ на e-mail)" if p["delivery"] == "electronic" else "коробка",
        "url": p["url"],
    }
    return {k: v for k, v in view.items() if v not in ("", None) or k in ("price_rub",)}


# ---------------- Поиск ----------------

def _fts_query(query: str) -> str:
    terms: set[str] = set()
    for tok in tokens(query):
        for variant in [tok] + SYNONYMS.get(tok, "").split():
            for w in re.findall(r"\w+", variant):
                if any(c.isdigit() for c in w) or len(w) == 2:  # «р7», «1с», «dr», «ос» — точное совпадение
                    terms.add(f'"{w}"')
                elif len(w) >= 3:
                    terms.add(f'"{stem(w)}"*')
    return " OR ".join(sorted(terms))


def _filters_sql(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    sql, params = "", []
    f = filters or {}
    if f.get("vendor"):
        sql += " AND p.vendor LIKE ?"
        params.append(f"%{f['vendor']}%")
    if f.get("russian_only"):
        sql += " AND p.is_russian = 1"
    if f.get("deployment") in ("cloud", "on-prem"):
        sql += " AND p.deployment IN (?, 'hybrid', '')"
        params.append(f["deployment"])
    if f.get("category"):
        sql += " AND p.category LIKE ?"
        params.append(f"%{f['category']}%")
    if f.get("purchase_type") in ("new", "renewal", "upgrade"):
        sql += " AND p.purchase_type IN (?, '')"
        params.append(f["purchase_type"])
    if f.get("exclude_family"):
        sql += " AND p.family != ?"
        params.append(f["exclude_family"])
    return sql, params


def search(query: str, limit: int = 10, filters: dict[str, Any] | None = None,
           candidates: int = 300) -> list[dict[str, Any]]:
    """Поиск SKU. Возвращает строки каталога с полем score (0..1)."""
    exact = get_by_sku(query)
    if exact and exact["active"]:
        return [{**exact, "score": 1.0}]
    fts = _fts_query(query)
    if not fts:
        return []
    extra, params = _filters_sql(filters)
    # Сначала лучшие совпадения из полнотекстового индекса (без соединения — быстро и на 200 000 SKU),
    # затем фильтры и данные товаров. С фильтрами берём больший запас кандидатов.
    pool = candidates * (8 if extra else 2)
    rows = db.query(
        "WITH top AS (SELECT rowid, bm25(products_fts, 3.0, 4.0, 8.0, 4.0, 2.0, 3.0, 1.0) AS rank "
        "FROM products_fts WHERE products_fts MATCH ? ORDER BY rank LIMIT ?) "
        "SELECT p.*, top.rank FROM top JOIN products p ON p.id = top.rowid "
        f"WHERE p.active = 1{extra} ORDER BY top.rank LIMIT ?",
        (fts, pool, *params, candidates))
    if not rows:
        return []
    # Переранжирование: близость запроса к названию и, отдельно, к ключевым словам товара
    expansion = " ".join(SYNONYMS.get(tok, "") for tok in tokens(query))  # «фотошоп» → и «photoshop»
    q = prep_chars(f"{query} {expansion}")
    names = [prep_chars(f"{r['family']} {r['name']} {r['vendor']}") for r in rows]
    tags = [prep_chars(r["tags"] or r["category"]) for r in rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    mx = vec.fit_transform(names + tags + [q])
    n = len(rows)
    sims = linear_kernel(mx[-1], mx[:-1]).ravel()
    cos = [max(sims[i], 0.85 * sims[n + i]) for i in range(n)]
    best_rank = min(r["rank"] for r in rows) or -1.0
    # Редкие слова запроса («р7», «mdaemon», «redcheck») — сильный признак нужного продукта
    df = family_token_df()
    q_words = set(re.findall(r"\w+", normalize(f"{query} {expansion}")))
    distinct = {w for w in q_words if len(w) >= 2 and 0 < df.get(w, 0) <= 12}
    for r, c in zip(rows, cos):
        bm = (r["rank"] / best_rank) if best_rank < 0 else 0.0
        # небольшой приоритет вариантам, которые можно купить прямо сейчас
        boost = 0.04 if r.get("available", 1) and r["price"] > 0 else 0.0
        if distinct and distinct & set(re.findall(r"\w+", normalize(f"{r['family']} {r['vendor']}"))):
            boost += 0.12
        r["score"] = round(float(0.65 * c + 0.35 * bm + boost), 3)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:limit]


def family_rows(family: str, only_active: bool = True) -> list[dict[str, Any]]:
    return db.query(
        "SELECT * FROM products WHERE family = ? COLLATE NOCASE" + (" AND active = 1" if only_active else "")
        + " ORDER BY edition, purchase_type, term_months, users_min, users_max, price", (family,))


def family_summary(family: str, rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    rows = rows if rows is not None else family_rows(family)
    if not rows:
        return None
    first = rows[0]
    on_sale = [r for r in rows if r.get("available", 1)]
    priced = [r for r in (on_sale or rows) if r["price"] > 0]
    per_unit = [r for r in priced if r["pricing"] == "per_unit"]

    def distinct(key: str, labels: dict[str, str] | None = None) -> list[str]:
        vals = sorted({r[key] for r in rows if r[key] not in ("", None)}, key=str)
        return [labels.get(v, v) if labels else v for v in vals]

    price_from = ""
    if per_unit:
        cheapest = min(per_unit, key=lambda r: r["price"])
        price_from = f"от {fmt_money(cheapest['price'])} ₽ за 1 {cheapest['unit']}"
    elif priced:
        price_from = f"от {fmt_money(min(r['price'] for r in priced))} ₽ за пакет"
    return {
        "family": first["family"],
        "vendor": first["vendor"],
        "category": first["category"],
        "description": first["description"],
        "russian_software": bool(first["is_russian"]),
        "certificates": first["certificates"],
        "deployment": DEPLOY_LABELS.get(first["deployment"], first["deployment"]),
        "platforms": first["platforms"],
        "url": first["url"],
        "price_from": price_from or "цена по запросу",
        "in_stock": bool(on_sale),
        "availability_note": "" if on_sale else "сейчас не продаётся на витрине — наличие и цену уточнит менеджер",
        "organizations": distinct("org_type", ORG_LABELS),
        "editions": distinct("edition"),
        "purchase_types": distinct("purchase_type", PURCHASE_LABELS),
        "terms": [term_label(m) for m in sorted({r["term_months"] for r in rows})],
        "users_tiers": sorted({users_label(r) for r in rows if users_label(r)}, key=_tier_key),
        "variants_count": len(rows),
    }


def search_families(query: str, limit: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Поиск продуктов (семейств): группирует найденные SKU по продукту."""
    found = search(query, limit=200, filters=filters)
    best: dict[str, float] = {}
    for r in found:
        fam = r["family"] or r["name"]
        best[fam] = max(best.get(fam, 0.0), r["score"])
    out = []
    for fam, score in sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:limit]:
        summary = family_summary(fam) if fam else None
        if summary:
            out.append({**summary, "relevance": round(score, 3)})
    return out


def resolve_family(name_or_sku: str) -> str | None:
    p = get_by_sku(name_or_sku)
    if p:
        return p["family"]
    row = db.query_one("SELECT family FROM products WHERE family = ? COLLATE NOCASE AND active = 1 LIMIT 1",
                       (name_or_sku.strip(),))
    if row:
        return row["family"]
    found = search_families(name_or_sku, limit=1)
    return found[0]["family"] if found and found[0]["relevance"] >= 0.25 else None


def _fits_users(r: dict[str, Any], n: int) -> bool:
    lo, hi = int(r["users_min"] or 0), int(r["users_max"] or 0)
    if r["pricing"] == "package":
        return hi == 0 or n <= hi
    return (lo == 0 or n >= lo) and (hi == 0 or n <= hi)


def resolve_variant(family: str, users: int | None = None, term_months: int | None = None,
                    purchase_type: str | None = None, edition: str | None = None,
                    deployment: str | None = None, license_type: str | None = None,
                    org_type: str | None = None) -> dict[str, Any]:
    """Сверка версий: подобрать артикул под требования и определить, каких данных не хватает.

    Возвращает selected (SKU) — если вариант определён однозначно, иначе missing: {параметр: варианты}.
    """
    fam = resolve_family(family) or family
    rows = family_rows(fam)
    if not rows:
        return {"family": family, "found": False, "selected": None, "missing": {}, "conflicts": [],
                "note": "Продукт не найден в каталоге"}
    conflicts: list[str] = []

    def apply(rows_: list[dict[str, Any]], key: str, value: Any, label: str) -> list[dict[str, Any]]:
        if value in (None, ""):
            return rows_
        if key == "edition":
            matched = [r for r in rows_ if str(value).lower() in r["edition"].lower()]
        else:
            matched = [r for r in rows_ if r[key] == value or r[key] in ("", None)]
        if not matched:
            show = {"term_months": term_label, "purchase_type": lambda v: PURCHASE_LABELS.get(v, v),
                    "org_type": lambda v: ORG_LABELS.get(v, v), "deployment": lambda v: DEPLOY_LABELS.get(v, v),
                    "license_type": lambda v: LICENSE_LABELS.get(v, v)}.get(key, str)
            options = [show(v) for v in sorted({r[key] for r in rows_ if r[key] not in ("", None)}, key=str)]
            conflicts.append(f"{label} «{show(value)}» для этого продукта нет; есть: {', '.join(options) or '—'}")
            return rows_
        return matched

    rows = apply(rows, "edition", edition, "редакция")
    rows = apply(rows, "purchase_type", purchase_type, "тип покупки")
    rows = apply(rows, "term_months", term_months, "срок")
    rows = apply(rows, "deployment", deployment, "развёртывание")
    rows = apply(rows, "license_type", license_type, "тип лицензии")
    rows = apply(rows, "org_type", org_type, "тип организации")
    on_sale = [r for r in rows if r.get("available", 1)]
    if on_sale:  # предлагаем только то, что сейчас можно купить
        rows = on_sale
    has_tiers = any(r["users_min"] or r["users_max"] for r in rows)
    users_conflict = False
    if users:
        fitting = [r for r in rows if _fits_users(r, users)]
        if fitting:
            rows = fitting
        else:
            users_conflict = True
            conflicts.append(f"Нет варианта на {users} {rows[0]['unit']}; доступно: "
                             + ", ".join(sorted({users_label(r) for r in rows}, key=_tier_key)))

    missing: dict[str, list[str]] = {}
    labels = {"org_type": ORG_LABELS, "edition": None, "purchase_type": PURCHASE_LABELS, "term_months": None,
              "deployment": DEPLOY_LABELS, "license_type": LICENSE_LABELS}
    for key, lab in labels.items():
        vals = sorted({r[key] for r in rows if r[key] not in ("", None)}, key=str)
        if len(vals) > 1:
            missing[key] = [term_label(v) if key == "term_months" else (lab.get(v, v) if lab else v)
                            for v in vals][:10]
    if has_tiers and (not users or users_conflict):
        missing["users"] = sorted({users_label(r) for r in rows if users_label(r)}, key=_tier_key)[:12]

    selected = None
    quantity = None
    if not missing and rows:
        if users:
            pkg = [r for r in rows if r["pricing"] == "package"]
            chosen = min(pkg, key=lambda r: r["users_max"] or 10 ** 9) if pkg else rows[0]
        else:
            chosen = rows[0]
        selected = public_view(chosen)
        quantity = 1 if chosen["pricing"] == "package" else max(users or 1, int(chosen["min_order_qty"] or 1))
        selected["quantity"] = quantity
        selected["availability"] = availability(chosen, quantity)["summary"]
        if chosen["price"] > 0:
            selected["estimated_total_rub"] = round(chosen["price"] * quantity, 2)
    return {
        "family": fam,
        "found": True,
        "selected": selected,
        "missing": missing,
        "missing_labels": [DIM_LABELS[k] for k in missing],
        "conflicts": conflicts,
        "candidates": [public_view(r) for r in rows[:6]],
        "note": ("Вариант определён однозначно" if selected else
                 "Уточните у клиента: " + ", ".join(DIM_LABELS[k] for k in missing) if missing else ""),
    }


def find_alternatives(family: str, russian_only: bool = True, limit: int = 3) -> dict[str, Any]:
    fam = resolve_family(family) or family
    base = family_summary(fam)
    if not base:
        return {"base": None, "alternatives": []}
    rows = family_rows(fam)
    query = f"{base['category']} {rows[0]['tags']} {base['family']} {base['description'][:200]}"
    filters = {"russian_only": russian_only, "exclude_family": fam}
    # Сначала аналоги той же категории; другие категории — только если в своей ничего нет
    alts = search_families(query, limit=limit + 3, filters={**filters, "category": base["category"]})
    if not alts:
        alts = search_families(query, limit=limit + 3, filters=filters)
    return {"base": {k: base[k] for k in ("family", "vendor", "category", "russian_software")},
            "alternatives": alts[:limit]}


def availability(p: dict[str, Any], quantity: int, today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    quantity = max(1, int(quantity))
    days = max(0, int(p["lead_time_days"] or 0))
    eta = add_workdays(today, days) if days else today
    warnings = []
    if quantity < int(p["min_order_qty"] or 1):
        warnings.append(f"минимальное количество — {p['min_order_qty']} {p['unit']}")
    if not p.get("available", 1):
        summary = "сейчас не продаётся на витрине — наличие, цену и срок уточнит менеджер"
    elif p["price"] <= 0:
        summary = "цена и срок — по запросу у менеджера"
    elif p["delivery"] == "electronic" or int(p["stock_qty"]) < 0:
        summary = (f"электронная поставка: лицензия/ключ на e-mail после оплаты, "
                   f"{'в течение дня' if days <= 1 else f'до {days} раб. дн.'} (ориентировочно к {fmt_date(eta)})")
    else:
        stock = int(p["stock_qty"])
        if quantity <= stock:
            summary = f"в наличии, отгрузка {fmt_date(add_workdays(today, 1))}"
        else:
            summary = f"под заказ, ориентировочно {fmt_date(eta)}"
    return {"sku": p["sku"], "requested": quantity, "summary": summary, "warnings": warnings,
            "eta": eta.isoformat()}


# ---------------- Запись ----------------

def normalize_product(data: dict[str, Any]) -> dict[str, Any]:
    def num(v: Any, default: float = 0.0) -> float:
        if v in (None, ""):
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(" ", "").replace(" ", "").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return default

    vat = data.get("vat_rate")
    clean = {
        "sku": str(data.get("sku") or "").strip(),
        "name": str(data.get("name") or "").strip(),
        "vendor": str(data.get("vendor") or "").strip(),
        "family": str(data.get("family") or "").strip(),
        "edition": str(data.get("edition") or "").strip(),
        "category": str(data.get("category") or "").strip(),
        "description": str(data.get("description") or "").strip(),
        "tags": str(data.get("tags") or "").strip(),
        "unit": str(data.get("unit") or "лиц.").strip() or "лиц.",
        "pricing": "package" if str(data.get("pricing")) == "package" else "per_unit",
        "price": num(data.get("price")),
        "purchase_type": str(data.get("purchase_type") or "").strip(),
        "license_type": str(data.get("license_type") or "").strip(),
        "term_months": int(num(data.get("term_months"))),
        "users_min": int(num(data.get("users_min"))),
        "users_max": int(num(data.get("users_max"))),
        "deployment": str(data.get("deployment") or "").strip(),
        "org_type": str(data.get("org_type") or "").strip(),
        "available": 0 if str(data.get("available", 1)).strip().lower() in ("0", "false", "нет") else 1,
        "platforms": str(data.get("platforms") or "").strip(),
        "is_russian": 1 if str(data.get("is_russian", 0)).strip().lower() in ("1", "true", "да", "yes") else 0,
        "certificates": str(data.get("certificates") or "").strip(),
        "vat_rate": None if vat in (None, "") else num(vat),
        "url": str(data.get("url") or "").strip(),
        "delivery": "box" if str(data.get("delivery")) == "box" else "electronic",
        "stock_qty": int(num(data.get("stock_qty"), -1)),
        "lead_time_days": int(num(data.get("lead_time_days"), 1)),
        "min_order_qty": max(1, int(num(data.get("min_order_qty"), 1))),
        "active": 0 if str(data.get("active", 1)).strip().lower() in ("0", "false", "нет") else 1,
        "attributes": data.get("attributes") if isinstance(data.get("attributes"), str)
        else db.dumps(data.get("attributes") or {}),
    }
    if not clean["family"]:
        clean["family"] = clean["name"]
    return clean


_UPSERT = (f"INSERT INTO products ({', '.join(PRODUCT_FIELDS)}, updated_at) "
           f"VALUES ({', '.join('?' * (len(PRODUCT_FIELDS) + 1))}) "
           f"ON CONFLICT(sku) DO UPDATE SET "
           + ", ".join(f"{f} = excluded.{f}" for f in PRODUCT_FIELDS if f != "sku")
           + ", updated_at = excluded.updated_at")


def bulk_upsert(items: list[dict[str, Any]]) -> dict[str, int]:
    """Пакетная загрузка фида. Полные строки (с наименованием) — вставка/обновление,
    частичные (например, только артикул + цена) — обновление указанных полей."""
    now = db.now_iso()
    full, partial, skipped = [], [], 0
    for it in items:
        if not str(it.get("sku") or "").strip():
            skipped += 1
        elif str(it.get("name") or "").strip():
            c = normalize_product(it)
            full.append(tuple(c[f] for f in PRODUCT_FIELDS) + (now,))
        else:
            partial.append(it)
    updated_partial = 0
    with db.connect() as conn:
        conn.executemany(_UPSERT, full)
        for it in partial:
            fields = {k: v for k, v in it.items() if k in PRODUCT_FIELDS and k != "sku" and v not in (None, "")}
            if not fields:
                skipped += 1
                continue
            cur = conn.execute(
                f"UPDATE products SET {', '.join(k + ' = ?' for k in fields)}, updated_at = ? WHERE sku = ? COLLATE NOCASE",
                (*fields.values(), now, str(it["sku"]).strip()))
            if cur.rowcount:
                updated_partial += 1
            else:
                skipped += 1
    return {"created_or_updated": len(full) + updated_partial, "skipped": skipped}


def save_product(data: dict[str, Any], product_id: int | None = None) -> int:
    clean = normalize_product(data)
    if not clean["sku"] or not clean["name"]:
        raise ValueError("Артикул и наименование обязательны")
    if product_id:
        existing = get(product_id)
        if existing and existing["sku"] != clean["sku"]:
            db.execute("UPDATE products SET sku = ? WHERE id = ?", (clean["sku"], product_id))
    bulk_upsert([{**clean}])
    return get_by_sku(clean["sku"])["id"]


def list_products(search_text: str = "", limit: int = 200) -> list[dict[str, Any]]:
    if search_text:
        return search(search_text, limit=limit)
    return db.query("SELECT * FROM products ORDER BY vendor, family, edition, purchase_type, term_months, users_min "
                    "LIMIT ?", (limit,))


def count() -> int:
    return db.scalar("SELECT COUNT(*) FROM products") or 0


# ---------------- Контекст страницы сайта ----------------

GENERIC_PAGES = {"search", "basket", "writeorder", "contacts", "catalog", "delivery_info", "personal", "order",
                 "b2b", "tenders", "about", "news", "actions", "promo", "help", "login", "registration"}


def page_hint(url: str) -> dict[str, Any] | None:
    """С какой страницы магазина пишет клиент: вендор и (если это страница продукта) сам продукт.

    store.softline.ru/alteks-soft/redcheck/ → продукт «RedCheck»; store.softline.ru/kaspersky/ → вендор.
    """
    m = re.match(r"^(?:https?://)?[^/]*softline\.ru/([^?#]*)", (url or "").strip(), re.IGNORECASE)
    if not m:
        return None
    parts = [p for p in m.group(1).split("/") if p]
    if not parts or parts[0] in GENERIC_PAGES:
        return None
    vendor_slug = parts[0]
    product_slug = parts[1] if len(parts) > 1 else ""
    words = re.sub(r"[-_]+", " ", product_slug)
    words = re.sub(r"\b\d+\b|\b[0-9a-f]{10,}\b", " ", words).strip()
    hint: dict[str, Any] = {"vendor_slug": vendor_slug, "url": url}
    if words:
        found = search_families(f"{vendor_slug} {words}", limit=1)
        if found and found[0]["relevance"] >= 0.3:
            hint["family"] = found[0]["family"]
            hint["vendor"] = found[0]["vendor"]
    if "vendor" not in hint:
        row = db.query_one("SELECT vendor FROM products WHERE active = 1 AND url LIKE ? LIMIT 1",
                           (f"%softline.ru/{vendor_slug}/%",))
        if row:
            hint["vendor"] = row["vendor"]
    return hint if ("family" in hint or "vendor" in hint) else None


# ---------------- Отличительные слова в названиях продуктов ----------------

_token_df: dict[str, int] = {}
_token_df_key: tuple[int, str] | None = None
_token_df_checked = 0.0


def family_token_df() -> dict[str, int]:
    """У скольких вендоров слово встречается в названиях продуктов. Слова, общие для многих вендоров
    («новая», «лицензия», «коммерческая»), не считаются названием продукта; бренд («kaspersky») — считается,
    даже если у вендора сотня продуктов."""
    global _token_df, _token_df_key, _token_df_checked
    if _token_df and time.monotonic() - _token_df_checked < 60:
        return _token_df
    _token_df_checked = time.monotonic()
    key = (count(), db.scalar("SELECT MAX(updated_at) FROM products") or "")
    if key != _token_df_key:
        vendors: dict[str, set[str]] = {}
        for r in db.query("SELECT DISTINCT family, vendor FROM products WHERE active = 1"):
            for w in set(re.findall(r"\w+", (r["family"] or "").lower().replace("ё", "е"))):
                vendors.setdefault(w, set()).add(r["vendor"])
        _token_df, _token_df_key = {w: len(v) for w, v in vendors.items()}, key
    return _token_df


_vendor_tokens: set[str] = set()
_vendor_tokens_checked = 0.0


def vendor_token_set() -> set[str]:
    """Слова из названий вендоров каталога (кэш на минуту)."""
    global _vendor_tokens, _vendor_tokens_checked
    if not _vendor_tokens or time.monotonic() - _vendor_tokens_checked > 60:
        words: set[str] = set()
        for r in db.query("SELECT DISTINCT vendor FROM products WHERE active = 1"):
            words.update(re.findall(r"\w+", (r["vendor"] or "").lower().replace("ё", "е")))
        _vendor_tokens, _vendor_tokens_checked = words, time.monotonic()
    return _vendor_tokens
