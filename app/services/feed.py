"""Импорт товарного фида: YML (Яндекс.Маркет), Google Merchant XML, CSV/XLSX, JSON.

Атрибуты ПО (редакция, срок, число пользователей, новая лицензия/продление, тип развёртывания,
российское ПО, НДС) берутся из параметров оффера, а если их нет — извлекаются из наименования.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from . import catalog
from .textutil import normalize

EDITION_WORDS = [
    "Standard", "Professional", "Pro", "Enterprise", "Business", "Premium", "Basic", "Advanced", "Ultimate",
    "Стандартный", "Стандартная", "Расширенный", "Расширенная", "Расширенный Плюс", "Базовый", "Базовая",
    "Профессиональный", "Корпоративный", "Оптимальный", "Минимальный", "Основной", "Стартовый", "Плюс",
]

PARAM_MAP = {
    "edition": ("редакция", "edition", "версия продукта", "издание"),
    "term": ("срок", "срок действия", "срок лицензии", "период", "срок подписки", "term", "duration"),
    "users": ("количество пользователей", "кол-во пользователей", "пользователей", "количество узлов", "узлов",
              "количество устройств", "устройств", "количество лицензий", "users", "seats", "nodes"),
    "purchase_type": ("тип покупки", "тип лицензии", "вид лицензии", "license type", "тип"),
    "deployment": ("развертывание", "развёртывание", "размещение", "тип развертывания", "deployment"),
    "platforms": ("платформа", "ос", "операционная система", "platform"),
    "russian": ("реестр", "российское по", "реестр российского по", "страна", "страна производителя"),
    "vat": ("ндс", "ставка ндс", "vat"),
    "delivery": ("тип поставки", "способ поставки", "вид поставки", "delivery"),
    "certificates": ("сертификат", "сертификаты", "сертификация"),
}

TERM_RE = re.compile(r"(\d+)\s*(год|года|лет|year|years|мес|месяц\w*|month\w*)", re.IGNORECASE)
USERS_RE = re.compile(
    r"(?:(\d+)\s*[-–—]\s*)?(\d+)\s*(?:пользовател\w*|польз\.?|узл\w*|устройств\w*|users?|nodes?|devices?|"
    r"ПК|рабоч\w* мест\w*|почтов\w* ящик\w*|mailbox\w*|лиценз\w*|сотрудник\w*)", re.IGNORECASE)


def _param(params: dict[str, str], key: str) -> str:
    for alias in PARAM_MAP[key]:
        for name, value in params.items():
            if normalize(name) == alias:
                return value
    for alias in PARAM_MAP[key]:
        for name, value in params.items():
            if alias in normalize(name):
                return value
    return ""


def parse_term(text: str) -> int | None:
    t = normalize(text)
    if "бессроч" in t or "perpetual" in t:
        return 0
    m = TERM_RE.search(text or "")
    if not m:
        return None
    n = int(m.group(1))
    return n * 12 if m.group(2).lower().startswith(("год", "лет", "year")) else n


def parse_users(text: str) -> tuple[int, int] | None:
    m = USERS_RE.search(text or "")
    if not m:
        return None
    hi = int(m.group(2))
    lo = int(m.group(1)) if m.group(1) else 0
    return lo, hi


def parse_purchase_type(text: str) -> str:
    t = normalize(text)
    if re.search(r"продлен|renewal|продление|extension", t):
        return "renewal"
    if re.search(r"переход|миграц|upgrade|обновлен|cross-?grade", t):
        return "upgrade"
    if re.search(r"нов\w+ лиценз|new|первичн|покупка", t):
        return "new"
    return ""


def parse_edition(text: str) -> str:
    found = ""
    for word in sorted(EDITION_WORDS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text or "", re.IGNORECASE):
            found = word
            break
    return found


FAMILY_CUT_RE = re.compile(
    r",|\(|\bна\s+(?:\d|одн\w*)|\s(?:продлени\w*|renewal|base license|educational|upgrade|russian edition|для академ\w*|"
    r"для государ\w*|для коммерч\w*|для образоват\w*|для некоммерч\w*|подписк\w* на|версия на|лицензи\w* на)\b",
    re.IGNORECASE)
TAIL_STOP = {"для", "на", "с", "и", "продление", "лицензии", "лицензия", "лицензий", "версия", "подписка", "подписки",
             "обновление", "-", "–", "—", "license", "renewal", "russian", "edition"}
NULL_VALUES = {"", "другое или отсутствует", "не определено", "не определен", "нет данных"}

# Вендоры российского ПО — запасной признак, если в фиде не указан номер записи в реестре
RUSSIAN_VENDORS = re.compile(
    r"касперск|kaspersky|доктор веб|dr\.?\s?web|астра|astra|базальт|basealt|ред софт|red ?soft|редсофт|криптопро|"
    r"crypto-?pro|код безопасности|positive technologies|позитив|мойофис|новые облачные|\bр7\b|\br7\b|\b1с\b|"
    r"аладдин|aladdin|infowatch|инфовотч|киберпротект|cyberprotect|алтэкс|altx|яндекс|yandex|\bсбк\b|communigate|"
    r"контур|searchinform|серчинформ|staffcop|солар|usergate|юзергейт|инфотекс|infotecs|газинформсервис|эшелон|"
    r"postgres pro|постгрес|content ai|контент ай|ростелеком|vk cloud|vk workspace|т1 облако|группа астра",
    re.IGNORECASE)
ORG_MAP = {"коммерческая": "commercial", "академическая": "academic", "государственная": "government",
           "некоммерческая": "nonprofit", "для дома": "home"}


def strip_vendor(name: str, vendor: str) -> str:
    """Убрать дублирующее имя вендора: «MDaemon Technologies, Ltd. MDaemon Messaging Server» → «MDaemon
    Messaging Server», «Adobe Systems Adobe InCopy» → «Adobe InCopy». «КриптоПро CSP» не трогаем: без вендора
    название продукта теряет смысл."""
    v = (vendor or "").strip()
    if v and name.lower().startswith(v.lower()):
        rest = name[len(v):].strip(" ,.-–")
        brand = re.findall(r"\w{3,}", v.lower())
        legal_name = re.search(r"\b(?:ltd|inc|llc|gmbh|corp|corporation|ао|ооо|пао|зао|technologies|systems)\b",
                               v.lower())  # «MDaemon Technologies, Ltd.», «АО Р7» — юрлицо, а не бренд продукта
        if len(rest) >= 3 and ((brand and brand[0] in rest.lower()) or legal_name):
            return rest
    return name


def derive_family(name: str, vendor: str = "") -> str:
    """Название продукта без срока, числа пользователей, типа покупки и типа организации."""
    s = strip_vendor(name, vendor)
    s = re.sub(r"\s[-–—]\s", " ", s).replace("«", "").replace("»", "")
    s = FAMILY_CUT_RE.split(s)[0]
    s = TERM_RE.sub("", s)
    s = USERS_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -–—.,")
    return s or name


def group_family(names: list[str], vendor: str) -> str:
    """Название продукта по группе вариантов (group_id): общий префикс наименований."""
    if len(names) == 1:
        return derive_family(names[0], vendor)
    words = [strip_vendor(n, vendor).split() for n in names]
    prefix = words[0]
    for w in words[1:]:
        i = 0
        while i < min(len(prefix), len(w)) and prefix[i] == w[i]:
            i += 1
        prefix = prefix[:i]
    text = " ".join(prefix)
    if text.count("(") > text.count(")"):
        text = text[:text.rfind("(")]
    text = FAMILY_CUT_RE.split(text)[0]
    tokens_ = text.split()
    while tokens_ and (tokens_[-1].lower().strip(",.") in TAIL_STOP or not re.search(r"\w", tokens_[-1])):
        tokens_.pop()
    text = " ".join(tokens_).strip(" ,.-–—")
    return text if len(text) >= 4 and len(tokens_) >= 1 else ""


def _clean(v: str | None) -> str:
    v = (v or "").strip()
    return "" if v.lower() in NULL_VALUES else v


def softline_fields(name: str, vendor: str, params: dict[str, str], description: str) -> dict[str, Any]:
    """Атрибуты ПО из параметров фида store.softline.ru."""
    item: dict[str, Any] = {}
    lic = _clean(params.get("Тип лицензии")).lower()
    term_raw = _clean(params.get("Срок действия"))
    term = parse_term(term_raw) if term_raw else None
    if term is None:
        term = parse_term(name)
    item["term_months"] = term or 0
    if re.search(r"продлен", lic) or re.search(r"\bпродлени|renewal", name, re.IGNORECASE):
        item["purchase_type"] = "renewal"
    elif re.search(r"обновлен|миграц|upgrade", lic):
        item["purchase_type"] = "upgrade"
    elif lic:
        item["purchase_type"] = "new" if lic not in ("техподдержка", "лицензионный платеж") else ""
    if lic == "техподдержка":
        item["license_type"] = "service"
    elif "подписк" in lic or (term and term > 0):
        item["license_type"] = "subscription"
    elif term_raw.lower().startswith("бессроч") or lic in ("полная версия", "full", "базовая"):
        item["license_type"] = "perpetual"
    nodes = _clean(params.get("К-во узлов"))
    users_to = _clean(params.get("К-во пользователей до"))
    if users_to.isdigit():  # «до N пользователей» — цена за пакет
        item.update(users_min=0, users_max=int(users_to), pricing="package", unit="пакет")
    elif nodes.isdigit():  # «К-во узлов» — нижняя граница диапазона, цена за 1 узел
        item.update(users_min=int(nodes), users_max=0, pricing="per_unit", unit="узел")
    org = ORG_MAP.get(_clean(params.get("Тип организации")).lower(), "")
    if not org:
        n = name.lower()
        org = ("academic" if re.search(r"академ|образоват|education", n) else
               "government" if re.search(r"государ|government", n) else "")
    item["org_type"] = org
    registry = _clean(params.get("Номер записи в реестре отечественного ПО"))
    is_soft = _clean(params.get("Тип продукта")).lower() != "электроника"
    item["is_russian"] = 1 if (re.search(r"\d", registry) or (is_soft and RUSSIAN_VENDORS.search(f"{vendor} {name}"))) else 0
    certs = [c for c in [_clean(params.get("Сертификаты и стандарты"))] if c]
    if re.search(r"\d", registry):
        certs.append(f"реестр отечественного ПО № {registry}")
    item["certificates"] = "; ".join(certs)
    if is_soft and item["is_russian"]:
        item["vat_rate"] = 0  # права на ПО из реестра не облагаются НДС (пп. 26 п. 2 ст. 149 НК РФ)
    item["delivery"] = "box" if _clean(params.get("Тип поставки")).lower() == "физическая" else "electronic"
    if not is_soft:
        item.update(unit="шт", pricing="per_unit", stock_qty=0)
    platform = _clean(params.get("Совместимая архитектура/платформа"))
    if not platform and (m := re.search(r"Платформа:\s*([^•]+)", description)):
        platform = m.group(1).strip()
    item["platforms"] = platform
    item["edition"] = parse_edition(name)
    return item


def enrich(item: dict[str, Any], params: dict[str, str] | None = None) -> dict[str, Any]:
    """Дополнить оффер атрибутами ПО из параметров и наименования."""
    params = params or {}
    name = str(item.get("name") or "")
    if not item.get("edition"):
        item["edition"] = _param(params, "edition") or parse_edition(name)
    if item.get("term_months") in (None, ""):
        t = parse_term(_param(params, "term")) if _param(params, "term") else None
        item["term_months"] = t if t is not None else (parse_term(name) or 0)
    if not item.get("users_max") and not item.get("users_min"):
        u = parse_users(_param(params, "users")) if _param(params, "users") else None
        if u is None and _param(params, "users").strip().isdigit():
            u = (0, int(_param(params, "users").strip()))
        u = u or parse_users(name)
        if u:
            item["users_min"], item["users_max"] = u
            # «на 50 пользователей» — обычно цена за пакет; «25-49 узлов» — цена за 1 узел в диапазоне
            item.setdefault("pricing", "package" if u[0] == 0 else "per_unit")
    if not item.get("purchase_type"):
        item["purchase_type"] = parse_purchase_type(_param(params, "purchase_type") + " " + name)
    if not item.get("license_type"):
        lt = normalize(_param(params, "purchase_type") + " " + name)
        item["license_type"] = ("perpetual" if "бессроч" in lt or "perpetual" in lt
                                else "subscription" if (item.get("term_months") or re.search(r"подписк|subscription", lt))
                                else "")
    if not item.get("deployment"):
        d = normalize(_param(params, "deployment") + " " + name)
        item["deployment"] = "cloud" if re.search(r"облак|cloud|saas", d) else ("on-prem" if re.search(r"on-?prem", d) else "")
    if not item.get("platforms"):
        item["platforms"] = _param(params, "platforms")
    if "is_russian" not in item:
        r = normalize(_param(params, "russian"))
        item["is_russian"] = 1 if r and re.search(r"да|yes|1|росси|рф|включ", r) and "не " not in r else 0
    if item.get("vat_rate") in (None, "") and _param(params, "vat"):
        v = normalize(_param(params, "vat"))
        item["vat_rate"] = 0 if "без" in v else (re.search(r"\d+", v) and float(re.search(r"\d+", v).group(0)))
    if not item.get("delivery"):
        item["delivery"] = "box" if re.search(r"короб|box|физическ", normalize(_param(params, "delivery"))) else "electronic"
    if not item.get("certificates"):
        item["certificates"] = _param(params, "certificates")
    if not item.get("family"):
        item["family"] = derive_family(name, str(item.get("vendor") or "")) or name
    if params:
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        item["attributes"] = {**attrs, "params": params}
    return item


# ---------------- Форматы ----------------

def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def open_source(src: Any) -> BinaryIO:
    """bytes, путь к файлу или файловый объект; gzip распаковывается на лету."""
    if isinstance(src, (bytes, bytearray)):
        f: BinaryIO = io.BytesIO(src)
    elif isinstance(src, (str, Path)):
        f = open(src, "rb")
    else:
        f = src
    head = f.read(2)
    f.seek(0)
    if head == b"\x1f\x8b":
        return gzip.GzipFile(fileobj=f)  # type: ignore[return-value]
    return f


def parse_yml(source: Any) -> Iterable[dict[str, Any]]:
    """Потоковый разбор YML: офферы обрабатываются по одному, память не растёт с размером фида."""
    cats: dict[str, tuple[str, str]] = {}
    path_cache: dict[str, str] = {}
    seen_codes: set[str] = set()

    def cat_path(cid: str) -> str:
        if cid in path_cache:
            return path_cache[cid]
        names, seen, cur = [], set(), cid
        while cur and cur in cats and cur not in seen:
            seen.add(cur)
            names.insert(0, cats[cur][0])
            cur = cats[cur][1]
        path_cache[cid] = " / ".join(names)
        return path_cache[cid]

    offers_parent = None
    for event, el in ET.iterparse(open_source(source), events=("start", "end")):
        if event == "start":
            if el.tag == "offers":
                offers_parent = el
            continue
        if el.tag == "category":
            cats[el.get("id", "")] = (_text(el), el.get("parentId", ""))
            continue
        if el.tag != "offer":
            continue
        params: dict[str, str] = {}
        for p in el.findall("param"):
            params.setdefault(p.get("name", ""), (p.text or "").strip())
        vendor = _text(el.find("vendor"))
        name = _text(el.find("name")) or " ".join(filter(None, [_text(el.find("typePrefix")), vendor,
                                                                 _text(el.find("model"))]))
        offer_id = el.get("id", "")
        code = _text(el.find("vendorCode"))
        sku = code if code and code not in seen_codes else (f"{code}-{offer_id}" if code else f"SL-{offer_id}")
        seen_codes.add(sku)
        description = re.sub(r"<[^>]+>", " ", _text(el.find("description")))[:3000]
        group_id = el.get("group_id") or _text(el.find("group_id"))
        vendor_slug = re.sub(r"^https?://[^/]+/([^/]+)/.*$", r"\1", _text(el.find("vendorUrl")))
        category = cat_path(_text(el.find("categoryId")))
        item: dict[str, Any] = {
            "sku": sku,
            "name": name,
            "vendor": vendor,
            "category": category,
            "description": description,
            "price": _text(el.find("price")),
            "url": _text(el.find("url")),
            "available": 0 if el.get("available", "true").lower() == "false" else 1,
            "active": 1,
            "tags": " ".join(filter(None, [vendor_slug if "/" not in vendor_slug else "", _text(el.find("model")),
                                           category.split(" / ")[-1] if category else ""])),
            "attributes": {"offer_id": offer_id, "group_id": group_id, "vendor_code": code},
        }
        if "Тип продукта" in params or "К-во узлов" in params or "Тип организации" in params:
            item.update(softline_fields(name, vendor, params, description))
        yield enrich(item, params if "Тип продукта" not in params else {})
        el.clear()
        if offers_parent is not None:
            offers_parent.clear()


def finalize_groups() -> dict[str, int]:
    """После загрузки фида: названия продуктов по группам вариантов и верхние границы диапазонов узлов."""
    from collections import defaultdict

    from .. import db
    rows = db.query("SELECT id, name, vendor, family, json_extract(attributes, '$.group_id') AS g FROM products "
                    "WHERE json_extract(attributes, '$.group_id') IS NOT NULL AND json_extract(attributes, '$.group_id') != ''")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["g"]].append(r)
    updates = []
    for items in groups.values():
        # Внутри группы вариантов: если большинство вариантов названо одинаково, а остальные лишь
        # продолжают это название («… Стандартный Russian Edition»), приводим их к общему названию
        counts: dict[str, int] = defaultdict(int)
        for i in items:
            counts[i["family"]] += 1
        mode, n = max(counts.items(), key=lambda kv: kv[1])
        if n / len(items) < 0.5 or len(mode.split()) < 2:
            continue
        updates.extend((mode, i["id"]) for i in items
                       if i["family"] != mode and i["family"].lower().startswith(mode.lower()))
    with db.connect() as conn:
        conn.executemany("UPDATE products SET family = ? WHERE id = ?", updates)
    # Диапазоны «К-во узлов»: 10, 15, 20 … → 10–14, 15–19, 20–24 …
    tiers = db.query("SELECT id, family, purchase_type, term_months, org_type, edition, license_type, users_min "
                     "FROM products WHERE pricing = 'per_unit' AND users_min > 0 AND users_max = 0")
    keyed: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for t in tiers:
        keyed[(t["family"], t["purchase_type"], t["term_months"], t["org_type"], t["edition"], t["license_type"])].append(t)
    tier_updates = []
    for items in keyed.values():
        mins = sorted({i["users_min"] for i in items})
        nxt = {m: (mins[k + 1] - 1 if k + 1 < len(mins) else 0) for k, m in enumerate(mins)}
        tier_updates.extend((nxt[i["users_min"]], i["id"]) for i in items if nxt[i["users_min"]])
    with db.connect() as conn:
        conn.executemany("UPDATE products SET users_max = ? WHERE id = ?", tier_updates)
    return {"families_renamed": len(updates), "tiers": len(tier_updates)}


def parse_google_xml(content: bytes) -> Iterable[dict[str, Any]]:
    root = ET.fromstring(content)
    ns = {"g": "http://base.google.com/ns/1.0"}
    for it in list(root.iter("item")) + list(root.iter("{http://www.w3.org/2005/Atom}entry")):
        def g(tag: str) -> str:
            return _text(it.find(f"g:{tag}", ns)) or _text(it.find(tag))
        price = re.sub(r"[^\d.,]", "", g("price")).replace(",", ".")
        item = {"sku": g("id") or g("mpn"), "name": g("title"), "vendor": g("brand"),
                "category": g("product_type") or g("google_product_category"),
                "description": g("description")[:3000], "price": price, "url": g("link"),
                "active": 0 if g("availability") == "out of stock" else 1}
        yield enrich(item)


COLUMN_ALIASES = {
    "sku": ("sku", "артикул", "код", "id", "vendorcode", "article"),
    "name": ("name", "наименование", "название", "товар", "title"),
    "vendor": ("vendor", "производитель", "вендор", "brand", "бренд"),
    "family": ("family", "продукт", "линейка", "model", "модель"),
    "edition": ("edition", "редакция"),
    "category": ("category", "категория", "раздел", "группа"),
    "description": ("description", "описание"),
    "price": ("price", "цена", "цена, руб", "цена с ндс", "стоимость"),
    "url": ("url", "ссылка", "link"),
    "term": ("срок", "term", "срок лицензии", "период"),
    "users": ("пользователей", "количество пользователей", "users", "узлов", "устройств"),
    "purchase_type": ("тип покупки", "тип лицензии", "purchase_type"),
    "is_russian": ("is_russian", "российское по", "реестр"),
    "vat_rate": ("vat_rate", "ндс"),
    "tags": ("tags", "теги", "синонимы"),
    "deployment": ("deployment", "развертывание", "развёртывание"),
    "unit": ("unit", "ед", "ед.", "единица"),
    "pricing": ("pricing",),
    "active": ("active", "активен", "доступен"),
}


def _rows_to_items(rows: list[list[Any]]) -> Iterable[dict[str, Any]]:
    header = [normalize(str(h or "")) for h in rows[0]]
    mapping: dict[int, str] = {}
    for idx, h in enumerate(header):
        for field, aliases in COLUMN_ALIASES.items():
            if h in aliases:
                mapping[idx] = field
                break
    if "sku" not in mapping.values():
        raise ValueError("В файле нужна колонка «Артикул» (sku)")
    for row in rows[1:]:
        raw = {f: row[i] for i, f in mapping.items() if i < len(row) and row[i] not in (None, "")}
        if not raw:
            continue
        params = {}
        if "term" in raw:
            params["срок"] = str(raw.pop("term"))
        if "users" in raw:
            params["пользователей"] = str(raw.pop("users"))
        if "purchase_type" in raw and raw["purchase_type"] not in ("new", "renewal", "upgrade"):
            params["тип лицензии"] = str(raw.pop("purchase_type"))
        if "is_russian" in raw:
            raw["is_russian"] = 1 if re.search(r"1|да|true|yes", normalize(str(raw["is_russian"]))) else 0
        yield enrich(raw, params) if raw.get("name") else raw


def parse_table(filename: str, content: bytes) -> Iterable[dict[str, Any]]:
    if filename.lower().endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        ws = load_workbook(io.BytesIO(content), read_only=True, data_only=True).active
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("cp1251")
        dialect = csv.Sniffer().sniff(text[:4000], delimiters=";,\t")
        rows = list(csv.reader(io.StringIO(text), dialect=dialect))
    return _rows_to_items(rows) if rows else []


def parse_json(content: bytes) -> Iterable[dict[str, Any]]:
    data = json.loads(content.decode("utf-8-sig"))
    items = data.get("offers") or data.get("products") or data.get("items") if isinstance(data, dict) else data
    for it in items or []:
        params = it.pop("params", None) or {}
        yield enrich(dict(it), params if isinstance(params, dict) else {})


def import_feed(filename: str, content: Any, hide_demo: bool = True) -> dict[str, int]:
    """Загрузить фид. content — bytes, путь или файловый объект (большие фиды читаются потоково)."""
    from .. import db
    name = filename.lower()

    def reopen() -> BinaryIO:
        if hasattr(content, "seek"):
            content.seek(0)
        return open_source(content)

    head = reopen().read(4000)
    src = reopen()
    is_xml = name.endswith((".xml", ".yml", ".gz")) or head.lstrip().lower().startswith((b"<?xml", b"<yml"))
    if is_xml:
        items = parse_google_xml(src.read()) if b"base.google.com" in head else parse_yml(src)
    else:
        raw = src.read()
        items = parse_json(raw) if name.endswith(".json") else parse_table(filename, raw)
    batch: list[dict[str, Any]] = []
    total = {"created_or_updated": 0, "skipped": 0}
    for item in items:
        batch.append(item)
        if len(batch) >= 5000:
            res = catalog.bulk_upsert(batch)
            total = {k: total[k] + res[k] for k in total}
            batch = []
    if batch:
        res = catalog.bulk_upsert(batch)
        total = {k: total[k] + res[k] for k in total}
    if is_xml:
        total.update(finalize_groups())
    if hide_demo and total["created_or_updated"] >= 50:
        # Загружен настоящий каталог — демо-товары скрываются, чтобы не смешивать условные цены с реальными
        db.execute("UPDATE products SET active = 0 WHERE json_extract(attributes, '$.demo') = 1")
        total["demo_hidden"] = db.scalar(
            "SELECT COUNT(*) FROM products WHERE json_extract(attributes, '$.demo') = 1 AND active = 0") or 0
    return total
