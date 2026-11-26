from __future__ import annotations

from datetime import date
from pathlib import Path

from app.services import catalog, feed, pricing, settings_store, support_examples
from app.services.textutil import add_workdays, amount_in_words, fmt_money, make_inn_checksum, validate_inn

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# ---------------- Утилиты ----------------

def test_amount_in_words():
    assert amount_in_words(1234.5) == "Одна тысяча двести тридцать четыре рубля 50 копеек"
    assert amount_in_words(21) == "Двадцать один рубль 00 копеек"
    assert amount_in_words(2_000_000.01) == "Два миллиона рублей 01 копейка"


def test_money_inn_workdays():
    assert fmt_money(1234567.891) == "1 234 567,89"
    assert validate_inn("1234567894") and not validate_inn("1234567890")
    assert validate_inn(make_inn_checksum("500100732"))
    assert add_workdays(date(2026, 11, 27), 1) == date(2026, 11, 30)


# ---------------- Каталог: поиск и сверка версий ----------------

def test_search_finds_products_by_name_transliteration_and_task():
    assert catalog.search_families("нужен MDaemon Standard", limit=1)[0]["family"] == "MDaemon Standard"
    assert catalog.search_families("касперский продление", limit=1)[0]["vendor"] == "Лаборатория Касперского"
    assert catalog.search_families("доктор веб для почты", limit=1)[0]["family"] == "Dr.Web Mail Security Suite"
    top = {f["family"] for f in catalog.search_families("российский почтовый сервер", limit=2)}
    assert top == {"CommuniGate Pro", "RuPost"}
    assert catalog.search_families("резервное копирование", limit=1)[0]["family"] == "Кибер Бэкап Стандартный"
    only_ru = catalog.search_families("почтовый сервер", limit=5, filters={"russian_only": True})
    assert only_ru and all(f["russian_software"] for f in only_ru)


def test_resolve_variant_asks_only_missing_parameters():
    res = catalog.resolve_variant("MDaemon Standard")
    assert res["selected"] is None
    assert set(res["missing"]) == {"users", "purchase_type"}  # срок у MDaemon один — не спрашиваем

    res = catalog.resolve_variant("MDaemon Standard", users=30, purchase_type="new")
    assert res["selected"]["sku"] == "MD-STD-N-1Y-50"  # наименьший пакет, вмещающий 30 пользователей
    assert res["selected"]["quantity"] == 1

    res = catalog.resolve_variant("касперский стандартный", users=40, purchase_type="renewal")
    assert res["family"] == "Kaspersky Endpoint Security для бизнеса Стандартный"
    assert list(res["missing"]) == ["term_months"]

    res = catalog.resolve_variant("Kaspersky Endpoint Security для бизнеса Стандартный", users=40,
                                  purchase_type="renewal", term_months=12)
    sel = res["selected"]
    assert sel["sku"] == "KESB-STD-R-1Y-25-49" and sel["quantity"] == 40
    assert sel["vat"].startswith("без НДС")

    res = catalog.resolve_variant("MDaemon Standard", users=30, purchase_type="new", term_months=36)
    assert res["conflicts"]  # трёхлетней лицензии нет — сообщаем и предлагаем доступное


def test_find_russian_alternatives():
    res = catalog.find_alternatives("MDaemon Standard", russian_only=True)
    names = [a["family"] for a in res["alternatives"]]
    assert set(names) == {"CommuniGate Pro", "RuPost"}
    assert all(a["russian_software"] for a in res["alternatives"])


# ---------------- Цены и скидки ----------------

def test_vat_per_line_russian_software_exempt():
    calc = pricing.calculate([{"sku": "KESB-STD-R-1Y-25-49", "quantity": 40}, {"sku": "MD-STD-N-1Y-50", "quantity": 1}])
    kes, md = calc["lines"]
    assert kes["vat_rate"] == 0 and kes["vat_amount"] == 0
    assert md["vat_amount"] == round(79900 * 22 / 122, 2)
    assert calc["vat_amount"] == md["vat_amount"]
    assert calc["payment_terms"] == "100% предоплата"


def test_agent_discount_limit_and_manager_approval():
    items = [{"sku": "MD-STD-N-1Y-50", "quantity": 1}]
    assert pricing.calculate(items, 3)["discount_percent"] == 3
    calc = pricing.calculate(items, 10)
    assert calc["discount_percent"] == 5 and calc["requires_approval"]
    approved = pricing.calculate(items, 10, approved_discount_percent=10)
    assert approved["discount_percent"] == 10 and not approved["requires_approval"]
    assert approved["total"] == 71910.0
    settings_store.update("pricing", {"max_agent_discount_percent": 3})
    assert pricing.calculate(items, 5)["discount_percent"] == 3


def test_price_on_request_and_unknown_sku():
    calc = pricing.calculate([{"sku": "CGP-ENT-USER", "quantity": 50}, {"sku": "NOPE", "quantity": 1}])
    assert calc["lines"] == [] and calc["price_on_request"][0]["sku"] == "CGP-ENT-USER"
    assert calc["not_found"] == ["NOPE"]


# ---------------- Импорт фида и переписок ----------------

def test_yml_feed_import_with_params_and_name_parsing():
    res = feed.import_feed("feed.yml", (SAMPLES / "feed_sample.yml").read_bytes(), hide_demo=False)
    assert res["created_or_updated"] == 3 and res["skipped"] == 0
    new = catalog.get_by_sku("DS-EP-N-1Y-25-49")
    assert new["family"] == "Демо Endpoint Protection Business" and new["edition"] == "Business"
    assert (new["users_min"], new["users_max"], new["term_months"]) == (25, 49, 12)
    assert new["purchase_type"] == "new" and new["is_russian"] == 1 and new["vat_rate"] == 0
    assert new["category"] == "Антивирусы и безопасность / Антивирусы для организаций"
    assert catalog.get_by_sku("DS-EP-R-1Y-25-49")["purchase_type"] == "renewal"
    mail = catalog.get_by_sku("DM-SRV-50")
    assert mail["pricing"] == "package" and mail["users_max"] == 50 and mail["deployment"] == "on-prem"
    res = catalog.resolve_variant("Демо Endpoint Protection Business", users=30, term_months=12)
    assert res["missing"] == {"purchase_type": ["новая лицензия", "продление"]}


def test_softline_feed_format_gzip_tiers_and_availability(tmp_path):
    import gzip
    gz = tmp_path / "feed.yml.gz"
    gz.write_bytes(gzip.compress((SAMPLES / "softline_feed_sample.xml").read_bytes()))
    res = feed.import_feed("feed.yml.gz", gz)  # путь к файлу, потоковый разбор, gzip
    assert res["created_or_updated"] == 6
    assert "demo_hidden" not in res  # меньше 50 позиций — демо-каталог не трогаем

    kes = catalog.get_by_sku("DL-R-1Y-10")
    assert kes["family"] == "Демо Endpoint Защита для бизнеса Стандартный"
    assert (kes["users_min"], kes["users_max"], kes["pricing"]) == (10, 24, "per_unit")  # 10 → 10–24
    assert kes["purchase_type"] == "renewal" and kes["org_type"] == "commercial" and kes["term_months"] == 12
    assert kes["is_russian"] == 1 and kes["vat_rate"] == 0 and "12345" in kes["certificates"]
    assert catalog.get_by_sku("DL-RA-1Y-10")["org_type"] == "academic"

    mail = catalog.get_by_sku("SL-9101")  # пустой vendorCode → артикул по id оффера
    assert mail["family"] == "Mailer Server"  # юрлицо вендора убрано из названия продукта
    assert (mail["pricing"], mail["users_max"]) == ("package", 10)
    assert catalog.get_by_sku("SL-9103")["available"] == 0

    res = catalog.resolve_variant("Демо Endpoint Защита для бизнеса Стандартный", users=30, purchase_type="renewal")
    assert list(res["missing"]) == ["org_type"]  # коммерческая или академическая — спрашиваем
    res = catalog.resolve_variant("Демо Endpoint Защита для бизнеса Стандартный", users=30, purchase_type="renewal",
                                  org_type="commercial")
    assert res["selected"]["sku"] == "DL-R-1Y-25" and res["selected"]["quantity"] == 30

    res = catalog.resolve_variant("Mailer Server", users=80)  # пакет на 100 не продаётся — предлагать нельзя
    assert res["selected"] is None and res["conflicts"]
    assert catalog.resolve_variant("Mailer Server", users=30)["selected"]["sku"] == "SL-9102"


def test_chat_export_format_and_page_hint():
    feed.import_feed("feed.xml", (SAMPLES / "softline_feed_sample.xml").read_bytes())
    n = support_examples.import_file("chats.txt", (SAMPLES / "chats_sample.txt").read_bytes())
    assert n == 3
    cats = {c["category"] for c in support_examples.categories()}
    assert "Follow-up / не дозвонился" in cats
    hint = catalog.page_hint("store.softline.ru/demolab/demo-endpoint/")
    assert hint and hint["family"] == "Демо Endpoint Защита для бизнеса Стандартный"
    assert catalog.page_hint("store.softline.ru/search/") is None


def test_csv_partial_price_update():
    res = feed.import_feed("prices.csv", "Артикул;Цена\nMD-STD-N-1Y-50;81000\nUNKNOWN;1\n".encode("utf-8"))
    assert res["created_or_updated"] == 1 and res["skipped"] == 1
    p = catalog.get_by_sku("MD-STD-N-1Y-50")
    assert p["price"] == 81000 and p["family"] == "MDaemon Standard"


def test_support_examples_import_and_scrub():
    n = support_examples.import_file("chats.txt", (SAMPLES / "support_sample.txt").read_bytes())
    assert n == 3  # в первом диалоге две пары «вопрос — ответ», во втором одна
    found = support_examples.search("антивирус на 12 компьютеров", top_k=1)[0]
    assert "Small Office" in found["answer"]
    docs = support_examples.search("закрывающие документы", top_k=1)[0]
    assert "[email]" in docs["question"] and "@" not in docs["question"]
    csv_data = "chat_id;author;text\n1;client;Как продлить лицензию?\n1;operator;В личном кабинете, раздел Продукты.\n"
    assert support_examples.import_file("c.csv", csv_data.encode("utf-8")) == 1
