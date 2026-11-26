"""Офлайн-ассистент: работает без ИИ — на правилах, поиске по каталогу и базе знаний.

Используется, когда ИИ не настроен или недоступен, и покрывает сценарии задачи:
1. Клиент знает продукт → сверка версий (select_variant) и вопросы только про недостающие параметры.
2. Клиент не знает решения → до 3 вопросов (задача, масштаб, требования) → 2 варианта с обоснованием.
3. Ответы на вопросы по сведениям о продуктах и базе знаний, российские аналоги.
4. Скидка до 5% сразу, больше — менеджеру; только предоплата; КП/счёт; передача менеджеру.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from ..services import catalog, customers, dialogs, knowledge, pricing, settings_store
from ..services.feed import parse_term
from ..services.textutil import (COMPANY_RE, EMAIL_RE, INN_RE, KPP_RE, PHONE_RE, fmt_date, fmt_money,
                                 normalize)
from .llm_agent import AgentReply
from .tools import KB_RELEVANCE, ToolContext, run_tool

PRODUCT_MIN_SCORE = 0.45
CONTACT_LABELS = {"company": "", "inn": "ИНН", "kpp": "КПП", "name": "", "phone": "тел.", "email": "email",
                  "address": "адрес"}

RX = {
    "human": re.compile(r"оператор|менеджер|живо\w* человек|с человеком|позвон|перезвон|свяж\w+ со мной|"
                        r"связаться|дозвон"),
    "complaint": re.compile(r"претензи|жалоб|не работает ключ|ключ не (?:подходит|активир)|не пришел|не пришёл|"
                            r"верните деньги|обман|списали"),
    "special": re.compile(r"тендер|госзакуп|44-?фз|223-?фз|дилер|партн[её]рск|индивидуальн\w* услов|договор поставки"),
    "followup": re.compile(r"вопрос реш|ждем|ждём|жду (?:ответ|сч)|отправлял|отправляла|отправил\w* (?:\w+ )?запрос|"
                           r"ранее (?:платили|оплачивали|работали)|почему (?:увелич|вырос|изменил)|"
                           r"не получил|номер заказа|по заказу|заказ №|по сч[её]ту №|оплатил|не дозвон|не берут трубку"),
    "deferral": re.compile(r"отсрочк|постоплат|рассрочк|в кредит|оплат\w* (?:позже|потом|после)|заплат\w* (?:позже|потом)"),
    "discount": re.compile(r"скидк|дешевле|подешевле|уступ"),
    "invoice": re.compile(r"сч[её]т\w*\s+(?:на оплату|выстав)|выстав\w*\s+сч[её]т|\bсч[её]т\b"),
    "quote": re.compile(r"\bкп\b|коммерческ\w* предложени|сформир\w* предложени|оформ\w* предложени"),
    "clear": re.compile(r"очист\w*|начн\w* заново|сброс"),
    "skip": re.compile(r"без (?:контакт|реквизит|данн)|не (?:хочу|буду) оставлять|не нужно|пропуст"),
    "analog": re.compile(r"аналог|отечествен|импортозамещ|замен\w* на росс|российск\w* (?:вариант|решени|продукт|по\b)"),
    "need": re.compile(r"не знаю|не уверен|подбер|посоветуй|посоветуете|порекоменд|что выбрать|какой выбрать|"
                       r"нужно защитить|хотим защитить|надо защитить|нужно решение|ищем решение|помогите выбрать|"
                       r"что (?:нам )?(?:нужно|подойд)"),
    "details": re.compile(r"что входит|чем отлича|подробнее|расскажите|поддерживает|платформ|linux|линукс|windows|"
                          r"macos|сертифик|фстэк|реестр|как работает|что умеет"),
    "about_price": re.compile(r"сколько стоит|стоимост|цен[ауы]|прайс|купить|приобрест|заказать|лиценз|в наличии|"
                              r"есть ли|можно ли|срок|продлить|продлени"),
    "choice": re.compile(r"(?:вариант|опци\w*)\s*(?:№\s*)?([12])|\b(перв|втор)\w*"),
    "greeting": re.compile(r"^(здравствуй\w*|добр\w+ (?:день|утро|вечер)|привет\w*|доброго времени)[\s!.,]*$"),
    "thanks": re.compile(r"^(спасибо|благодар\w*|ок|понятно|хорошо)[\s!.,]*\w*[\s!.,]*$"),
    "policy": re.compile(r"доставк|оплат|документ|упд|эдо|ндс|возврат|верну|продлени\w* лиценз|личн\w* кабинет|ключ|"
                         r"режим работы|контакт|телефон|отмен|предзаказ|доверенност|подписк"),
}

USERS_RE = re.compile(
    r"(\d[\d\s]{0,6})\s*(?:пользовател\w*|польз\.?|сотрудник\w*|человек|чел\.?|компьютер\w*|комп\w*|пк\b|узл\w*|"
    r"устройств\w*|рабоч\w* мест\w*|ящик\w*|лиценз\w*|машин\w*|ноутбук\w*|сервер\w*|рабочих станц\w*)",
    re.IGNORECASE)
WORD_TERM = {"на год": 12, "на один год": 12, "на 1 год": 12, "на два года": 24, "на 2 года": 24,
             "на три года": 36, "на 3 года": 36, "бессрочн": 0}

TOPICS = {
    "mail": (re.compile(r"почт|e-?mail|емейл|exchange|почтов|переписк"), "защита почты почтовый сервер"),
    "files": (re.compile(r"файл|данн\w|документы на сервер"), "защита файлов файловый сервер"),
    "endpoint": (re.compile(r"компьютер|\bпк\b|антивирус|вирус|рабоч\w* станц|ноутбук|рабоч\w* мест"),
                 "антивирус защита компьютеров рабочих станций"),
    "backup": (re.compile(r"бэкап|backup|резервн|восстановлен|шифровальщик"), "резервное копирование"),
    "office": (re.compile(r"офис|таблиц|ворд|word|excel|редактор документ"), "офисный пакет документы"),
}
TOPIC_LABELS = {"mail": "почта", "files": "файлы", "endpoint": "компьютеры", "backup": "резервные копии",
                "office": "офисные программы"}
GENERIC_WORDS = {
    "для", "бизнеса", "защита", "защиты", "security", "suite", "сервер", "серверов", "почтовых", "почтовый", "почта",
    "standard", "стандартный", "стандартная", "расширенный", "pro", "профессиональный", "for", "plus", "desktop",
    "mail", "server", "лиценз", "лицензия", "лицензии", "офис", "офисный", "small", "office", "кибер", "endpoint",
    "станция", "связь", "обратная", "версия", "версии", "программа", "система", "решение", "платформа", "модуль",
    "комплект", "пакет", "подписка", "продление", "сервис", "услуга", "работы", "рабочая", "рабочее", "место",
    "edition", "professional", "enterprise", "business", "premium", "basic", "home", "cloud", "антивирус",
    "ноутбук", "планшет", "компьютер", "software", "technologies", "systems", "ltd", "inc", "ооо", "ао", "пао",
    "the", "and", "или", "на", "по", "с", "и", "в", "от", "до", "все", "new", "base", "license",
    # слова-параметры лицензии: их называют, уточняя вариант, а не продукт
    "редакция", "редакции", "расширенная", "стандартная", "базовая", "корпоративная", "сертификат", "сертификата",
    "сопровождение", "сопровождения", "техподдержка", "поддержка", "коммерческая", "коммерческой", "организация",
    "организации", "образовательная", "государственная", "новая", "новый", "новую", "продажа", "покупка",
    "обновление", "переход", "миграция", "подписки", "год", "года", "лет", "месяц", "месяцев", "пользователей",
    "пользователя", "узлов", "узел", "устройств", "рабочих", "мест", "сервера", "серверная", "десктоп",
    # части названий вендоров, которые встречаются в обычной речи («для ИП», «Я люблю ИП», «Группа …»)
    "ип", "я", "люблю", "группа", "компания", "технологии", "решения", "лаборатория", "центр", "студия",
}
PLACEHOLDERS = re.compile(r"\[(?:email|тел|телефон|номер|инн|отправил файл|отправила файл|файл)\]", re.IGNORECASE)


# ---------------- Извлечение параметров ----------------

def extract_contacts(text: str, expect_fio: bool = False) -> dict[str, str]:
    found: dict[str, str] = {}
    if m := EMAIL_RE.search(text):
        found["email"] = m.group(0)
    if m := PHONE_RE.search(text):
        found["phone"] = re.sub(r"[^\d+]", "", m.group(0))
    if m := INN_RE.search(text):
        found["inn"] = m.group(1)
    if m := KPP_RE.search(text):
        found["kpp"] = m.group(1)
    if m := COMPANY_RE.search(text):
        name = m.group(2).strip().strip("«»\"“” ")
        if len(name) <= 50 and "?" not in name and "софтлайн" not in name.lower():
            found["company"] = f"{m.group(1).upper()} «{name}»" if m.group(1).upper() != "ИП" else f"ИП {name}"
    if m := re.search(r"(?:меня зовут|мое имя|моё имя|контактное лицо)\s*:?\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)", text):
        found["name"] = m.group(1)
    elif expect_fio and (m := re.match(r"^\s*([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s*$", text)):
        found["name"] = m.group(1)
    return found


NUM_WORDS = {
    "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12, "пятнадцать": 15,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
    "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200, "триста": 300, "пятьсот": 500,
}
UNITS_AFTER = re.compile(
    r"(?:пользовател\w*|сотрудник\w*|человек|компьютер\w*|пк|узл\w*|устройств\w*|ящик\w*)\s+(?:на|примерно|около)?\s*(\d{1,6})")


def words_to_digits(text: str) -> str:
    """«двадцать пять человек» → «25 человек» (для расшифровок звонков и разговорной речи)."""
    out, acc, last = [], 0, 0
    for tok in re.findall(r"\w+|\W+", text):
        val = NUM_WORDS.get(tok.lower())
        # «сто двадцать пять»: каждое следующее слово — меньший разряд
        if val is not None:
            if acc and len(str(val)) < len(str(last)):
                acc += val
            else:
                if acc:
                    out.append(f"{acc} ")
                acc = val
            last = val
            continue
        if acc and not tok.strip():
            continue
        if acc:
            out.append(f"{acc} ")
            acc = 0
        out.append(tok)
    if acc:
        out.append(str(acc))
    return "".join(out)


def extract_params(text: str, focus_rows: list[dict[str, Any]] | None = None,
                   expect_number: bool = False) -> dict[str, Any]:
    text = words_to_digits(text)
    t = normalize(text)
    params: dict[str, Any] = {}
    if m := USERS_RE.search(text):
        params["users"] = int(re.sub(r"\D", "", m.group(1)))
    elif m := UNITS_AFTER.search(t):  # «человек на тридцать»
        params["users"] = int(m.group(1))
    elif expect_number and (m := re.fullmatch(r"\s*(\d{1,6})\s*", text)):
        params["users"] = int(m.group(1))
    for phrase, months in WORD_TERM.items():
        if phrase in t:
            params["term_months"] = months
    term = parse_term(text)
    if term is not None and "term_months" not in params and re.search(r"год|лет|мес|бессроч", t):
        params["term_months"] = term
    if re.search(r"продл|renew|истека|заканчива", t):
        params["purchase_type"] = "renewal"
    elif re.search(r"нов\w+ лиценз|впервые|первичн|ещ[её] нет|нет лицензии|купить нов", t):
        params["purchase_type"] = "new"
    if re.search(r"облак|cloud|saas", t):
        params["deployment"] = "cloud"
    elif re.search(r"сво\w* сервер|сервер\w* (?:у нас )?сво|локальн|on-?prem|у себя|свой сервер|своя инфраструкт", t):
        params["deployment"] = "on-prem"
    if re.search(r"некоммерч|\bнко\b", t):
        params["org_type"] = "nonprofit"
    elif re.search(r"коммерческ|\bооо\b|\bип\b|компани", t):
        params["org_type"] = "commercial"
    elif re.search(r"образоват|академ|учебн|\bвуз|школ|университет|колледж", t):
        params["org_type"] = "academic"
    elif re.search(r"государ|бюджетн|\bгос\b|муниципал", t):
        params["org_type"] = "government"
    if re.search(r"российск|отечествен|реестр|импортозамещ", t):
        params["russian"] = True
    elif re.search(r"не обязательно российск|любое|не важно|неважно", t):
        params["russian"] = False
    if focus_rows:
        for ed in sorted({r["edition"] for r in focus_rows if r["edition"]}, key=len, reverse=True):
            e = normalize(ed)
            stem_e = e[:6] if len(e) > 6 else e
            if re.search(rf"(?<!\w){re.escape(stem_e)}", t) or (e == "standard" and "стандарт" in t) or \
                    (e == "pro" and re.search(r"\bпро\b|\bpro\b", t)):
                params["edition"] = ed
                break
    return params


def describe(fam: dict[str, Any]) -> str:
    """Короткое описание продукта: текст описания или (если в фиде описание — список атрибутов) категория."""
    desc = fam.get("description") or ""
    if not desc or "•" in desc or re.match(r"^[А-ЯA-Z][\w ]{2,30}:", desc):
        return (fam.get("category") or "").split(" / ")[-1] or fam["family"]
    return short(desc)


def short(text: str, limit: int = 170) -> str:
    """Первое предложение описания, без обрыва посреди слова."""
    first = text.split(". ")[0].strip()
    if len(first) > limit:
        first = first[:limit].rsplit(" ", 1)[0] + "…"
    return first.rstrip(".") + "."


ENDINGS = {"", "а", "я", "у", "ю", "е", "и", "ы", "о", "ом", "ем", "ой", "ей", "ам", "ям", "ах", "ях", "ов", "ев",
           "ий", "ый", "ого", "его", "ому", "ему", "ым", "им", "ых", "их", "ую", "юю", "ая", "яя", "ое", "ее", "ми"}


def _word_match(a: str, b: str) -> bool:
    """Совпадение слов с учётом падежных окончаний («касперский» ≈ «касперского»), но «фотошоп» ≠ «фотошоу»."""
    if a == b:
        return True
    short_, long_ = sorted((a, b), key=len)
    if len(short_) >= 5 and long_.startswith(short_) and len(long_) - len(short_) <= 4:
        return True
    i = 0
    while i < len(short_) and short_[i] == long_[i]:
        i += 1
    return i >= 5 and short_[i:] in ENDINGS and long_[i:] in ENDINGS


def match_strength(text: str, family: dict[str, Any]) -> int:
    """Насколько явно клиент назвал продукт: 2 — бренд вендора, код модели или очень редкое слово названия;
    1 — совпало относительно редкое слово названия (возможна случайность); 0 — не назвал."""
    t = normalize(text)
    query_words = set(re.findall(r"\w+", t))
    for tok in list(query_words):
        query_words.update(re.findall(r"\w+", catalog.SYNONYMS.get(tok, "")))
    df = catalog.family_token_df()
    vendor_words = set(re.findall(r"\w+", normalize(family["vendor"])))
    best = 0
    for w in re.findall(r"\w+", normalize(f"{family['family']} {family['vendor']}")):
        if w in GENERIC_WORDS or w.isdigit() or len(w) < 2:
            continue
        # бренд вендора отличителен всегда; слово из названия — только если оно редкое в каталоге
        if w not in vendor_words and df.get(w, 0) > 12:
            continue
        if any(_word_match(w, q) for q in query_words if len(q) >= 2):
            # сильное совпадение — бренд вендора, код модели или редкое латинское название; обычное русское
            # слово (даже редкое в каталоге: «прямая», «обложки») — только повод предложить варианты
            strong = (w in vendor_words or (any(c.isdigit() for c in w) and len(w) >= 4)
                      or (df.get(w, 0) <= 2 and re.fullmatch(r"[a-z][a-z0-9-]+", w) is not None))
            best = max(best, 2 if strong else 1)
    return best


def names_product(text: str, family: dict[str, Any]) -> bool:
    return match_strength(text, family) > 0


COMMON_LATIN = {
    "windows", "linux", "server", "office", "home", "cloud", "edition", "license", "licence", "standard",
    "professional", "enterprise", "business", "basic", "plus", "premium", "version", "subscription", "with",
    "full", "user", "users", "device", "year", "years", "email", "mail", "web", "online", "client", "support",
    "pro", "the", "and", "for", "new", "vpn", "https", "http", "www", "com", "softline", "store", "html",
}


def model_codes(text: str) -> list[str]:
    """Коды моделей и артикулов из сообщения: буквы и цифры вместе (ux8406ca, mfc-l2700dwr, sfn300t-ap)."""
    t = normalize(re.sub(r"https?://\S+", " ", text))
    codes = re.findall(r"\b(?=[\w-]*\d)(?=[\w-]*[a-z])[a-z0-9][a-z0-9-]{3,19}\b", t)
    return [c for c in dict.fromkeys(codes) if c.count("-") <= 2 and not re.fullmatch(r"\d+[a-z]{1,2}", c)]


def unknown_brands(text: str) -> list[str]:
    """Латинские названия из сообщения, которых нет ни в одном продукте каталога (anydesk, uipath…)."""
    df = catalog.family_token_df()
    vendors = catalog.vendor_token_set()
    words = re.findall(r"\b[a-z][a-z0-9\-]{3,}\b", normalize(re.sub(r"https?://\S+", " ", text)))
    return [w for w in dict.fromkeys(words)
            if w not in COMMON_LATIN and w not in vendors and not df.get(w) and not any(c.isdigit() for c in w)]


class OfflineAgent:
    def reply(self, dialog_id: int, client_text: str, ctx: ToolContext) -> AgentReply:
        self.dialog_id = dialog_id
        state = dialogs.get(dialog_id)["state"]
        client_text = PLACEHOLDERS.sub(" ", client_text)  # маски обезличивания — не часть вопроса
        t = normalize(client_text)
        parts: list[str] = []

        def finish(*extra: str) -> AgentReply:
            parts.extend(e for e in extra if e)
            return AgentReply(text="\n\n".join(p for p in parts if p).strip(), mode="offline",
                              escalation=ctx.escalation, documents=ctx.documents, kb_used=ctx.kb_used,
                              tools_used=ctx.tools_used)

        # 1. Контакты и реквизиты — из любого сообщения
        contact = extract_contacts(client_text, expect_fio=state.get("pending") in ("invoice", "quote"))
        if contact:
            result, err = run_tool(ctx, "save_customer_contact", contact)
            if not err:
                saved = ", ".join(f"{lab} {contact[k]}".strip() for k, lab in CONTACT_LABELS.items() if k in contact)
                parts.append(f"Записал: {saved}.")
                parts.extend(f"Обратите внимание: {w}." for w in result.get("warnings", []))

        # Названный в сообщении продукт (нужен и для писем с несколькими просьбами сразу)
        focus = state.get("focus") or {}
        found = catalog.search_families(client_text, limit=3)
        # Явное упоминание вендора/продукта — сильный сигнал, поэтому порог ниже (длинные письма «размывают» оценку)
        named = next((f for f in found if f["relevance"] >= 0.4 and names_product(client_text, f)), None)
        named_strength = match_strength(client_text, named) if named else 0
        if named and focus.get("family") and named["family"] != focus["family"] and \
                extract_params(client_text) and named_strength < 2 and named["relevance"] < 0.75:
            # «на 30 пользователей, новая лицензия» — уточнение текущего продукта, а не новый; но если клиент
            # явно назвал вендора или модель («продлить Касперский Стандартный») — это новый запрос
            named, named_strength = None, 0
        if not named and not focus.get("family") and RX["about_price"].search(t):
            # «Сколько стоит лицензия?» со страницы продукта — значит, про этот продукт
            dlg = dialogs.get(dialog_id)
            hint = catalog.page_hint(dlg["subject"]) if dlg["channel"] == "web" and dlg["subject"] else None
            if hint and hint.get("family"):
                named = catalog.family_summary(hint["family"])
                named_strength = 2  # страница продукта — надёжный признак
        missing_brands = [] if focus.get("family") else unknown_brands(client_text)
        if named and missing_brands and not any(b in normalize(f"{named['family']} {named['vendor']}")
                                                for b in missing_brands):
            named = None  # клиент назвал продукт, которого нет в каталоге, — не подменяем его похожим
        models = [] if focus.get("family") else model_codes(client_text)
        if named and models:
            hay = normalize(" ".join(r["name"] for r in catalog.family_rows(named["family"])[:200]) + named["family"])
            if not any(m in hay.replace(" ", "") or m in hay for m in models):
                # спросили конкретную модель (UX8406CA, MFC-L2700DWR), а нашли другую — не выдаём её за нужную
                missing_brands = [m.upper() for m in models]
                named = None

        def product_part() -> str:
            """Подбор варианта для продукта, названного в том же сообщении."""
            nonlocal state
            f = focus if named and named["family"] == focus.get("family") else {"family": named["family"]}
            text = self._variant(ctx, f, client_text, state)
            state = dialogs.get(dialog_id)["state"]
            return text

        # 2. Явные запросы
        if RX["human"].search(t):
            return self._escalate(ctx, parts, "client_request", client_text, state)
        if RX["followup"].search(t) and not named:
            parts.append("Уточню статус у менеджера — он ответит здесь же.")
            return self._escalate(ctx, parts, "other", client_text, state)
        if missing_brands and not named:
            parts.append(f"«{', '.join(missing_brands)}» в каталоге магазина не нашёл. Уточню у менеджера, можно ли "
                         f"поставить этот продукт или подобрать аналог.")
            return self._escalate(ctx, parts, "no_answer", client_text, state)
        if RX["complaint"].search(t):
            parts.append("Сожалею, что так вышло — разберёмся.")
            return self._escalate(ctx, parts, "complaint", client_text, state)
        if RX["discount"].search(t):
            if named and not state.get("cart"):
                parts.append(product_part())
            return finish(self._discount(ctx, client_text, state, parts))
        if RX["deferral"].search(t):
            kb, _ = self._kb_answer(ctx, "отсрочка постоплата оплатить позже", KB_RELEVANCE)
            payment = kb or ("Мы работаем по 100% предоплате, отсрочки и постоплаты нет. Если хотите обсудить "
                             "особые условия с менеджером — напишите «менеджер», подключу его.")
            if named:
                return finish(product_part(), "По оплате: " + payment[0].lower() + payment[1:])
            return finish(payment)
        if RX["special"].search(t):
            return self._escalate(ctx, parts, "special_terms", client_text, state)
        if RX["clear"].search(t) and state.get("cart"):
            dialogs.update_state(dialog_id, cart=[], pending=None, focus=None, need=None)
            return finish("Расчёт очищен. Что подобрать?")

        # 3. КП и счёт
        if state.get("pending") == "quote" and (contact or RX["skip"].search(t)):
            return finish(self._quote(ctx, state, force=True))
        wants_invoice = RX["invoice"].search(t) or (state.get("pending") == "invoice" and contact)
        wants_quote = RX["quote"].search(t)
        if (wants_invoice or wants_quote) and named and named_strength == 2 and not state.get("cart"):
            # «Пришлите КП на Сканер-ВС 7» — сначала подбираем вариант названного продукта
            parts.append(product_part())
            if not state.get("cart"):
                dialogs.update_state(dialog_id, pending="invoice" if wants_invoice else "quote")
                return finish("Как только определимся с вариантом, сразу подготовлю "
                              + ("счёт." if wants_invoice else "коммерческое предложение."))
        if wants_invoice:
            return finish(self._invoice(ctx, state))
        if wants_quote:
            return finish(self._quote(ctx, state, force=bool(contact)))

        # 4. Выбор из предложенных вариантов
        if state.get("offer") and (m := RX["choice"].search(t)):
            idx = 0 if (m.group(1) == "1" or (m.group(2) or "").startswith("перв")) else 1
            return finish(self._choose_option(ctx, state, idx))

        # 5. Российский аналог
        if RX["analog"].search(t) and (state.get("focus") or state.get("offer")):
            return finish(self._alternatives(ctx, state))

        # 6. Продукт назван явно / уточнение параметров выбранного продукта
        if named and named_strength == 1 and named["family"] != focus.get("family") and not RX["need"].search(t):
            # Совпадение по общему слову: не утверждаем, что это нужный товар, — предлагаем выбрать
            options = [f for f in found if f["relevance"] >= 0.4][:3] or [named]
            dialogs.update_state(dialog_id, focus={"family": options[0]["family"]} if len(options) == 1 else None)
            return finish(self._products_list(options))
        if named and not RX["need"].search(t):
            if named["family"] != focus.get("family"):
                focus = {"family": named["family"]}
            if RX["details"].search(t) and not extract_params(client_text):
                dialogs.update_state(dialog_id, focus=focus, need=None)
                return finish(self._details(named["family"]),
                              "Подсказать стоимость? Напишите количество пользователей/устройств и срок.")
            return finish(self._variant(ctx, focus, client_text, state))
        focus_rows = catalog.family_rows(focus["family"]) if focus.get("family") else []
        params = extract_params(client_text, focus_rows, expect_number=bool(focus.get("asked_users")))
        if focus.get("family") and RX["details"].search(t):
            return finish(self._details(focus["family"]))
        if focus.get("family") and not state.get("need") and any(k in params for k in
                                                                ("users", "term_months", "purchase_type",
                                                                 "edition", "deployment", "org_type")):
            return finish(self._variant(ctx, focus, client_text, state))

        # 7. Подбор решения, когда клиент не знает продукт
        topics = [k for k, (rx, _) in TOPICS.items() if rx.search(t)]
        need = state.get("need")
        if RX["need"].search(t) or (topics and not RX["policy"].search(t) and not named) or \
                (need and (params or topics)):
            return finish(self._need(ctx, state, topics, params))

        if RX["greeting"].match(t):
            return finish(settings_store.get("assistant")["greeting"])
        if RX["thanks"].match(t):
            return finish("Пожалуйста! Если появятся вопросы — пишите.")
        if contact:
            return finish("Чем ещё помочь?")

        # 8. Вопросы об условиях (база знаний) или о продуктах
        kb_text, kb_score = self._kb_answer(
            ctx, client_text, float(settings_store.get("assistant")["kb_min_score"]), track=False)
        top = found[0] if found else None
        if kb_text and (RX["policy"].search(t) or not top or kb_score >= top["relevance"]):
            self._kb_answer(ctx, client_text, 0)
            return finish(kb_text)
        if top and top["relevance"] >= PRODUCT_MIN_SCORE:
            dialogs.update_state(dialog_id, focus={"family": top["family"]})
            return finish(self._products_list(found))
        if kb_text:
            self._kb_answer(ctx, client_text, 0)
            return finish(kb_text)

        knowledge.register_miss(dialog_id, client_text)
        parts.append("Точного ответа в сведениях о продуктах и базе знаний у меня нет.")
        return self._escalate(ctx, parts, "no_answer", client_text, state)

    # ---------------- Сценарий «знает продукт» ----------------

    def _variant(self, ctx: ToolContext, focus: dict[str, Any], text: str, state: dict[str, Any]) -> str:
        rows = catalog.family_rows(focus["family"])
        params = extract_params(text, rows, expect_number=bool(focus.get("asked_users")))
        for k in ("users", "term_months", "purchase_type", "edition", "deployment", "org_type"):
            if k in params:
                focus[k] = params[k]
        res, _ = run_tool(ctx, "select_variant", {"product": focus["family"], **{
            k: focus[k] for k in ("users", "term_months", "purchase_type", "edition", "deployment", "org_type")
            if focus.get(k) not in (None, "")}})
        if "error" in res:
            return f"Не нашёл продукт «{focus['family']}» в каталоге. Уточните название?"
        lines = []
        for c in res["conflicts"]:
            lines.append(f"Обратите внимание: {c}.")
        if res["selected"]:
            s = res["selected"]
            focus["asked_users"] = False
            cart = [i for i in state.get("cart", []) if not catalog.get_by_sku(i["sku"])
                    or catalog.get_by_sku(i["sku"])["family"] != res["family"]]
            cart.append({"sku": s["sku"], "quantity": s["quantity"]})
            dialogs.update_state(self.dialog_id, focus=focus, cart=cart, need=None)
            price = (f"{fmt_money(s['price_rub'])} ₽ ({s['pricing']})" if s.get("price_rub") else "цена по запросу")
            total = f"\nСумма: {fmt_money(s['estimated_total_rub'])} ₽, {s['vat']}." if s.get("estimated_total_rub") else ""
            lines.append(f"Подходит: {s['name']} [{s['sku']}]\n• Цена: {price}, количество: {s['quantity']}{total}\n"
                         f"• Наличие и срок: {s['availability']}"
                         + (f"\n• Страница продукта: {s['url']}" if s.get("url") else ""))
            if not focus.get("users") and s["pricing"].startswith("цена за 1"):
                focus["asked_users"] = True
                dialogs.update_state(self.dialog_id, focus=focus)
                lines.append(f"Сколько нужно ({s['pricing'].replace('цена за 1 ', '')})? Пересчитаю сумму. "
                             "Или сразу сформирую КП.")
            else:
                lines.append("Сформировать коммерческое предложение? Также могу рассказать подробнее о продукте "
                             "или подобрать российский аналог.")
        else:
            questions = []
            opts = res["missing"]
            if "users" in opts:
                focus["asked_users"] = True
                questions.append(f"• Сколько пользователей/устройств нужно защитить или подключить? "
                                 f"(варианты лицензий: {', '.join(opts['users'][:6])})")
            if "purchase_type" in opts:
                questions.append(f"• Это новая лицензия или продление действующей? ({', '.join(opts['purchase_type'])})")
            if "term_months" in opts:
                questions.append(f"• На какой срок? ({', '.join(opts['term_months'])})")
            if "edition" in opts:
                questions.append(f"• Какая редакция? ({', '.join(opts['edition'])})")
            if "deployment" in opts:
                questions.append(f"• Где развернуть? ({', '.join(opts['deployment'])})")
            if "license_type" in opts:
                questions.append(f"• Тип лицензии? ({', '.join(opts['license_type'])})")
            if "org_type" in opts:
                questions.append(f"• Для какой организации лицензия? ({', '.join(opts['org_type'])}) — цены отличаются")
            dialogs.update_state(self.dialog_id, focus=focus, need=None)
            summary = catalog.family_summary(res["family"])
            lines.append(f"{res['family']} есть в каталоге ({summary['price_from']}). Чтобы подобрать точный "
                         f"вариант лицензии, уточните:\n" + "\n".join(questions))
        return "\n\n".join(lines)

    def _details(self, family: str) -> str:
        s = catalog.family_summary(family)
        if not s:
            return "Не нашёл сведений о продукте."
        parts = [f"{s['family']} ({s['vendor']}): {s['description']}"]
        facts = []
        if s["platforms"]:
            facts.append(f"платформы: {s['platforms']}")
        if s["deployment"]:
            facts.append(f"развёртывание: {s['deployment']}")
        facts.append("российское ПО из реестра" if s["russian_software"] else "зарубежное ПО")
        if s["certificates"]:
            facts.append(f"сертификаты: {s['certificates']}")
        if s["editions"]:
            facts.append(f"редакции: {', '.join(s['editions'])}")
        parts.append("• " + "\n• ".join(facts))
        if s["url"]:
            parts.append(f"Подробнее на странице продукта: {s['url']}")
        return "\n".join(parts)

    def _alternatives(self, ctx: ToolContext, state: dict[str, Any]) -> str:
        families = [state["focus"]["family"]] if state.get("focus") else \
            [f for opt in state.get("offer", []) for f in opt["families"]]
        out = []
        for fam in families[:2]:
            base = catalog.family_summary(fam)
            if base and base["russian_software"]:
                out.append(f"{fam} — уже российское ПО из реестра.")
                continue
            res, err = run_tool(ctx, "find_alternatives", {"product": fam, "russian_only": True})
            alts = [] if err else res["alternatives"]
            if not alts:
                out.append(f"Для {fam} российских аналогов в каталоге не нашёл — могу уточнить у менеджера.")
                continue
            items = "\n".join(f"• {a['family']} ({a['vendor']}) — {describe(a)} "
                              + (f"Цена {a['price_from'].rstrip('.')}." if a["price_from"] != "цена по запросу" else
                                 "Цена по запросу.") + (f" {a['url']}" if a.get("url") else "") for a in alts)
            out.append(f"Российские аналоги {fam}:\n{items}")
        out.append("Рассчитать стоимость какого-то из вариантов?")
        return "\n\n".join(out)

    # ---------------- Сценарий «не знает продукт» ----------------

    def _need(self, ctx: ToolContext, state: dict[str, Any], topics: list[str], params: dict[str, Any]) -> str:
        need = state.get("need") or {"topics": [], "asked": 0}
        for tp in topics:
            if tp not in need["topics"]:
                need["topics"].append(tp)
        for k in ("users", "russian", "deployment"):
            if k in params:
                need[k] = params[k]
        questions = []
        if not need["topics"]:
            questions.append("• Что нужно защитить или автоматизировать: компьютеры сотрудников, почту, файлы на "
                             "серверах, резервные копии, офисные программы?")
        if not need.get("users"):
            questions.append("• Сколько сотрудников (или компьютеров, почтовых ящиков) нужно охватить?")
        if need.get("russian") is None:
            questions.append("• Обязательно ли российское ПО из реестра (например, для импортозамещения или "
                             "госзаказчиков)?")
        if "mail" in need["topics"] and not need.get("deployment") and len(questions) < 3:
            questions.append("• Почта у вас на своём сервере или в облаке?")
        if questions and need["asked"] < 2 and not (need["topics"] and need.get("users") and need["asked"] >= 1):
            need["asked"] += 1
            dialogs.update_state(self.dialog_id, need=need, focus=None)
            intro = ("Помогу подобрать решение. " if need["asked"] == 1 else "") + "Уточните, пожалуйста:"
            return intro + "\n" + "\n".join(questions[:3])
        return self._recommend(ctx, need)

    def _recommend(self, ctx: ToolContext, need: dict[str, Any]) -> str:
        users = int(need.get("users") or 10)
        filters = {"russian_only": bool(need.get("russian")),
                   "deployment": need.get("deployment") if need.get("deployment") in ("cloud", "on-prem") else None}
        topics = need["topics"] or ["endpoint"]
        per_topic: dict[str, list[dict[str, Any]]] = {}
        for tp in topics:
            found = [f for f in catalog.search_families(TOPICS[tp][1], limit=8, filters=filters)
                     if f["relevance"] >= 0.3]
            buyable = [f for f in found if f["in_stock"] and f["price_from"] != "цена по запросу"]
            found = (buyable or found)[:4]  # сначала то, что можно купить сразу
            per_topic[tp] = found
        offers = []
        for idx in (0, 1):
            fams, items, total, notes = [], [], 0.0, []
            for tp, found in per_topic.items():
                if not found:
                    continue
                fam = found[min(idx, len(found) - 1)]
                if fam["family"] in fams:
                    continue
                per_server = any(r["unit"] == "сервер" for r in catalog.family_rows(fam["family"]))
                n = 1 if per_server else users  # лицензия «за сервер»: по умолчанию один файловый сервер
                res = catalog.resolve_variant(fam["family"], users=n, purchase_type="new", term_months=12)
                # Сумму считаем только при однозначном совпадении варианта, иначе показываем «цену от»:
                # угадывание пакета/диапазона на реальном фиде даёт ошибки в разы
                sel = res["selected"] if not res["conflicts"] else None
                fams.append(fam["family"])
                if sel and sel.get("price_rub"):
                    qty = sel.get("quantity") or (1 if sel["pricing"] == "цена за пакет" else n)
                    items.append({"sku": sel["sku"], "quantity": qty})
                    total += sel["price_rub"] * qty
                why = describe(fam).rstrip(".")
                flags = ", ".join(filter(None, ["российское ПО" if fam["russian_software"] else "",
                                                fam["certificates"] and f"сертификат {fam['certificates']}",
                                                fam["deployment"]]))
                qty_note = " (расчёт на 1 сервер)" if per_server else ""
                notes.append(f"   – {TOPIC_LABELS.get(tp, tp)}: {fam['family']} — {why} ({flags}); "
                             f"{fam['price_from']}{qty_note}")
            if fams and (not offers or fams != offers[0]["families"]):
                offers.append({"families": fams, "items": items, "total": total, "notes": notes})
        if not offers:
            dialogs.update_state(self.dialog_id, need=need)
            return ("Не нашёл в каталоге подходящих решений под эти требования. Передать запрос менеджеру? "
                    "Напишите «менеджер».")
        dialogs.update_state(self.dialog_id, need=None, offer=[{"families": o["families"], "items": o["items"]}
                                                              for o in offers])
        lines = [f"Под задачу ({', '.join(TOPIC_LABELS.get(t, t) for t in topics)}, {users} польз.) предлагаю:"]
        for i, o in enumerate(offers, 1):
            est = f" — ориентировочно {fmt_money(o['total'])} ₽ за год" if o["total"] else ""
            lines.append(f"Вариант {i}{est}:\n" + "\n".join(o["notes"]))
        lines.append("Какой вариант ближе? Могу рассказать подробнее, найти российский аналог или сразу "
                     "сформировать КП.")
        return "\n\n".join(lines)

    def _choose_option(self, ctx: ToolContext, state: dict[str, Any], idx: int) -> str:
        offer = state["offer"][min(idx, len(state["offer"]) - 1)]
        dialogs.update_state(self.dialog_id, cart=offer["items"],
                             focus={"family": offer["families"][0]} if offer["families"] else None)
        calc, _ = run_tool(ctx, "calculate_offer", {"items": offer["items"]}) if offer["items"] else ({}, True)
        text = pricing.summary_text(calc) if calc.get("lines") else ""
        return (f"Отлично, вариант {idx + 1}: {', '.join(offer['families'])}.\n\n{text}\n\nСформировать КП? "
                "Укажите, на какую организацию.").strip()

    # ---------------- Скидка, КП, счёт, менеджер ----------------

    def _discount(self, ctx: ToolContext, text: str, state: dict[str, Any], parts: list[str]) -> str:
        limit = float(settings_store.get("pricing").get("max_agent_discount_percent", 5))
        approved = ctx.approved_discount
        m = re.search(r"(\d{1,2}(?:[.,]\d)?)\s*%", text)
        wanted = float(m.group(1).replace(",", ".")) if m else limit
        give = min(wanted, max(limit, approved))
        dialogs.update_state(self.dialog_id, discount_percent=give)
        lines = [f"Могу сразу предоставить скидку {give:g}%."]
        if state.get("cart"):
            calc, _ = run_tool(ctx, "calculate_offer", {"items": state["cart"], "discount_percent": give})
            if calc.get("lines"):
                lines.append(pricing.summary_text(calc))
        if wanted > max(limit, approved):
            res, _ = run_tool(ctx, "escalate_to_manager", {
                "reason": "special_terms", "requested_discount_percent": wanted,
                "summary": f"Клиент просит скидку {wanted:g}% (дано {give:g}%). Сообщение: {text[:500]}"})
            lines.append(f"Скидку {wanted:g}% должен согласовать менеджер. {res.get('tell_client', '')}")
        elif not state.get("cart"):
            lines.append("Скидка будет учтена в расчёте и КП.")
        return "\n\n".join(lines)

    def _quote(self, ctx: ToolContext, state: dict[str, Any], force: bool = False) -> str:
        if not state.get("cart"):
            dialogs.update_state(self.dialog_id, pending="quote")
            return "Подготовлю КП. Какой продукт и на сколько пользователей/устройств нужен?"
        c = customers.get(ctx.customer_id)
        if not force and not (c and (c["company"] or c["name"])):
            dialogs.update_state(self.dialog_id, pending="quote")
            return ("Как указать получателя в КП — название организации? И оставьте, пожалуйста, телефон или "
                    "e-mail для связи. Если не хотите — напишите «без контактов», сформирую так.")
        res, err = run_tool(ctx, "create_quote", {"items": state["cart"],
                                                  "discount_percent": state.get("discount_percent") or None})
        if err:
            return f"Не получилось сформировать КП: {res['error']}"
        dialogs.update_state(self.dialog_id, pending=None)
        sla = settings_store.get("assistant")["operator_sla_minutes"]
        text = (f"Черновик коммерческого предложения № {res['number']} на {fmt_money(res['total'])} ₽ готов: "
                f"{res['url']}\nМенеджер проверит его в течение {sla} мин. Предложение действует до "
                f"{fmt_date(date.fromisoformat(res['valid_until']))}, оплата — 100% предоплата.")
        if res.get("price_on_request"):
            text += "\nПозиции с ценой по запросу менеджер добавит отдельно."
        return text

    def _invoice(self, ctx: ToolContext, state: dict[str, Any]) -> str:
        if not state.get("cart"):
            dialogs.update_state(self.dialog_id, pending="invoice")
            return "Выставлю счёт. Какой продукт и на сколько пользователей/устройств нужен?"
        c = customers.get(ctx.customer_id)
        if not c or not (c["company"] or c["name"]) or (c["company"] and not c["inn"]):
            dialogs.update_state(self.dialog_id, pending="invoice")
            return ("Для счёта напишите название организации и ИНН (например: ООО «Альфа», ИНН 1234567894). "
                    "Для частного лица — ФИО.")
        res, err = run_tool(ctx, "create_invoice", {"quote_number": state.get("last_quote", ""), "items": state["cart"]})
        if err:
            return f"Не получилось выставить счёт: {res['error']}"
        dialogs.update_state(self.dialog_id, pending=None)
        return (f"Счёт № {res['number']} на {fmt_money(res['total'])} ₽ подготовлен: {res['url']}\n"
                f"Оплатить до {fmt_date(date.fromisoformat(res['pay_until']))}. После оплаты лицензии придут на "
                f"e-mail.")

    def _escalate(self, ctx: ToolContext, parts: list[str], reason: str, client_text: str,
                  state: dict[str, Any] | None = None) -> AgentReply:
        summary = f"Сообщение клиента: {client_text[:800]}"
        if state and state.get("cart"):
            calc = pricing.calculate(state["cart"], state.get("discount_percent"), ctx.approved_discount)
            summary += f"\nВ расчёте: {calc['items_count']} поз. на {fmt_money(calc['total'])} ₽"
        if state and state.get("focus"):
            summary += f"\nОбсуждаемый продукт: {state['focus'].get('family')}"
        already = dialogs.get(self.dialog_id)["status"] == "waiting_operator"
        result, _ = run_tool(ctx, "escalate_to_manager", {"reason": reason, "summary": summary})
        parts.append("Добавил этот вопрос для менеджера — он ответит здесь же, в чате." if already
                     else result.get("tell_client", ""))
        return AgentReply(text="\n\n".join(p for p in parts if p), mode="offline", escalation=ctx.escalation,
                          documents=ctx.documents, kb_used=ctx.kb_used, tools_used=ctx.tools_used)

    # ---------------- Справка ----------------

    def _kb_answer(self, ctx: ToolContext, query: str, min_score: float, track: bool = True) -> tuple[str, float]:
        if track:
            result, _ = run_tool(ctx, "search_knowledge_base", {"query": query[:500]})
            results = result.get("results", [])
        else:
            results = [{"question": r["question"], "answer": r["answer"], "relevance": r["score"], "source": r["url"]}
                       for r in knowledge.search(query, top_k=3, track=False)]
        if not results or results[0]["relevance"] < max(min_score, KB_RELEVANCE):
            return "", 0.0
        text = results[0]["answer"]
        if results[0].get("source", "").startswith("http"):
            text += f"\nИсточник: {results[0]['source']}"
        return text, results[0]["relevance"]

    def _products_list(self, products: list[dict[str, Any]]) -> str:
        lines = ["Вот что нашёл в каталоге:"]
        for p in products:
            if p["relevance"] < PRODUCT_MIN_SCORE * 0.6:
                continue
            ru = ", российское ПО" if p["russian_software"] else ""
            lines.append(f"• {p['family']} ({p['vendor']}{ru}) — {p['price_from']}")
        lines.append("Какой продукт интересует? Подскажите количество пользователей/устройств и срок — подберу "
                     "точный вариант лицензии.")
        return "\n".join(lines)
