"""Текстовые утилиты: нормализация, деньги, сумма прописью, ИНН, рабочие дни."""
from __future__ import annotations

import re
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def normalize(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[ \s]+", " ", text)
    return text.strip()


def money(value: float | Decimal | int) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def fmt_money(value: float | Decimal | int) -> str:
    """12345.5 -> '12 345,50'"""
    d = money(value)
    sign = "-" if d < 0 else ""
    d = abs(d)
    integer, frac = f"{d:.2f}".split(".")
    groups = []
    while integer:
        groups.insert(0, integer[-3:])
        integer = integer[:-3]
    return f"{sign}{' '.join(groups)},{frac}"


def fmt_qty(value: float | int) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", ",")


def fmt_date(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def fmt_date_wd(d: date) -> str:
    return f"{fmt_date(d)} ({WEEKDAYS[d.weekday()]})"


def add_workdays(start: date, days: int) -> date:
    """Прибавить рабочие дни (пн–пт). Праздники не учитываются — сроки ориентировочные."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


_UNITS_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_UNITS_F = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
          "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]
# (формы, женский род)
_ORDERS = [
    (("", "", ""), False),
    (("тысяча", "тысячи", "тысяч"), True),
    (("миллион", "миллиона", "миллионов"), False),
    (("миллиард", "миллиарда", "миллиардов"), False),
    (("триллион", "триллиона", "триллионов"), False),
]


def _triad_words(n: int, feminine: bool) -> list[str]:
    words = []
    h, rest = divmod(n, 100)
    if h:
        words.append(_HUNDREDS[h])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
    else:
        t, u = divmod(rest, 10)
        if t:
            words.append(_TENS[t])
        if u:
            words.append((_UNITS_F if feminine else _UNITS_M)[u])
    return words


def int_to_words(n: int, feminine: bool = False) -> str:
    if n == 0:
        return "ноль"
    if n < 0:
        return "минус " + int_to_words(-n, feminine)
    words: list[str] = []
    order = 0
    while n:
        n, triad = divmod(n, 1000)
        if triad:
            (forms, fem) = _ORDERS[order]
            part = _triad_words(triad, fem if order else feminine)
            if order:
                part.append(plural(triad, *forms))
            words = part + words
        order += 1
    return " ".join(words)


def amount_in_words(value: float | Decimal) -> str:
    """1234.5 -> 'Одна тысяча двести тридцать четыре рубля 50 копеек'"""
    d = money(value)
    rubles = int(d)
    kopecks = int((d - rubles) * 100)
    text = f"{int_to_words(rubles)} {plural(rubles, 'рубль', 'рубля', 'рублей')} " \
           f"{kopecks:02d} {plural(kopecks, 'копейка', 'копейки', 'копеек')}"
    return text[0].upper() + text[1:]


def validate_inn(inn: str) -> bool:
    """Проверка контрольных цифр ИНН (10 знаков — юрлицо, 12 — ИП/физлицо)."""
    if not inn or not inn.isdigit() or len(inn) not in (10, 12):
        return False
    digits = [int(c) for c in inn]

    def check(coeffs: list[int]) -> int:
        return sum(c * d for c, d in zip(coeffs, digits)) % 11 % 10

    if len(inn) == 10:
        return check([2, 4, 10, 3, 5, 9, 4, 6, 8]) == digits[9]
    return (check([7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == digits[10]
            and check([3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == digits[11])


def make_inn_checksum(prefix9: str) -> str:
    """Дописать контрольную цифру к 9 цифрам — для демо-данных и тестов."""
    digits = [int(c) for c in prefix9]
    coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
    return prefix9 + str(sum(c * d for c, d in zip(coeffs, digits)) % 11 % 10)


QTY_RE = re.compile(
    r"(\d[\d\s]*(?:[.,]\d+)?)\s*(шт|штук|штуки|м\b|метр\w*|бухт\w*|упак\w*|уп\b|компл\w*|кг|рулон\w*|пар\w*)",
    re.IGNORECASE,
)


def extract_quantity(text: str) -> int | None:
    m = QTY_RE.search(text)
    if not m:
        m = re.search(r"(?:x|х|\*|×|кол-?во|количеств\w*)\s*:?\s*(\d+)", text, re.IGNORECASE)
        if not m:
            return None
    raw = m.group(1).replace(" ", "").replace(",", ".")
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return None


INN_RE = re.compile(r"(?<!\d)(\d{10}|\d{12})(?!\d)")
KPP_RE = re.compile(r"кпп\s*:?\s*(\d{9})", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")
COMPANY_RE = re.compile(r"\b(ООО|ОАО|АО|ПАО|ЗАО|ИП|НКО|ГУП|МУП)\s+([«\"“]?[^,\n;]+?[»\"”]?)(?=,|;|\n|\s+инн|\s+кпп|$)",
                        re.IGNORECASE)
