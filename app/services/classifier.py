"""Классификатор обращений: тип обращения и IT-тема.

Как устроен:
- две «головы»: тип обращения (обучается на размеченных переписках Softline) и IT-тема (справочник
  ключевых слов, 31 категория, файл app/resources/it_topics_handbook.json);
- обучаемая модель на TF-IDF; по умолчанию «hybrid» — слова и символьные n-граммы с логистической
  регрессией плюс нейросеть rubert-tiny2, дообученная на тех же переписках (neural.py). Есть и базовый
  вариант «nb» (TF-IDF + наивный Байес), сравнение — scripts/evaluate_classifier.py;
- справочник ключевых слов важнее модели: нашлось однозначное ключевое слово — берём его категорию;
- ключевые слова сверяются по границам слов с учётом окончаний («ПО» не находится в «почему»),
  при совпадении слов нескольких категорий побеждает категория с более специфичными словами;
- ниже 80 % уверенности обращение попадает менеджеру на проверку; подтверждение или исправление
  становится обучающим примером, модель дообучается;
- категории с одним примером не обучаются, для обучения нужно не меньше 10 примеров.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .. import db
from . import neural

log = logging.getLogger(__name__)

HEADS = {"request_type": "Тип обращения", "it_topic": "IT-тема"}
ALGORITHMS = {
    "hybrid": "Нейросеть rubert-tiny2 + слова и символьные n-граммы",
    "logreg": "Слова + символьные n-граммы, логистическая регрессия",
    "nb": "Базовая: TF-IDF (слова и биграммы) + наивный Байес",
}
DEFAULT_ALGORITHM = "hybrid"
NEURAL_WEIGHT = 0.3    # доля нейросети в итоговой вероятности (подобрано на переписках Softline)
LOW_CONFIDENCE = 0.8        # ниже — «низкая уверенность», тип проверяет менеджер
KEYWORD_CONFIDENCE = 0.95   # категория определена по однозначному ключевому слову справочника
DECISIVE = {"request_type": 1.0, "it_topic": 2.0}  # вес ключевых слов, достаточный для решения
MIN_PER_CATEGORY = 2
MIN_SAMPLES = 10
MAX_DIALOG_TEXT = 2000
HANDBOOK_FILE = Path(__file__).resolve().parent.parent / "resources" / "it_topics_handbook.json"

# Справочник типов обращений: ключевые фразы, по которым тип определяется сразу (правится в админке).
# Категории совпадают с разметкой переписок Softline.
REQUEST_TYPE_KEYWORDS: dict[str, list[str]] = {
    "Follow-up / не дозвонился": [
        "не дозвонился", "не могу дозвониться", "не получается дозвониться", "никто не отвечает",
        "не берут трубку", "до сих пор не получил", "до сих пор нет ответа", "жду ответа", "ждем ответа",
        "уже оплатили", "оплатили счет", "ранее отправлял запрос", "отправлял запрос", "когда ждать ключ",
    ],
    "Госзакупки / процедуры": [
        "44-ФЗ", "223-ФЗ", "госзакупка", "госзакупки", "госконтракт", "тендер", "аукцион", "котировка",
        "конкурентная закупка", "бюджетная организация", "НМЦК", "обоснование цены", "ЕИС",
        "закупочная процедура",
    ],
    "Физлицо / самозанятый / срок": [
        "физлицо", "физ лицо", "физическое лицо", "частное лицо", "самозанятый", "для домашнего использования",
        "для дома", "для личных целей", "для личного использования", "банковской картой",
    ],
    "Скидки / льготы / спецусловия": [
        "скидка", "скидку", "льгота", "по акции", "спецусловия", "специальные условия", "специальная цена",
        "партнерская цена", "являемся партнерами", "бонус",
    ],
    "Счет / КП как документ (с реквизитами)": [
        "выставить счет", "выставьте счет", "счет на оплату", "реквизиты", "коммерческое предложение", "КП",
        "счет-оферта", "запрос на КП", "прислать КП",
    ],
    "Доступность вендора (можно ли купить)": [
        "можно ли купить", "можно ли приобрести", "возможно ли приобрести", "есть возможность приобрести",
        "можно у вас купить", "можно у вас приобрести", "продаете ли", "работаете ли с", "доступен ли",
        "есть ли у вас",
    ],
    "Конфигурация / спецификация": [
        "какую лицензию выбрать", "какие лицензии нужны", "в чем отличие", "чем отличается", "спецификация",
        "конфигурация", "какая редакция",
    ],
}

_PLACEHOLDER = re.compile(r"\[[^\]]{1,30}\]")
_lock = threading.RLock()
_models: dict[str, tuple[tuple, "TextClassifier | None"]] = {}
_matchers: dict[str, tuple[tuple, "KeywordMatcher"]] = {}
_disputes: dict[str, tuple[tuple, list[dict[str, Any]]]] = {}


# ----------------------------------------------------------------------------- текст и ключевые слова

def normalize(text: str, lower: bool = True) -> str:
    """Нормализация: тире, NFKC, нижний регистр, ё→е и удаление плейсхолдеров вида [email]."""
    text = (text or "").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = unicodedata.normalize("NFKC", text)
    text = (text.lower() if lower else text).replace("ё", "е").replace("Ё", "Е")
    text = _PLACEHOLDER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# В справочнике IT-тем бренды записаны латиницей, клиенты пишут и по-русски («Касперский», «зум»)
_TRANSLIT = {"касперск": "kaspersky", "есет": "eset", "зум": "zoom", "тимс": "teams", "скайп": "skype",
             "вебекс": "webex", "аутлук": "outlook", "линукс": "linux", "виндовс": "windows", "винд": "windows",
             "докер": "docker", "вмвар": "vmware", "эксчейндж": "exchange", "вайфай": "wifi", "андроид": "android",
             "айос": "ios", "тандерберд": "thunderbird", "хайпер": "hyper-v", "1с": "1c", "сап": "sap"}


def _with_translit(norm: str) -> str:
    extra = {v for t in re.findall(r"[a-zа-я0-9]+", norm) for k, v in _TRANSLIT.items()
             if t == k or (len(k) >= 4 and t.startswith(k))}
    return f"{norm} {' '.join(sorted(extra))}" if extra else norm


# Аббревиатуры, совпадающие с обычными словами («ПО» — «по»): ищутся только заглавными буквами
_AMBIGUOUS_SHORT = {"по", "на", "от", "до", "не", "из", "за", "он", "ад", "ис", "ит", "ad", "it", "is", "in", "on",
                    "or", "to", "as", "at", "id"}


_STOP = {
    "это", "что", "как", "для", "или", "если", "когда", "где", "почему", "который", "которая", "которое",
    "которые", "может", "должен", "нужно", "можно", "нельзя", "будет", "было", "есть", "быть", "мой", "моя",
    "мое", "мои", "меня", "тебя", "вас", "нас", "их", "какой", "какая", "какое", "какие", "добрый", "день",
    "здравствуйте", "подскажите", "пожалуйста", "спасибо", "вечер", "утро", "доброе", "уважением", "нам",
    "нас", "так", "уже", "еще", "все", "вам", "мне", "при", "под", "над", "без", "про", "чем", "ваш", "наш",
}


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """Ключевые слова обращения: частотные слова от 3 букв без стоп-слов."""
    words = re.findall(r"[a-zа-я0-9][a-zа-я0-9\-]{2,}", normalize(text))
    words = [w.strip("-") for w in words if w.strip("-") not in _STOP and not w.isdigit()]
    return [w for w, _ in Counter(w for w in words if len(w) >= 3).most_common(max_keywords)]


def _needle(keyword: str) -> str:
    """Буквальный фрагмент, без которого ключевое слово не может совпасть: быстрая проверка подстрокой
    перед регулярным выражением (основа первого слова)."""
    if is_case_sensitive(keyword):
        return normalize(keyword, lower=False)
    tokens = re.findall(r"[a-zа-я0-9]+", normalize(keyword))
    if not tokens:
        return ""
    t = tokens[0]
    if re.fullmatch(r"[а-я]+", t) and len(t) > 3:
        return t[:len(t) - 1 if len(t) <= 5 else max(5, len(t) - 3)]
    return t


def _token_pattern(token: str) -> str:
    """Шаблон одного слова ключевой фразы: русские слова — с любым окончанием, остальное — точно."""
    if re.fullmatch(r"[а-я]+", token) and len(token) > 3:
        stem_len = len(token) - 1 if len(token) <= 5 else max(5, len(token) - 3)
        stem = token[:stem_len]
        return re.escape(stem) + r"[а-я]{0,%d}" % (len(token) - stem_len + 3)
    return re.escape(token)


def is_case_sensitive(keyword: str) -> bool:
    return normalize(keyword) in _AMBIGUOUS_SHORT and keyword.strip().isupper()


def keyword_pattern(keyword: str) -> re.Pattern[str] | None:
    """Шаблон ключевого слова. Регистрозависимые аббревиатуры («ПО») сверяются с текстом без перевода
    в нижний регистр — см. KeywordMatcher.match."""
    if is_case_sensitive(keyword):
        token = normalize(keyword, lower=False)
        return re.compile(r"(?<![A-Za-zА-Яа-я0-9])" + re.escape(token) + r"(?![A-Za-zА-Яа-я0-9])")
    tokens = re.findall(r"[a-zа-я0-9]+", normalize(keyword))
    if not tokens:
        return None
    gap = r"(?:[\s\-./,]+[a-zа-я0-9]+){0,2}[\s\-./,]*" if len(tokens) > 1 else ""
    body = gap.join(_token_pattern(t) for t in tokens)
    return re.compile(r"(?<![a-zа-я0-9])" + body + r"(?![a-zа-я0-9])")


class KeywordMatcher:
    """Справочник «категория → ключевые слова» с весами: чем в большем числе категорий встречается
    слово, тем меньше его вес. Названия и аббревиатуры (Kaspersky, VMware, ИБП, 44-ФЗ) весят 2,
    фразы — 1.5, обычные слова — 1. Категория выбирается по справочнику, если её вес не меньше
    decisive и в полтора раза больше, чем у следующей."""

    def __init__(self, handbook: dict[str, list[str]], decisive: float = 1.0,
                 strong: set[tuple[str, str]] | None = None):
        """strong — пары (категория, слово), заданные как однозначные (например, бренд, перенесённый из
        облака слов): им вес 2 независимо от написания."""
        self.decisive = decisive
        strong = strong or set()
        owners: dict[str, set[str]] = {}
        for category, words in handbook.items():
            for w in words:
                owners.setdefault(normalize(w), set()).add(category)
        self.rules: list[tuple[str, str, re.Pattern[str], float, bool, str]] = []
        for category, words in handbook.items():
            for w in words:
                pattern = keyword_pattern(w)
                if pattern is None:
                    continue
                key = normalize(w)
                if (category, w) in strong or re.search(r"[a-z0-9]", key) or (w.isupper() and len(w) >= 2):
                    strength = 2.0
                elif re.search(r"[\s\-]", key):
                    strength = 1.5
                else:
                    strength = 1.0
                self.rules.append((category, w, pattern, strength / len(owners[key]), is_case_sensitive(w),
                                   _needle(w)))

    def match(self, text: str) -> dict[str, Any]:
        """Возвращает {scores: {категория: вес}, hits: {категория: [слова]}, winner: категория|None}."""
        norm, original = _with_translit(normalize(text)), normalize(text, lower=False)
        scores: dict[str, float] = {}
        hits: dict[str, list[str]] = {}
        for category, word, pattern, weight, case_sensitive, needle in self.rules:
            if word in hits.get(category, []):
                continue
            source = original if case_sensitive else norm
            if needle in source and pattern.search(source):
                scores[category] = scores.get(category, 0.0) + weight
                hits.setdefault(category, []).append(word)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        winner = None
        if ranked and ranked[0][1] >= self.decisive and (len(ranked) == 1 or ranked[0][1] >= 1.5 * ranked[1][1]):
            winner = ranked[0][0]
        return {"scores": dict(ranked), "hits": hits, "winner": winner}


# ----------------------------------------------------------------------------- модель

class TextClassifier:
    """Обучаемая модель.

    hybrid — взвешенное среднее вероятностей TF-IDF-модели (слова и символьные n-граммы) и нейросети
    rubert-tiny2, дообученной на тех же переписках (neural.py). TF-IDF-часть переобучается мгновенно,
    нейросеть — в фоне; до первого дообучения или без PyTorch hybrid работает как logreg.
    """

    def __init__(self, algorithm: str = DEFAULT_ALGORITHM):
        self.algorithm = algorithm if algorithm in ALGORITHMS else DEFAULT_ALGORITHM
        self.pipeline = None
        self.neural = None
        self.categories: list[str] = []
        self.removed: list[str] = []
        self.metrics: dict[str, Any] = {}

    @property
    def uses_neural(self) -> bool:
        return self.neural is not None

    def _make_pipeline(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline

        if self.algorithm == "nb":
            from sklearn.naive_bayes import MultinomialNB
            return Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)),
                ("classifier", MultinomialNB(alpha=0.1)),
            ])
        from sklearn.linear_model import LogisticRegression
        return Pipeline([
            ("tfidf", FeatureUnion([
                ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
                ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
            ])),
            # C=50: при пороге 0.8 уверенных ответов втрое больше, чем при C=5, точность почти та же
            ("classifier", LogisticRegression(max_iter=3000, C=50.0, class_weight="balanced")),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> "TextClassifier":
        self.pipeline = self._make_pipeline().fit([normalize(t) for t in texts], labels)
        self.categories = [str(c) for c in self.pipeline.classes_]
        return self

    def predict_many(self, texts: list[str]) -> list[dict[str, float]]:
        if self.pipeline is None:
            return [{} for _ in texts]
        proba = self.pipeline.predict_proba([normalize(t) for t in texts])
        if self.neural is not None and self.algorithm == "hybrid":
            proba = blend(proba, self.categories, self.neural.predict(texts), self.neural.labels)
        return [{c: float(p) for c, p in zip(self.categories, row)} for row in proba]

    def predict_proba(self, text: str) -> dict[str, float]:
        return self.predict_many([text])[0]

    def train(self, texts: list[str], labels: list[str], low_confidence: float = LOW_CONFIDENCE,
              keywords: "KeywordMatcher | None" = None) -> dict[str, Any]:
        """Обучить модель. Точность считается на отложенных 20 % вместе со справочником ключевых слов —
        так, как классификатор работает с обращениями."""
        from sklearn.model_selection import train_test_split

        counts = Counter(labels)
        self.removed = sorted(c for c, n in counts.items() if n < MIN_PER_CATEGORY)
        data = [(t, c) for t, c in zip(texts, labels) if c not in self.removed and normalize(t)]
        if len(data) < MIN_SAMPLES or len({c for _, c in data}) < 2:
            raise ValueError(f"Недостаточно данных для обучения: {len(data)} примеров "
                             f"(нужно не меньше {MIN_SAMPLES} и хотя бы 2 категории)")
        x, y = [d[0] for d in data], [d[1] for d in data]
        # Отложенная выборка 20 % для быстрой оценки точности…
        stratify = y if min(Counter(y).values()) >= 2 and len(y) >= 20 else None
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
        probe = TextClassifier(self.algorithm).fit(x_tr, y_tr)
        no_keywords = {"winner": None, "scores": {}}
        pred, conf = [], []
        for text, proba in zip(x_te, probe.predict_many(x_te)):
            category, confidence, _ = combine(proba, keywords.match(text) if keywords else no_keywords)
            pred.append(category)
            conf.append(confidence)
        right = [p == t for p, t in zip(pred, y_te)]
        sure = [c >= low_confidence for c in conf]
        n_sure = sum(sure)
        # …а рабочая модель обучается на всех примерах
        self.fit(x, y)
        self.metrics = {
            "samples": len(y),
            "test_size": len(y_te),
            "accuracy": round(sum(right) / len(right), 3),
            "confident_share": round(n_sure / len(sure), 3),
            "confident_accuracy": round(sum(r for r, s in zip(right, sure) if s) / n_sure, 3) if n_sure else None,
        }
        return self.metrics


def blend(proba, categories: list[str], neural_proba, neural_labels: list[str]):
    """Смешать вероятности TF-IDF-модели и нейросети. Категории нейросети сопоставляются по названию:
    если после её обучения появилась новая категория, для неё берётся только TF-IDF-модель."""
    import numpy as np
    aligned = np.zeros_like(proba)
    for j, c in enumerate(categories):
        if c in neural_labels:
            aligned[:, j] = neural_proba[:, neural_labels.index(c)]
    known = np.array([c in neural_labels for c in categories])
    weight = np.where(known, NEURAL_WEIGHT, 0.0)
    mixed = proba * (1 - weight) + aligned * weight
    return mixed / mixed.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- данные и справочник

def settings() -> dict[str, Any]:
    from . import settings_store
    return settings_store.get("classifier")


def training_data(head: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    seen: dict[str, int] = {}

    def add(text: str, label: str) -> None:
        key = normalize(text)
        if not key or not label:
            return
        if key in seen:  # более поздняя разметка (исправление менеджера) важнее
            labels[seen[key]] = label
            return
        seen[key] = len(texts)
        texts.append(text)
        labels.append(label)

    if head == "request_type":
        for r in db.query("SELECT question, category FROM support_examples WHERE category != '' ORDER BY id"):
            add(r["question"], r["category"])
    for r in db.query("SELECT text, category FROM clf_training WHERE head = ? AND source IN "
                      f"({_training_sources()}) ORDER BY id", (head,)):
        add(r["text"], r["category"])
    return texts, labels


def train_on_reviewed() -> bool:
    """Учить ли модель на проверенной разметке спорных чатов. По умолчанию нет: пока по новому правилу
    исправлена только часть чатов, противоречивые примеры снижают точность (проверено на 21 исправлении).
    Проверенные метки в любом случае служат эталоном для оценки точности."""
    return bool(settings().get("train_on_reviewed"))


def _training_sources() -> str:
    return "'manager', 'import', 'review'" if train_on_reviewed() else "'manager', 'import'"


def labeled_chats() -> list[dict[str, Any]]:
    """Размеченные переписки (без дублей) с учётом проверки разметки человеком."""
    reviewed = {normalize(r["text"]): r["category"] for r in db.query(
        "SELECT text, category FROM clf_training WHERE head = 'request_type' AND source = 'review'")}
    out, seen = [], set()
    for r in db.query("SELECT id, question, category FROM support_examples WHERE category != '' ORDER BY id"):
        key = normalize(r["question"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"id": r["id"], "text": r["question"], "original": r["category"],
                    "category": reviewed.get(key, r["category"]), "reviewed": key in reviewed})
    return out


def disputed(limit: int = 30) -> dict[str, Any]:
    """Спорная разметка (подход confident learning): переписки, где модель, не видевшая пример при обучении
    (кросс-валидация, 5 блоков), уверенно ставит другую категорию. Это кандидаты на проверку человеком:
    исправленная разметка сразу улучшает обучающую выборку."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    algorithm = settings().get("algorithm") or DEFAULT_ALGORITHM
    algorithm = "logreg" if algorithm == "hybrid" else algorithm  # для поиска спорных хватает быстрой модели
    se = db.query_one("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM support_examples WHERE category != ''")
    reviewed_n = db.scalar("SELECT COUNT(*) FROM clf_training WHERE head = 'request_type' AND source = 'review'") or 0
    # пересчёт — при новых чатах и после каждых 10 проверенных меток (а не после каждого клика: ~6 с)
    key = (_db_id(), algorithm, train_on_reviewed(), se["n"], se["m"], reviewed_n // 10)
    with _lock:
        cached = _disputes.get("request_type")
        if not cached or cached[0] != key:
            items = labeled_chats()
            found: list[dict[str, Any]] = []
            labels = [i["category"] for i in items]  # эталон — с учётом уже проверенных меток
            train_labels = labels if train_on_reviewed() else [i["original"] for i in items]
            if len(items) >= 50 and min(Counter(labels).values()) >= 2:
                y = np.array(labels)
                kw = matcher("request_type")
                low = float(settings().get("low_confidence", LOW_CONFIDENCE))
                folds = min(5, min(Counter(labels).values()))
                for tr, te in StratifiedKFold(folds, shuffle=True, random_state=42).split(labels, y):
                    clf = TextClassifier(algorithm).fit([items[i]["text"] for i in tr], [train_labels[i] for i in tr])
                    for i, proba in zip(te, clf.predict_many([items[i]["text"] for i in te])):
                        category, confidence, source = combine(proba, kw.match(items[i]["text"]))
                        if category != labels[i] and confidence >= low and not items[i]["reviewed"]:
                            found.append({"id": items[i]["id"], "suggestion": category,
                                          "confidence": round(confidence, 3), "source": source})
            _disputes["request_type"] = (key, found)
        found = _disputes["request_type"][1]
    by_id = {i["id"]: i for i in labeled_chats()}
    rows = [{**by_id[f["id"]], **f} for f in found if f["id"] in by_id and not by_id[f["id"]]["reviewed"]]
    rows.sort(key=lambda r: r["confidence"], reverse=True)
    reviewed = sum(1 for i in by_id.values() if i["reviewed"])
    changed = sum(1 for i in by_id.values() if i["reviewed"] and i["category"] != i["original"])
    return {"items": rows[:limit], "total": len(rows), "reviewed": reviewed, "changed": changed}


def review_label(example_id: int, category: str) -> None:
    """Человек проверил разметку переписки: подтвердил исходную категорию или поставил другую."""
    row = db.query_one("SELECT question FROM support_examples WHERE id = ?", (example_id,))
    if row is None or not category.strip():
        raise ValueError("Пример не найден")
    db.execute("INSERT OR REPLACE INTO clf_training (head, text, category, source, created_at) "
               "VALUES ('request_type', ?, ?, 'review', ?)", (row["question"], category.strip(), db.now_iso()))


def _data_key(head: str, algorithm: str) -> tuple:
    se = db.query_one("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM support_examples "
                      "WHERE category != ''") if head == "request_type" else {"n": 0, "m": 0}
    tr = db.query_one("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m FROM clf_training WHERE head = ? AND "
                      f"source IN ({_training_sources()})", (head,))
    return (_db_id(), algorithm, train_on_reviewed(), se["n"], se["m"], tr["n"], tr["m"])


def _db_id() -> str:
    from ..config import get_settings
    return str(get_settings().db_path)


def _seed_keywords() -> None:
    if db.scalar("SELECT value FROM settings WHERE key = 'clf_seeded'"):
        return
    rows = [("request_type", c, w) for c, words in REQUEST_TYPE_KEYWORDS.items() for w in words]
    if HANDBOOK_FILE.exists():
        data = json.loads(HANDBOOK_FILE.read_text(encoding="utf-8"))
        rows += [("it_topic", c["name"], w) for c in data["categories"] for w in c["keywords"]]
    with db.connect() as conn:
        conn.executemany("INSERT OR IGNORE INTO clf_keywords (head, category, keyword) VALUES (?, ?, ?)", rows)
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('clf_seeded', '1')")


def load_handbook(replace: bool = False) -> int:
    """Восстановить исходный справочник IT-тем (app/resources/it_topics_handbook.json)."""
    data = json.loads(HANDBOOK_FILE.read_text(encoding="utf-8"))
    rows = [("it_topic", c["name"], w) for c in data["categories"] for w in c["keywords"]]
    with db.connect() as conn:
        if replace:
            conn.execute("DELETE FROM clf_keywords WHERE head = 'it_topic'")
        conn.executemany("INSERT OR IGNORE INTO clf_keywords (head, category, keyword) VALUES (?, ?, ?)", rows)
    return len(rows)


def keywords(head: str) -> dict[str, list[dict[str, Any]]]:
    _seed_keywords()
    out: dict[str, list[dict[str, Any]]] = {}
    for r in db.query("SELECT id, category, keyword FROM clf_keywords WHERE head = ? "
                      "ORDER BY category, keyword COLLATE NOCASE", (head,)):
        out.setdefault(r["category"], []).append({"id": r["id"], "keyword": r["keyword"]})
    return out


def add_keyword(head: str, category: str, keyword: str, strong: bool = False) -> None:
    """strong — однозначное слово (бренд, продукт): одного такого слова достаточно для уверенной темы."""
    category, keyword = category.strip(), keyword.strip()
    if head not in HEADS or not category or not keyword_pattern(keyword):
        raise ValueError("Укажите категорию и ключевое слово")
    _seed_keywords()
    db.execute("INSERT INTO clf_keywords (head, category, keyword, strong) VALUES (?, ?, ?, ?) "
               "ON CONFLICT(head, category, keyword) DO UPDATE SET strong = MAX(strong, excluded.strong)",
               (head, category, keyword, int(strong)))


def delete_keyword(keyword_id: int) -> None:
    db.execute("DELETE FROM clf_keywords WHERE id = ?", (keyword_id,))


# ----------------------------------------------------------------------------- облако слов IT-тем

# Слова о намерении и покупке, а не о технической теме: в облаке IT-тем они только мешают
_CLOUD_STOP_STEMS = (
    "цен", "стоим", "стои", "куп", "приобр", "лиценз", "предлож", "коммерч", "получ", "необход", "возмож", "через",
    "може", "мож", "нуж", "организ", "запрос", "прош", "вопрос", "хоте", "хочу", "могу", "верси", "подпис", "добр",
    "здравств", "подскаж", "пожалуй", "спасиб", "интерес", "сколь", "сейчас", "сч", "инн", "прям", "компан",
    "день", "дом", "работ", "сайт", "заказ", "оплат", "ваш", "наш", "этот", "есть", "буд", "также", "ооо", "email",
    "номер", "тел", "для", "год", "мес", "шту", "срок", "поставк", "услов", "бизнес", "пользоват", "клиент",
    "сотрудник", "человек", "вариант", "подойд", "подход", "какой", "какая", "какие", "каким", "нам", "вас", "мне",
    "уже", "ещё", "еще", "очень", "просто", "только", "пишу", "отправ", "файл", "ответ", "звон", "оператор",
    "менеджер", "привет", "уважен", "кп", "pro", "the", "for", "and", "edition", "использ", "http", "www", "нет",
    "ндс", "хот", "чтоб", "store", "softline", "софтлайн", "бессроч", "как", "ключ", "присл", "пришл", "програм",
    "продукт", "узна", "виж", "вчера", "зов", "скаж", "треб", "хорош", "благодар", "его", "могл", "налич",
    "несколь", "образ", "сам", "связ", "систем", "так", "докуп", "решени", "вид", "лучш", "ряд", "тип",
    "выстав", "данн", "дозвон", "ждем", "ждём", "зна", "лица", "лицо", "позиц", "поним", "правил", "прос",
    "расч", "руб", "трубк", "модел", "артикул", "выбра",
)


def _cloud_word_ok(word: str) -> bool:
    return len(word) >= 3 and not word.startswith(_CLOUD_STOP_STEMS) and not re.fullmatch(r"[\d\-]+", word)


def cloud_ignored() -> set[str]:
    return set(db.loads(db.scalar("SELECT value FROM settings WHERE key = 'clf_cloud_ignored'"), []) or [])


def ignore_cloud_word(word: str) -> None:
    words = cloud_ignored() | {normalize(word)}
    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('clf_cloud_ignored', ?)",
               (db.dumps(sorted(words)),))


def topic_cloud(limit: int = 70) -> dict[str, Any]:
    """Облако слов для IT-тем: частые слова из обращений,
    для которых справочник не нашёл уверенной темы. Слово перетаскивают в тему — оно становится её
    ключевым словом и сразу работает, без переобучения.

    Для типа обращения облако не используется: проверка на 495 чатах показала, что перенос слова
    в его самую частую категорию верен лишь в 41 % случаев — хуже самой модели (69 % на тех же обращениях)."""
    key = (_db_id(), limit, tuple(db.query_one(
        "SELECT (SELECT COUNT(*) FROM support_examples) AS a, (SELECT COALESCE(MAX(id), 0) FROM messages) AS b, "
        "(SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0) FROM clf_keywords WHERE head = 'it_topic') AS c, "
        "(SELECT value FROM settings WHERE key = 'clf_cloud_ignored') AS d").values()))
    cached = _cloud_cache.get("it_topic")
    if cached and cached[0] == key:
        return cached[1]
    result = _build_cloud(limit)
    _cloud_cache["it_topic"] = (key, result)
    return result


_cloud_cache: dict[str, tuple[tuple, dict[str, Any]]] = {}


def _build_cloud(limit: int) -> dict[str, Any]:
    m = matcher("it_topic")
    texts = [r["question"] for r in db.query("SELECT question FROM support_examples WHERE category != ''")]
    texts += [dialog_text(r["id"]) for r in db.query(
        "SELECT DISTINCT dialog_id AS id FROM messages WHERE role = 'client' ORDER BY dialog_id DESC LIMIT 500")]
    ignored = cloud_ignored()
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    known: dict[str, bool] = {}
    unmatched = 0
    for text in texts:
        if not text.strip() or m.match(text)["winner"]:
            continue
        unmatched += 1
        for word in set(extract_keywords(text, 8)):
            if word in ignored or not _cloud_word_ok(word):
                continue
            if word not in known:
                known[word] = bool(m.match(word)["scores"])
            if known[word]:
                continue  # уже есть в справочнике
            counts[word] += 1
            if len(examples.setdefault(word, [])) < 2:
                examples[word].append(re.sub(r"\s+", " ", text)[:140])
    words = [{"word": w, "count": n, "examples": examples[w]} for w, n in counts.most_common(limit) if n >= 2]
    top = words[0]["count"] if words else 1
    for w in words:
        w["size"] = round(12 + 18 * (w["count"] / top) ** 0.6)
    return {"words": sorted(words, key=lambda w: w["word"]), "unmatched": unmatched, "total": len(texts)}


def refresh_topics(limit: int = 1000) -> int:
    """Пересчитать IT-тему у диалогов после правки справочника (кроме тех, где тему поставил менеджер)."""
    changed = 0
    for r in db.query("SELECT id, it_topic, classification FROM dialogs WHERE it_topic_source != 'manager' "
                      "ORDER BY id DESC LIMIT ?", (limit,)):
        text = dialog_text(r["id"])
        if not text.strip():
            continue
        result = classify_head("it_topic", text)
        saved = db.loads(r["classification"], {}) or {}
        saved["it_topic"] = result
        db.execute("UPDATE dialogs SET it_topic = ?, it_topic_conf = ?, it_topic_source = ?, classification = ? "
                   "WHERE id = ?", (result["category"] if result else "", result["confidence"] if result else 0,
                                    result["source"] if result else "", db.dumps(saved), r["id"]))
        changed += (result["category"] if result else "") != r["it_topic"]
    return changed


def matcher(head: str) -> KeywordMatcher:
    _seed_keywords()
    key = (_db_id(), *db.query_one("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m, "
                                   "COALESCE(SUM(LENGTH(keyword) + strong), 0) AS s FROM clf_keywords WHERE head = ?",
                                   (head,)).values())
    with _lock:
        cached = _matchers.get(head)
        if cached and cached[0] == key:
            return cached[1]
        handbook: dict[str, list[str]] = {}
        strong: set[tuple[str, str]] = set()
        for r in db.query("SELECT category, keyword, strong FROM clf_keywords WHERE head = ?", (head,)):
            handbook.setdefault(r["category"], []).append(r["keyword"])
            if r["strong"]:
                strong.add((r["category"], r["keyword"]))
        # В справочнике IT-тем есть общие слова («вложение», «настройка», «не могу»),
        # поэтому одного обычного слова для решения там мало
        m = KeywordMatcher(handbook, decisive=DECISIVE.get(head, 1.0), strong=strong)
        _matchers[head] = (key, m)
        return m


def model(head: str) -> TextClassifier | None:
    """Модель из кэша или обученная заново, если изменились данные."""
    algorithm = settings().get("algorithm") or DEFAULT_ALGORITHM
    key = _data_key(head, algorithm)
    with _lock:
        cached = _models.get(head)
        if cached and cached[0] == key:
            clf = cached[1]
        else:
            texts, labels = training_data(head)
            clf = TextClassifier(algorithm)
            info: dict[str, Any] = {"algorithm": clf.algorithm}
            try:
                info.update(clf.train(texts, labels, float(settings().get("low_confidence", LOW_CONFIDENCE)),
                                      matcher(head)))
                info["categories"] = clf.categories
                info["removed"] = clf.removed
            except ValueError as exc:
                clf = None
                info.update({"error": str(exc), "samples": len(texts)})
            db.execute("INSERT OR REPLACE INTO clf_models (head, info, trained_at) VALUES (?, ?, ?)",
                       (head, db.dumps(info), db.now_iso()))
            _models[head] = (key, clf)
            if clf is not None and algorithm == "hybrid" and neural_stale(head):
                neural.start_training(head)  # в фоне; до конца обучения работает прежняя версия
    if clf is not None:
        # дообученная нейросеть подхватывается, как только фоновое обучение её сохранило
        clf.neural = neural.load(head) if clf.algorithm == "hybrid" and neural.available() else None
    return clf


def neural_stale(head: str) -> bool:
    """Нужно ли (до)обучить нейросеть: её ещё нет или после обучения появилось достаточно новых
    размеченных примеров (исправления менеджеров, проверенная разметка, загруженные файлы и чаты)."""
    if not neural.available():
        return False
    meta = neural.meta(head)
    if meta is None:
        return True
    since = meta["trained_at"]
    fresh = db.scalar("SELECT COUNT(*) FROM clf_training WHERE head = ? AND created_at > ? AND "
                      f"source IN ({_training_sources()})", (head, since)) or 0
    if head == "request_type":
        fresh += db.scalar("SELECT COUNT(*) FROM support_examples WHERE category != '' AND created_at > ?",
                           (since,)) or 0
    return fresh >= neural.RETRAIN_AFTER_NEW


def train_neural(head: str = "request_type") -> bool:
    """Запустить дообучение нейросети вручную (кнопка в панели)."""
    return neural.start_training(head)


def retrain(head: str | None = None) -> None:
    with _lock:
        for h in ([head] if head else list(HEADS)):
            _models.pop(h, None)
            model(h)


def warm_up() -> threading.Thread:
    def run() -> None:
        try:
            for h in HEADS:
                model(h)
            disputed()
            # диалоги, начатые до появления классификатора
            for r in db.query("SELECT DISTINCT d.id FROM dialogs d JOIN messages m ON m.dialog_id = d.id "
                              "AND m.role = 'client' WHERE d.request_type = '' AND d.request_type_source = '' "
                              "ORDER BY d.id DESC LIMIT 500"):
                classify_dialog(r["id"])
        except Exception:  # noqa: BLE001 — прогрев не должен мешать запуску
            log.exception("Не удалось обучить классификатор при запуске")
    t = threading.Thread(target=run, name="classifier-warmup", daemon=True)
    t.start()
    return t


# ----------------------------------------------------------------------------- классификация

def combine(proba: dict[str, float], kw: dict[str, Any]) -> tuple[str, float, str] | None:
    """Итог по модели и справочнику: (категория, уверенность, источник).

    Ключевое слово справочника важнее модели. Уверенность 0.95 ставится, когда модель
    согласна со справочником или модели нет; если модель считает иначе, категория остаётся по справочнику,
    но с уверенностью модели (не меньше 0.6) — такое обращение попадёт менеджеру на проверку.
    """
    top = max(proba.items(), key=lambda kv: kv[1]) if proba else None
    if kw["winner"]:
        category = kw["winner"]
        agree = top is None or top[0] == category
        confidence = max(KEYWORD_CONFIDENCE, proba.get(category, 0.0)) if agree else \
            max(0.6, proba.get(category, 0.0))
        return category, confidence, "keyword"
    if top:
        return top[0], top[1], "model"
    if kw["scores"]:  # только справочник, без модели, и совпадение неоднозначное
        category, score = next(iter(kw["scores"].items()))
        return category, min(0.75, 0.35 + 0.4 * score / (sum(kw["scores"].values()) + 0.5)), "keyword"
    return None


def classify_head(head: str, text: str) -> dict[str, Any] | None:
    low = float(settings().get("low_confidence", LOW_CONFIDENCE))
    kw = matcher(head).match(text)
    clf = model(head)
    proba = clf.predict_proba(text) if clf else {}
    top = sorted(proba.items(), key=lambda kv: kv[1], reverse=True)[:3]
    combined = combine(proba, kw)
    if combined is None:
        return None
    category, confidence, source = combined
    return {
        "category": category,
        "confidence": round(float(confidence), 3),
        "low_confidence": confidence < low,
        "source": source,
        "top": [{"category": c, "p": round(p, 3)} for c, p in top],
        "matched": kw["hits"].get(category, []),
    }


def classify(text: str) -> dict[str, Any]:
    return {**{h: classify_head(h, text) for h in HEADS}, "keywords": extract_keywords(text)}


def dialog_text(dialog_id: int) -> str:
    rows = db.query("SELECT text FROM messages WHERE dialog_id = ? AND role = 'client' ORDER BY id LIMIT 8",
                    (dialog_id,))
    return " ".join(r["text"] for r in rows)[:MAX_DIALOG_TEXT]


def classify_dialog(dialog_id: int) -> dict[str, Any] | None:
    """Классифицировать диалог по сообщениям клиента и сохранить результат.
    Категорию, которую поставил менеджер, классификатор не меняет."""
    text = dialog_text(dialog_id)
    if not text.strip():
        return None
    row = db.query_one("SELECT request_type_source, it_topic_source, classification FROM dialogs WHERE id = ?",
                       (dialog_id,))
    if row is None:
        return None
    result = classify(text)
    saved = db.loads(row["classification"], {}) or {}
    sets, params = [], []
    for head in HEADS:
        if row[f"{head}_source"] == "manager":
            result[head] = saved.get(head)
            continue
        r = result[head]
        sets += [f"{head} = ?", f"{head}_conf = ?", f"{head}_source = ?"]
        params += [r["category"], r["confidence"], r["source"]] if r else ["", 0, ""]
    sets.append("classification = ?")
    params.append(db.dumps(result))
    db.execute(f"UPDATE dialogs SET {', '.join(sets)} WHERE id = ?", (*params, dialog_id))
    return result


def correct_dialog(dialog_id: int, head: str, category: str) -> None:
    """Менеджер исправил категорию: фиксируем её в диалоге и добавляем пример в обучающую выборку."""
    if head not in HEADS:
        raise ValueError("Неизвестный классификатор")
    category = category.strip()
    row = db.query_one("SELECT classification FROM dialogs WHERE id = ?", (dialog_id,))
    if row is None:
        raise ValueError("Диалог не найден")
    result = db.loads(row["classification"], {}) or {}
    if not category:  # вернуть автоматическую классификацию
        db.execute(f"UPDATE dialogs SET {head}_source = '' WHERE id = ?", (dialog_id,))
        db.execute("DELETE FROM clf_training WHERE head = ? AND dialog_id = ?", (head, dialog_id))
        classify_dialog(dialog_id)
        return
    previous = (result.get(head) or {}).get("category", "")
    result[head] = {"category": category, "confidence": 1.0, "low_confidence": False, "source": "manager",
                    "top": (result.get(head) or {}).get("top", []), "matched": [], "previous": previous}
    db.execute(f"UPDATE dialogs SET {head} = ?, {head}_conf = 1, {head}_source = 'manager', classification = ? "
               f"WHERE id = ?", (category, db.dumps(result), dialog_id))
    text = dialog_text(dialog_id)
    if text.strip():
        db.execute("INSERT OR REPLACE INTO clf_training (head, text, category, source, dialog_id, created_at) "
                   "VALUES (?, ?, ?, 'manager', ?, ?)", (head, text, category, dialog_id, db.now_iso()))
    db.log_event("classification_corrected", dialog_id, head=head, category=category, previous=previous)


def categories(head: str) -> list[str]:
    names = set(keywords(head))
    if head == "request_type":
        names |= {r["category"] for r in db.query(
            "SELECT DISTINCT category FROM support_examples WHERE category != ''")}
    names |= {r["category"] for r in db.query("SELECT DISTINCT category FROM clf_training WHERE head = ?", (head,))}
    return sorted(names)


def import_training(head: str, rows: Iterable[tuple[str, str]], source: str = "import") -> int:
    """Загрузить размеченные примеры (текст, категория) из файла."""
    if head not in HEADS:
        raise ValueError("Неизвестный классификатор")
    now = db.now_iso()
    data = [(head, t.strip()[:MAX_DIALOG_TEXT], c.strip(), source, now) for t, c in rows if t.strip() and c.strip()]
    with db.connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO clf_training (head, text, category, source, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", data)
    return len(data)


def parse_training_file(filename: str, content: bytes, text_col: int = 0, category_col: int = 1
                        ) -> list[tuple[str, str]]:
    """CSV/XLSX: колонка с текстом и колонка с категорией (номера колонок с нуля)."""
    rows: list[list[str]] = []
    if filename.lower().endswith((".xlsx", ".xlsm")):
        import io

        from openpyxl import load_workbook
        ws = load_workbook(io.BytesIO(content), read_only=True, data_only=True).active
        rows = [["" if v is None else str(v) for v in r] for r in ws.iter_rows(values_only=True)]
    else:
        import csv
        import io
        text = content.decode("utf-8-sig", errors="replace")
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t") if text.strip() else csv.excel
        rows = list(csv.reader(io.StringIO(text), dialect))
    out = []
    for r in rows:
        if len(r) > max(text_col, category_col):
            out.append((r[text_col], r[category_col]))
    if out and re.search(r"текст|text|вопрос|обращ", out[0][0].lower()) and \
            re.search(r"катег|category|тип|класс", out[0][1].lower()):
        out = out[1:]  # строка заголовков
    return out


def status() -> list[dict[str, Any]]:
    out = []
    for head, title in HEADS.items():
        row = db.query_one("SELECT info, trained_at FROM clf_models WHERE head = ?", (head,))
        info = db.loads(row["info"], {}) if row else {}
        out.append({
            "head": head, "title": title, "info": info, "trained_at": row["trained_at"] if row else None,
            "keywords": db.scalar("SELECT COUNT(*) FROM clf_keywords WHERE head = ?", (head,)) or 0,
            "keyword_categories": db.scalar("SELECT COUNT(DISTINCT category) FROM clf_keywords WHERE head = ?",
                                            (head,)) or 0,
            "manager_samples": db.scalar("SELECT COUNT(*) FROM clf_training WHERE head = ? AND source = 'manager'",
                                         (head,)) or 0,
            "imported_samples": db.scalar("SELECT COUNT(*) FROM clf_training WHERE head = ? AND source = 'import'",
                                          (head,)) or 0,
            "reviewed_samples": db.scalar("SELECT COUNT(*) FROM clf_training WHERE head = ? AND source = 'review'",
                                          (head,)) or 0,
            "neural": {"installed": neural.installed(), "available": neural.available(),
                       "meta": neural.meta(head), "training": neural.training_status(head)},
        })
    return out


def distribution(head: str, days: int = 30) -> list[dict[str, Any]]:
    since = db.query_one("SELECT datetime('now', 'localtime', ?) AS s", (f"-{days} days",))["s"]
    return db.query(
        f"SELECT {head} AS category, COUNT(*) AS n, "
        f"SUM(CASE WHEN {head}_source != 'manager' AND {head}_conf < ? THEN 1 ELSE 0 END) AS low "
        f"FROM dialogs WHERE {head} != '' AND created_at >= ? GROUP BY {head} ORDER BY n DESC",
        (float(settings().get("low_confidence", LOW_CONFIDENCE)), since))


def review_queue(limit: int = 50) -> list[dict[str, Any]]:
    """Обращения с низкой уверенностью — их тип проверяет менеджер."""
    low = float(settings().get("low_confidence", LOW_CONFIDENCE))
    return db.query(
        "SELECT d.id, d.channel, d.request_type, d.request_type_conf, d.it_topic, d.created_at, "
        "(SELECT text FROM messages m WHERE m.dialog_id = d.id AND m.role = 'client' ORDER BY m.id LIMIT 1) "
        "AS first_text FROM dialogs d WHERE d.request_type != '' AND d.request_type_source != 'manager' "
        "AND d.request_type_conf < ? ORDER BY d.id DESC LIMIT ?", (low, limit))


def review_count() -> int:
    """Сколько обращений ждут подтверждения типа (для счётчика в меню)."""
    low = float(settings().get("low_confidence", LOW_CONFIDENCE))
    return db.scalar("SELECT COUNT(*) FROM dialogs WHERE request_type != '' AND request_type_source != 'manager' "
                     "AND request_type_conf < ?", (low,)) or 0


def summary_line(result: dict[str, Any] | None) -> str:
    """Короткая строка для менеджера и для ИИ-ассистента."""
    if not result:
        return ""
    parts = []
    for head, title in HEADS.items():
        r = result.get(head)
        if r:
            parts.append(f"{title}: {r['category']} ({round(r['confidence'] * 100)}%)")
    return "; ".join(parts)
