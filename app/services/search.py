"""Локальный полнотекстовый поиск: TF-IDF по символьным n-граммам + по словам с усечением окончаний.

Символьные n-граммы устойчивы к опечаткам и форматам артикулов («3х2,5», «16А»), а усечение слов до
основы («гарантией» → «гаран») и стоп-слова снимают влияние русской морфологии. Работает локально,
без внешних сервисов; на тысячах статей/товаров индекс строится за доли секунды.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .textutil import normalize

STOPWORDS = set("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее ей
если есть еще же за здесь и из или им их к как ко когда кто ли либо мне меня мной мы на над надо не него нее
нет ни них но ну о об однако он она они оно от очень по под при про с со так также такой там те тем то того
тоже той только том ты у уже хотя чего чей чем что чтобы чье чья эта эти это этот я ваш ваша ваше ваши наш
наша наше наши мой моя мое мои свой свою можно нужно ли подскажите скажите пожалуйста здравствуйте добрый день
хочу хотим хотел хотела интересует какой какая какие какое каков сколько бывает имеется
нужен нужна нужны надо купить приобрести подобрать подберите посоветуйте
""".split())

WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(normalize(text)) if w not in STOPWORDS]


def stem(word: str, size: int = 5) -> str:
    """Грубое усечение окончаний: «файлы»/«файлов» → «файл», «гарантией» → «гаран»."""
    if any(c.isdigit() for c in word) or len(word) <= size - 1:
        return word
    return word[:size - 1] if len(word) == size else word[:size]


def prep_chars(text: str) -> str:
    return " ".join(tokens(text))


def prep_words(text: str) -> str:
    return " ".join(stem(w) for w in tokens(text))


class TextIndex:
    def __init__(self, loader: Callable[[], list[tuple[int, str]]], char_weight: float = 0.5):
        self._loader = loader
        self._char_weight = char_weight
        self._lock = threading.Lock()
        self._dirty = True
        self._ids: list[int] = []
        self._char_vec = self._word_vec = None
        self._char_mx = self._word_mx = None

    def invalidate(self) -> None:
        self._dirty = True

    def _build(self) -> None:
        docs = self._loader()
        self._ids = [doc_id for doc_id, _ in docs]
        if not docs:
            self._char_vec = self._word_vec = None
            self._dirty = False
            return
        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                         token_pattern=r"(?u)\b\w+\b")
        self._char_mx = self._char_vec.fit_transform([prep_chars(t) for _, t in docs])
        self._word_mx = self._word_vec.fit_transform([prep_words(t) for _, t in docs])
        self._dirty = False

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_chars, q_words = prep_chars(query), prep_words(query)
        if not q_chars:
            return []
        with self._lock:
            if self._dirty:
                self._build()
            if self._char_vec is None:
                return []
            char_scores = linear_kernel(self._char_vec.transform([q_chars]), self._char_mx).ravel()
            word_scores = linear_kernel(self._word_vec.transform([q_words]), self._word_mx).ravel()
            ids = list(self._ids)
        scores = self._char_weight * char_scores + (1 - self._char_weight) * word_scores
        order = scores.argsort()[::-1][:top_k]
        return [(ids[i], float(scores[i])) for i in order if scores[i] > 0]
