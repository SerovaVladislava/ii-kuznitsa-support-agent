"""Оценка классификатора типов обращений на размеченных переписках Softline
(кросс-валидация: 5 блоков, 3 разных разбиения — результат усредняется).

Сравнивает базовый алгоритм (TF-IDF + наивный Байес) и логистическую регрессию — без
справочника ключевых слов и с ним. С --neural добавляется гибрид с нейросетью rubert-tiny2: она дообучается
заново в каждом блоке (около 5 минут на блок на CPU), поэтому гибрид оценивается на одном разбиении.
Метки, проверенные менеджером на странице «Классификатор» (спорная разметка), заменяют исходные;
--original — оценить по исходной разметке. База только читается. Отчёт: reports/classifier_eval.md

    python scripts/evaluate_classifier.py [--original] [--neural]
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services import neural  # noqa: E402
from app.services.classifier import (LOW_CONFIDENCE, REQUEST_TYPE_KEYWORDS, KeywordMatcher,  # noqa: E402
                                     TextClassifier, combine, normalize)


REVIEWED = 0
CHANGED = 0
TRAIN_ON_REVIEWED = False


def load(original: bool = False) -> tuple[list[str], list[str]]:
    """Тексты и эталонные метки (с учётом проверенной разметки, если не original).
    Метки для обучения — load_train_labels()."""
    global REVIEWED, CHANGED, TRAIN_ON_REVIEWED, _ORIGINAL
    conn = sqlite3.connect(f"file:{get_settings().db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT question, category FROM support_examples WHERE category != '' "
                            "ORDER BY id").fetchall()
        reviewed = {} if original else {normalize(t): c for t, c in conn.execute(
            "SELECT text, category FROM clf_training WHERE head = 'request_type' AND source = 'review'")}
        cfg = conn.execute("SELECT value FROM settings WHERE key = 'classifier'").fetchone()
    finally:
        conn.close()
    TRAIN_ON_REVIEWED = bool(cfg and json.loads(cfg[0]).get("train_on_reviewed"))
    seen, texts, labels, _ORIGINAL = set(), [], [], []
    REVIEWED = CHANGED = 0
    for q, c in rows:
        k = normalize(q)
        if k and k not in seen:
            seen.add(k)
            texts.append(q)
            labels.append(reviewed.get(k, c))
            _ORIGINAL.append(c)
            REVIEWED += k in reviewed
            CHANGED += reviewed.get(k, c) != c
    return texts, labels


_ORIGINAL: list[str] = []


def load_train_labels(labels: list[str]) -> list[str]:
    """Как обучается рабочая модель: на проверенной разметке, только если это включено в настройках."""
    return labels if TRAIN_ON_REVIEWED else list(_ORIGINAL)


SEEDS = (1, 2, 3)


def evaluate(texts, labels, algorithm: str, use_keywords: bool, seeds=SEEDS, train_labels=None) -> dict:
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    matcher = KeywordMatcher(REQUEST_TYPE_KEYWORDS)
    y = np.array(labels)
    y_train = np.array(train_labels if train_labels is not None else labels)
    pred = np.empty(len(y), dtype=object)
    conf = np.zeros(len(y))
    kw_used = np.zeros(len(y), dtype=bool)
    runs = []
    for seed in seeds:
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y)):
            clf = TextClassifier(algorithm).fit([texts[i] for i in tr], list(y_train[tr]))
            if algorithm == "hybrid":
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp) / "fold"
                    neural.finetune([texts[i] for i in tr], list(y_train[tr]), folder)
                    clf.neural = neural.FineTuned(folder)
                print(f"  нейросеть: блок {k + 1}/5 разбиения {seed}", flush=True)
            for i, proba in zip(te, clf.predict_many([texts[i] for i in te])):
                kw = matcher.match(texts[i]) if use_keywords else {"winner": None, "scores": {}}
                cat, p, source = combine(proba, kw)
                pred[i], conf[i], kw_used[i] = cat, p, source == "keyword"
        sure = conf >= LOW_CONFIDENCE
        runs.append({
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred.astype(str), average="macro")),
            "confident_share": float(sure.mean()),
            "confident_accuracy": float((pred[sure] == y[sure]).mean()) if sure.any() else 0.0,
            "keyword_share": float(kw_used.mean()),
            "keyword_accuracy": float((pred[kw_used] == y[kw_used]).mean()) if kw_used.any() else 0.0,
        })
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out["accuracy_std"] = float(np.std([r["accuracy"] for r in runs]))
    return out


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    warnings.filterwarnings("ignore")
    texts, labels = load(original="--original" in sys.argv)
    if len(texts) < 50:
        sys.exit("Мало размеченных переписок: загрузите выгрузку чатов в «Переписки поддержки».")
    counts = Counter(labels)
    baseline = max(counts.values()) / len(labels)
    lines = [
        "# Классификатор типов обращений: оценка на переписках Softline",
        "",
        f"Размеченных переписок: {len(texts)} (без дублей), категорий: {len(counts)}. "
        f"Кросс-валидация: 5 блоков × 3 разбиения. Если всегда отвечать самой частой категорией, точность будет {baseline:.0%}.",
        "",
        (f"Эталон — проверенная разметка: спорных чатов проверено {REVIEWED}, метка изменена у {CHANGED}. "
         f"Обучение — на {'проверенной' if TRAIN_ON_REVIEWED else 'исходной'} разметке, как у рабочей модели "
         f"(настройка «Учить на проверенной разметке»)." if REVIEWED
         else "Эталон и обучение — исходная разметка выгрузки."),
        "",
        "| Вариант | Точность | Macro-F1 | Уверенных (≥80%) | Точность уверенных | По ключевым словам | Их точность |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"nb": "базовая: TF-IDF + наивный Байес", "logreg": "слова + символы, лог. регрессия",
             "hybrid": "гибрид: дообученная rubert-tiny2 + слова и символы"}
    with_neural = "--neural" in sys.argv
    if with_neural and not neural.available():
        sys.exit("Нейросеть не установлена: pip install -r requirements-ml.txt и scripts/download_model.py")
    runs = [(a, kw) for a in ("nb", "logreg") for kw in (False, True)]
    if with_neural:
        runs.append(("hybrid", True))
    for algorithm, use_kw in runs:
        seeds = SEEDS[:1] if algorithm == "hybrid" else SEEDS
        m = evaluate(texts, labels, algorithm, use_kw, seeds, load_train_labels(labels))
        label = names[algorithm] + (" + справочник" if use_kw else "") + \
            (" (одно разбиение)" if algorithm == "hybrid" else "")
        kw = f"{m['keyword_share']:.0%} | {m['keyword_accuracy']:.0%}" if use_kw else "— | —"
        spread = f" ± {m['accuracy_std']:.1%}" if len(seeds) > 1 else ""
        lines.append(f"| {label} | {m['accuracy']:.1%}{spread} | {m['macro_f1']:.3f} | "
                     f"{m['confident_share']:.0%} | {m['confident_accuracy']:.0%} | {kw} |")
        print(lines[-1])
    lines += ["", "Категории:", ""] + [f"- {c}: {n}" for c, n in counts.most_common()]
    out = ROOT / "reports" / "classifier_eval.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
