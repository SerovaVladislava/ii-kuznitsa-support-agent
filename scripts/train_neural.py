"""Дообучить нейросеть классификатора (rubert-tiny2) на размеченных обращениях — синхронно, с прогрессом.

Нужно после переноса на новый компьютер, если папка data\\models не скопирована. Обычно сервер делает это
сам в фоне при запуске, а этот скрипт удобен, чтобы обучить заранее и увидеть результат.
На CPU занимает 5–10 минут.

    .venv\\Scripts\\python scripts\\train_neural.py
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import classifier, neural  # noqa: E402


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if not neural.available():
        sys.exit("Нейросеть не установлена: pip install -r requirements-ml.txt и python scripts/download_model.py")
    texts, labels = classifier.training_data("request_type")
    keep = {c for c, n in classifier.Counter(labels).items() if n >= classifier.MIN_PER_CATEGORY}
    data = [(t, c) for t, c in zip(texts, labels) if c in keep]
    if len(data) < classifier.MIN_SAMPLES:
        sys.exit("Мало размеченных обращений: загрузите выгрузку чатов в «Переписки поддержки».")
    print(f"Обучаю на {len(data)} обращениях, {len(keep)} категорий…", flush=True)
    target = neural.models_root() / f"request_type-rubert-{datetime.now():%Y%m%d-%H%M%S}"
    meta = neural.finetune([t for t, _ in data], [c for _, c in data], target,
                           lambda e, n: print(f"  эпоха {e}/{n}", flush=True))
    neural._activate("request_type", target)
    print(f"Готово за {meta['seconds']} с: {target}")


if __name__ == "__main__":
    main()
