"""Скачать нейросеть rubert-tiny2 для классификатора обращений (~120 МБ, один раз).

Нужны пакеты из requirements-ml.txt (PyTorch CPU и transformers):

    .venv\\Scripts\\pip install -r requirements-ml.txt
    .venv\\Scripts\\python scripts\\download_model.py

Модель сохраняется в data\\models\\rubert-tiny2 (только веса safetensors и словарь токенизатора).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import neural  # noqa: E402


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    if neural.available():
        print(f"Модель уже скачана: {neural.base_dir()}")
        return
    path = neural.download()
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"Готово: {path} ({size:.0f} МБ)")


if __name__ == "__main__":
    main()
