"""Нейросеть для классификатора обращений: русская модель rubert-tiny2 (cointegrated/rubert-tiny2),
дообученная на размеченных переписках Softline.

- Базовая модель скачивается один раз (scripts/download_model.py) в data/models/rubert-tiny2.
- Дообучение (fine-tuning) на CPU занимает несколько минут, поэтому идёт в фоне: кнопка на странице
  «Классификатор», скрипт scripts/train_neural.py или автоматически, когда накопилось достаточно новых примеров.
  Пока идёт обучение, классификатор работает на прежней версии нейросети.
- Если PyTorch/transformers не установлены (requirements-ml.txt) или модель не скачана, available()
  возвращает False, и классификатор работает без нейросети.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "cointegrated/rubert-tiny2"
EPOCHS = 12
LEARNING_RATE = 1.5e-4
MAX_TOKENS = 160
BATCH = 16
RETRAIN_AFTER_NEW = 30  # дообучать заново, когда накопилось столько новых размеченных примеров

_lock = threading.Lock()
_loaded: dict[str, tuple[str, "FineTuned"]] = {}
_training: dict[str, dict[str, Any]] = {}


def models_root() -> Path:
    return Path(os.getenv("MODELS_DIR") or Path(__file__).resolve().parent.parent.parent / "data" / "models")


def base_dir() -> Path:
    return models_root() / "rubert-tiny2"


def _pointer(head: str) -> Path:
    return models_root() / f"{head}-rubert.current"


def finetuned_dir(head: str) -> Path | None:
    """Каталог действующей версии дообученной модели. Версии лежат в отдельных каталогах, а указатель
    переключается одной записью файла: Windows не даёт удалить или перезаписать файлы загруженной модели."""
    try:
        name = _pointer(head).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    path = models_root() / name
    return path if (path / "meta.json").exists() else None


def _activate(head: str, path: Path) -> None:
    tmp = _pointer(head).with_suffix(".tmp")
    tmp.write_text(path.name, encoding="utf-8")
    os.replace(tmp, _pointer(head))
    for old in models_root().glob(f"{head}-rubert-*"):  # старые версии — по возможности удалить
        if old.is_dir() and old != path:
            shutil.rmtree(old, ignore_errors=True)


def installed() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def available() -> bool:
    """Можно ли дообучать и использовать нейросеть."""
    return (base_dir() / "config.json").exists() and installed()


def download() -> Path:
    """Скачать базовую модель с Hugging Face (только safetensors) и сохранить локально."""
    from transformers import AutoModel, AutoTokenizer
    target = base_dir()
    target.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(target)
    AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).save_pretrained(target, safe_serialization=True)
    return target


def _clean(texts: list[str]) -> list[str]:
    from .classifier import normalize
    return [normalize(t, lower=False) or "-" for t in texts]


def finetune(texts: list[str], labels: list[str], out_dir: Path,
             progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Дообучить rubert-tiny2 на размеченных текстах и сохранить в out_dir.

    Потери взвешены по частоте категорий (редкие категории не теряются), AdamW, разогрев 10 % шагов
    и линейное затухание скорости обучения. Параметры подобраны на переписках Softline."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    started = time.monotonic()
    torch.manual_seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    classes = sorted(set(labels))
    y = np.array([classes.index(c) for c in labels])
    tok = AutoTokenizer.from_pretrained(base_dir())
    model = AutoModelForSequenceClassification.from_pretrained(
        base_dir(), num_labels=len(classes), use_safetensors=True,
        id2label=dict(enumerate(classes)), label2id={c: i for i, c in enumerate(classes)})
    counts = np.bincount(y, minlength=len(classes)).astype(float)
    weights = torch.tensor(len(y) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    per_epoch = int(np.ceil(len(y) / BATCH))
    steps = EPOCHS * per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
    clean = _clean(texts)
    target = torch.tensor(y)
    rng = np.random.default_rng(0)
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(y))
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            enc = tok([clean[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt")
            loss = loss_fn(model(**enc).logits, target[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if progress:
            progress(epoch + 1, EPOCHS)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    meta = {"labels": classes, "samples": len(y), "epochs": EPOCHS, "base": MODEL_NAME,
            "trained_at": datetime.now().replace(microsecond=0).isoformat(sep=" "),
            "seconds": round(time.monotonic() - started)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


class FineTuned:
    """Дообученная модель: вероятности категорий для текстов."""

    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        self.labels: list[str] = self.meta["labels"]
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, use_safetensors=True).eval()
        self._torch = torch
        self._lock = threading.Lock()

    def predict(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        with self._lock, torch.no_grad():
            clean = _clean(texts)
            for b in range(0, len(clean), 32):
                enc = self.tok(clean[b:b + 32], padding=True, truncation=True, max_length=MAX_TOKENS,
                               return_tensors="pt")
                out.append(torch.softmax(self.model(**enc).logits, -1).numpy())
        return np.vstack(out)


def meta(head: str) -> dict[str, Any] | None:
    path = finetuned_dir(head)
    if path is None:
        return None
    try:
        return json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(head: str) -> FineTuned | None:
    """Действующая дообученная модель (перечитывается, когда фоновое обучение сохранило новую версию)."""
    path = finetuned_dir(head)
    if path is None or not installed():
        return None
    stamp = path.name
    with _lock:
        cached = _loaded.get(head)
        if cached and cached[0] == stamp:
            return cached[1]
        try:
            model = FineTuned(path)
        except Exception:  # noqa: BLE001 — повреждённая модель не должна ломать классификатор
            log.exception("Не удалось загрузить нейросеть классификатора")
            return None
        _loaded[head] = (stamp, model)
        return model


def training_status(head: str) -> dict[str, Any] | None:
    return _training.get(head)


def start_training(head: str, on_done: Callable[[], None] | None = None) -> bool:
    """Запустить дообучение в фоне. False — если уже идёт или нейросеть недоступна."""
    if not available():
        return False
    with _lock:
        if _training.get(head, {}).get("running"):
            return False
        _training[head] = {"running": True, "epoch": 0, "epochs": EPOCHS,
                           "started": datetime.now().replace(microsecond=0).isoformat(sep=" ")}

    def progress(epoch: int, total: int) -> None:
        _training[head]["epoch"] = epoch

    def run() -> None:
        from . import classifier
        try:
            texts, labels = classifier.training_data(head)
            keep = {c for c, n in classifier.Counter(labels).items() if n >= classifier.MIN_PER_CATEGORY}
            data = [(t, c) for t, c in zip(texts, labels) if c in keep]
            if len(data) < classifier.MIN_SAMPLES or len(keep) < 2:
                raise ValueError("Недостаточно размеченных примеров для нейросети")
            target = models_root() / f"{head}-rubert-{datetime.now():%Y%m%d-%H%M%S}"
            info = finetune([t for t, _ in data], [c for _, c in data], target, progress)
            _activate(head, target)
            _training[head].update({"running": False, "finished": info["trained_at"], "error": ""})
            log.info("Нейросеть классификатора дообучена: %s примеров за %s с", info["samples"], info["seconds"])
            if on_done:
                on_done()
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка дообучения нейросети")
            _training[head].update({"running": False, "error": str(exc)[:300]})

    threading.Thread(target=run, name=f"neural-train-{head}", daemon=True).start()
    return True
