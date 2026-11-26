"""Нагрузочная проверка поиска на каталоге масштаба задачи (~200 000 SKU).

Запуск:  .venv\\Scripts\\python scripts\\bench_search.py [количество_SKU]
Создаёт временную БД с синтетическим фидом, замеряет загрузку и время поиска/подбора варианта.
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="bench_")
os.environ["LLM_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

from app import seed  # noqa: E402
from app.services import catalog  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
random.seed(1)
VENDORS = [f"Вендор{i}" for i in range(400)] + ["Microsoft", "Adobe", "Autodesk", "1С", "Astra", "Positive"]
KINDS = ["Антивирус", "Почтовый сервер", "Офисный пакет", "Резервное копирование", "САПР", "Графический редактор",
         "СУБД", "Операционная система", "VPN-шлюз", "Система мониторинга", "Бухгалтерия", "Виртуализация"]
EDITIONS = ["Standard", "Pro", "Enterprise", "Базовый", "Расширенный"]
TIERS = [(1, 9), (10, 24), (25, 49), (50, 99), (100, 249)]


def synthetic(n: int):
    i = 0
    while i < n:
        vendor = random.choice(VENDORS)
        kind = random.choice(KINDS)
        family = f"{vendor} {kind} {random.randint(1, 9999)}"
        for ed in random.sample(EDITIONS, 2):
            for ptype in ("new", "renewal"):
                for term in (12, 24):
                    for lo, hi in TIERS:
                        if i >= n:
                            return
                        yield {"sku": f"SYN-{i:07d}", "name": f"{family} {ed}, {lo}-{hi} польз., {term // 12} г.",
                               "vendor": vendor, "family": f"{family} {ed}", "edition": ed, "category": kind,
                               "description": f"{kind} для организаций, редакция {ed}", "tags": kind.lower(),
                               "price": round(random.uniform(500, 50000), 2), "purchase_type": ptype,
                               "license_type": "subscription", "term_months": term, "users_min": lo, "users_max": hi,
                               "is_russian": int(random.random() < 0.4)}
                        i += 1


def main() -> None:
    seed.seed_if_empty()
    t = time.perf_counter()
    batch = []
    for item in synthetic(N):
        batch.append(item)
        if len(batch) == 10_000:
            catalog.bulk_upsert(batch)
            batch = []
    if batch:
        catalog.bulk_upsert(batch)
    load = time.perf_counter() - t
    print(f"Загружено {catalog.count():,} SKU за {load:.1f} с".replace(",", " "))

    queries = ["нужен MDaemon Standard", "касперский продление 40 компьютеров", "защита почты", "российский почтовый сервер",
               "офисный пакет", "резервное копирование", "Вендор17 САПР Pro", "антивирус для организаций", "kes"]
    timings = []
    for q in queries:
        t = time.perf_counter()
        res = catalog.search_families(q, limit=5)
        ms = (time.perf_counter() - t) * 1000
        timings.append(ms)
        print(f"{ms:7.1f} мс  {q!r:42} → {res[0]['family'] if res else '—'}")
    t = time.perf_counter()
    catalog.resolve_variant("MDaemon Standard", users=30, purchase_type="new")
    print(f"{(time.perf_counter() - t) * 1000:7.1f} мс  сверка версий MDaemon Standard (30 польз.)")
    print(f"Поиск: медиана {statistics.median(timings):.0f} мс, максимум {max(timings):.0f} мс")


if __name__ == "__main__":
    main()
