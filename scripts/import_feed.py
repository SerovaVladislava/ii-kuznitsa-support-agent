"""Загрузка большого товарного фида из файла (быстрее, чем через форму в браузере).

Запуск:  .venv\\Scripts\\python scripts\\import_feed.py materials\\drive\\Mindbox.yml.gz.xml [--keep-demo]
Фид читается потоково (YML/XML, в т.ч. сжатый gzip), затем строятся названия продуктов по группам
вариантов и диапазоны «К-во узлов». Демо-каталог по умолчанию скрывается.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, seed  # noqa: E402
from app.services import catalog, feed  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    db.init_db()
    seed.seed_if_empty()
    started = time.perf_counter()
    res = feed.import_feed(path.name, path, hide_demo="--keep-demo" not in sys.argv)
    print(f"Готово за {time.perf_counter() - started:.0f} с: {res}")
    print(f"Активных артикулов в каталоге: {db.scalar('SELECT COUNT(*) FROM products WHERE active = 1')}, "
          f"продуктов: {db.scalar('SELECT COUNT(DISTINCT family) FROM products WHERE active = 1')}, "
          f"доступно к заказу: {db.scalar('SELECT COUNT(*) FROM products WHERE active = 1 AND available = 1')}")
    print(f"Всего в базе: {catalog.count()}")


if __name__ == "__main__":
    main()
