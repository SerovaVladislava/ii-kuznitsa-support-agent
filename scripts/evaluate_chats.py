"""Прогон агента по реальным обращениям из чатов поддержки (обезличенная выгрузка задачи).

Запуск:  .venv\\Scripts\\python scripts\\evaluate_chats.py materials\\drive\\chats [--limit N]
Работает на КОПИИ рабочей базы (каталог из фида уже должен быть загружен) и не трогает рабочие данные.
Каждое обращение подаётся как первое сообщение нового диалога со страницы, с которой писал клиент.
Результат: reports/chats_eval.md (сводка по категориям) и reports/chats_eval.csv (каждое обращение).

Это оценка покрытия, а не качества: «ответил без менеджера» не значит «ответил верно». Офлайн-режим
отвечает по правилам; ИИ-режим обрабатывает свободные формулировки лучше — прогоните с ключом (EVAL_LLM_API_KEY).
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTCOMES = {
    "variant": "подобран вариант с ценой",
    "clarify": "уточняющие вопросы по продукту",
    "need": "выяснение требований",
    "products": "предложены продукты",
    "kb": "ответ из базы знаний",
    "document": "КП / счёт",
    "escalated": "передано менеджеру",
    "other": "прочее",
}


def classify(text: str, meta: dict, status: str) -> str:
    if status == "waiting_operator":
        return "escalated"
    if meta.get("documents"):
        return "document"
    if "Подходит:" in text:
        return "variant"
    if "Чтобы подобрать точный вариант" in text:
        return "clarify"
    if "Помогу подобрать решение" in text or "Уточните, пожалуйста" in text:
        return "need"
    if "Вот что нашёл" in text or "Российские аналоги" in text:
        return "products"
    if meta.get("kb_used") or "Источник:" in text:
        return "kb"
    return "other"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    chats_dir = Path(args[0]) if args else ROOT / "materials" / "drive" / "chats"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0

    # копия рабочей базы — оценка не засоряет рабочие диалоги и статистику
    src_db = ROOT / "data" / "store_agent.sqlite3"
    tmp = Path(tempfile.mkdtemp(prefix="eval_"))
    with __import__("sqlite3").connect(src_db) as conn_src, __import__("sqlite3").connect(tmp / "store_agent.sqlite3") as dst:
        conn_src.backup(dst)
    os.environ["DATA_DIR"] = str(tmp)
    os.environ["LLM_API_KEY"] = os.environ.get("EVAL_LLM_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = ""

    import logging
    logging.disable(logging.WARNING)
    from app.assistant import engine
    from app.services import dialogs, support_examples

    chats = []
    for f in sorted(chats_dir.glob("*.txt")):
        chats += support_examples.parse_chat_export(f.read_text(encoding="utf-8"))
    if limit:
        chats = chats[:limit]

    rows = []
    started = time.perf_counter()
    for c in chats:
        page = c["page_url"]
        d = dialogs.create("web", subject=f"https://{page}" if page and not page.startswith("http") else page)
        t0 = time.perf_counter()
        out = engine.handle_client_message(d["id"], c["question"])
        ms = int((time.perf_counter() - t0) * 1000)
        msg = out[0] if out else {"text": "", "meta": {}}
        dlg = dialogs.get(d["id"])
        outcome = classify(msg["text"], msg.get("meta") or {}, dlg["status"])
        rows.append({"id": c["external_id"], "category": c["category"], "outcome": outcome,
                     "escalation_reason": dlg["escalation_reason"] if outcome == "escalated" else "",
                     "ms": ms, "question": c["question"][:500], "answer": msg["text"][:700]})
    total_s = time.perf_counter() - started

    by_cat: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_cat[r["category"]][r["outcome"]] += 1
    overall = Counter(r["outcome"] for r in rows)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with open(reports / "chats_eval.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    no_mgr = n - overall["escalated"]
    lines = [
        "# Прогон агента по реальным обращениям",
        "",
        f"Обращений: **{n}** из обезличенных чатов поддержки. Режим: "
        f"{'ИИ-режим' if os.environ['LLM_API_KEY'] else 'офлайн (правила, без ИИ)'}. "
        f"Среднее время ответа: {sum(r['ms'] for r in rows) / n / 1000:.2f} с (всего {total_s:.0f} с).",
        "",
        f"**Без передачи менеджеру: {no_mgr} ({no_mgr / n:.0%})** — агент сам ответил, подобрал вариант, "
        f"предложил продукты или задал уточняющие вопросы. Передано менеджеру: {overall['escalated']} "
        f"({overall['escalated'] / n:.0%}).",
        "",
        "Это оценка покрытия, а не качества: обращения — склеенные реплики целого чата, многие — продолжения "
        "разговора («вопрос решён?», «счёт ждём»). Корректность ответов проверяется выборочно (см. CSV).",
        "",
        "## Итоги по исходам",
        "",
        "| Исход | Обращений | Доля |",
        "|---|---:|---:|",
    ]
    for key, label in OUTCOMES.items():
        if overall[key]:
            lines.append(f"| {label} | {overall[key]} | {overall[key] / n:.0%} |")
    lines += ["", "## По категориям обращений", "",
              "| Категория | Всего | Без менеджера | Основной исход |", "|---|---:|---:|---|"]
    for cat, cnt in sorted(by_cat.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(cnt.values())
        main_outcome = OUTCOMES[cnt.most_common(1)[0][0]]
        lines.append(f"| {cat} | {tot} | {(tot - cnt['escalated']) / tot:.0%} | {main_outcome} |")
    lines += ["", "## Примеры", ""]
    shown = Counter()
    for r in rows:
        if shown[r["outcome"]] >= 2 or r["outcome"] == "other":
            continue
        shown[r["outcome"]] += 1
        lines += [f"**#{r['id']} · {r['category']} → {OUTCOMES[r['outcome']]}**", "",
                  f"> Клиент: {r['question'][:300]}", ">",
                  "> Агент: " + r["answer"][:400].replace("\n", "\n> "), ""]
    (reports / "chats_eval.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
