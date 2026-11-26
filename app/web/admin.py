"""Панель менеджера: сводка, единый inbox диалогов, база знаний, каталог, документы, клиенты, настройки."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import db
from ..assistant import engine
from ..config import get_settings
from ..services import (analytics, catalog, classifier, customers, dialogs, documents, feed, knowledge, neural,
                        pricing, settings_store, support_examples)
from .auth import COOKIE, TTL, check_password, make_token, require_admin
from .public import public_message
from .templating import templates

router = APIRouter(prefix="/admin")
guard = [Depends(require_admin)]


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    ctx.setdefault("nav_waiting", db.scalar("SELECT COUNT(*) FROM dialogs WHERE status = 'waiting_operator'") or 0)
    ctx.setdefault("nav_drafts", db.scalar("SELECT COUNT(*) FROM documents WHERE status = 'draft'") or 0)
    ctx.setdefault("nav_candidates", db.scalar("SELECT COUNT(*) FROM kb_candidates WHERE status = 'new'") or 0)
    ctx.setdefault("nav_review", classifier.review_count())
    ctx.setdefault("assistant_mode", engine.assistant_mode())
    ctx.setdefault("flash", request.query_params.get("msg", ""))
    return templates.TemplateResponse(request, f"admin/{name}", ctx)


def back(url: str, msg: str = "") -> RedirectResponse:
    """Редирект с сообщением. Сообщение ставится до #якоря — иначе браузер не передаст его серверу."""
    if msg:
        path, _, anchor = url.partition("#")
        url = path + ("&" if "?" in path else "?") + "msg=" + quote(msg) + (f"#{anchor}" if anchor else "")
    return RedirectResponse(url, status_code=303)


# ---------- Вход ----------

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = "") -> HTMLResponse:
    return templates.TemplateResponse(request, "admin/login.html", {"error": error})


@router.post("/login")
def login(password: str = Form(...)) -> RedirectResponse:
    if not check_password(password):
        return RedirectResponse("/admin/login?error=1", status_code=303)
    resp = RedirectResponse("/admin", status_code=303)
    resp.set_cookie(COOKIE, make_token(), max_age=TTL, httponly=True, samesite="lax",
                    secure=get_settings().public_base_url.startswith("https"))
    return resp


@router.get("/logout")
def logout() -> RedirectResponse:
    resp = RedirectResponse("/admin/login", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


# ---------- Сводка ----------

@router.get("", response_class=HTMLResponse, dependencies=guard)
def dashboard(request: Request, days: int = 30) -> HTMLResponse:
    days = days if days in (1, 7, 30, 90, 365) else 30
    stats = analytics.dashboard(days)
    max_daily = max([d["dialogs"] for d in stats["daily"]] + [1])
    return render(request, "dashboard.html", stats=stats, max_daily=max_daily,
                  assistant=settings_store.get("assistant"))


# ---------- Диалоги ----------

@router.get("/dialogs", response_class=HTMLResponse, dependencies=guard)
def dialogs_list(request: Request, status: str = "", channel: str = "") -> HTMLResponse:
    return render(request, "dialogs.html", items=dialogs.list_dialogs(status, channel), status=status,
                  channel=channel, low_conf=float(classifier.settings().get("low_confidence", 0.8)))


@router.get("/dialogs/{dialog_id}", response_class=HTMLResponse, dependencies=guard)
def dialog_page(request: Request, dialog_id: int) -> HTMLResponse:
    dialog = dialogs.get(dialog_id)
    if not dialog:
        raise HTTPException(404)
    dialogs.mark_read(dialog_id)
    state = dialog["state"]
    cart = state.get("cart") or []
    calc = pricing.calculate(cart, state.get("discount_percent"), state.get("approved_discount_percent", 0))         if cart else None
    return render(request, "dialog.html", dialog=dialog, messages=dialogs.list_messages(dialog_id),
                  customer=customers.get(dialog["customer_id"]), docs=documents.list_documents(dialog_id),
                  cart=calc, max_discount=settings_store.get("pricing").get("max_agent_discount_percent", 5),
                  clf_heads=classifier.HEADS, clf_categories={h: classifier.categories(h) for h in classifier.HEADS})


@router.post("/dialogs/{dialog_id}/reply", dependencies=guard)
def dialog_reply(dialog_id: int, text: str = Form(...)) -> RedirectResponse:
    if text.strip():
        engine.operator_reply(dialog_id, text)
    return back(f"/admin/dialogs/{dialog_id}")


@router.post("/dialogs/{dialog_id}/classify", dependencies=guard)
def dialog_classify(dialog_id: int, head: str = Form(...), category: str = Form(""),
                    next: str = Form("")) -> RedirectResponse:
    """Менеджер подтверждает или исправляет тип обращения / IT-тему — пример уходит в обучающую выборку."""
    target = next if next.startswith("/admin/") else f"/admin/dialogs/{dialog_id}"
    try:
        classifier.correct_dialog(dialog_id, head, category)
    except ValueError as exc:
        return back(target, str(exc))
    return back(target, f"Сохранено: {category.strip()} — пример добавлен в обучение" if category.strip()
                else "Вернул автоматическую классификацию")


@router.post("/dialogs/{dialog_id}/{action}", dependencies=guard)
def dialog_action(dialog_id: int, action: str) -> RedirectResponse:
    actions = {"take": engine.take_over, "return": engine.return_to_bot, "close": engine.close_dialog}
    if action not in actions:
        raise HTTPException(404)
    actions[action](dialog_id)
    return back(f"/admin/dialogs/{dialog_id}" if action != "close" else "/admin/dialogs?status=active")


@router.post("/dialogs/{dialog_id}/customer", dependencies=guard)
def dialog_customer(dialog_id: int, name: str = Form(""), phone: str = Form(""), email: str = Form(""),
                    company: str = Form(""), inn: str = Form(""), kpp: str = Form(""),
                    address: str = Form("")) -> RedirectResponse:
    dialog = dialogs.get(dialog_id)
    cid, warnings = customers.upsert(dict(name=name, phone=phone, email=email, company=company, inn=inn, kpp=kpp,
                                          address=address), dialog["customer_id"])
    dialogs.set_customer(dialog_id, cid)
    return back(f"/admin/dialogs/{dialog_id}", "; ".join(warnings) or "Карточка клиента сохранена")


@router.get("/api/dialogs/{dialog_id}/messages", dependencies=guard)
def api_dialog_messages(dialog_id: int, after: int = 0) -> dict[str, Any]:
    dialog = dialogs.get(dialog_id)
    if not dialog:
        raise HTTPException(404)
    msgs = dialogs.list_messages(dialog_id, after_id=after)
    if msgs:
        dialogs.mark_read(dialog_id)
    return {"status": dialog["status"], "messages": [public_message(m) for m in msgs]}


@router.post("/api/dialogs/{dialog_id}/draft", dependencies=guard)
def api_dialog_draft(dialog_id: int) -> dict[str, Any]:
    return engine.operator_draft(dialog_id)


@router.get("/api/counters", dependencies=guard)
def api_counters() -> dict[str, Any]:
    return {"waiting": db.scalar("SELECT COUNT(*) FROM dialogs WHERE status = 'waiting_operator'") or 0,
            "drafts": db.scalar("SELECT COUNT(*) FROM documents WHERE status = 'draft'") or 0,
            "unread": db.scalar("SELECT COALESCE(SUM(unread), 0) FROM dialogs WHERE status != 'closed'") or 0}


# ---------- База знаний ----------

@router.get("/kb", response_class=HTMLResponse, dependencies=guard)
def kb_list(request: Request, q: str = "") -> HTMLResponse:
    return render(request, "kb.html", items=knowledge.list_articles(q), q=q)


@router.get("/kb/candidates", response_class=HTMLResponse, dependencies=guard)
def kb_candidates(request: Request) -> HTMLResponse:
    return render(request, "kb_candidates.html", items=knowledge.list_candidates())


@router.post("/kb/candidates/{candidate_id}", dependencies=guard)
def kb_candidate_resolve(candidate_id: int, action: str = Form(...), question: str = Form(""),
                         answer: str = Form(""), tags: str = Form("")) -> RedirectResponse:
    knowledge.resolve_candidate(candidate_id, action == "approve", question, answer, tags)
    return back("/admin/kb/candidates", "Добавлено в базу знаний" if action == "approve" else "Отклонено")


@router.get("/kb/new", response_class=HTMLResponse, dependencies=guard)
def kb_new(request: Request, question: str = "") -> HTMLResponse:
    return render(request, "kb_edit.html", item={"id": None, "question": question, "answer": "", "tags": "",
                                                 "active": 1})


@router.get("/kb/{article_id}", response_class=HTMLResponse, dependencies=guard)
def kb_edit(request: Request, article_id: int) -> HTMLResponse:
    item = knowledge.get_article(article_id)
    if not item:
        raise HTTPException(404)
    return render(request, "kb_edit.html", item=item)


@router.post("/kb/save", dependencies=guard)
def kb_save(question: str = Form(...), answer: str = Form(...), tags: str = Form(""),
            article_id: str = Form(""), active: str = Form("")) -> RedirectResponse:
    knowledge.save_article(question, answer, tags, article_id=int(article_id) if article_id else None,
                           active=bool(active))
    return back("/admin/kb", "Статья сохранена")


@router.post("/kb/{article_id}/delete", dependencies=guard)
def kb_delete(article_id: int) -> RedirectResponse:
    knowledge.delete_article(article_id)
    return back("/admin/kb", "Статья удалена")


@router.post("/kb/import", dependencies=guard)
async def kb_import(file: UploadFile = File(...)) -> RedirectResponse:
    try:
        n = knowledge.import_csv(await file.read())
        return back("/admin/kb", f"Импортировано статей: {n}")
    except Exception as exc:  # noqa: BLE001
        return back("/admin/kb", f"Ошибка импорта: {exc}")


# ---------- Классификатор обращений ----------

@router.get("/classifier", response_class=HTMLResponse, dependencies=guard)
def classifier_page(request: Request, text: str = "", tab: str = "check") -> HTMLResponse:
    """Вкладки: проверка (работа менеджера), IT-темы, статистика, справочник, модель. Данные считаются
    только для открытой вкладки — спорная разметка и облако слов не тормозят остальные."""
    tab = "stats" if text.strip() else tab
    tab = tab if tab in ("check", "topics", "stats", "handbook", "model") else "check"
    ctx: dict[str, Any] = {"tab": tab, "cfg": classifier.settings(), "heads": classifier.HEADS,
                           "algorithms": classifier.ALGORITHMS, "text": text}
    if tab == "check":
        ctx.update(queue=classifier.review_queue(), disputes=classifier.disputed(),
                   categories={"request_type": classifier.categories("request_type")})
    elif tab == "topics":
        ctx.update(cloud=classifier.topic_cloud(), topics=sorted(classifier.keywords("it_topic")))
    elif tab == "stats":
        ctx.update(dist={h: classifier.distribution(h) for h in classifier.HEADS},
                   result=classifier.classify(text) if text.strip() else None)
    elif tab == "handbook":
        ctx.update(handbook={h: classifier.keywords(h) for h in classifier.HEADS},
                   categories={h: classifier.categories(h) for h in classifier.HEADS})
    else:
        ctx.update(status=classifier.status(), neural_weight=classifier.NEURAL_WEIGHT)
    return render(request, "classifier.html", **ctx)


@router.post("/classifier/review", dependencies=guard)
def classifier_review(example_id: int = Form(...), category: str = Form(...)) -> RedirectResponse:
    """Проверка спорной разметки: подтвердить исходную категорию переписки или заменить."""
    try:
        classifier.review_label(example_id, category)
    except ValueError as exc:
        return back("/admin/classifier?tab=check#disputed", str(exc))
    return back("/admin/classifier?tab=check#disputed", f"Разметка сохранена: {category}")


@router.post("/classifier/settings", dependencies=guard)
def classifier_settings(algorithm: str = Form(...), low_confidence: float = Form(0.8),
                        train_on_reviewed: str = Form("")) -> RedirectResponse:
    if algorithm not in classifier.ALGORITHMS:
        raise HTTPException(400)
    settings_store.update("classifier", {"algorithm": algorithm,
                                         "low_confidence": min(0.99, max(0.1, low_confidence)),
                                         "train_on_reviewed": bool(train_on_reviewed)})
    classifier.retrain()
    return back("/admin/classifier?tab=model", "Настройки сохранены, модель переобучена")


@router.post("/classifier/neural", dependencies=guard)
def classifier_neural() -> RedirectResponse:
    if not neural.available():
        return back("/admin/classifier?tab=model", "Нейросеть не установлена: pip install -r requirements-ml.txt, "
                                         "затем scripts/download_model.py")
    started = classifier.train_neural()
    return back("/admin/classifier?tab=model", "Дообучение нейросети запущено в фоне — несколько минут" if started
                else "Нейросеть уже обучается")


@router.post("/classifier/retrain", dependencies=guard)
def classifier_retrain() -> RedirectResponse:
    classifier.retrain()
    return back("/admin/classifier?tab=model", "Модель переобучена")


@router.post("/classifier/keywords", dependencies=guard)
def classifier_keyword_add(head: str = Form(...), category: str = Form(...), keyword: str = Form(...)
                           ) -> RedirectResponse:
    try:
        classifier.add_keyword(head, category, keyword)
    except ValueError as exc:
        return back("/admin/classifier?tab=handbook#handbook", str(exc))
    return back("/admin/classifier?tab=handbook#handbook", f"Ключевое слово «{keyword.strip()}» добавлено")


@router.post("/classifier/cloud/assign", dependencies=guard)
def classifier_cloud_assign(word: str = Form(...), category: str = Form(...)) -> RedirectResponse:
    """Облако слов: слово перенесено в IT-тему (существующую или новую)."""
    try:
        classifier.add_keyword("it_topic", category, word, strong=True)
    except ValueError as exc:
        return back("/admin/classifier?tab=topics#cloud", str(exc))
    changed = classifier.refresh_topics()
    return back("/admin/classifier?tab=topics#cloud", f"«{word.strip()}» → «{category.strip()}». "
                                           f"Тема обновилась у диалогов: {changed}")


@router.post("/classifier/cloud/ignore", dependencies=guard)
def classifier_cloud_ignore(word: str = Form(...)) -> RedirectResponse:
    classifier.ignore_cloud_word(word)
    return back("/admin/classifier?tab=topics#cloud", f"Слово «{word.strip()}» скрыто из облака")


@router.post("/classifier/keywords/{keyword_id}/delete", dependencies=guard)
def classifier_keyword_delete(keyword_id: int) -> RedirectResponse:
    classifier.delete_keyword(keyword_id)
    return back("/admin/classifier?tab=handbook#handbook", "Ключевое слово удалено")


@router.post("/classifier/handbook", dependencies=guard)
def classifier_handbook_reload() -> RedirectResponse:
    n = classifier.load_handbook(replace=True)
    return back("/admin/classifier?tab=handbook#handbook", f"Исходный справочник IT-тем восстановлен: {n} ключевых слов")


@router.post("/classifier/upload", dependencies=guard)
async def classifier_upload(file: UploadFile = File(...), head: str = Form("request_type"),
                            text_col: int = Form(0), category_col: int = Form(1)) -> RedirectResponse:
    try:
        rows = classifier.parse_training_file(file.filename or "", await file.read(), text_col, category_col)
        n = classifier.import_training(head, rows)
        classifier.retrain(head)
        return back("/admin/classifier?tab=model", f"Загружено примеров: {n}, модель переобучена")
    except Exception as exc:  # noqa: BLE001
        return back("/admin/classifier?tab=model", f"Ошибка загрузки: {exc}")


# ---------- Каталог ----------

@router.get("/catalog", response_class=HTMLResponse, dependencies=guard)
def catalog_list(request: Request, q: str = "") -> HTMLResponse:
    return render(request, "catalog.html", items=catalog.list_products(q), q=q, total=catalog.count())


@router.get("/catalog/new", response_class=HTMLResponse, dependencies=guard)
def catalog_new(request: Request) -> HTMLResponse:
    item = {f: "" for f in catalog.PRODUCT_FIELDS}
    item.update(id=None, unit="лиц.", pricing="per_unit", price=0, term_months=12, users_min=0, users_max=0,
                vat_rate=None, lead_time_days=1, min_order_qty=1, active=1, is_russian=0)
    return render(request, "product_edit.html", item=item)


@router.get("/catalog/{product_id}", response_class=HTMLResponse, dependencies=guard)
def catalog_edit(request: Request, product_id: int) -> HTMLResponse:
    item = catalog.get(product_id)
    if not item:
        raise HTTPException(404)
    return render(request, "product_edit.html", item=item)


@router.post("/catalog/save", dependencies=guard)
async def catalog_save(request: Request) -> RedirectResponse:
    form = dict(await request.form())
    product_id = int(form.pop("product_id") or 0) or None
    form["active"] = 1 if form.get("active") else 0
    form["is_russian"] = 1 if form.get("is_russian") else 0
    try:
        catalog.save_product(form, product_id)
    except ValueError as exc:
        return back("/admin/catalog", str(exc))
    return back("/admin/catalog", "Товар сохранён")


@router.post("/catalog/import", dependencies=guard)
async def catalog_import(file: UploadFile = File(...)) -> RedirectResponse:
    try:
        res = feed.import_feed(file.filename or "", await file.read())
        return back("/admin/catalog", f"Обновлено товаров: {res['created_or_updated']}, пропущено: {res['skipped']}")
    except Exception as exc:  # noqa: BLE001
        return back("/admin/catalog", f"Ошибка импорта: {exc}")


# ---------- Документы и клиенты ----------

@router.get("/documents", response_class=HTMLResponse, dependencies=guard)
def documents_list(request: Request) -> HTMLResponse:
    return render(request, "documents.html", items=documents.list_documents())


@router.post("/documents/{doc_id}/status", dependencies=guard)
def document_status(doc_id: int, status: str = Form(...)) -> RedirectResponse:
    if status in documents.STATUS_LABELS:
        documents.set_status(doc_id, status)
    return back("/admin/documents", "Статус обновлён")


@router.post("/documents/{doc_id}/approve", dependencies=guard)
def document_approve(doc_id: int, next: str = Form("/admin/decisions")) -> RedirectResponse:
    engine.approve_document(doc_id)
    return back(next if next.startswith("/admin") else "/admin/decisions", "Документ утверждён, клиент уведомлён")


# ---------- «Ждут решения» ----------

@router.get("/decisions", response_class=HTMLResponse, dependencies=guard)
def decisions_page(request: Request) -> HTMLResponse:
    return render(request, "decisions.html", data=engine.pending_decisions(),
                  max_discount=settings_store.get("pricing").get("max_agent_discount_percent", 5))


@router.post("/decisions/{dialog_id}/approve-discount", dependencies=guard)
def decision_discount(dialog_id: int, percent: float = Form(...), next: str = Form("/admin/decisions")) -> RedirectResponse:
    if 0 < percent <= 100:
        engine.approve_discount(dialog_id, percent)
    return back(next if next.startswith("/admin") else "/admin/decisions", f"Скидка {percent:g}% согласована")


@router.post("/decisions/{dialog_id}/decline", dependencies=guard)
def decision_decline(dialog_id: int, text: str = Form(""), next: str = Form("/admin/decisions")) -> RedirectResponse:
    engine.decline_request(dialog_id, text)
    return back(next if next.startswith("/admin") else "/admin/decisions", "Ответ отправлен клиенту")


# ---------- Входящее обращение: письмо или звонок (имитация каналов для демонстрации) ----------

@router.get("/email", response_class=HTMLResponse, dependencies=guard)
def email_page(request: Request, channel: str = "email") -> HTMLResponse:
    return render(request, "email.html", result=None, form={"channel": channel},
                  env_smtp=get_settings().smtp_enabled)


@router.post("/email", response_class=HTMLResponse, dependencies=guard)
def email_simulate(request: Request, sender: str = Form(...), sender_name: str = Form(""), subject: str = Form(""),
                   body: str = Form(...), channel: str = Form("email")) -> HTMLResponse:
    if channel == "phone":
        # Звонок: текст — расшифровка речи клиента (в продуктиве приходит от телефонии через /api/channels/inbound)
        dialog = dialogs.get_or_create_by_external("phone", sender.strip(), "Звонок")
        if not dialog["customer_id"]:
            cid, _ = customers.upsert({"phone": sender, "name": sender_name})
            dialogs.set_customer(dialog["id"], cid)
        replies = engine.handle_client_message(dialog["id"], body, {"channel": "phone"})
        result = {"dialog_id": dialog["id"], "replies": [r["text"] for r in replies], "sent": False}
    else:
        from ..channels.email_channel import handle_incoming_email
        result = handle_incoming_email(sender, sender_name, subject, body)
    return render(request, "email.html", result=result,
                  form={"sender": sender, "sender_name": sender_name, "subject": subject, "body": body,
                        "channel": channel},
                  env_smtp=get_settings().smtp_enabled)


# ---------- Переписки поддержки ----------

@router.get("/support", response_class=HTMLResponse, dependencies=guard)
def support_page(request: Request, q: str = "") -> HTMLResponse:
    items = support_examples.search(q, top_k=50) if q else db.query(
        "SELECT * FROM support_examples ORDER BY id DESC LIMIT 200")
    return render(request, "support.html", items=items, q=q, total=support_examples.count(),
                  categories=support_examples.categories())


@router.post("/support/import", dependencies=guard)
async def support_import(file: UploadFile = File(...)) -> RedirectResponse:
    try:
        n = support_examples.import_file(file.filename or "", await file.read())
        return back("/admin/support", f"Загружено пар «вопрос — ответ»: {n}")
    except Exception as exc:  # noqa: BLE001
        return back("/admin/support", f"Ошибка импорта: {exc}")


@router.post("/support/{example_id}/to-kb", dependencies=guard)
def support_to_kb(example_id: int) -> RedirectResponse:
    ex = db.query_one("SELECT * FROM support_examples WHERE id = ?", (example_id,))
    if ex:
        knowledge.add_candidate(ex["question"], ex["answer"], None)
    return back("/admin/support", "Добавлено в кандидаты базы знаний")


@router.get("/customers", response_class=HTMLResponse, dependencies=guard)
def customers_list(request: Request) -> HTMLResponse:
    items = db.query(
        "SELECT c.*, (SELECT MAX(d.id) FROM dialogs d WHERE d.customer_id = c.id) AS last_dialog_id, "
        "(SELECT COUNT(*) FROM documents x WHERE x.customer_id = c.id AND x.kind = 'quote') AS quotes, "
        "(SELECT COUNT(*) FROM documents x WHERE x.customer_id = c.id AND x.kind = 'invoice') AS invoices "
        "FROM customers c ORDER BY c.id DESC LIMIT 500")
    return render(request, "customers.html", items=items)


# ---------- Настройки ----------

SECTION_TITLES = {"company": "Компания и шаблон КП", "pricing": "Цены, скидки, оплата",
                  "assistant": "Ассистент и SLA", "economics": "Экономика: стоимость ИИ и сотрудников"}


@router.get("/settings", response_class=HTMLResponse, dependencies=guard)
def settings_page(request: Request) -> HTMLResponse:
    sections = {k: settings_store.get(k) for k in SECTION_TITLES}
    return render(request, "settings.html", sections=sections, titles=SECTION_TITLES,
                  env=get_settings())


@router.post("/settings/{section}", dependencies=guard)
async def settings_save(request: Request, section: str) -> RedirectResponse:
    if section not in SECTION_TITLES:
        raise HTTPException(404)
    current = settings_store.get(section)
    form = await request.form()
    values: dict[str, Any] = {}
    try:
        for key, old in current.items():
            if isinstance(old, bool):
                values[key] = key in form
            elif key not in form:
                continue
            elif isinstance(old, (list, dict)):
                values[key] = json.loads(str(form[key]))
            elif isinstance(old, (int, float)):
                values[key] = type(old)(str(form[key]).replace(",", ".")) if isinstance(old, float) \
                    else int(float(str(form[key]).replace(",", ".")))
            else:
                values[key] = str(form[key])
    except (ValueError, json.JSONDecodeError) as exc:
        return back("/admin/settings", f"Ошибка в поле {key}: {exc}")
    settings_store.update(section, values)
    return back("/admin/settings", "Настройки сохранены")
