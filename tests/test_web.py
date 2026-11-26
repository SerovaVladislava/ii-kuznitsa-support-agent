from __future__ import annotations

from pathlib import Path

from app.services import catalog, dialogs, documents

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _quote_via_chat(client) -> tuple[str, dict]:
    pid = client.post("/api/chat/start", json={}).json()["dialog_id"]
    client.post(f"/api/chat/{pid}/messages", json={"text": "Продлить Касперский Стандартный на 40 ПК на год"})
    client.post(f"/api/chat/{pid}/messages", json={"text": "КП"})
    r = client.post(f"/api/chat/{pid}/messages", json={"text": "ООО Альфа, a@alfa.ru"})
    return pid, r.json()["messages"][0]


def test_widget_chat_flow_and_operator_polling(client):
    data = client.post("/api/chat/start", json={"page_url": "https://shop/x"}).json()
    pid = data["dialog_id"]
    assert "подобрать" in data["greeting"]
    r = client.post(f"/api/chat/{pid}/messages", json={"text": "Нужен MDaemon Standard"})
    assert "Сколько пользователей" in r.json()["messages"][0]["text"]
    assert len(client.post("/api/chat/start", json={"dialog_id": pid}).json()["messages"]) == 2

    from app.assistant import engine
    d = dialogs.get_by_public_id(pid)
    last_id = dialogs.list_messages(d["id"])[-1]["id"]
    engine.operator_reply(d["id"], "Здравствуйте, это менеджер.")
    r = client.get(f"/api/chat/{pid}/messages", params={"after": last_id})
    assert r.json()["messages"][0]["role"] == "operator" and r.json()["status"] == "operator"


def test_quote_document_follows_company_template(client):
    _, msg = _quote_via_chat(client)
    assert msg["documents"] and msg["documents"][0]["number"].startswith("КП-")
    doc = documents.list_documents()[0]
    html = client.get(f"/d/{doc['token']}").text
    for fragment in ("ЧЕРНОВИК", "Куда: ООО «Альфа»", f"Исх. № {doc['number']}", "Уважаемый заказчик!",
                     "ПАО «Софтлайн» предлагает Вашему вниманию коммерческое предложение",
                     "В т.ч. НДС, руб.", "Без НДС", "Итого", "Условия оплаты: Предоплата 100%",
                     "Срок поставки:", "Срок действия коммерческого предложения - до"):
        assert fragment in html, fragment
    docx = client.get(f"/d/{doc['token']}.docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"
    assert client.get("/d/wrong-token").status_code == 404


def test_inbound_api_requires_token(client):
    body = {"channel": "phone", "external_id": "+79990000000", "text": "Какие способы оплаты?", "name": "Иван"}
    assert client.post("/api/channels/inbound", json=body).status_code == 401
    r = client.post("/api/channels/inbound", json=body, headers={"Authorization": "Bearer tok"})
    assert r.status_code == 200 and "СБП" in r.json()["replies"][0]


def test_admin_requires_login(client):
    r = client.get("/admin", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/admin/login"
    assert client.get("/admin/api/counters").status_code == 401


def test_admin_pages_render(admin):
    pid, _ = _quote_via_chat(admin)
    admin.post(f"/api/chat/{pid}/messages", json={"text": "дайте скидку 12%"})
    d = dialogs.get_by_public_id(pid)
    for url in ["/admin", "/admin?days=7", "/admin/decisions", "/admin/dialogs", "/admin/dialogs?status=active",
                f"/admin/dialogs/{d['id']}", "/admin/kb", "/admin/kb?q=ключ", "/admin/kb/new", "/admin/kb/1",
                "/admin/kb/candidates", "/admin/support", "/admin/support?q=продление", "/admin/catalog",
                "/admin/catalog?q=касперский", "/admin/catalog/1", "/admin/catalog/new", "/admin/documents",
                "/admin/customers", "/admin/email", "/admin/settings", "/demo"]:
        r = admin.get(url)
        assert r.status_code == 200, url
    board = admin.get("/admin/decisions").text
    assert "Особые условия" in board and "Согласовать скидку" in board and "Утвердить и отправить" in board
    counters = admin.get("/admin/api/counters").json()
    assert counters["waiting"] == 1 and counters["drafts"] == 1


def test_admin_decisions_flow(admin):
    pid, _ = _quote_via_chat(admin)
    admin.post(f"/api/chat/{pid}/messages", json={"text": "дайте скидку 8%"})
    d = dialogs.get_by_public_id(pid)
    assert d["status"] == "waiting_operator"
    r = admin.post(f"/admin/decisions/{d['id']}/approve-discount", data={"percent": "8"}, follow_redirects=False)
    assert r.status_code == 303
    assert dialogs.get(d["id"])["status"] == "bot"
    msgs = admin.get(f"/api/chat/{pid}/messages").json()["messages"]
    assert "согласовал скидку 8%" in msgs[-1]["text"]

    docs = documents.list_documents(d["id"])
    assert docs[0]["status"] == "approved" and docs[0]["totals"]["discount_percent"] == 8
    assert all(x["status"] != "draft" for x in docs)  # прежний черновик заменён

    admin.post(f"/api/chat/{pid}/messages", json={"text": "сделайте ещё КП"})  # новый черновик — со скидкой 8%
    draft = next(x for x in documents.list_documents(d["id"]) if x["status"] == "draft")
    assert draft["totals"]["discount_percent"] == 8
    r = admin.post(f"/admin/documents/{draft['id']}/approve", follow_redirects=False)
    assert r.status_code == 303 and documents.get(draft["id"])["status"] == "approved"


def test_admin_email_simulation(admin):
    r = admin.post("/admin/email", data={"sender": "olga@example.com", "sender_name": "Ольга",
                                         "subject": "Продление",
                                         "body": "Добрый день! Заканчивается лицензия Касперского на 40 компьютеров, "
                                                 "нужно продлить на год, редакция Стандартный. Можно ли оплатить с "
                                                 "отсрочкой 30 дней? Ольга, ООО «Альфа»"})
    # Письмо с двумя просьбами: подобран артикул продления и дан ответ по оплате
    assert r.status_code == 200 and "KESB-STD-R-1Y-25-49" in r.text and "предоплат" in r.text
    assert "не отправлен" in r.text
    d = dialogs.find_open_by_external("email", "olga@example.com")
    assert d and d["channel"] == "email"


def test_admin_phone_call_simulation(admin):
    r = admin.post("/admin/email", data={"channel": "phone", "sender": "+79000000000", "sender_name": "Сергей",
                                         "body": "Алло, нам нужен почтовый сервер, человек на тридцать, "
                                                 "свой сервер на Windows есть. Что посоветуете?"})
    assert r.status_code == 200 and "российское ПО" in r.text
    assert "Сколько сотрудников" not in r.text  # число из речи («на тридцать») распознано
    d = dialogs.find_open_by_external("phone", "+79000000000")
    assert d and d["state"]["need"]["users"] == 30 and d["state"]["need"]["deployment"] == "on-prem"


def test_admin_feed_and_support_upload(admin):
    r = admin.post("/admin/catalog/import", files={"file": ("feed.yml", (SAMPLES / "feed_sample.yml").read_bytes())},
                   follow_redirects=False)
    assert "msg=" in r.headers["location"] and catalog.get_by_sku("DM-SRV-50")
    r = admin.post("/admin/support/import",
                   files={"file": ("chats.txt", (SAMPLES / "support_sample.txt").read_bytes())}, follow_redirects=False)
    assert r.status_code == 303


def test_rate_limit(client):
    from app.web import public
    pid = client.post("/api/chat/start", json={}).json()["dialog_id"]
    codes = [client.post(f"/api/chat/{pid}/messages", json={"text": "привет"}).status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and codes[-1] == 429
    public.message_limiter._hits.clear()
