"""Классификатор обращений."""
from __future__ import annotations

from pathlib import Path

from app.assistant import engine
from app.services import classifier, dialogs, settings_store, support_examples

LABELED = {
    "Просто цена конкретного товара": [
        "сколько стоит касперский на 10 компьютеров", "какая цена у ред ос на год", "сколько стоит р7 офис",
        "цена astra linux смоленск", "стоимость dr.web на 5 пк", "сколько стоит лицензия windows 11 pro",
    ],
    "Физлицо / самозанятый / срок": [
        "хочу купить как физическое лицо для дома", "можно оплатить картой физлица", "я самозанятый, нужна лицензия",
        "покупаю для домашнего использования", "нужна программа для себя как физ лицо", "физлицо может купить?",
    ],
    "Госзакупки / процедуры": [
        "закупка по 44-фз, нужно кп для обоснования", "мы бюджетная организация, тендер на антивирус",
        "участвуем в аукционе по 223-фз", "нужна котировка для госзакупки", "госконтракт на поставку по",
        "конкурентная закупка лицензий",
    ],
}


def _label_chats() -> None:
    for category, texts in LABELED.items():
        for i, t in enumerate(texts):
            support_examples.add(t, "", source="тест", category=category, external_id=f"{category}-{i}")


def test_keywords_match_whole_words_not_substrings():
    m = classifier.KeywordMatcher({"Закупка ПО": ["ПО"], "Доступы": ["права"], "Антивирусы": ["антивирус"],
                                   "Счет": ["выставить счет"]})
    assert m.match("почему по договору не работает")["hits"] == {}  # «ПО» не находится в «почему»
    assert m.match("правительство области")["hits"] == {}  # а «права» — в «правительство»
    assert m.match("Нужно ПО для офиса")["winner"] == "Закупка ПО"
    assert m.match("нужна антивирусная защита")["winner"] == "Антивирусы"
    assert m.match("прошу выставить, пожалуйста, счет")["winner"] == "Счет"  # фраза с разрывом


def test_it_topics_handbook_gives_it_topic():
    topics = classifier.keywords("it_topic")
    assert len(topics) == 31 and "Антивирусная защита" in topics
    r = classifier.classify("Нужна лицензия Zoom для видеоконференций на 20 человек")
    assert r["it_topic"]["category"] == "Конференцсвязь / ВКС"
    assert r["it_topic"]["source"] == "keyword" and not r["it_topic"]["low_confidence"]


def test_model_trains_on_chats_and_learns_from_manager():
    _label_chats()
    clf = classifier.model("request_type")
    assert clf is not None and set(clf.categories) == set(LABELED)
    r = classifier.classify("подскажите сколько стоит ред ос")["request_type"]
    assert r["category"] == "Просто цена конкретного товара" and r["source"] == "model"
    # Ключевое слово справочника важнее модели
    r = classifier.classify("Закупаем по 44-ФЗ, пришлите котировку")["request_type"]
    assert r["category"] == "Госзакупки / процедуры" and r["source"] == "keyword"

    dialog = dialogs.create("web")
    engine.handle_client_message(dialog["id"], "сколько стоит касперский на 20 компьютеров?")
    d = dialogs.get(dialog["id"])
    assert d["request_type"] == "Просто цена конкретного товара"
    assert d["classification"]["request_type"]["top"]

    classifier.correct_dialog(dialog["id"], "request_type", "Госзакупки / процедуры")
    engine.handle_client_message(dialog["id"], "и ещё вопрос по цене")
    d = dialogs.get(dialog["id"])
    assert d["request_type"] == "Госзакупки / процедуры" and d["request_type_source"] == "manager"
    assert classifier.status()[0]["manager_samples"] == 1
    assert classifier.model("request_type") is not clf  # новые данные → модель переобучена

    classifier.correct_dialog(dialog["id"], "request_type", "")  # вернуть автоматическую
    assert dialogs.get(dialog["id"])["request_type_source"] != "manager"


def test_baseline_nb_algorithm_available():
    _label_chats()
    settings_store.update("classifier", {"algorithm": "nb"})
    clf = classifier.model("request_type")
    assert clf.algorithm == "nb"
    assert classifier.classify("сколько стоит р7 офис на год")["request_type"]["category"] == \
        "Просто цена конкретного товара"


def test_training_file_upload():
    rows = classifier.parse_training_file("t.csv", "Текст;Категория\nнужен zoom;ВКС\nне печатает принтер;Оргтехника\n"
                                          .encode("utf-8"))
    assert rows == [("нужен zoom", "ВКС"), ("не печатает принтер", "Оргтехника")]
    assert classifier.import_training("it_topic", rows) == 2


def test_admin_classifier_pages(admin):
    _label_chats()
    r = admin.get("/admin/classifier")
    assert r.status_code == 200 and "Классификатор обращений" in r.text and "Новые обращения" in r.text
    for tab in ("topics", "stats", "handbook", "model"):
        assert admin.get("/admin/classifier", params={"tab": tab}).status_code == 200
    assert "Антивирусная защита" in admin.get("/admin/classifier", params={"tab": "handbook"}).text
    r = admin.get("/admin/classifier", params={"text": "Сколько стоит Kaspersky для 30 компьютеров?"})
    assert "Антивирусная защита" in r.text and "result-box" in r.text

    pid = admin.post("/api/chat/start", json={}).json()["dialog_id"]
    admin.post(f"/api/chat/{pid}/messages", json={"text": "хочу купить для дома как физлицо"})
    did = dialogs.get_by_public_id(pid)["id"]
    page = admin.get(f"/admin/dialogs/{did}")
    assert "Классификация" in page.text and "Физлицо / самозанятый / срок" in page.text
    assert "Тип обращения" in admin.get("/admin/dialogs?status=").text

    r = admin.post(f"/admin/dialogs/{did}/classify", data={"head": "it_topic", "category": "Антивирусная защита"},
                   follow_redirects=False)
    assert r.status_code == 303 and dialogs.get(did)["it_topic"] == "Антивирусная защита"

    admin.post("/admin/classifier/keywords", data={"head": "it_topic", "category": "Антивирусная защита",
                                                   "keyword": "антивирь"})
    assert any(w["keyword"] == "антивирь" for w in classifier.keywords("it_topic")["Антивирусная защита"])
    r = admin.post("/admin/classifier/upload", data={"head": "request_type", "text_col": "0", "category_col": "1"},
                   files={"file": ("x.csv", "цена ред ос,Просто цена конкретного товара\n".encode("utf-8"))},
                   follow_redirects=False)
    assert r.status_code == 303 and "msg=" in r.headers["location"]
    assert admin.post("/admin/classifier/retrain", follow_redirects=False).status_code == 303


def test_label_review_overrides_training_and_disputes():
    for i in range(20):
        support_examples.add(f"сколько стоит касперский на {i + 3} компьютеров", "", source="тест",
                             category="Просто цена конкретного товара", external_id=f"p{i}")
        support_examples.add(f"выставьте счет на оплату, реквизиты ооо ромашка {i}", "", source="тест",
                             category="Счет / КП как документ (с реквизитами)", external_id=f"s{i}")
        support_examples.add(f"хочу купить для дома как физлицо вариант {i}", "", source="тест",
                             category="Физлицо / самозанятый / срок", external_id=f"f{i}")
    # явная ошибка разметки: просьба выставить счёт размечена как «цена»
    support_examples.add("выставьте счет на оплату, реквизиты во вложении", "", source="тест",
                         category="Просто цена конкретного товара", external_id="bad")
    bad = next(i for i in classifier.labeled_chats() if i["text"].endswith("во вложении"))

    found = classifier.disputed()
    assert any(d["id"] == bad["id"] and d["suggestion"].startswith("Счет") for d in found["items"])

    classifier.review_label(bad["id"], "Счет / КП как документ (с реквизитами)")
    item = next(i for i in classifier.labeled_chats() if i["id"] == bad["id"])
    assert item["reviewed"] and item["category"].startswith("Счет") and item["original"].startswith("Просто")
    texts, labels = classifier.training_data("request_type")
    assert labels[texts.index(bad["text"])].startswith("Просто")  # по умолчанию учимся на исходной разметке
    settings_store.update("classifier", {"train_on_reviewed": True})
    texts, labels = classifier.training_data("request_type")
    assert labels[texts.index(bad["text"])].startswith("Счет")  # а при включённой настройке — на проверенной
    after = classifier.disputed()
    assert all(d["id"] != bad["id"] for d in after["items"]) and after["reviewed"] == 1 and after["changed"] == 1


def test_hybrid_without_neural_model_falls_back():
    _label_chats()
    settings_store.update("classifier", {"algorithm": "hybrid"})
    clf = classifier.model("request_type")
    assert clf.algorithm == "hybrid" and not clf.uses_neural  # в тестах нейросеть не загружается
    assert classifier.classify("сколько стоит р7 офис")["request_type"]["category"] == "Просто цена конкретного товара"


def test_neural_versions_switch_by_pointer():
    from app.services import neural
    def version(name: str, samples: int) -> Path:
        path = neural.models_root() / name
        path.mkdir(parents=True)
        (path / "meta.json").write_text('{"labels": ["a", "b"], "samples": %d, "trained_at": "2026-01-01"}'
                                        % samples, encoding="utf-8")
        return path

    old = version("request_type-rubert-20260101-000000", 10)
    assert neural.meta("request_type") is None  # пока ни одна версия не включена
    neural._activate("request_type", old)
    assert neural.meta("request_type")["samples"] == 10
    new = version("request_type-rubert-20260102-000000", 20)
    neural._activate("request_type", new)  # новая версия включается, старая удаляется
    assert neural.meta("request_type")["samples"] == 20 and not old.exists()
    assert not neural.available()  # базовая модель в тестах не скачана — нейросеть не используется


def test_one_click_confirm_from_review_queue(admin):
    _label_chats()
    pid = admin.post("/api/chat/start", json={}).json()["dialog_id"]
    admin.post(f"/api/chat/{pid}/messages", json={"text": "подскажите по лицензиям"})
    did = dialogs.get_by_public_id(pid)["id"]
    predicted = dialogs.get(did)["request_type"]
    page = admin.get("/admin/classifier")
    assert "✓ Верно" in page.text and classifier.review_count() >= 1
    r = admin.post(f"/admin/dialogs/{did}/classify",
                   data={"head": "request_type", "category": predicted, "next": "/admin/classifier?tab=check#queue"},
                   follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/admin/classifier?tab=check")
    assert r.headers["location"].index("msg=") < r.headers["location"].index("#")  # сообщение не за якорем
    assert dialogs.get(did)["request_type_source"] == "manager"
    assert classifier.status()[0]["manager_samples"] == 1  # подтверждение тоже идёт в обучение


def test_topic_cloud_assign_and_hide(admin):
    for i, t in enumerate(["нужна криптопро на 5 рабочих мест", "криптопро csp для организации, цена?",
                           "продлить криптопро на год", "лицензия эшелон сканер нужна", "эшелон сканер-вс на 32 ip"]):
        support_examples.add(t, "", source="тест", category="Прочее (покупка)", external_id=f"c{i}")
    cloud = classifier.topic_cloud()
    words = {w["word"]: w for w in cloud["words"]}
    assert "криптопро" in words and words["криптопро"]["count"] == 3 and words["криптопро"]["examples"]
    assert "лицензия" not in words and "нужна" not in words  # слова о покупке в облако не попадают

    r = admin.post("/admin/classifier/cloud/assign", data={"word": "криптопро", "category": "Электронная подпись"},
                   follow_redirects=False)
    assert r.status_code == 303 and "#cloud" in r.headers["location"]
    assert "криптопро" not in {w["word"] for w in classifier.topic_cloud()["words"]}
    topic = classifier.classify("сколько стоит криптопро?")["it_topic"]
    assert topic["category"] == "Электронная подпись" and not topic["low_confidence"]  # одного слова из облака хватает

    admin.post("/admin/classifier/cloud/ignore", data={"word": "эшелон"})
    assert "эшелон" not in {w["word"] for w in classifier.topic_cloud()["words"]}
    assert 'id="cloud"' in admin.get("/admin/classifier", params={"tab": "topics"}).text
    assert "Электронная подпись" in admin.get("/admin/classifier", params={"tab": "handbook"}).text
