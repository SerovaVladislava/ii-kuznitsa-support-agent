from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import seed  # noqa: E402
from app.assistant import engine  # noqa: E402
from app.config import Settings, override_settings  # noqa: E402
from app.services import knowledge, settings_store, support_examples  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_env(tmp_path, monkeypatch):
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "EMAIL_IMAP_HOST", "MANAGER_EMAIL",
                "MANAGER_TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))  # тесты не загружают нейросеть
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("INTEGRATION_TOKEN", "tok")
    settings = Settings()
    override_settings(settings)
    settings_store.reset_cache()
    knowledge.index.invalidate()
    support_examples.index.invalidate()
    engine.set_llm_agent(None)
    seed.seed_if_empty()
    yield settings
    engine.set_llm_agent(None)


@pytest.fixture
def client(fresh_env):
    from fastapi.testclient import TestClient

    from app.web import public
    from app.web.main import create_app

    public.message_limiter._hits.clear()
    public.start_limiter._hits.clear()
    with TestClient(create_app(start_background=False)) as c:
        yield c


@pytest.fixture
def admin(client):
    r = client.post("/admin/login", data={"password": "secret"}, follow_redirects=False)
    assert r.status_code == 303
    return client
