"""Конфигурация из переменных окружения и файла .env (без внешних зависимостей)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BASE_DIR / "data"))))
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    public_base_url: str = field(default_factory=lambda: _env("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/"))
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY", "change-me-please"))
    admin_password: str = field(default_factory=lambda: _env("ADMIN_PASSWORD", "admin"))
    integration_token: str = field(default_factory=lambda: _env("INTEGRATION_TOKEN"))
    widget_allowed_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in _env("WIDGET_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    )

    # ключ языковой модели для ИИ-режима; без ключа работает офлайн-ассистент
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY") or _env("ANTHROPIC_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "claude-opus-5-5"))
    llm_effort: str = field(default_factory=lambda: _env("LLM_EFFORT", "low"))
    llm_fallbacks: bool = field(default_factory=lambda: _env("LLM_FALLBACKS", "1") == "1")
    llm_disabled: bool = field(default_factory=lambda: _env("LLM_DISABLED", "0") == "1")

    email_imap_host: str = field(default_factory=lambda: _env("EMAIL_IMAP_HOST"))
    email_imap_port: int = field(default_factory=lambda: _env_int("EMAIL_IMAP_PORT", 993))
    email_smtp_host: str = field(default_factory=lambda: _env("EMAIL_SMTP_HOST"))
    email_smtp_port: int = field(default_factory=lambda: _env_int("EMAIL_SMTP_PORT", 465))
    email_user: str = field(default_factory=lambda: _env("EMAIL_USER"))
    email_password: str = field(default_factory=lambda: _env("EMAIL_PASSWORD"))
    email_poll_seconds: int = field(default_factory=lambda: _env_int("EMAIL_POLL_SECONDS", 60))

    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    manager_telegram_chat_id: str = field(default_factory=lambda: _env("MANAGER_TELEGRAM_CHAT_ID"))
    manager_email: str = field(default_factory=lambda: _env("MANAGER_EMAIL"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "store_agent.sqlite3"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key) and not self.llm_disabled

    @property
    def email_enabled(self) -> bool:
        return bool(self.email_imap_host and self.email_smtp_host and self.email_user and self.email_password)

    @property
    def smtp_enabled(self) -> bool:
        return bool(self.email_smtp_host and self.email_user and self.email_password)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _load_dotenv(BASE_DIR / ".env")
        _settings = Settings()
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
    return _settings


def override_settings(settings: Settings) -> None:
    """Для тестов: подменить конфигурацию."""
    global _settings
    _settings = settings
    settings.data_dir.mkdir(parents=True, exist_ok=True)
