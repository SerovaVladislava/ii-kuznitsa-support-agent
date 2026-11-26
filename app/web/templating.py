from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi.templating import Jinja2Templates

from ..services import dialogs, documents
from ..services.textutil import fmt_money, fmt_qty

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def _dt(value: str | None) -> str:
    if not value:
        return ""
    try:
        d = datetime.fromisoformat(value)
    except ValueError:
        return value
    return d.strftime("%H:%M") if d.date() == datetime.now().date() else d.strftime("%d.%m %H:%M")


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


templates.env.filters.update(money=fmt_money, qty=fmt_qty, dt=_dt, pct=_pct)
templates.env.globals.update(STATUSES=dialogs.STATUSES, CHANNELS=dialogs.CHANNELS,
                             REASONS=dialogs.ESCALATION_REASONS, STATUS_LABELS=documents.STATUS_LABELS)


def _asset_version() -> str:
    """Версия статики по времени изменения файлов: браузер не держит устаревшие стили и виджет после обновления."""
    static = Path(__file__).resolve().parent / "static"
    try:
        return str(int(max((static / n).stat().st_mtime for n in ("admin.css", "widget.js"))))
    except OSError:
        return "0"


templates.env.globals["asset_v"] = _asset_version
