"""Простая авторизация админки: пароль из ADMIN_PASSWORD, подписанная HMAC cookie."""
from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import HTTPException, Request

from ..config import get_settings

COOKIE = "consultant_admin"
TTL = 12 * 3600


def _sign(payload: str) -> str:
    return hmac.new(get_settings().secret_key.encode(), payload.encode(), hashlib.sha256).hexdigest()


def make_token() -> str:
    expires = str(int(time.time()) + TTL)
    return f"{expires}.{_sign(expires)}"


def check_password(password: str) -> bool:
    return hmac.compare_digest(password.encode(), get_settings().admin_password.encode())


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(COOKIE, "")
    expires, _, sig = token.partition(".")
    if not expires.isdigit() or int(expires) < time.time():
        return False
    return hmac.compare_digest(sig, _sign(expires))


class LoginRequired(Exception):
    pass


def require_admin(request: Request) -> None:
    if not is_authenticated(request):
        if request.url.path.startswith("/admin/api/"):
            raise HTTPException(status_code=401, detail="Требуется вход")
        raise LoginRequired()
