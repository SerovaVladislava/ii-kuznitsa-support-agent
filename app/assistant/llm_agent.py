"""ИИ-режим консультанта: языковая модель по API, цикл tool use поверх инструментов компании.

- История диалога хранится в БД в формате API и только дополняется (append-only), включая блоки
  размышлений, — так её можно продолжать в следующих сообщениях без потери контекста.
- Сообщения, появившиеся в диалоге без участия модели (ответы менеджера), подмешиваются служебной
  вставкой в следующий пользовательский ход.
- При недоступности API движок переключается на офлайн-ассистента (см. engine.py).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import anthropic

from ..config import get_settings
from ..services import dialogs
from ..services.textutil import WEEKDAYS, fmt_date
from . import prompts
from .tools import ToolContext, api_tools, run_tool

log = logging.getLogger("llm")

MAX_STEPS = 8
ROLE_LABELS = {"client": "Клиент", "bot": "Ассистент", "operator": "Менеджер", "system": "Система"}


class LLMUnavailable(Exception):
    def __init__(self, message: str, reset_history: bool = False):
        super().__init__(message)
        self.reset_history = reset_history


@dataclass
class AgentReply:
    text: str
    mode: str
    escalation: tuple[str, str] | None = None
    documents: list[dict[str, Any]] = field(default_factory=list)
    kb_used: list[int] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)


def _service_note(channel: str) -> str:
    today = date.today()
    return f"[Служебно: сегодня {fmt_date(today)} ({WEEKDAYS[today.weekday()]}). Канал: {channel}.]"


def format_transcript(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{ROLE_LABELS.get(m['role'], m['role'])}: {m['text']}" for m in messages)


class LLMAgent:
    def __init__(self, client: Any = None):
        s = get_settings()
        self.settings = s
        self.client = client or anthropic.Anthropic(api_key=s.llm_api_key, max_retries=2, timeout=120)

    # ---- низкоуровневый вызов ----
    def _create(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        params: dict[str, Any] = {
            "model": self.settings.llm_model,
            "max_tokens": 16000,
            # tools + system — стабильный префикс, кэшируется; переписка кэшируется автоматически
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "cache_control": {"type": "ephemeral"},
            "tools": tools,
            "messages": messages,
            "output_config": {"effort": self.settings.llm_effort},
        }
        if self.settings.llm_fallbacks:
            # При отказе модели запрос автоматически повторяется на резервной модели на стороне API
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        try:
            return self.client.beta.messages.create(**params)
        except anthropic.BadRequestError as exc:
            log.error("LLM API 400: %s", exc)
            raise LLMUnavailable(f"bad request: {exc}", reset_history=True) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            log.error("LLM API: ошибка доступа: %s", exc)
            raise LLMUnavailable(f"auth: {exc}") from exc
        except anthropic.RateLimitError as exc:
            log.warning("LLM API: превышен лимит запросов")
            raise LLMUnavailable("rate limit") from exc
        except anthropic.APIStatusError as exc:
            log.error("LLM API %s: %s", exc.status_code, exc)
            raise LLMUnavailable(f"status {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            log.error("LLM API недоступен: %s", exc)
            raise LLMUnavailable("connection") from exc

    # ---- цикл tool use ----
    def _run(self, system: str, messages: list[dict[str, Any]], ctx: ToolContext,
             tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any], str]:
        working = list(messages)
        texts: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0, "requests": 0, "model": ""}
        stop = "max_steps"
        for _ in range(MAX_STEPS):
            resp = self._create(system, working, tools)
            usage["requests"] += 1
            usage["model"] = getattr(resp, "model", "")
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage[key] += getattr(resp.usage, key, 0) or 0

            if resp.stop_reason == "refusal":
                stop = "refusal"
                break
            # Полный content (включая блоки размышлений) — обязательно без изменений
            working.append({"role": "assistant", "content": [b.to_dict(mode="json") for b in resp.content]})
            texts.extend(b.text.strip() for b in resp.content if b.type == "text" and b.text.strip())

            if resp.stop_reason == "tool_use":
                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    result, is_error = run_tool(ctx, block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(result, ensure_ascii=False, default=str),
                                    "is_error": is_error})
                working.append({"role": "user", "content": results})
                continue
            if resp.stop_reason == "pause_turn":
                continue
            stop = resp.stop_reason or "end_turn"
            break
        return working, texts, usage, stop

    # ---- ответ клиенту ----
    def reply(self, dialog_id: int, client_msg_id: int, client_text: str, ctx: ToolContext) -> AgentReply:
        history, synced = dialogs.load_llm_history(dialog_id)
        earlier = [m for m in dialogs.list_messages(dialog_id, after_id=synced) if m["id"] < client_msg_id]
        content = [{"type": "text", "text": _service_note(ctx.channel)}]
        dialog = dialogs.get(dialog_id)
        if not history and dialog["channel"] == "web" and dialog["subject"]:
            from ..services import catalog
            hint = catalog.page_hint(dialog["subject"])
            if hint:
                about = (f"странице продукта «{hint['family']}» ({hint.get('vendor', '')})" if hint.get("family")
                         else f"странице вендора {hint.get('vendor', hint['vendor_slug'])}")
                content.append({"type": "text", "text": f"[Служебно: клиент пишет со {about}: {hint['url']}. "
                                                        f"Вопросы без названия продукта, скорее всего, о нём.]"})
        if not history:
            from ..services import classifier
            kind = classifier.summary_line(dialog.get("classification"))
            if kind:
                content.append({"type": "text", "text": f"[Служебно: классификатор обращений — {kind}. "
                                                        f"Это подсказка, а не факт: ориентируйся на слова клиента.]"})
        if earlier:
            content.append({"type": "text", "text": "[Служебно: сообщения в диалоге, которых ты не видел "
                                                    "(например, ответы менеджера):\n"
                                                    + format_transcript(earlier) + "]"})
        content.append({"type": "text", "text": client_text})

        messages = history + [{"role": "user", "content": content}]
        working, texts, usage, stop = self._run(prompts.system_prompt(), messages, ctx, api_tools())

        text = "\n\n".join(texts).strip()
        if stop in ("refusal", "max_steps") or not text:
            reason = "error" if stop == "max_steps" else "other"
            if ctx.escalation is None:
                ctx.escalation = (reason, f"Ассистент не смог обработать сообщение ({stop}). "
                                          f"Последнее сообщение клиента: {client_text[:500]}")
            text = text or "Уточню этот вопрос у менеджера — он подключится к диалогу."
        for doc in ctx.documents:
            if doc["url"] not in text:
                text += f"\n\n{doc['title']} № {doc['number']}: {doc['url']}"

        if working[-1]["role"] != "assistant":
            # Ход завершился без финального ответа модели — фиксируем, что сказали клиенту, чтобы
            # сохранить чередование ролей в истории
            working.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        dialogs.save_llm_history(dialog_id, working, client_msg_id)
        return AgentReply(text=text, mode="llm", escalation=ctx.escalation, documents=ctx.documents,
                          kb_used=ctx.kb_used, tools_used=ctx.tools_used, usage=usage)

    # ---- подсказка менеджеру ----
    def draft_for_operator(self, dialog_id: int) -> str:
        transcript = format_transcript(dialogs.list_messages(dialog_id)[-40:])
        ctx = ToolContext(dialog_id=dialog_id, read_only=True)
        messages = [{"role": "user", "content": f"{_service_note('admin')}\nПереписка:\n{transcript}\n\n"
                                                f"Подготовь черновик следующего ответа менеджера клиенту."}]
        _, texts, _, _ = self._run(prompts.copilot_prompt(), messages, ctx, api_tools(read_only=True))
        return "\n\n".join(texts).strip()
