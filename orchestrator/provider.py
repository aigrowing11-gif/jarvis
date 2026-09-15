"""NVIDIA-hosted API adapter; transport retries never replay a tool handler."""

import asyncio
import math
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from orchestrator.config import NVIDIA_BASE_URL, Settings
from orchestrator.types import Message, ModelReply, ProviderError, RateLimited, ToolCall, encode


def parse_reply(payload: Any) -> ModelReply:
    try:
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        message = choice["message"]
        if message["role"] != "assistant":
            raise ValueError
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list) or len(raw_calls) > 8:
            raise ValueError
        calls = []
        ids = set()
        for raw in raw_calls:
            fn = raw["function"]
            call = ToolCall(raw["id"], fn["name"], fn["arguments"])
            if raw["type"] != "function" or not all(
                isinstance(x, str) and x for x in (call.id, call.name, call.arguments)
            ):
                raise ValueError
            if call.id in ids or len(call.id) > 200 or len(call.name) > 64:
                raise ValueError
            if len(call.arguments.encode()) > 16384:
                raise ValueError
            ids.add(call.id)
            calls.append(call)
        # Reject partial completions rather than executing truncated actions.
        if choice["finish_reason"] != ("tool_calls" if calls else "stop"):
            raise ValueError
        if not calls and (not content or not content.strip()):
            raise ValueError
        if content and len(content.encode()) > 32768:
            raise ValueError
        return ModelReply(content, tuple(calls))
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError):
        raise ProviderError(
            "NVIDIA returned an invalid or incomplete response; no new tools ran."
        ) from None


def retry_seconds(value: str | None, attempt: int) -> float:
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0.0, seconds)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                return max(0.0, (date - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return 2**attempt + random.uniform(0, 0.5)


class NvidiaProvider:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        if not settings.api_key:
            raise ValueError("Set NVIDIA_API_KEY in .env before using the live provider")
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=NVIDIA_BASE_URL + "/",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.request_timeout,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )
        self._lock = asyncio.Lock()
        self._next_request = 0.0
        self.last_status = "not_checked"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self, messages: list[Message], tools: list[Message], *, tool_choice: Any = "auto"
    ) -> ModelReply:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": self.settings.max_tokens,
        }
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        if len(encode(body).encode()) > 131072:
            raise ProviderError("Request budget exceeded; start a new session or narrow the task.")
        # Serial API access makes cooldown account-wide for this provider instance.
        async with self._lock:
            for attempt in range(self.settings.max_retries + 1):
                wait = self._next_request - time.monotonic()
                if wait > self.settings.max_retry_wait:
                    raise RateLimited(
                        "NVIDIA is cooling down; try again later. No paid fallback used."
                    )
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next_request = time.monotonic() + self.settings.request_interval
                try:
                    async with self._client.stream(
                        "POST", "chat/completions", json=body
                    ) as response:
                        status = response.status_code
                        retry_after = response.headers.get("Retry-After")
                        if status == 200:
                            chunks = bytearray()
                            async for chunk in response.aiter_bytes():
                                chunks.extend(chunk)
                                if len(chunks) > 262144:
                                    raise ProviderError(
                                        "NVIDIA response exceeded the safe size limit."
                                    )
                            import json

                            try:
                                reply = parse_reply(json.loads(chunks))
                            except (ValueError, RecursionError):
                                raise ProviderError("NVIDIA returned invalid JSON.") from None
                            self.last_status = "ok"
                            return reply
                except httpx.TransportError:
                    status, retry_after = 503, None
                except ProviderError:
                    self.last_status = "invalid_response"
                    raise
                self.last_status = f"http_{status}"
                if status in (401, 403):
                    raise ProviderError(
                        "NVIDIA access denied; check the API key and account entitlement."
                    )
                if status not in (429, 500, 502, 503, 504):
                    raise ProviderError(
                        f"NVIDIA rejected the request (HTTP {status}); check model support."
                    )
                delay = retry_seconds(retry_after, attempt)
                self._next_request = max(self._next_request, time.monotonic() + delay)
                if attempt == self.settings.max_retries or delay > self.settings.max_retry_wait:
                    if status == 429:
                        raise RateLimited(
                            "NVIDIA rate limit reached; try again later. No paid fallback used."
                        )
                    raise ProviderError(
                        "NVIDIA is temporarily unavailable; retry later. No paid fallback used."
                    )
        raise AssertionError("Retry loop must return or raise")
