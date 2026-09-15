"""Provider-neutral messages and trusted transport context."""

import json
from dataclasses import dataclass
from typing import Any, Protocol

Message = dict[str, Any]


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class SessionKey:
    # Adapters must authenticate these values, never take identity from model arguments.
    source: str
    user_id: str
    conversation_id: str

    def __post_init__(self):
        if any(not x or len(x) > 200 for x in (self.source, self.user_id, self.conversation_id)):
            raise ValueError("Session identity fields must be 1..200 characters")


@dataclass(frozen=True)
class ToolContext:
    session: SessionKey
    call_id: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def message(self) -> Message:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ModelReply:
    content: str | None = None
    calls: tuple[ToolCall, ...] = ()

    def message(self) -> Message:
        message: Message = {"role": "assistant", "content": self.content}
        if self.calls:
            message["tool_calls"] = [call.message() for call in self.calls]
        return message


class ProviderError(Exception):
    """Safe public message; never includes an HTTP body, key, or prompt."""


class RateLimited(ProviderError):
    pass


class Provider(Protocol):
    async def complete(
        self, messages: list[Message], tools: list[Message], *, tool_choice: Any = "auto"
    ) -> ModelReply: ...
