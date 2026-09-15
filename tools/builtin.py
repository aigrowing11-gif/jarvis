"""Read-only diagnostics: no shell, file access, browser, or screen capture."""

from collections.abc import Callable
from datetime import UTC, datetime

from orchestrator.registry import Risk, Tool, ToolRegistry

EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def register_builtins(registry: ToolRegistry, status: Callable[[], dict]) -> None:
    async def get_time(arguments, context):
        return {"utc": datetime.now(UTC).isoformat()}

    async def assistant_status(arguments, context):
        return status()

    registry.register(
        Tool("get_time", "Read the current UTC date and time.", EMPTY_SCHEMA, get_time, Risk.SAFE)
    )
    registry.register(
        Tool(
            "assistant_status",
            "Read this Jarvis process's health and implemented capabilities.",
            EMPTY_SCHEMA,
            assistant_status,
            Risk.SAFE,
        )
    )
