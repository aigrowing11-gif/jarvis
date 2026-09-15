"""Explicit offline demo and live compatibility probe; no credentials in tests."""

import json

from orchestrator.registry import ToolRegistry
from orchestrator.types import (
    ModelReply,
    Provider,
    ProviderError,
    SessionKey,
    ToolCall,
    ToolContext,
)


class DemoProvider:
    """Scripted status demo, NOT a local LLM or a fallback after a live error."""

    last_status = "offline_demo_no_network"

    async def complete(self, messages, tools, *, tool_choice="auto"):
        if messages[-1]["role"] == "user":
            return ModelReply(calls=(ToolCall("demo_status", "assistant_status", "{}"),))
        status = json.loads(messages[-1]["content"])["data"]
        return ModelReply(
            f"Offline demo: Component {status['component']} is running. "
            f"Registered tools: {', '.join(status['registered_tools'])}. "
            "Sensitive actions are blocked. This was a scripted tool-loop check, not an AI reply."
        )


async def check_provider(provider: Provider, registry: ToolRegistry) -> str:
    """Force one harmless tool call, then prove role=tool results are accepted."""
    messages = [
        {"role": "system", "content": "Follow the tool protocol; report only observed results."},
        {"role": "user", "content": "Use get_time to read the current UTC time."},
    ]
    tools = [schema for schema in registry.schemas() if schema["function"]["name"] == "get_time"]
    first = await provider.complete(
        messages, tools, tool_choice={"type": "function", "function": {"name": "get_time"}}
    )
    if len(first.calls) != 1 or first.calls[0].name != "get_time":
        raise ProviderError("Compatibility check failed: the endpoint did not emit get_time.")
    call = first.calls[0]
    result = await registry.execute(
        call, ToolContext(SessionKey("probe", "local", "probe"), call.id)
    )
    if not json.loads(result)["ok"]:
        raise ProviderError("Compatibility check failed: tool arguments were invalid.")
    messages.extend([first.message(), {"role": "tool", "tool_call_id": call.id, "content": result}])
    final = await provider.complete(messages, tools, tool_choice="none")
    if final.calls or not final.content:
        raise ProviderError("Compatibility check failed: no final answer after the tool result.")
    return "PASS: NVIDIA emitted a schema-valid tool call and accepted its result in the next turn."
