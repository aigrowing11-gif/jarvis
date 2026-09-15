"""Component 0 development entrypoint; the background daemon comes in Component 8."""

import argparse
import asyncio
import logging
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.diagnostics import DemoProvider, check_provider
from orchestrator.engine import Orchestrator
from orchestrator.provider import NvidiaProvider
from orchestrator.registry import ToolRegistry
from orchestrator.types import ProviderError, SessionKey, encode
from tools.builtin import register_builtins


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jarvis Component 0 — local text development harness"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Offline scripted tool-loop demo; no key")
    mode.add_argument("--check-provider", action="store_true", help="Live two-step tool-call probe")
    mode.add_argument(
        "--status", action="store_true", help="Inspect this new CLI process, not a daemon"
    )
    parser.add_argument(
        "--message", help="One message, then exit (otherwise interactive text mode)"
    )
    parser.add_argument("--session", default="default", help="Process-local conversation label")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


async def run(args: argparse.Namespace) -> int:
    provider = None
    try:
        if args.demo or args.status:
            provider = DemoProvider()
        else:
            provider = NvidiaProvider(Settings.load(args.env_file))
        registry = ToolRegistry()
        assistant = Orchestrator(provider, registry)
        register_builtins(registry, assistant.status)
        key = SessionKey("local", "owner", args.session)
        if args.status:
            print(encode(assistant.status()))
            return 0
        if args.check_provider:
            print(await check_provider(provider, registry))
            return 0
        if args.demo or args.message is not None:
            reply = await assistant.respond(key, args.message or "Check your status")
            print(reply.text)
            return 0 if reply.ok else 1
        print("Jarvis Component 0 text harness. /quit exits; /reset forgets this session.")
        print("Voice, Discord, desktop controls and background startup are not built yet.")
        while True:
            try:
                text = await asyncio.to_thread(input, "You> ")
            except EOFError:
                break
            if text.strip() == "/quit":
                break
            if text.strip() == "/reset":
                await assistant.sessions.reset(key)
                print("Session cleared.")
                continue
            reply = await assistant.respond(key, text)
            print(f"Jarvis> {reply.text}")
        return 0
    except (ValueError, ProviderError) as exc:
        print(f"Jarvis: {exc}")
        return 1
    finally:
        if isinstance(provider, NvidiaProvider):
            await provider.aclose()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        code = asyncio.run(run(make_parser().parse_args()))
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
