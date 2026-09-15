"""Bounded, process-local sessions with complete-turn eviction and per-session locks."""

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field

from orchestrator.types import Message, SessionKey, encode


class SessionBusy(Exception):
    pass


@dataclass
class Session:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turns: list[list[Message]] = field(default_factory=list)
    leases: int = 0

    def remember(self, turn: list[Message], max_turns: int, max_bytes: int) -> None:
        self.turns.append(deepcopy(turn))
        # Never truncate individual tool messages or orphan tool_call_id references.
        while self.turns and (
            len(self.turns) > max_turns or len(encode(self.turns).encode()) > max_bytes
        ):
            self.turns.pop(0)


class SessionStore:
    def __init__(self, max_sessions: int = 32, max_pending: int = 4):
        if max_sessions < 1 or max_pending < 1:
            raise ValueError("Session limits must be positive")
        self.max_sessions = max_sessions
        self.max_pending = max_pending
        self._sessions: OrderedDict[SessionKey, Session] = OrderedDict()

    @asynccontextmanager
    async def acquire(self, key: SessionKey):
        # This section has no await: safe within the single application event loop.
        session = self._sessions.get(key)
        if session is None:
            if len(self._sessions) >= self.max_sessions:
                candidate = next((k for k, s in self._sessions.items() if s.leases == 0), None)
                if candidate is None:
                    raise SessionBusy("All sessions are busy; try again shortly.")
                del self._sessions[candidate]
            session = self._sessions[key] = Session()
        self._sessions.move_to_end(key)
        if session.leases >= self.max_pending:
            raise SessionBusy("This session already has queued messages; wait for the reply.")
        session.leases += 1
        try:
            async with session.lock:
                yield session
        finally:
            session.leases -= 1

    async def reset(self, key: SessionKey) -> None:
        async with self.acquire(key) as session:
            session.turns.clear()
