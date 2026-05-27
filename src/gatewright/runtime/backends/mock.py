from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import uuid4

from ..models import (
    AgentConfig,
    AgentEvent,
    AgentNotRunningError,
    DecisionAction,
    EventType,
    ForkAgentRequest,
    PendingRequest,
    PendingRequestNotFoundError,
    RunRequest,
    UserDecision,
)
from .base import ProviderBackend


@dataclass
class _RunState:
    run_id: str
    interrupted: bool = False
    pending: dict[str, asyncio.Future[UserDecision]] = field(default_factory=dict)


class MockBackend(ProviderBackend):
    provider_name = "mock"

    def __init__(self) -> None:
        self._runs: dict[int, _RunState] = {}

    async def run(self, agent: object, request: RunRequest) -> AsyncIterator[AgentEvent]:
        if getattr(agent, "context_id") is None:
            setattr(agent, "context_id", f"mock-{uuid4().hex}")

        run_id = f"run-{uuid4().hex}"
        state = _RunState(run_id=run_id)
        self._runs[id(agent)] = state

        yield self._event(agent, run_id, EventType.TEXT, {"text": request.prompt, "delta": False})

        if "tool" in request.prompt or "approval" in request.prompt:
            request_id = f"request-{uuid4().hex}"
            future: asyncio.Future[UserDecision] = asyncio.get_running_loop().create_future()
            state.pending[request_id] = future
            pending = PendingRequest(
                id=request_id,
                tool_name="MockTool",
                input={"prompt": request.prompt},
            )
            yield self._event(agent, run_id, EventType.NEEDS_DECISION, {"request": pending})
            decision = await future
            if decision.action == DecisionAction.DENY:
                yield self._event(
                    agent,
                    run_id,
                    EventType.FAILED,
                    {"reason": "error", "message": "request denied"},
                )
                self._runs.pop(id(agent), None)
                return
            yield self._event(
                agent,
                run_id,
                EventType.TOOL,
                {
                    "phase": "result",
                    "tool_name": "MockTool",
                    "input": decision.input or {},
                    "output": "approved",
                },
            )

        if "wait" in request.prompt:
            while not state.interrupted:
                await asyncio.sleep(0.01)

        if state.interrupted:
            yield self._event(
                agent,
                run_id,
                EventType.FAILED,
                {"reason": "cancelled", "message": "interrupted"},
            )
        else:
            yield self._event(agent, run_id, EventType.COMPLETED, {})

        self._runs.pop(id(agent), None)

    async def resolve_request(self, agent: object, decision: UserDecision) -> None:
        state = self._runs.get(id(agent))
        if state is None:
            raise AgentNotRunningError("agent has no active run")
        future = state.pending.get(decision.request_id)
        if future is None:
            raise PendingRequestNotFoundError(decision.request_id)
        if not future.done():
            future.set_result(decision)

    async def interrupt(self, agent: object, reason: str | None = None) -> None:
        state = self._runs.get(id(agent))
        if state is None:
            raise AgentNotRunningError("agent has no active run")
        state.interrupted = True

    async def fork(self, agent: object, request: ForkAgentRequest) -> object:
        from ..agent import Agent

        return Agent(
            AgentConfig(
                provider="mock",
                context_id=f"mock-{uuid4().hex}",
                cwd=getattr(agent, "cwd"),
                policy=getattr(agent, "policy"),
            ),
            backend=self,
        )

    def _event(
        self,
        agent: object,
        run_id: str,
        type: EventType,
        payload: dict,
    ) -> AgentEvent:
        return AgentEvent.now(
            type=type,
            provider=getattr(agent, "provider"),
            context_id=getattr(agent, "context_id"),
            run_id=run_id,
            payload=payload,
        )
