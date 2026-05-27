from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..models import (
    AgentConfig,
    AgentEvent,
    AgentMode,
    AgentNotRunningError,
    AgentRuntimeError,
    DecisionAction,
    EventType,
    ForkAgentRequest,
    PendingRequest,
    PendingRequestNotFoundError,
    RunRequest,
    UserDecision,
)
from ._sdk_utils import event, get_field, to_plain
from .base import ProviderBackend


@dataclass
class _ClaudePendingDecision:
    future: asyncio.Future[UserDecision]


@dataclass
class _ClaudeState:
    client: Any | None = None
    active: bool = False
    resume_context_id: str | None = None
    fork_from_context_id: str | None = None
    fork_session: bool = False
    agent: object | None = None
    run_id: str | None = None
    event_queue: asyncio.Queue[AgentEvent | object] | None = None
    partial_text_seen: bool = False
    pending: dict[str, _ClaudePendingDecision] = field(default_factory=dict)


_STREAM_END = object()


class ClaudeBackend(ProviderBackend):
    provider_name = "claude"

    def __init__(self, *, client_factory: Callable[[object, _ClaudeState], Any] | None = None) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._states: dict[int, _ClaudeState] = {}

    async def run(self, agent: object, request: RunRequest) -> AsyncIterator[AgentEvent]:
        state = self._states.setdefault(
            id(agent),
            _ClaudeState(resume_context_id=getattr(agent, "context_id", None)),
        )
        if state.active:
            raise AgentRuntimeError("agent already has an active run")
        state.active = True
        state.agent = agent
        state.event_queue = asyncio.Queue()
        state.partial_text_seen = False
        state.pending = {}
        run_id = f"claude-run-{uuid4().hex}"
        state.run_id = run_id

        try:
            client = await self._ensure_client(agent, state)
            await client.query(request.prompt)
            async for item in self._stream_response(agent, state, run_id):
                yield item
        except asyncio.CancelledError:
            await self.interrupt(agent, "cancelled")
            raise
        except Exception as exc:
            yield event(
                agent,
                EventType.FAILED,
                run_id,
                {"reason": "error", "message": str(exc)},
            )
        finally:
            if state.client is not None:
                await self._disconnect_client(state.client)
                state.client = None
            state.active = False
            state.agent = None
            state.run_id = None
            state.event_queue = None

    async def resolve_request(self, agent: object, decision: UserDecision) -> None:
        state = self._states.get(id(agent))
        if state is None or not state.active:
            raise AgentNotRunningError("agent has no active run")
        pending = state.pending.pop(decision.request_id, None)
        if pending is None:
            raise PendingRequestNotFoundError(decision.request_id)
        if not pending.future.done():
            pending.future.set_result(decision)

    async def interrupt(self, agent: object, reason: str | None = None) -> None:
        state = self._states.get(id(agent))
        if state is None or state.client is None or not state.active:
            raise AgentNotRunningError("agent has no active run")
        await self._call_if_exists(state.client, "interrupt", "cancel", "abort")

    async def fork(self, agent: object, request: ForkAgentRequest) -> object:
        from ..agent import Agent

        child = Agent(
            AgentConfig(
                provider="claude",
                context_id=None,
                cwd=getattr(agent, "cwd"),
                policy=getattr(agent, "policy"),
            ),
            backend=self,
        )
        self._states[id(child)] = _ClaudeState(
            fork_from_context_id=getattr(agent, "context_id", None),
            fork_session=True,
        )
        return child

    async def _ensure_client(self, agent: object, state: _ClaudeState) -> Any:
        if state.client is None:
            state.client = self._client_factory(agent, state)
            connect = getattr(state.client, "connect", None)
            if connect is not None:
                result = connect()
                if hasattr(result, "__await__"):
                    await result
        return state.client

    async def _disconnect_client(self, client: object) -> None:
        disconnect = getattr(client, "disconnect", None)
        if disconnect is None:
            return
        result = disconnect()
        if hasattr(result, "__await__"):
            await result

    async def _stream_response(
        self,
        agent: object,
        state: _ClaudeState,
        run_id: str,
    ) -> AsyncIterator[AgentEvent]:
        if state.event_queue is None:
            raise AgentRuntimeError("Claude event queue is not initialized")

        poller = asyncio.create_task(self._poll_messages(agent, state, run_id))
        try:
            while True:
                item = await state.event_queue.get()
                if item is _STREAM_END:
                    return
                yield item
                if item.type in {EventType.COMPLETED, EventType.FAILED}:
                    return
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller

    async def _poll_messages(self, agent: object, state: _ClaudeState, run_id: str) -> None:
        assert state.event_queue is not None
        terminal = False
        try:
            async for message in state.client.receive_response():
                events = self._map_message(agent, state, run_id, message)
                for item in events:
                    await state.event_queue.put(item)
                    if item.type in {EventType.COMPLETED, EventType.FAILED}:
                        terminal = True
                if terminal:
                    return
            await state.event_queue.put(event(agent, EventType.COMPLETED, run_id, {}))
        except Exception as exc:
            await state.event_queue.put(
                event(
                    agent,
                    EventType.FAILED,
                    run_id,
                    {"reason": "error", "message": str(exc)},
                )
            )
        finally:
            await state.event_queue.put(_STREAM_END)

    def _map_message(
        self,
        agent: object,
        state: _ClaudeState,
        run_id: str,
        message: Any,
    ) -> list[AgentEvent]:
        message_type = type(message).__name__
        payload = to_plain(message)

        session_id = get_field(message, "session_id", "sessionId", default=None)
        if session_id:
            setattr(agent, "context_id", str(session_id))

        if message_type == "StreamEvent" or "event" in payload:
            mapped = self._map_stream_event(agent, run_id, payload)
            if mapped:
                state.partial_text_seen = True
            return mapped

        if message_type == "ResultMessage" or "ResultMessage" in message_type:
            if self._is_error_result(message, payload):
                return [
                    event(
                        agent,
                        EventType.FAILED,
                        run_id,
                        {"reason": "error", "message": str(payload)},
                    )
                ]
            return [event(agent, EventType.COMPLETED, run_id, payload)]

        content = get_field(message, "content", default=None)
        if content is None and isinstance(payload, dict):
            content = payload.get("content")

        if isinstance(content, str):
            if state.partial_text_seen:
                return []
            return [event(agent, EventType.TEXT, run_id, {"text": content, "delta": False})]
        if isinstance(content, list):
            mapped: list[AgentEvent] = []
            for block in content:
                mapped.extend(
                    self._map_content_block(
                        agent,
                        run_id,
                        block,
                        skip_text=state.partial_text_seen,
                    )
                )
            return mapped

        text = self._extract_stream_text(payload)
        if text:
            state.partial_text_seen = True
            return [event(agent, EventType.TEXT, run_id, {"text": text, "delta": True})]

        return []

    def _map_content_block(
        self,
        agent: object,
        run_id: str,
        block: Any,
        *,
        skip_text: bool = False,
    ) -> list[AgentEvent]:
        block_type = type(block).__name__
        payload = to_plain(block)
        kind = str(get_field(block, "type", default=payload.get("type", "")))

        if block_type == "TextBlock" or kind == "text":
            if skip_text:
                return []
            text = get_field(block, "text", default=payload.get("text"))
            if text:
                return [event(agent, EventType.TEXT, run_id, {"text": text, "delta": False})]

        if block_type == "ToolUseBlock" or kind == "tool_use":
            return [
                event(
                    agent,
                    EventType.TOOL,
                    run_id,
                    {
                        "phase": "call",
                        "tool_name": str(payload.get("name") or payload.get("tool_name") or "tool"),
                        "input": payload.get("input", payload),
                        "output": None,
                    },
                )
            ]

        if block_type == "ToolResultBlock" or kind == "tool_result":
            return [
                event(
                    agent,
                    EventType.TOOL,
                    run_id,
                    {
                        "phase": "result",
                        "tool_name": str(payload.get("tool_use_id") or "tool"),
                        "input": payload,
                        "output": payload.get("content"),
                    },
                )
            ]

        return []

    def _make_can_use_tool(self, state: _ClaudeState) -> Callable[..., Any]:
        async def can_use_tool(tool_name: str, input_data: dict[str, Any], context: Any) -> Any:
            request_id = f"claude-request-{uuid4().hex}"
            future: asyncio.Future[UserDecision] = asyncio.get_running_loop().create_future()
            state.pending[request_id] = _ClaudePendingDecision(future=future)
            pending = PendingRequest(
                id=request_id,
                tool_name=tool_name,
                input={"input": to_plain(input_data), "context": to_plain(context)},
            )
            if state.agent is None or state.run_id is None or state.event_queue is None:
                raise AgentRuntimeError("Claude permission request arrived outside an active run")
            await state.event_queue.put(
                event(
                    state.agent,
                    EventType.NEEDS_DECISION,
                    state.run_id,
                    {"request": pending},
                )
            )
            decision = await future
            return self._claude_permission_result(decision)

        return can_use_tool

    def _default_client_factory(self, agent: object, state: _ClaudeState) -> Any:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        except ModuleNotFoundError as exc:
            raise AgentRuntimeError(
                "claude_agent_sdk is required for the Claude backend"
            ) from exc

        options_kwargs: dict[str, Any] = {
            "cwd": str(getattr(agent, "cwd")),
            "permission_mode": self._permission_mode(agent),
            "can_use_tool": self._make_can_use_tool(state),
            "include_partial_messages": True,
        }
        if getattr(agent, "policy").tools.allowed_tools:
            options_kwargs["allowed_tools"] = list(getattr(agent, "policy").tools.allowed_tools)
        if getattr(agent, "policy").tools.denied_tools:
            options_kwargs["disallowed_tools"] = list(getattr(agent, "policy").tools.denied_tools)
        if state.fork_from_context_id:
            options_kwargs["resume"] = state.fork_from_context_id
            options_kwargs["fork_session"] = True
        elif state.resume_context_id:
            options_kwargs["resume"] = state.resume_context_id

        return ClaudeSDKClient(options=ClaudeAgentOptions(**options_kwargs))

    def _permission_mode(self, agent: object) -> str:
        if getattr(agent, "policy").mode == AgentMode.AUTO:
            return "acceptEdits"
        return "default"

    def _claude_permission_result(self, decision: UserDecision) -> Any:
        try:
            from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny
        except ModuleNotFoundError:
            if decision.action == DecisionAction.DENY:
                return {"behavior": "deny"}
            return {"behavior": "allow", "updated_input": decision.input}

        if decision.action == DecisionAction.DENY:
            return PermissionResultDeny()
        kwargs = {}
        if decision.input is not None:
            kwargs["updated_input"] = decision.input
        return PermissionResultAllow(**kwargs)

    async def _call_if_exists(self, target: object, *names: str) -> None:
        for name in names:
            method = getattr(target, name, None)
            if method is not None:
                result = method()
                if hasattr(result, "__await__"):
                    await result
                return
        raise AgentRuntimeError("Claude SDK client does not expose interrupt/cancel/abort")

    def _is_error_result(self, message: Any, payload: dict[str, Any]) -> bool:
        if get_field(message, "is_error", "error", default=False):
            return True
        subtype = str(get_field(message, "subtype", default=payload.get("subtype", "")))
        return subtype not in {"", "success", "completed"}

    def _extract_stream_text(self, payload: dict[str, Any]) -> str | None:
        delta = payload.get("delta")
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str):
                return text
        text = payload.get("text")
        if isinstance(text, str):
            return text
        return None

    def _map_stream_event(
        self,
        agent: object,
        run_id: str,
        payload: dict[str, Any],
    ) -> list[AgentEvent]:
        raw_event = payload.get("event")
        if not isinstance(raw_event, dict):
            return []

        event_type = raw_event.get("type")
        if event_type == "content_block_delta":
            delta = raw_event.get("delta")
            if isinstance(delta, dict):
                text = delta.get("text")
                if isinstance(text, str) and text:
                    return [event(agent, EventType.TEXT, run_id, {"text": text, "delta": True})]

        if event_type == "content_block_start":
            block = raw_event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    return [event(agent, EventType.TEXT, run_id, {"text": text, "delta": True})]

        return []
