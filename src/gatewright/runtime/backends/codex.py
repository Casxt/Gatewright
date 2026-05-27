from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
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
    SandboxMode,
    UserDecision,
)
from ._sdk_utils import event, get_field, to_plain
from .base import ProviderBackend


@dataclass
class _PendingDecision:
    provider_future: Future[dict[str, Any]]


@dataclass
class _CodexRun:
    client: Any
    thread_id: str
    turn_id: str
    event_queue: asyncio.Queue[AgentEvent | object] = field(default_factory=asyncio.Queue)
    streamed_agent_message_item_ids: set[str] = field(default_factory=set)
    pending: dict[str, _PendingDecision] = field(default_factory=dict)


_STREAM_END = object()


class CodexBackend(ProviderBackend):
    provider_name = "codex"

    def __init__(
        self,
        *,
        client_factory: Callable[[object, Callable[..., dict[str, Any]]], Any] | None = None,
        sync_runner: Callable[[Callable[..., Any], tuple[Any, ...]], Any] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._sync_runner = sync_runner
        self._runs: dict[int, _CodexRun] = {}

    async def run(self, agent: object, request: RunRequest) -> AsyncIterator[AgentEvent]:
        run = await self._start_run(agent, request)
        self._runs[id(agent)] = run
        try:
            async for item in self._stream_run(agent, run):
                yield item
        except asyncio.CancelledError:
            await self.interrupt(agent, "cancelled")
            raise
        except Exception as exc:
            yield event(
                agent,
                EventType.FAILED,
                run.turn_id,
                {"reason": "error", "message": str(exc)},
            )
        finally:
            self._runs.pop(id(agent), None)
            await self._call_sync(run.client.close)

    async def resolve_request(self, agent: object, decision: UserDecision) -> None:
        run = self._runs.get(id(agent))
        if run is None:
            raise AgentNotRunningError("agent has no active run")
        pending = run.pending.pop(decision.request_id, None)
        if pending is None:
            raise PendingRequestNotFoundError(decision.request_id)
        if not pending.provider_future.done():
            pending.provider_future.set_result(self._codex_decision(decision))

    async def interrupt(self, agent: object, reason: str | None = None) -> None:
        run = self._runs.get(id(agent))
        if run is None:
            raise AgentNotRunningError("agent has no active run")
        await self._call_sync(run.client.turn_interrupt, run.thread_id, run.turn_id)

    async def fork(self, agent: object, request: ForkAgentRequest) -> object:
        from ..agent import Agent

        client = self._client_factory(agent, self._make_approval_handler(agent, None, None))
        try:
            await self._call_sync(client.start)
            await self._call_sync(client.initialize)
            thread_id = getattr(agent, "context_id", None)
            if thread_id is None:
                raise AgentRuntimeError("cannot fork Codex agent before it has a context_id")
            forked = await self._call_sync(client.thread_fork, thread_id)
            forked_thread_id = self._extract_id(forked, "thread")
        finally:
            await self._call_sync(client.close)

        return Agent(
            AgentConfig(
                provider="codex",
                context_id=forked_thread_id,
                cwd=getattr(agent, "cwd"),
                policy=getattr(agent, "policy"),
            )
        )

    async def _start_run(self, agent: object, request: RunRequest) -> _CodexRun:
        holder: dict[str, _CodexRun] = {}
        client = self._client_factory(
            agent,
            self._make_approval_handler(agent, holder, asyncio.get_running_loop()),
        )
        await self._call_sync(client.start)
        await self._call_sync(client.initialize)

        thread_id = getattr(agent, "context_id", None)
        if thread_id is None:
            started = await self._call_sync(client.thread_start, self._thread_start_params(agent))
            thread_id = self._extract_id(started, "thread")
            setattr(agent, "context_id", thread_id)
        else:
            resumed = await self._call_sync(
                client.thread_resume,
                thread_id,
                self._thread_resume_params(agent),
            )
            thread_id = self._extract_id(resumed, "thread")
            setattr(agent, "context_id", thread_id)

        turn = await self._call_sync(
            client.turn_start,
            thread_id,
            request.prompt,
            self._turn_start_params(agent),
        )
        turn_id = self._extract_id(turn, "turn")
        run = _CodexRun(client=client, thread_id=thread_id, turn_id=turn_id)
        holder["run"] = run
        return run

    async def _stream_run(self, agent: object, run: _CodexRun) -> AsyncIterator[AgentEvent]:
        run.client.acquire_turn_consumer(run.turn_id)
        poller = asyncio.create_task(self._poll_notifications(agent, run))
        try:
            while True:
                item = await run.event_queue.get()
                if item is _STREAM_END:
                    return
                yield item
                if item.type in {EventType.COMPLETED, EventType.FAILED}:
                    return
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
            run.client.release_turn_consumer(run.turn_id)

    async def _poll_notifications(self, agent: object, run: _CodexRun) -> None:
        try:
            while True:
                notification = await self._call_sync(run.client.next_notification)
                events = self._map_notification(agent, run.turn_id, notification)
                for item in events:
                    await run.event_queue.put(item)
                if self._is_terminal_notification(notification, events):
                    return
        except Exception as exc:
            await run.event_queue.put(
                event(
                    agent,
                    EventType.FAILED,
                    run.turn_id,
                    {"reason": "error", "message": str(exc)},
                )
            )
        finally:
            await run.event_queue.put(_STREAM_END)

    def _make_approval_handler(
        self,
        agent: object,
        holder: dict[str, _CodexRun] | None,
        loop: asyncio.AbstractEventLoop | None,
    ) -> Callable[..., dict[str, Any]]:
        def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
            if holder is None or loop is None:
                return {"decision": "accept"}
            run = holder.get("run")
            if run is None:
                return {"decision": "accept"}

            request_id = f"codex-request-{uuid4().hex}"
            provider_future: Future[dict[str, Any]] = Future()
            run.pending[request_id] = _PendingDecision(provider_future=provider_future)
            pending = PendingRequest(
                id=request_id,
                tool_name=self._approval_tool_name(method, params),
                input={"method": method, "params": to_plain(params)},
            )
            pending_event = event(
                agent,
                EventType.NEEDS_DECISION,
                run.turn_id,
                {"request": pending},
            )
            loop.call_soon_threadsafe(run.event_queue.put_nowait, pending_event)
            return provider_future.result()

        return handler

    def _is_terminal_notification(self, notification: Any, events: list[AgentEvent]) -> bool:
        if any(item.type in {EventType.COMPLETED, EventType.FAILED} for item in events):
            return True
        method = get_field(notification, "method", default="")
        return method in {"turn/completed", "turn/complete"} or "error" in method.lower()

    def _map_notification(
        self,
        agent: object,
        run_id: str,
        notification: Any,
    ) -> list[AgentEvent]:
        method = get_field(notification, "method", default="")
        payload = get_field(notification, "payload", "params", default={})
        payload_dict = to_plain(payload)
        if not self._belongs_to_turn(payload, run_id):
            return []

        # Reasoning summary / full reasoning streaming. Codex emits these for
        # OpenAI reasoning models (o*-family). We surface them as TEXT events
        # with a kind=reasoning marker so the scheduler / TUI can render them
        # distinctly from the final answer.
        if method in {"item/reasoning/summaryTextDelta", "item/reasoning/textDelta"}:
            delta = get_field(payload, "delta", default="")
            if delta:
                item_id = get_field(payload, "item_id", "itemId", default=None)
                if item_id:
                    run = self._runs.get(id(agent))
                    if run is not None:
                        run.streamed_agent_message_item_ids.add(str(item_id))
                return [
                    event(
                        agent,
                        EventType.TEXT,
                        run_id,
                        {"text": delta, "delta": True, "kind": "reasoning"},
                    )
                ]

        if method == "item/reasoning/summaryPartAdded":
            # A new reasoning summary section is starting. Emit a separator so
            # successive sections do not run together in the output_tail.
            return [
                event(
                    agent,
                    EventType.TEXT,
                    run_id,
                    {"text": "\n\n", "delta": True, "kind": "reasoning"},
                )
            ]

        if method.endswith("/agentMessage/delta") or method.endswith("/delta"):
            text = get_field(payload, "delta", "text", default="")
            if text:
                item_id = get_field(payload, "item_id", "itemId", default=None)
                if item_id:
                    run = self._runs.get(id(agent))
                    if run is not None:
                        run.streamed_agent_message_item_ids.add(str(item_id))
                return [event(agent, EventType.TEXT, run_id, {"text": text, "delta": True})]

        if method == "item/completed":
            return self._map_completed_item(agent, run_id, payload)

        if method in {"turn/completed", "turn/complete"}:
            turn = get_field(payload, "turn", default={})
            status = self._enum_value(get_field(turn, "status", default="completed"))
            if status in {"completed", "succeeded", "success"}:
                return [event(agent, EventType.COMPLETED, run_id, payload_dict)]
            return [
                event(
                    agent,
                    EventType.FAILED,
                    run_id,
                    {"reason": "cancelled" if status == "cancelled" else "error", "message": status},
                )
            ]

        if "error" in method.lower():
            return [
                event(
                    agent,
                    EventType.FAILED,
                    run_id,
                    {"reason": "error", "message": str(payload_dict)},
                )
            ]

        return []

    def _thread_start_params(self, agent: object) -> dict[str, Any]:
        return {
            "cwd": str(getattr(agent, "cwd")),
            "approvalPolicy": self._approval_policy(agent),
            "sandbox": "workspace-write",
        }

    def _thread_resume_params(self, agent: object) -> dict[str, Any]:
        return self._thread_start_params(agent)

    def _turn_start_params(self, agent: object) -> dict[str, Any]:
        # Enable reasoning summary streaming so the orchestrator surfaces the
        # model's chain of thought in the agent message list. Without
        # `summary` set, codex emits NO `item/reasoning/*` notifications and
        # the [think] block would always be empty. Override via env var if
        # you need to silence reasoning or bump effort.
        params: dict[str, Any] = {
            "cwd": str(getattr(agent, "cwd")),
            "approvalPolicy": self._approval_policy(agent),
            "sandboxPolicy": self._sandbox_policy(agent),
            "summary": os.environ.get("CODEX_REASONING_SUMMARY", "auto"),
        }
        effort = os.environ.get("CODEX_REASONING_EFFORT")
        if effort:
            params["effort"] = effort
        return params

    def _sandbox_policy(self, agent: object) -> dict[str, Any]:
        mode = getattr(agent, "policy").sandbox.mode
        if mode == SandboxMode.GLOBAL_READ_WORKSPACE_WRITE:
            return {
                "type": "workspaceWrite",
                "writableRoots": [str(getattr(agent, "cwd"))],
                "networkAccess": True,
            }
        return {
            "type": "workspaceWrite",
            "writableRoots": [str(getattr(agent, "cwd"))],
            "networkAccess": True,
        }

    def _approval_policy(self, agent: object) -> str:
        return "on-request"

    def _codex_decision(self, decision: UserDecision) -> dict[str, Any]:
        if decision.action == DecisionAction.DENY:
            return {"decision": "deny"}
        if decision.input is not None:
            return {"decision": "accept", "input": to_plain(decision.input)}
        return {"decision": "accept"}

    def _default_client_factory(
        self,
        agent: object,
        approval_handler: Callable[..., dict[str, Any]],
    ) -> Any:
        try:
            from codex_app_server import AppServerConfig, AppServerClient
        except ModuleNotFoundError as exc:
            raise AgentRuntimeError(
                "codex_app_server is required for the Codex backend"
            ) from exc

        codex_bin = os.environ.get("CODEX_BIN") or shutil.which("codex")
        return AppServerClient(
            config=AppServerConfig(
                cwd=str(getattr(agent, "cwd")),
                codex_bin=codex_bin or None,
            ),
            approval_handler=approval_handler,
        )

    async def _call_sync(self, func: Callable[..., Any], *args: Any) -> Any:
        if self._sync_runner is not None:
            result = self._sync_runner(func, args)
            if hasattr(result, "__await__"):
                return await result
            return result
        return await asyncio.to_thread(func, *args)

    def _extract_id(self, value: Any, field: str) -> str:
        current = get_field(value, field, default=value)
        item_id = get_field(current, "id", default=None)
        if item_id is None:
            raise AgentRuntimeError(f"Codex SDK response did not include {field}.id")
        return str(item_id)

    def _approval_tool_name(self, method: str, params: dict[str, Any]) -> str:
        item = get_field(params, "item", default={})
        return str(
            get_field(item, "toolName", "tool_name", "type", default=None)
            or method.rsplit("/", 1)[-1]
        )

    def _tool_phase(self, method: str) -> str:
        if method.endswith("/completed") or method.endswith("/result"):
            return "result"
        return "call"

    def _tool_name(self, method: str, payload: dict[str, Any]) -> str:
        item = payload.get("item", payload)
        return str(item.get("toolName") or item.get("tool_name") or item.get("type") or method)

    def _tool_output(self, payload: dict[str, Any]) -> Any:
        return (
            payload.get("output")
            or payload.get("result")
            or payload.get("content")
            or payload.get("aggregatedOutput")
            or payload.get("aggregated_output")
            or payload.get("exitCode")
            or payload.get("exit_code")
        )

    def _extract_text(self, payload: dict[str, Any]) -> str | None:
        for key in ("text", "delta"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        item = payload.get("item")
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [part.get("text") for part in content if isinstance(part, dict)]
                joined = "".join(text for text in texts if isinstance(text, str))
                return joined or None
        return None

    def _enum_value(self, value: Any) -> str:
        return str(get_field(value, "value", default=value))

    def _belongs_to_turn(self, payload: Any, turn_id: str) -> bool:
        payload_turn_id = get_field(payload, "turn_id", "turnId", default=None)
        if payload_turn_id is not None:
            return str(payload_turn_id) == turn_id
        turn = get_field(payload, "turn", default=None)
        if turn is not None:
            return str(get_field(turn, "id", default="")) == turn_id
        return True

    def _map_completed_item(self, agent: object, run_id: str, payload: Any) -> list[AgentEvent]:
        item = get_field(payload, "item", default=None)
        if item is None:
            return []
        item = get_field(item, "root", default=item)
        item_type = str(get_field(item, "type", default=""))
        item_payload = to_plain(item)

        if item_type == "agentMessage":
            item_id = get_field(item, "id", default=None)
            run = self._runs.get(id(agent))
            if item_id and run is not None and str(item_id) in run.streamed_agent_message_item_ids:
                return []
            text = get_field(item, "text", default=None)
            phase = get_field(item, "phase", default=None)
            if text and phase is not None and str(get_field(phase, "value", default=phase)) == "final_answer":
                return [event(agent, EventType.TEXT, run_id, {"text": text, "delta": False})]
            return []

        if item_type == "reasoning":
            # If the reasoning was streamed via summaryTextDelta / textDelta we
            # already surfaced it; do not duplicate. Otherwise fall back to the
            # aggregated summary / content fields on the completed item.
            item_id = get_field(item, "id", default=None)
            run = self._runs.get(id(agent))
            if item_id and run is not None and str(item_id) in run.streamed_agent_message_item_ids:
                return []
            summary = get_field(item, "summary", default=None) or []
            content = get_field(item, "content", default=None) or []
            parts = list(summary) if summary else list(content)
            text = "\n\n".join(str(part) for part in parts if part)
            if text:
                return [
                    event(
                        agent,
                        EventType.TEXT,
                        run_id,
                        {"text": text, "delta": False, "kind": "reasoning"},
                    )
                ]
            return []

        if item_type in {"commandExecution", "fileChange", "mcpToolCall"}:
            return [
                event(
                    agent,
                    EventType.TOOL,
                    run_id,
                    {
                        "phase": "result",
                        "tool_name": item_type,
                        "input": item_payload,
                        "output": self._tool_output(item_payload),
                    },
                )
            ]
        return []
