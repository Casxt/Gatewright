from __future__ import annotations

import asyncio
from types import SimpleNamespace

from gatewright.runtime import (
    Agent,
    AgentRuntimeError,
    DecisionAction,
    EventType,
    RunRequest,
    UserDecision,
)
from gatewright.runtime.backends.claude import ClaudeBackend
from gatewright.runtime.backends.codex import CodexBackend


class FakeCodexClient:
    def __init__(self, approval_handler):
        self.approval_handler = approval_handler
        self.notification_count = 0
        self.interrupted = False
        self.closed = False

    def start(self):
        return None

    def initialize(self):
        return None

    def close(self):
        self.closed = True

    def thread_start(self, params):
        return SimpleNamespace(thread=SimpleNamespace(id="thread-1"))

    def thread_resume(self, thread_id, params):
        return SimpleNamespace(thread=SimpleNamespace(id=thread_id))

    def thread_fork(self, thread_id):
        return SimpleNamespace(thread=SimpleNamespace(id=f"{thread_id}-fork"))

    def turn_start(self, thread_id, prompt, params):
        return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))

    def turn_interrupt(self, thread_id, turn_id):
        self.interrupted = True

    def acquire_turn_consumer(self, turn_id):
        self.active_turn = turn_id

    def release_turn_consumer(self, turn_id):
        self.active_turn = None

    def next_notification(self):
        self.notification_count += 1
        if self.notification_count == 1:
            return SimpleNamespace(
                method="item/agentMessage/delta",
                payload={"delta": "hello", "turnId": "turn-1"},
            )
        if self.notification_count == 2:
            return SimpleNamespace(
                method="item/completed",
                payload={
                    "turnId": "turn-1",
                    "item": {
                        "type": "commandExecution",
                        "exitCode": 0,
                        "aggregatedOutput": "done",
                    },
                },
            )
        return SimpleNamespace(
            method="turn/completed",
            payload={"turn": {"id": "turn-1", "status": "completed"}},
        )


def _direct_runner(func, args):
    return func(*args)


def test_codex_backend_streams_notifications(tmp_path) -> None:
    def factory(agent, approval_handler):
        return FakeCodexClient(approval_handler)

    async def scenario():
        backend = CodexBackend(client_factory=factory, sync_runner=_direct_runner)
        agent = Agent(provider="codex", cwd=tmp_path, backend=backend)
        events = []

        async for event in agent.run(RunRequest(prompt="run command")):
            events.append(event)

        return agent, events

    agent, events = asyncio.run(scenario())

    assert agent.context_id == "thread-1"
    assert [event.type for event in events] == [
        EventType.TEXT,
        EventType.TOOL,
        EventType.COMPLETED,
    ]
    assert events[0].payload == {"text": "hello", "delta": True}
    assert events[1].payload["output"] == "done"


def test_codex_backend_turn_start_params_enable_reasoning_summary(tmp_path, monkeypatch) -> None:
    """Codex must be configured with `summary: "auto"` so the model emits
    `item/reasoning/*` notifications — otherwise the orchestrator's [think]
    block is always empty regardless of UI rendering. Locks in the default
    AND the env-var override paths."""
    monkeypatch.delenv("CODEX_REASONING_SUMMARY", raising=False)
    monkeypatch.delenv("CODEX_REASONING_EFFORT", raising=False)

    backend = CodexBackend()
    fake_agent = SimpleNamespace(
        cwd=tmp_path,
        policy=SimpleNamespace(sandbox=SimpleNamespace(mode="global_read_workspace_write")),
    )
    params = backend._turn_start_params(fake_agent)

    assert params["summary"] == "auto", "default reasoning summary must be 'auto'"
    assert "effort" not in params, "effort should default to codex's own default"

    # Env-var override paths
    monkeypatch.setenv("CODEX_REASONING_SUMMARY", "detailed")
    monkeypatch.setenv("CODEX_REASONING_EFFORT", "high")
    params = backend._turn_start_params(fake_agent)
    assert params["summary"] == "detailed"
    assert params["effort"] == "high"

    # Setting CODEX_REASONING_SUMMARY=none disables reasoning streaming.
    monkeypatch.setenv("CODEX_REASONING_SUMMARY", "none")
    params = backend._turn_start_params(fake_agent)
    assert params["summary"] == "none"


def test_codex_backend_forks_thread(tmp_path) -> None:
    def factory(agent, approval_handler):
        return FakeCodexClient(approval_handler)

    async def scenario():
        backend = CodexBackend(client_factory=factory, sync_runner=_direct_runner)
        parent = Agent(provider="codex", context_id="thread-1", cwd=tmp_path, backend=backend)
        child = await parent.fork()
        return child

    child = asyncio.run(scenario())

    assert child.provider == "codex"
    assert child.context_id == "thread-1-fork"


def test_codex_backend_requires_existing_context_to_fork(tmp_path) -> None:
    def factory(agent, approval_handler):
        return FakeCodexClient(approval_handler)

    async def scenario():
        backend = CodexBackend(client_factory=factory, sync_runner=_direct_runner)
        parent = Agent(provider="codex", cwd=tmp_path, backend=backend)
        await parent.fork()

    try:
        asyncio.run(scenario())
    except AgentRuntimeError as exc:
        assert "context_id" in str(exc)
    else:
        raise AssertionError("expected AgentRuntimeError")


class FakeClaudeClient:
    def __init__(self, can_use_tool):
        self.can_use_tool = can_use_tool
        self.interrupted = False
        self.disconnected = False
        self.prompt = None

    async def connect(self):
        return None

    async def query(self, prompt):
        self.prompt = prompt

    async def receive_response(self):
        yield SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])
        decision = await self.can_use_tool("Bash", {"command": "echo hi"}, {"cwd": "/tmp"})
        yield SimpleNamespace(content=[SimpleNamespace(type="tool_result", content=decision)])
        yield SimpleNamespace(session_id="claude-session-1", subtype="success")

    async def interrupt(self):
        self.interrupted = True

    async def disconnect(self):
        self.disconnected = True


def test_claude_backend_streams_and_resolves_permission(tmp_path) -> None:
    backend = ClaudeBackend()
    clients = []

    def factory(agent, state):
        client = FakeClaudeClient(backend._make_can_use_tool(state))
        clients.append(client)
        return client

    backend._client_factory = factory

    async def scenario():
        agent = Agent(provider="claude", cwd=tmp_path, backend=backend)
        events = []

        async for event in agent.run(RunRequest(prompt="run command")):
            events.append(event)
            if event.type == EventType.NEEDS_DECISION:
                pending = event.payload["request"]
                await agent.resolve_request(
                    UserDecision(
                        request_id=pending.id,
                        action=DecisionAction.APPROVE,
                        input={"command": "echo hi"},
                    )
                )

        return agent, events

    agent, events = asyncio.run(scenario())

    assert agent.context_id == "claude-session-1"
    assert [event.type for event in events] == [
        EventType.TEXT,
        EventType.NEEDS_DECISION,
        EventType.TOOL,
        EventType.COMPLETED,
    ]
    assert clients[0].disconnected is True


def test_claude_backend_streams_partial_text_and_filters_final_duplicate(tmp_path) -> None:
    backend = ClaudeBackend()

    class PartialClaudeClient:
        async def connect(self):
            return None

        async def query(self, prompt):
            return None

        async def disconnect(self):
            return None

        async def receive_response(self):
            yield SimpleNamespace(
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "hel"},
                }
            )
            yield SimpleNamespace(
                event={
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "lo"},
                }
            )
            yield SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])
            yield SimpleNamespace(session_id="claude-session-1", subtype="success")

    def factory(agent, state):
        return PartialClaudeClient()

    backend._client_factory = factory

    async def scenario():
        agent = Agent(provider="claude", cwd=tmp_path, backend=backend)
        return [event async for event in agent.run(RunRequest(prompt="say hello"))]

    events = asyncio.run(scenario())

    assert [(event.type, event.payload) for event in events] == [
        (EventType.TEXT, {"text": "hel", "delta": True}),
        (EventType.TEXT, {"text": "lo", "delta": True}),
        (EventType.COMPLETED, {}),
    ]


def test_claude_backend_fork_starts_new_session_from_parent_context(tmp_path) -> None:
    seen_states = []
    backend = ClaudeBackend()

    class ForkedClaudeClient:
        async def connect(self):
            return None

        async def query(self, prompt):
            return None

        async def receive_response(self):
            yield SimpleNamespace(session_id="claude-child-session", subtype="success")

    def factory(agent, state):
        seen_states.append(state)
        return ForkedClaudeClient()

    backend._client_factory = factory

    async def scenario():
        parent = Agent(
            provider="claude",
            context_id="claude-parent-session",
            cwd=tmp_path,
            backend=backend,
        )
        child = await parent.fork()
        assert child.context_id is None
        events = [event async for event in child.run(RunRequest(prompt="continue"))]
        return child, events

    child, events = asyncio.run(scenario())

    assert child.context_id == "claude-child-session"
    assert seen_states[0].fork_from_context_id == "claude-parent-session"
    assert seen_states[0].fork_session is True
    assert [event.type for event in events] == [EventType.COMPLETED]
