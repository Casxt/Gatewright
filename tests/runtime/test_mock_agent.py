from __future__ import annotations

import asyncio

import pytest

from gatewright.runtime import (
    Agent,
    AgentConfig,
    AgentMode,
    AgentNotRunningError,
    DecisionAction,
    EventType,
    RunRequest,
    SandboxMode,
    UserDecision,
    default_policy,
)


async def _collect(agent: Agent, prompt: str):
    return [event async for event in agent.run(RunRequest(prompt=prompt))]


def test_default_policy_matches_provider_defaults() -> None:
    codex_policy = default_policy("codex")
    claude_policy = default_policy("claude")

    assert codex_policy.mode == AgentMode.AUTO
    assert codex_policy.sandbox.mode == SandboxMode.GLOBAL_READ_WORKSPACE_WRITE
    assert claude_policy.mode == AgentMode.AUTO
    assert claude_policy.sandbox.mode == SandboxMode.WORKSPACE_WRITE


def test_agent_can_run_from_direct_constructor(tmp_path) -> None:
    agent = Agent(provider="mock", cwd=tmp_path)

    events = asyncio.run(_collect(agent, "hello"))

    assert agent.context_id is not None
    assert [event.type for event in events] == [EventType.TEXT, EventType.COMPLETED]
    assert events[0].payload == {"text": "hello", "delta": False}


def test_agent_can_run_from_config(tmp_path) -> None:
    agent = Agent(AgentConfig(provider="mock", cwd=tmp_path))

    events = asyncio.run(_collect(agent, "hello"))

    assert [event.type for event in events] == [EventType.TEXT, EventType.COMPLETED]


def test_pending_request_can_be_approved(tmp_path) -> None:
    async def scenario():
        agent = Agent(provider="mock", cwd=tmp_path)
        events = []

        async for event in agent.run(RunRequest(prompt="tool please")):
            events.append(event)
            if event.type == EventType.NEEDS_DECISION:
                pending = event.payload["request"]
                await agent.resolve_request(
                    UserDecision(
                        request_id=pending.id,
                        action=DecisionAction.APPROVE,
                        input={"approved": True},
                    )
                )

        return events

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [
        EventType.TEXT,
        EventType.NEEDS_DECISION,
        EventType.TOOL,
        EventType.COMPLETED,
    ]
    assert events[2].payload["input"] == {"approved": True}


def test_pending_request_can_be_denied(tmp_path) -> None:
    async def scenario():
        agent = Agent(provider="mock", cwd=tmp_path)
        events = []

        async for event in agent.run(RunRequest(prompt="approval please")):
            events.append(event)
            if event.type == EventType.NEEDS_DECISION:
                pending = event.payload["request"]
                await agent.resolve_request(
                    UserDecision(
                        request_id=pending.id,
                        action=DecisionAction.DENY,
                    )
                )

        return events

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [
        EventType.TEXT,
        EventType.NEEDS_DECISION,
        EventType.FAILED,
    ]
    assert events[-1].payload["reason"] == "error"


def test_interrupt_active_run(tmp_path) -> None:
    async def scenario():
        agent = Agent(provider="mock", cwd=tmp_path)
        events = []
        run_started = asyncio.Event()

        async def consume():
            async for event in agent.run(RunRequest(prompt="wait")):
                events.append(event)
                if event.type == EventType.TEXT:
                    run_started.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(run_started.wait(), timeout=1)
        await agent.interrupt("test")
        await asyncio.wait_for(task, timeout=1)
        return events

    events = asyncio.run(scenario())

    assert [event.type for event in events] == [EventType.TEXT, EventType.FAILED]
    assert events[-1].payload["reason"] == "cancelled"


def test_interrupt_without_active_run_raises(tmp_path) -> None:
    async def scenario():
        agent = Agent(provider="mock", cwd=tmp_path)
        with pytest.raises(AgentNotRunningError):
            await agent.interrupt("nothing to cancel")

    asyncio.run(scenario())


def test_fork_returns_separate_live_agent(tmp_path) -> None:
    async def scenario():
        parent = Agent(provider="mock", cwd=tmp_path)
        child = await parent.fork()
        return parent, child

    parent, child = asyncio.run(scenario())

    assert child is not parent
    assert child.provider == parent.provider
    assert child.context_id != parent.context_id
    assert child.cwd == parent.cwd
    assert child.policy == parent.policy
