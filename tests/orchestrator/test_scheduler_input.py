from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from gatewright.orchestrator.app import (
    OrchestratorTextualApp,
    _agent_text,
    _animated_status,
    _compact,
    _json_block,
    _live_agent_states,
    _message_body_renderables,
    _output_renderable,
    _output_text,
    _status_line_renderable,
    _workflow_renderable,
    _workflow_text,
)
from gatewright.orchestrator.scheduler import (
    WorkflowError,
    WorkflowNeedsDecision,
    WorkflowRunner,
    _format_tool_event,
    _prompt_excerpt,
    approve_all,
)
from gatewright.orchestrator.tui import AgentViewState, InteractionMessage
from gatewright.orchestrator.tui import agent_rows, agent_rows_from_state
from gatewright.orchestrator.workflow import (
    StepPlan,
    load_workflow,
    workflow_from_dict,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def base_spec(workspace: Path, workflow: list[dict]) -> dict:
    return {
        "version": 1,
        "name": "test-flow",
        "runtime": {
            "trace_root": str(workspace / "runs"),
            "default_workspace": str(workspace),
        },
        "agents": {
            "main": {"provider": "mock"},
            "worker": {"provider": "mock"},
        },
        "variables": {},
        "workflow": workflow,
    }


def test_step_records_agent_context(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "hello")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "main-agent"},
                    "prompt_template": "prompts/one.md",
                }
            ],
        )
    )

    events = []
    runner = WorkflowRunner(spec, workspace=tmp_path, event_handler=events.append)
    results = asyncio.run(runner.run())

    assert results[0].agent_id == "main-agent"
    assert results[0].context_id is not None
    assert runner.agent_states["main-agent"].status == "idle"
    assert "[step] one\n> hello" in runner.agent_states["main-agent"].output_tail
    assert runner.agent_states["main-agent"].output_tail.endswith("hello")
    assert any(event["kind"] == "agent_state" for event in events)
    assert any(event["kind"] == "agent_output" for event in events)
    state = json.loads((runner.store.node_dir("one") / "node_state.json").read_text(encoding="utf-8"))
    assert state["context_id"] == results[0].context_id

def test_queued_input_runs_after_agent_stops(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "hello")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "main-agent"},
                    "prompt_template": "prompts/one.md",
                }
            ],
        )
    )

    queued = False
    runner = WorkflowRunner(spec, workspace=tmp_path)

    def on_event(event: dict) -> None:
        nonlocal queued
        if event["kind"] == "agent_output" and not queued:
            queued = True
            runner.queue_input("main-agent", "second turn")

    runner.event_handler = on_event
    results = asyncio.run(runner.run())

    assert "hello" in results[0].output_text
    assert "hello\n\nUser follow-up input" in results[0].output_text
    assert "second turn" in results[0].output_text
    assert runner.agent_states["main-agent"].queued_inputs == 0

def test_input_after_workflow_completion_runs_immediately(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "hello")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "main-agent"},
                    "prompt_template": "prompts/one.md",
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    asyncio.run(runner.run())
    result = asyncio.run(runner.send_input("main-agent", "post workflow question"))

    assert result == "sent"
    assert runner.queued_inputs.get("main-agent") is None
    assert runner.agent_states["main-agent"].status == "idle"
    assert "[input] operator/main-agent\n> post workflow question" in runner.agent_states["main-agent"].output_tail
    assert "post workflow question" in runner.agent_states["main-agent"].output_tail

def test_input_while_agent_active_is_visible_immediately(tmp_path: Path) -> None:
    write(tmp_path / "prompts/wait.md", "wait")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "wait",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "main-agent"},
                    "prompt_template": "prompts/wait.md",
                }
            ],
        )
    )

    async def run_and_send() -> WorkflowRunner:
        runner = WorkflowRunner(spec, workspace=tmp_path)
        task = asyncio.create_task(runner.run())
        while "main-agent" not in runner.active_agents:
            await asyncio.sleep(0.01)
        result = await runner.send_input("main-agent", "visible queued input")
        assert result == "queued"
        assert "[input queued]" not in runner.agent_states["main-agent"].output_tail
        assert runner.agent_states["main-agent"].queued_input_tail == ["visible queued input"]
        await runner.cancel()
        try:
            await task
        except RuntimeError:
            pass
        return runner

    runner = asyncio.run(run_and_send())

    assert runner.agent_states["main-agent"].queued_input_tail == ["visible queued input"]

def test_cancel_queued_input_clears_runner_and_visible_tail(tmp_path: Path) -> None:
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)

    async def seed_and_cancel() -> int:
        await runner._set_agent_state("main-agent", provider="mock", status="running")
        runner.queue_input("main-agent", "queued question")
        return await runner.cancel_queued_input("main-agent")

    cancelled = asyncio.run(seed_and_cancel())

    assert cancelled == 1
    assert runner.queued_inputs.get("main-agent") is None
    assert runner.agent_states["main-agent"].queued_inputs == 0
    assert runner.agent_states["main-agent"].queued_input_tail == []

def test_queued_input_renders_after_running_output_without_mutating_history() -> None:
    console = Console(record=True, width=100)
    state = AgentViewState(
        agent_id="main-agent",
        provider="mock",
        status="running",
        output_tail="partial model output",
        current_node="step-one",
        current_prompt_excerpt="run the current step",
        queued_input_tail=["queued question"],
    )

    console.print(_output_renderable(_output_text(state, None), state, None, animation_frame=2))
    rendered = console.export_text()

    assert rendered.index("partial model output") < rendered.index("[input queued] pending")
    assert "Esc/" in rendered
    assert "queued question" in rendered
    assert "model output.." not in rendered
    assert "[input queued]" not in state.output_tail

    console.print(_status_line_renderable(state, None, step_mode=True, animation_frame=2))
    status = console.export_text()
    assert "mode step" in status
    assert "working.." in status
    assert "run the current step" in status

def test_output_tail_retains_large_recent_history(tmp_path: Path) -> None:
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)

    async def append_large_output() -> str:
        await runner._set_agent_state("main-agent", provider="mock")
        await runner._append_agent_output("main-agent", "a" * 10_000)
        return runner.agent_states["main-agent"].output_tail

    tail = asyncio.run(append_large_output())

    assert len(tail) == 10_000

def test_reused_agent_output_has_turn_boundary_between_steps(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "first")
    write(tmp_path / "prompts/two.md", "second")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "main-agent"},
                    "prompt_template": "prompts/one.md",
                },
                {
                    "id": "two",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "main-agent"},
                    "prompt_template": "prompts/two.md",
                },
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    asyncio.run(runner.run())

    assert (
        "[step] one\n> first\n\nfirst\n\n[step] two\n> second\n\nsecond"
        in runner.agent_states["main-agent"].output_tail
    )

def test_send_input_to_unknown_agent_raises(tmp_path: Path) -> None:
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)

    try:
        asyncio.run(runner.send_input("nope", "hi"))
    except WorkflowError as exc:
        assert "unknown live agent" in str(exc)
    else:
        raise AssertionError("send_input to unknown agent must raise WorkflowError")
