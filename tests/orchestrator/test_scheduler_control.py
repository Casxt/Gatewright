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


def test_step_confirm_pauses_before_each_step(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "first")
    write(tmp_path / "prompts/two.md", "second")
    confirmations: list[str] = []

    async def confirm(result) -> bool:
        confirmations.append(result.node_key)
        return True

    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "create", "agent_id": "one-agent"},
                    "prompt_template": "prompts/one.md",
                },
                {
                    "id": "two",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "create", "agent_id": "two-agent"},
                    "prompt_template": "prompts/two.md",
                },
            ],
        )
    )

    results = asyncio.run(WorkflowRunner(spec, workspace=tmp_path, step_confirm_handler=confirm).run())

    assert [result.node_id for result in results] == ["one", "two"]
    assert confirmations == ["one", "two"]

def test_step_confirm_can_stop_workflow(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "first")
    write(tmp_path / "prompts/two.md", "second")

    async def confirm(_) -> bool:
        return False

    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "create", "agent_id": "one-agent"},
                    "prompt_template": "prompts/one.md",
                },
                {
                    "id": "two",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "create", "agent_id": "two-agent"},
                    "prompt_template": "prompts/two.md",
                },
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path, step_confirm_handler=confirm)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "workflow stopped before step: one" in str(exc)
    else:
        raise AssertionError("step confirmation denial should stop workflow")
    assert "one-agent" not in runner.agent_states
    assert "two-agent" not in runner.agent_states

def test_cancel_interrupts_active_agent(tmp_path: Path) -> None:
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

    async def run_and_cancel() -> WorkflowRunner:
        runner = WorkflowRunner(spec, workspace=tmp_path)
        task = asyncio.create_task(runner.run())
        while "main-agent" not in runner.active_agents:
            await asyncio.sleep(0.01)
        await runner.cancel()
        try:
            await task
        except RuntimeError:
            pass
        return runner

    runner = asyncio.run(run_and_cancel())

    assert runner.cancel_requested
    assert runner.agent_states["main-agent"].status == "cancelled"

def test_render_prompt_failure_lands_in_agent_message_list(tmp_path: Path) -> None:
    """When _render_prompt fails (e.g. an input_file is missing on disk), the
    error should be appended to the failing step's agent output_tail and the
    agent state should become 'failed' — instead of crashing the TUI surface
    before any state exists."""
    write(tmp_path / "prompts/one.md", "first")
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
                    # This file does not exist; _render_prompt will FileNotFoundError.
                    "input_files": ["missing/does_not_exist.md"],
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except Exception as exc:
        assert isinstance(exc, FileNotFoundError) or "does_not_exist" in str(exc)
    else:
        raise AssertionError("missing input_file should still halt the workflow")

    state = runner.agent_states.get("main-agent")
    assert state is not None, "agent state must be created before the failure"
    assert state.status == "failed", f"expected failed, got {state.status!r}"
    assert "[error]" in state.output_tail
    assert "FileNotFoundError" in state.output_tail or "does_not_exist" in state.output_tail

def test_needs_decision_without_handler_raises(tmp_path: Path) -> None:
    """When the agent yields NEEDS_DECISION and no decision_handler is wired,
    the runner should raise WorkflowNeedsDecision and mark the agent failed."""
    # MockBackend yields NEEDS_DECISION when the prompt contains "tool" or "approval".
    write(tmp_path / "prompts/one.md", "please use a tool")
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

    runner = WorkflowRunner(spec, workspace=tmp_path)  # decision_handler=None
    try:
        asyncio.run(runner.run())
    except WorkflowNeedsDecision:
        pass
    except WorkflowError as exc:
        # Acceptable: WorkflowError wrapping the needs_decision error
        assert "decision" in str(exc).lower() or "needs" in str(exc).lower()
    else:
        raise AssertionError("missing decision_handler must raise WorkflowNeedsDecision")

    state = runner.agent_states.get("main-agent")
    assert state is not None
    assert state.status == "failed"
