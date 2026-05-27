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


def test_status_line_shows_open_interaction_without_error_details() -> None:
    """Pause/error details live in the agent message list. The status line
    should stay small and only indicate that the input box is attached to an
    open message."""
    state = AgentViewState(agent_id="main-agent", provider="mock", status="failed")
    state.messages.append(
        InteractionMessage(
            id="m1",
            agent_id="main-agent",
            node_key="company-research-loop/round-5/conclude-and-plan",
            kind="pause",
            text="codex network drop",
        )
    )
    line = _status_line_renderable(
        state,
        None,
        step_mode=False,
        animation_frame=0,
    )
    text = line.plain

    assert "open message" in text
    assert "a/abort" in text
    assert "codex network drop" not in text

def test_status_line_does_not_truncate_long_workflow_stop_reason() -> None:
    """The status line widget renders with `height: auto` so it can wrap onto
    multiple lines. The renderer therefore must NOT char-truncate the error
    message — earlier code did `reason[:120]` which silently hid the tail of
    long codex tracebacks even when the widget could have shown them."""
    long_reason = (
        "WorkflowError: codex_app_server.AppServerError: turn failed because "
        "the underlying model returned a contextWindowExceeded after 320k input "
        "tokens were assembled from the prior 11 supplement-collection turns; "
        "this thread is now non-resumable and must be re-forked from the round "
        "kickoff agent"
    )
    line = _status_line_renderable(
        AgentViewState(agent_id="main-agent", provider="codex", status="failed"),
        None,
        step_mode=False,
        animation_frame=0,
        workflow_stop_reason=long_reason,
        workflow_resume_hint="--start-from company-research-loop/build-model -- --var round=7",
    )
    text = line.plain

    # The full reason (every word) must be present — no [:120] clamp.
    assert "contextWindowExceeded" in text
    assert "re-forked from the round" in text
    assert "kickoff agent" in text
    # Resume hint still present too.
    assert "build-model" in text
    assert "round=7" in text

def test_open_interaction_message_keeps_full_long_error(tmp_path: Path) -> None:
    """Long pause/error details should be stored on the message itself, not
    squeezed through a status-line branch."""
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)
    awaitable = runner._set_agent_state(
        "main-agent",
        provider="codex",
        current_node="company-research-loop/round-7/conclude-and-plan",
        status="failed",
    )
    asyncio.run(awaitable)
    long_err = FileNotFoundError(
        "input_files declared but not found on disk: "
        "/very/long/research/path/round-7/build_model_changes.md, "
        "/very/long/research/path/round-7/evidence_matrix.md, "
        "/very/long/research/path/round-7/review_and_challenge.md"
    )
    message = asyncio.run(
        runner._open_step_interaction(
            StepPlan(
                node_id="conclude-and-plan",
                node_key="company-research-loop/round-7/conclude-and-plan",
                agent_id="main-agent",
                provider="codex",
            ),
            long_err,
        )
    )

    assert message.state == "open"
    assert "build_model_changes.md" in message.text
    assert "evidence_matrix.md" in message.text
    assert "review_and_challenge.md" in message.text
    assert runner.agent_states["main-agent"].messages[-1] is message
    assert "[pause] company-research-loop/round-7/conclude-and-plan" in runner.agent_states["main-agent"].output_tail

def test_status_line_includes_resume_hint_after_abort() -> None:
    """After a workflow is stopped, the status line must include the
    --start-from snippet so the user can copy-paste a resume command."""
    state = AgentViewState(agent_id="main-agent", provider="mock", status="failed")
    line = _status_line_renderable(
        state,
        None,
        step_mode=False,
        animation_frame=0,
        workflow_stop_reason="WorkflowError: agent aborted",
        workflow_resume_hint="--start-from company-research-loop/conclude-and-plan -- --var round=5",
    )
    text = line.plain

    assert "workflow stopped" in text
    assert "resume:" in text
    assert "--start-from company-research-loop/conclude-and-plan" in text
    assert "--var round=5" in text

def test_status_line_shows_workflow_stop_reason() -> None:
    """When the app sets _workflow_stop_reason after a failure, the status
    line must show the persistent red indicator pointing the user to the
    failed agent — not just the default mode chip."""
    state = AgentViewState(
        agent_id="main-agent",
        provider="mock",
        status="failed",
        output_tail="[error] one\nFileNotFoundError: missing\n",
    )
    line = _status_line_renderable(
        state,
        None,
        step_mode=False,
        animation_frame=0,
        workflow_stop_reason="FileNotFoundError: prompts/missing.md",
    )
    text = line.plain

    assert "workflow stopped" in text
    assert "FileNotFoundError" in text
    assert "select failed agent for details" in text

def test_status_line_shows_pending_step_confirmation() -> None:
    state = AgentViewState(agent_id="main-agent", provider="mock", status="idle")
    plan = StepPlan(node_id="one", node_key="one", agent_id="main-agent", provider="mock")
    line = _status_line_renderable(state, plan, step_mode=True, animation_frame=0)

    assert "start one" in line.plain
    assert "mode step" in line.plain

def test_workflow_renderable_shows_pending_running_done_markers() -> None:
    from gatewright.orchestrator.workflow import StepResult

    states: dict[str, AgentViewState] = {
        "running-agent": AgentViewState(
            agent_id="running-agent",
            provider="mock",
            status="running",
            current_node="middle",
        ),
    }
    completed = {
        "first": StepResult(node_id="first", node_key="first", status="completed", agent_id="a"),
    }
    nodes = [
        {"id": "first", "type": "step"},
        {"id": "middle", "type": "step"},
        {"id": "last", "type": "step"},
    ]

    console = Console(record=True, width=60, color_system=None)
    console.print(_workflow_renderable("test-flow", nodes, states, completed))
    text = console.export_text()

    assert "test-flow" in text
    assert ". first" in text  # done marker
    assert "> middle" in text  # running marker
    assert "- last" in text  # pending marker
    assert "active" in text

def test_workflow_renderable_shows_needs_decision_marker() -> None:
    states: dict[str, AgentViewState] = {
        "gate-agent": AgentViewState(
            agent_id="gate-agent",
            provider="mock",
            status="needs_decision",
            current_node="gate",
        ),
    }
    nodes = [{"id": "gate", "type": "step"}]

    console = Console(record=True, width=60, color_system=None)
    console.print(_workflow_renderable("test-flow", nodes, states, {}))
    text = console.export_text()

    assert "? gate" in text
