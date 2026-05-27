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


def test_textual_app_helpers_render_agent_list_and_output(tmp_path: Path) -> None:
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

    workflow_text = _workflow_text(spec.name, runner.agent_states)
    agent_text = _agent_text(runner.agent_states, "main-agent")
    output_text = _output_text(runner.agent_states["main-agent"], None)

    assert issubclass(OrchestratorTextualApp, object)
    assert "test-flow" in workflow_text
    assert "> . main-agent" in agent_text
    assert ". main-agent" in agent_text
    assert "hello" in output_text
    assert "[step] one\n> hello" in output_text

def test_agent_list_filters_to_live_agents_and_switches_output(tmp_path: Path) -> None:
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
                    "agent": "worker",
                    "context": {"mode": "create", "agent_id": "worker-agent"},
                    "prompt_template": "prompts/two.md",
                },
            ],
        )
    )
    runner = WorkflowRunner(spec, workspace=tmp_path)
    asyncio.run(runner.run())
    runner.agent_states["stale-agent"] = AgentViewState(
        agent_id="stale-agent",
        provider="mock",
        status="idle",
        output_tail="stale",
    )

    live_states = _live_agent_states(runner.agent_states, runner.agent_registry.keys())
    agent_text = _agent_text(live_states, "worker-agent")
    selected_output = _output_text(live_states["worker-agent"], None)

    assert set(live_states) == {"main-agent", "worker-agent"}
    assert "stale-agent" not in agent_text
    assert "> . worker-agent" in agent_text
    assert "\n  mock" not in agent_text
    assert "second" in selected_output
    assert "first" not in selected_output

def test_help_screen_lists_every_documented_shortcut() -> None:
    """The help modal is the user's discovery surface for keybindings — if
    something is added to BINDINGS without showing up here, users won't know
    it exists. Conversely, anything documented here that isn't actually
    bound will mislead users. Pin both directions."""
    from gatewright.orchestrator.app import (
        _help_renderable,
    )

    console = Console(record=True, width=120, color_system=None)
    console.print(_help_renderable())
    text = console.export_text()

    # The title and every section heading are present.
    assert "keyboard shortcuts" in text
    for heading in [
        "Navigation",
        "Visibility",
        "Workflow control",
        "Operator input",
        "step pauses",
        "step-confirm",
        "Help",
    ]:
        assert heading in text, f"help screen missing section: {heading}"

    # Each documented binding is present, including context-sensitive ones.
    for key in [
        "Tab",
        "Shift+Tab",
        "Ctrl+R",
        "Ctrl+T",
        "Ctrl+Y",
        "Ctrl+C",
        "q",
        "Enter",
        "Esc /",
        "F1",
        "Esc",
    ]:
        assert key in text, f"help screen missing binding: {key}"

    # Pause/error interaction is message-driven: normal input continues the
    # same agent, while a/abort stops the workflow.
    assert "same agent thread" in text
    assert "a / abort" in text
    assert "--start-from" in text

def test_help_screen_bindings_use_priority_so_input_focus_does_not_swallow_them() -> None:
    """F1 and `?` must be priority bindings — without that, an Input widget
    with focus (the common case) swallows the keypress and the help screen
    never opens."""
    from gatewright.orchestrator.app import (
        OrchestratorTextualApp,
    )

    help_bindings = [
        b for b in OrchestratorTextualApp.BINDINGS if b.action == "show_help"
    ]
    assert len(help_bindings) >= 2, "expected at least F1 and ? bindings"
    for b in help_bindings:
        assert b.priority is True, (
            f"help binding {b.key!r} must be priority=True so Input focus "
            "does not swallow it"
        )

def test_queued_input_cancel_bindings_use_priority() -> None:
    """Esc / Up cancel queued input even while the operator input has focus."""
    from gatewright.orchestrator.app import (
        OrchestratorTextualApp,
    )

    bindings = [
        b for b in OrchestratorTextualApp.BINDINGS if b.action == "cancel_queued_input"
    ]
    keys = {b.key for b in bindings}
    assert {"escape", "up"} <= keys
    assert all(b.priority is True for b in bindings)
