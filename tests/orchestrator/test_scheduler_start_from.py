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


def test_start_from_skips_previous_top_level_nodes(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "first")
    write(tmp_path / "prompts/two.md", "second")
    write(tmp_path / "prompts/three.md", "third")
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
                {
                    "id": "three",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "create", "agent_id": "three-agent"},
                    "prompt_template": "prompts/three.md",
                },
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path, start_from="two")
    results = asyncio.run(runner.run())

    assert [result.node_id for result in results] == ["two", "three"]
    assert "one-agent" not in runner.agent_states

def test_start_from_bare_nested_id_directs_user_to_loop_path_form(tmp_path: Path) -> None:
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "loop",
                    "type": "loop",
                    "body": [
                        {
                            "id": "body",
                            "type": "step",
                            "agent": "main",
                            "prompt_template": "prompts/body.md",
                        }
                    ],
                    "gate": {
                        "id": "gate",
                        "type": "step",
                        "agent": "main",
                        "prompt_template": "prompts/gate.md",
                        "decision": {
                            "parser": "json",
                            "exit_when": {"path": "decision", "equals": "exit"},
                        },
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path, start_from="body")
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        # User passed "body" but the loop body requires "loop/body" form.
        assert "loop_id" in str(exc) or "<loop_id>/<step_id>" in str(exc)
    else:
        raise AssertionError("bare nested id should fail with a hint about the loop/step form")

def test_start_from_loop_step_skips_earlier_body_steps_first_round_only(tmp_path: Path) -> None:
    """`--start-from loop_id/step_id` enters the loop and skips body steps
    before step_id on the FIRST iteration; the gate still runs. Subsequent
    iterations (if any) run the full body."""
    write(tmp_path / "prompts/a.md", "a")
    write(tmp_path / "prompts/b.md", "b")
    write(tmp_path / "prompts/c.md", "c")
    write(tmp_path / "prompts/gate.md", "")  # body fills it
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "loop",
                    "type": "loop",
                    "round_variable": "round",
                    "body": [
                        {
                            "id": "step-a",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/a.md",
                        },
                        {
                            "id": "step-b",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/b.md",
                        },
                        {
                            "id": "step-c",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/c.md",
                        },
                    ],
                    "gate": {
                        "id": "gate",
                        "type": "step",
                        "agent": "main",
                        "context": {"mode": "reuse", "agent_id": "main-agent"},
                        "prompt_template": "prompts/gate.md",
                        "decision": {
                            "parser": "json",
                            "exit_when": {"path": "decision", "equals": "exit"},
                        },
                    },
                }
            ],
        )
    )

    # Mock backend echoes prompt; the gate prompt is empty so we patch it.
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')

    runner = WorkflowRunner(
        spec,
        workspace=tmp_path,
        start_from="loop/step-c",
        variables={"round": "2"},
    )
    results = asyncio.run(runner.run())

    ran_ids = [r.node_id for r in results]
    # Only step-c and gate ran. step-a and step-b were skipped.
    assert "step-c" in ran_ids
    assert "gate" in ran_ids
    assert "step-a" not in ran_ids
    assert "step-b" not in ran_ids
    # current round was honored from --var round=2
    assert any("round-2" in r.node_key for r in results)

def test_start_from_loop_step_unknown_id_raises(tmp_path: Path) -> None:
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "loop",
                    "type": "loop",
                    "body": [
                        {
                            "id": "body",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/body.md",
                        }
                    ],
                    "gate": {
                        "id": "gate",
                        "type": "step",
                        "agent": "main",
                        "context": {"mode": "reuse", "agent_id": "main-agent"},
                        "prompt_template": "prompts/gate.md",
                        "decision": {
                            "parser": "json",
                            "exit_when": {"path": "decision", "equals": "exit"},
                        },
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path, start_from="loop/nonexistent")
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "nonexistent" in str(exc)
    else:
        raise AssertionError("unknown loop step in --start-from must raise")

def test_start_from_loop_step_unknown_loop_raises(tmp_path: Path) -> None:
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path, start_from="not-a-loop/foo")
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "unknown loop node" in str(exc) or "not-a-loop" in str(exc)
    else:
        raise AssertionError("unknown loop id in --start-from must raise")

def test_build_resume_hint_top_level_node(tmp_path: Path) -> None:
    """For a top-level (non-loop) failure, the resume hint should reference
    just the top-level node id, with no '/round-' segment."""
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)
    assert runner._build_resume_hint("initialize-company-context") == "--start-from initialize-company-context"
