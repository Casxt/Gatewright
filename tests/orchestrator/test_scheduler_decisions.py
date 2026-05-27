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


def test_gate_output_without_matching_rule_raises(tmp_path: Path) -> None:
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", '{"decision":"hold"}')  # neither continue nor exit
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
                            "continue_when": {"path": "decision", "equals": "continue"},
                            "exit_when": {"path": "decision", "equals": "exit"},
                        },
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "did not match" in str(exc) or "did not contain" in str(exc)
    else:
        raise AssertionError("gate without matching decision must raise")

def test_gate_output_without_json_raises(tmp_path: Path) -> None:
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", "just prose no json here")
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

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "did not contain a JSON object" in str(exc)
    else:
        raise AssertionError("gate without JSON must raise WorkflowError")

def test_decision_json_renders_as_decision_block() -> None:
    console = Console(record=True, width=80)
    block = _json_block('{"decision":"exit","reason":"done"}')

    assert block is not None
    console.print(block)
    rendered = console.export_text()
    assert "decision EXIT" in rendered
    assert "done" in rendered
    assert '{"decision"' not in rendered
