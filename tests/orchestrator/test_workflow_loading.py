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


def test_loads_yaml_workflow(tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        """
version: 1
name: yaml-flow
agents:
  main:
    provider: mock
workflow: []
""".lstrip(),
        encoding="utf-8",
    )

    spec = load_workflow(workflow_path)

    assert spec.name == "yaml-flow"
    assert spec.agents["main"].provider == "mock"

def test_every_public_example_loads_with_load_workflow() -> None:
    """All user-facing examples (everything under examples/ except the
    `_testing/` mock fixtures and the `_archive/` historical files) must
    parse cleanly via `load_workflow`. Catches the common breakage modes:
    bad YAML indentation, unknown node types, schema drift after renames,
    and stale path references after the examples directory is reorganized.
    """
    from gatewright.orchestrator.workflow import (
        load_workflow,
    )

    examples_root = (
        Path(__file__).resolve().parents[2] / "examples"
    )
    assert examples_root.is_dir(), examples_root

    user_facing_yamls = [
        p
        for p in sorted(examples_root.rglob("*.yaml"))
        if "/prompts/" not in str(p)
        and "/fixtures/" not in str(p)
        and "/_testing/" not in str(p)
        and "/_archive/" not in str(p)
    ]
    # Don't quietly pass if someone deleted all examples — the catalog is
    # part of the contract.
    assert len(user_facing_yamls) >= 5, user_facing_yamls

    for yaml_path in user_facing_yamls:
        spec = load_workflow(yaml_path)
        assert spec.name, f"{yaml_path}: workflow has empty name"
        assert spec.workflow, f"{yaml_path}: workflow has no nodes"

def test_complex_feature_flow_example_exercises_mvp_features() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    spec = load_workflow(repo_root / "examples/_testing/complex-feature-flow.yaml")
    events: list[dict] = []
    queued = False
    runner = WorkflowRunner(spec, workspace=repo_root, decision_handler=approve_all)

    def on_event(event: dict) -> None:
        nonlocal queued
        events.append(event)
        if event["kind"] == "agent_output" and event["agent_id"] == "main-agent" and not queued:
            queued = True
            runner.queue_input("main-agent", "queued operator note")

    runner.event_handler = on_event
    results = asyncio.run(runner.run())

    node_keys = [result.node_key for result in results]
    assert "kickoff" in node_keys
    assert "two-round-loop/round-0/round0-only" in node_keys
    assert "two-round-loop/round-0/fanout/collect-alpha" in node_keys
    assert "two-round-loop/round-0/fanout/collect-beta-approval" in node_keys
    assert "two-round-loop/round-1/fanout/collect-alpha" in node_keys
    assert "two-round-loop/round-1/fanout/collect-beta-approval" in node_keys
    assert "two-round-loop/round-1/round1-review" in node_keys
    assert "final-summary" in node_keys
    assert "two-round-loop/round-1/round0-only" not in node_keys

    assert runner.variables["round"] == 1
    assert any(result.node_id == "gate" and '"continue"' in result.output_text for result in results)
    assert any(result.node_id == "gate" and '"exit"' in result.output_text for result in results)
    assert any(event.get("event_type") == "needs_decision" for event in events)
    assert "queued operator note" in results[0].output_text
    assert runner.agent_states["main-agent"].queued_inputs == 0

    main_context = runner.agent_states["main-agent"].context_id
    assert runner.agent_states["collect-alpha-round-0"].context_id != main_context
    assert runner.agent_states["collect-beta-round-0"].context_id != main_context
    assert runner.agent_states["reviewer-round-1"].context_id != main_context
