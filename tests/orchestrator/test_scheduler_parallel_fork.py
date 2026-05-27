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


def test_fork_from_step_creates_new_context(tmp_path: Path) -> None:
    write(tmp_path / "prompts/parent.md", "parent")
    write(tmp_path / "prompts/child.md", "child")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "parent",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "parent-agent"},
                    "prompt_template": "prompts/parent.md",
                },
                {
                    "id": "child",
                    "type": "step",
                    "agent": "worker",
                    "context": {
                        "mode": "fork_from_step",
                        "from_step": "parent",
                        "agent_id": "child-agent",
                    },
                    "prompt_template": "prompts/child.md",
                },
            ],
        )
    )

    results = asyncio.run(WorkflowRunner(spec, workspace=tmp_path).run())

    assert results[0].agent_id == "parent-agent"
    assert results[1].agent_id == "child-agent"
    assert results[0].context_id != results[1].context_id

def test_parallel_runs_children_and_agent_rows(tmp_path: Path) -> None:
    write(tmp_path / "prompts/a.md", "a")
    write(tmp_path / "prompts/b.md", "b")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "parallel",
                    "type": "parallel",
                    "max_concurrency": 2,
                    "children": [
                        {
                            "id": "a",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "create", "agent_id": "a-agent"},
                            "prompt_template": "prompts/a.md",
                        },
                        {
                            "id": "b",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "create", "agent_id": "b-agent"},
                            "prompt_template": "prompts/b.md",
                        },
                    ],
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())
    rows = agent_rows(results)
    live_rows = agent_rows_from_state(runner.agent_states)

    assert {result.node_id for result in results} == {"a", "b"}
    assert {row.agent_id for row in rows} == {"a-agent", "b-agent"}
    assert {row.agent_id for row in live_rows} == {"a-agent", "b-agent"}
    assert all(row.status == "idle" for row in live_rows)

def test_parallel_collect_failures_does_not_abort_siblings(tmp_path: Path) -> None:
    write(tmp_path / "prompts/ok.md", "ok")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "fanout",
                    "type": "parallel",
                    "max_concurrency": 2,
                    "failure_policy": "collect_failures",
                    "children": [
                        {
                            "id": "ok",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "create", "agent_id": "ok-agent"},
                            "prompt_template": "prompts/ok.md",
                        },
                        {
                            "id": "bad",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "create", "agent_id": "bad-agent"},
                            # File does not exist — _render_prompt will FileNotFoundError.
                            "prompt_template": "prompts/missing.md",
                        },
                    ],
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())

    # Sibling completed despite the bad child failing.
    assert any(result.node_id == "ok" and result.status == "completed" for result in results)
    # Failed child wrote an error block to its agent.
    bad_state = runner.agent_states.get("bad-agent")
    assert bad_state is not None
    assert bad_state.status == "failed"
    assert "[error]" in bad_state.output_tail

def test_parallel_fail_fast_propagates_first_failure(tmp_path: Path) -> None:
    write(tmp_path / "prompts/slow.md", "wait")  # mock backend blocks until interrupted
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "fanout",
                    "type": "parallel",
                    "max_concurrency": 2,
                    # default failure_policy = fail_fast
                    "children": [
                        {
                            "id": "slow",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "create", "agent_id": "slow-agent"},
                            "prompt_template": "prompts/slow.md",
                        },
                        {
                            "id": "bad",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "create", "agent_id": "bad-agent"},
                            "prompt_template": "prompts/missing.md",  # FileNotFoundError
                        },
                    ],
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)

    async def run_then_cancel() -> None:
        task = asyncio.create_task(runner.run())
        # Give the fan-out a moment to start, then cancel so the "slow"
        # child does not hang the test if the failure happens to land after.
        for _ in range(50):
            if runner.agent_states.get("bad-agent") and runner.agent_states["bad-agent"].status == "failed":
                break
            await asyncio.sleep(0.02)
        await runner.cancel()
        try:
            await task
        except (WorkflowError, FileNotFoundError, RuntimeError):
            pass

    asyncio.run(run_then_cancel())

    bad_state = runner.agent_states.get("bad-agent")
    assert bad_state is not None
    assert bad_state.status == "failed"

def test_fork_from_step_missing_source_raises(tmp_path: Path) -> None:
    write(tmp_path / "prompts/child.md", "child")
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "child",
                    "type": "step",
                    "agent": "worker",
                    "context": {
                        "mode": "fork_from_step",
                        "from_step": "missing-parent",
                        "agent_id": "child-agent",
                    },
                    "prompt_template": "prompts/child.md",
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "cannot fork from missing step" in str(exc)
    else:
        raise AssertionError("fork_from_step with missing source must raise")

def test_fork_from_step_different_provider_raises(tmp_path: Path) -> None:
    write(tmp_path / "prompts/parent.md", "parent")
    write(tmp_path / "prompts/child.md", "child")
    spec = workflow_from_dict(
        {
            **base_spec(
                tmp_path,
                [
                    {
                        "id": "parent",
                        "type": "step",
                        "agent": "main",
                        "context": {"mode": "create", "agent_id": "parent-agent"},
                        "prompt_template": "prompts/parent.md",
                    },
                    {
                        "id": "child",
                        "type": "step",
                        "agent": "other",
                        "context": {
                            "mode": "fork_from_step",
                            "from_step": "parent",
                            "agent_id": "child-agent",
                        },
                        "prompt_template": "prompts/child.md",
                    },
                ],
            ),
            "agents": {
                "main": {"provider": "mock"},
                # Different provider so the fork check fires. We don't actually
                # need this provider to be usable — _select_agent raises before
                # any backend is constructed.
                "other": {"provider": "mock-other"},
            },
        }
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "same provider" in str(exc)
    else:
        raise AssertionError("fork_from_step across providers must raise")
