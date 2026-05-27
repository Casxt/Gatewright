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


def test_nested_variables_expand_at_runtime(tmp_path: Path) -> None:
    write(tmp_path / "prompts/one.md", "round file {round_file}")
    write(tmp_path / "prompts/gate.md", "gate round file {round_file}")
    spec = workflow_from_dict(
        {
            "version": 1,
            "name": "nested-vars",
            "runtime": {
                "trace_root": str(tmp_path / "runs"),
                "default_workspace": str(tmp_path),
            },
            "agents": {"main": {"provider": "mock"}},
            "variables": {
                "base": str(tmp_path / "output"),
                "round_file": "{base}/round-{round}/done.md",
            },
            "workflow": [
                {
                    "id": "loop",
                    "type": "loop",
                    "round_variable": "round",
                    "body": [
                        {
                            "id": "one",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/one.md",
                            "outputs": {"required_files": ["{round_file}"]},
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
        }
    )

    class FileWritingAgent:
        provider = "mock"
        context_id = "ctx"

        async def run(self, request):
            prompt = request.prompt
            path = Path(prompt.split("round file ", 1)[1].strip())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("done", encoding="utf-8")
            text = '{"decision":"exit"}' if "gate" in prompt else prompt
            from gatewright.runtime import AgentEvent, EventType

            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx",
                run_id="run",
                payload={"text": text},
            )

    class TestRunner(WorkflowRunner):
        async def _select_agent(self, node):
            agent = FileWritingAgent()
            self.agent_registry[self._agent_id(node)] = agent
            return agent

    results = asyncio.run(TestRunner(spec, workspace=tmp_path).run())

    assert (tmp_path / "output/round-0/done.md").exists()
    assert "{round}" not in results[0].output_text

def test_derived_round_variables_refresh_each_iteration(tmp_path: Path) -> None:
    """Regression: spec.variables that reference {round} (e.g. round_dir)
    must be re-resolved on every loop iteration. Previously they were
    eagerly resolved once at startup and froze on the entry round, so
    round 3 still saw round-2 paths."""
    write(tmp_path / "prompts/body.md", "body round={round} round_dir={round_dir}")
    write(tmp_path / "prompts/gate.md", "gate")  # filled per-round below

    spec_dict = {
        **base_spec(
            tmp_path,
            [
                {
                    "id": "loop",
                    "type": "loop",
                    "round_variable": "round",
                    "body": [
                        {
                            "id": "body",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/body.md",
                            "outputs": {
                                "required_files": ["{round_dir}/done.md"],
                            },
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
        ),
        "variables": {
            "base": str(tmp_path / "out"),
            "round_dir": "{base}/round-{round}",
        },
    }
    spec = workflow_from_dict(spec_dict)

    # Custom agent that writes {round_dir}/done.md from the path embedded in the
    # rendered prompt, and emits an "exit" gate after the second round.
    class RoundAwareAgent:
        provider = "mock"
        context_id = "ctx-r"

        def __init__(self) -> None:
            self.rounds_seen: list[str] = []

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            prompt = request.prompt
            if prompt.lstrip().startswith("gate"):
                # Exit after we've done two body iterations.
                decision = '{"decision":"exit"}' if len(self.rounds_seen) >= 2 else '{"decision":"continue"}'
                yield AgentEvent.now(
                    type=EventType.TEXT,
                    provider="mock",
                    context_id="ctx-r",
                    run_id="r",
                    payload={"text": decision, "delta": False},
                )
                return
            # body step — extract round_dir from prompt, write done.md there
            for token in prompt.split():
                if token.startswith("round_dir="):
                    round_dir = Path(token.split("=", 1)[1])
                    round_dir.mkdir(parents=True, exist_ok=True)
                    (round_dir / "done.md").write_text("ok", encoding="utf-8")
                    self.rounds_seen.append(round_dir.name)
                    break
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-r",
                run_id="r",
                payload={"text": prompt, "delta": False},
            )

    class RoundAwareRunner(WorkflowRunner):
        _shared_agent = RoundAwareAgent()

        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = self._shared_agent
            return self._shared_agent

    runner = RoundAwareRunner(spec, workspace=tmp_path)
    asyncio.run(runner.run())

    # Both round directories must exist with their own done.md — proving the
    # body re-evaluated {round_dir} for each iteration.
    assert (tmp_path / "out/round-0/done.md").exists(), "round 0 done.md missing"
    assert (tmp_path / "out/round-1/done.md").exists(), "round 1 done.md missing"
    # The runner's live variables reflect the final round. round_dir is kept
    # as a raw template (lazy resolution); expand it on demand.
    from gatewright.orchestrator.workflow import expand

    assert runner.variables["round"] == 1
    assert expand(runner.variables["round_dir"], runner.variables).endswith("round-1")

def test_derived_round_variables_refresh_when_started_mid_loop(tmp_path: Path) -> None:
    """Regression for the exact failure mode the user hit: starting with
    --var round=2, the first iteration runs round 2 (round_dir=round-2), and
    if the gate continues, the NEXT iteration must see round_dir=round-3 —
    not still round-2. Previously this stayed frozen at the entry value."""
    write(tmp_path / "prompts/body.md", "round={round} round_dir={round_dir}")
    write(tmp_path / "prompts/gate.md", "gate")

    spec = workflow_from_dict(
        {
            **base_spec(
                tmp_path,
                [
                    {
                        "id": "loop",
                        "type": "loop",
                        "round_variable": "round",
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
            ),
            "variables": {
                "base": str(tmp_path / "out"),
                "round_dir": "{base}/round-{round}",
            },
        }
    )

    body_round_dirs: list[str] = []

    class RecorderAgent:
        provider = "mock"
        context_id = "ctx-rec"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            prompt = request.prompt
            if prompt.lstrip().startswith("gate"):
                decision = '{"decision":"exit"}' if len(body_round_dirs) >= 2 else '{"decision":"continue"}'
                yield AgentEvent.now(
                    type=EventType.TEXT,
                    provider="mock",
                    context_id="ctx-rec",
                    run_id="r",
                    payload={"text": decision, "delta": False},
                )
                return
            for token in prompt.split():
                if token.startswith("round_dir="):
                    body_round_dirs.append(token.split("=", 1)[1])
                    break
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-rec",
                run_id="r",
                payload={"text": prompt, "delta": False},
            )

    class RecorderRunner(WorkflowRunner):
        _agent = RecorderAgent()

        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = self._agent
            return self._agent

    runner = RecorderRunner(spec, variables={"round": "2"}, workspace=tmp_path)
    asyncio.run(runner.run())

    # Two body iterations: round 2 then round 3 — each must see its own dir.
    assert len(body_round_dirs) == 2
    assert body_round_dirs[0].endswith("round-2")
    assert body_round_dirs[1].endswith("round-3"), (
        f"round 3 body still sees {body_round_dirs[1]} — round_dir froze at entry round"
    )

def test_run_when_false_skips_step(tmp_path: Path) -> None:
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
                    "context": {"mode": "create", "agent_id": "one-agent"},
                    "prompt_template": "prompts/one.md",
                    "run_when": {"expression": "False"},
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

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())

    assert [result.node_id for result in results] == ["two"]
    assert "one-agent" not in runner.agent_states

def test_run_when_true_executes_step(tmp_path: Path) -> None:
    write(tmp_path / "prompts/only.md", "only")
    spec = workflow_from_dict(
        {
            **base_spec(
                tmp_path,
                [
                    {
                        "id": "only",
                        "type": "step",
                        "agent": "main",
                        "context": {"mode": "create", "agent_id": "only-agent"},
                        "prompt_template": "prompts/only.md",
                        "run_when": {"expression": "flag == 'on'"},
                    }
                ],
            ),
            "variables": {"flag": "on"},
        }
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())

    assert [result.node_id for result in results] == ["only"]

def test_loop_round_variable_string_is_coerced_to_int(tmp_path: Path) -> None:
    """`--var round=1` arrives as a string. The loop must coerce to int so
    `run_when` expressions like `round > 0` evaluate without TypeError."""
    write(tmp_path / "prompts/body.md", "body round={round}")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')
    write(tmp_path / "prompts/iter.md", "iterate round={round}")
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
                            "id": "body",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/body.md",
                        },
                        {
                            "id": "only-after-zero",
                            "type": "step",
                            "agent": "main",
                            "context": {"mode": "reuse", "agent_id": "main-agent"},
                            "prompt_template": "prompts/iter.md",
                            "run_when": {"expression": "round > 0"},
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

    runner = WorkflowRunner(spec, variables={"round": "1"}, workspace=tmp_path)
    results = asyncio.run(runner.run())

    assert runner.variables["round"] == 1
    # round > 0 must hold on the first (and only) iteration → iterate ran.
    assert any(result.node_id == "only-after-zero" for result in results)

def test_loop_round_variable_nonint_raises_workflow_error(tmp_path: Path) -> None:
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')
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

    runner = WorkflowRunner(spec, variables={"round": "not-a-number"}, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "must be an integer" in str(exc)
    else:
        raise AssertionError("non-integer round must raise WorkflowError")

def test_loop_uses_agent_backed_quality_gate(tmp_path: Path) -> None:
    write(tmp_path / "prompts/body.md", "round {round}")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit","reason":"done"}')
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

    results = asyncio.run(WorkflowRunner(spec, workspace=tmp_path).run())

    assert [result.node_id for result in results] == ["body", "gate"]
    assert results[0].output_text == "round 0"
    assert results[-1].output_text == '{"decision":"exit","reason":"done"}'
