"""Additional unit tests covering branches the original test_scheduler.py
does not exercise: run_when filtering, loop variable coercion, _check_outputs
error surfacing, parallel failure policies, fork errors, gate decision errors,
needs_decision without handler, send_input edge cases, and status-line
rendering of workflow stop reason."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console

from gatewright.orchestrator.app import (
    _status_line_renderable,
    _workflow_renderable,
)
from gatewright.orchestrator.scheduler import (
    WorkflowError,
    WorkflowNeedsDecision,
    WorkflowRunner,
)
from gatewright.orchestrator.tui import AgentViewState, InteractionMessage
from gatewright.orchestrator.workflow import (
    StepPlan,
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


# ---------------------------------------------------------------------------
# run_when branching
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Loop variable coercion
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _check_outputs surfaces failure in agent message list
# ---------------------------------------------------------------------------

def test_check_outputs_missing_file_surfaces_in_agent_output(tmp_path: Path) -> None:
    """If a step finishes the LLM turn but its declared required_files are
    missing, _check_outputs raises — and the new error path should append
    a [error] block to the failing step's agent_state.output_tail (not crash
    the TUI before the error is visible)."""
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
                    "outputs": {
                        "required_files": [str(tmp_path / "outputs/missing.md")],
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except WorkflowError as exc:
        assert "missing required files" in str(exc)
    else:
        raise AssertionError("missing required_file must raise WorkflowError")

    state = runner.agent_states.get("main-agent")
    assert state is not None
    assert state.status == "failed"
    assert "[error]" in state.output_tail
    assert "WorkflowOutputError" in state.output_tail or "missing required files" in state.output_tail


# ---------------------------------------------------------------------------
# Parallel failure policies
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# fork_from_step error cases
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# send_input error cases
# ---------------------------------------------------------------------------

def test_send_input_to_unknown_agent_raises(tmp_path: Path) -> None:
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)

    try:
        asyncio.run(runner.send_input("nope", "hi"))
    except WorkflowError as exc:
        assert "unknown live agent" in str(exc)
    else:
        raise AssertionError("send_input to unknown agent must raise WorkflowError")


# ---------------------------------------------------------------------------
# Gate decision errors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# needs_decision without a handler
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Status line / workflow renderable
# ---------------------------------------------------------------------------

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


def test_message_body_renders_think_block_with_italic_dim_style() -> None:
    """A `[think]` ... `[/think]` segment inside output_tail must be rendered
    as a distinct block (not as part of the normal agent buffer)."""
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = [
        "[think]",
        "Step 1: check the file.",
        "Step 2: compute the answer.",
        "[/think]",
        "The answer is 42.",
    ]
    rendered = _message_body_renderables(lines, running=False)

    console = Console(record=True, width=80, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    assert "[think]" in text
    assert "Step 1: check the file." in text
    assert "Step 2: compute the answer." in text
    assert "The answer is 42." in text
    # The reasoning content precedes the final answer in render order.
    assert text.index("Step 1") < text.index("The answer is 42.")


def test_collapsed_reasoning_shows_indicator_with_line_count() -> None:
    """When show_reasoning=False, a [think] block must render as a single
    one-line indicator showing how much content is hidden, NOT inline the
    actual reasoning lines."""
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = [
        "[think]",
        "Step 1: collect facts.",
        "Step 2: assess gaps.",
        "Step 3: produce report.",
        "[/think]",
        "The final answer is X.",
    ]
    rendered = _message_body_renderables(lines, running=False, show_reasoning=False)

    console = Console(record=True, width=100, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    assert "[think]" in text
    assert "lines" in text
    assert "Ctrl+R to expand" in text
    # The actual reasoning content must NOT appear.
    assert "Step 1: collect facts." not in text
    assert "Step 2: assess gaps." not in text
    assert "Step 3: produce report." not in text
    # The final answer still shows.
    assert "The final answer is X." in text


def test_expanded_reasoning_default_shows_full_text() -> None:
    """Default show_reasoning=True keeps the full reasoning visible (matches
    the existing in-tree test for backwards compatibility)."""
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = ["[think]", "thought one", "thought two", "[/think]", "answer"]
    rendered = _message_body_renderables(lines, running=False, show_reasoning=True)

    console = Console(record=True, width=80, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    assert "thought one" in text
    assert "thought two" in text
    assert "Ctrl+R to collapse" in text


def test_collapsed_tools_groups_consecutive_calls_into_summary() -> None:
    """When show_tools=False, [tool ...] lines should NOT be rendered as
    individual one-liners. Instead a single grouped indicator with the count
    should appear."""
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = [
        "starting work",
        "[tool commandExecution] ls /tmp",
        "[tool commandExecution] cat /tmp/foo.md",
        "[tool fileChange] /tmp/foo.md",
        "done.",
    ]
    rendered = _message_body_renderables(lines, running=False, show_tools=False)

    console = Console(record=True, width=100, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    # No raw tool detail leaks
    assert "ls /tmp" not in text
    assert "cat /tmp/foo.md" not in text
    assert "/tmp/foo.md" not in text
    # Aggregated indicator present with count + hotkey hint
    assert "3 calls hidden" in text
    assert "Ctrl+T to expand" in text
    # Surrounding text still rendered
    assert "starting work" in text
    assert "done." in text


def test_collapsed_tools_does_not_split_surrounding_text_into_multiple_blocks() -> None:
    """Regression for the 'huge gap between paragraphs' visual bug: when
    tools are collapsed, a sequence of `text → [tool] → [tool] → text`
    must render as ONE continuous Markdown block — not two separate blocks
    with Rich's built-in inter-block spacing between them."""
    from rich.markdown import Markdown

    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = [
        "para one line 1.",
        "para one line 2.",
        "",
        "[tool commandExecution] ls /tmp",
        "",
        "[tool commandExecution] cat /tmp/a.md",
        "",
        "[tool fileChange] /tmp/a.md",
        "",
        "para two line 1.",
        "para two line 2.",
    ]
    rendered = _message_body_renderables(lines, running=False, show_tools=False)

    markdown_blocks = [item for item in rendered if isinstance(item, Markdown)]
    assert len(markdown_blocks) == 1, (
        f"surrounding text must render as a single Markdown block in collapsed "
        f"mode, got {len(markdown_blocks)} blocks"
    )
    body = markdown_blocks[0].markup
    # All four paragraph lines must live inside that single block.
    assert "para one line 1." in body
    assert "para one line 2." in body
    assert "para two line 1." in body
    assert "para two line 2." in body
    # The tool detail itself is gone.
    assert "ls /tmp" not in body
    assert "cat /tmp/a.md" not in body


def test_collapsed_tools_singular_label_for_one_call() -> None:
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = ["before", "[tool commandExecution] echo hi", "after"]
    rendered = _message_body_renderables(lines, running=False, show_tools=False)

    console = Console(record=True, width=80, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    assert "1 call hidden" in text
    assert "1 calls hidden" not in text  # singular not pluralized
    assert "echo hi" not in text


def test_collapse_independent_for_reasoning_and_tools() -> None:
    """A user expanding [think] should not be forced to also expand [tool]
    (and vice versa). The two flags are independent."""
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = [
        "[think]",
        "secret thought",
        "[/think]",
        "[tool commandExecution] secret command",
        "final.",
    ]
    rendered = _message_body_renderables(
        lines, running=False, show_reasoning=True, show_tools=False
    )

    console = Console(record=True, width=100, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    assert "secret thought" in text  # reasoning expanded
    assert "secret command" not in text  # tools collapsed
    assert "1 call hidden" in text


def test_message_body_renders_tool_block_distinctly() -> None:
    from gatewright.orchestrator.app import (
        _message_body_renderables,
    )

    lines = [
        "thinking before tool",
        "[tool commandExecution] ls -la /tmp",
        "now interpreting tool output",
    ]
    rendered = _message_body_renderables(lines, running=False)

    console = Console(record=True, width=80, color_system=None)
    for item in rendered:
        console.print(item)
    text = console.export_text()

    assert "[tool commandExecution]" in text
    assert "ls -la /tmp" in text
    assert "thinking before tool" in text
    assert "now interpreting tool output" in text
    # Tool line is rendered as its own block, sandwiched between the two text
    # segments.
    assert text.index("thinking before tool") < text.index("[tool commandExecution]")
    assert text.index("[tool commandExecution]") < text.index("now interpreting tool output")


def test_format_tool_event_command_execution_one_line() -> None:
    """Unit test of the helper that converts a TOOL event payload into the
    one-line summary that gets appended to output_tail."""
    from gatewright.orchestrator.scheduler import (
        _format_tool_event,
    )

    line = _format_tool_event(
        {
            "phase": "result",
            "tool_name": "commandExecution",
            "input": {"command": "ls -la /tmp"},
            "output": "...",
        }
    )
    assert line == "[tool commandExecution] ls -la /tmp"


def test_format_tool_event_truncates_long_detail() -> None:
    from gatewright.orchestrator.scheduler import (
        _format_tool_event,
    )

    payload = {
        "phase": "result",
        "tool_name": "commandExecution",
        "input": {"command": "echo " + ("x" * 500)},
    }
    line = _format_tool_event(payload, limit=80)
    assert line.startswith("[tool commandExecution]")
    assert line.endswith("…")
    # Just the bracketed prefix + space + truncated detail (80 chars + ellipsis).
    assert len(line) <= len("[tool commandExecution] ") + 80


def test_format_tool_event_mcp_tool_call_extracts_inner_name() -> None:
    from gatewright.orchestrator.scheduler import (
        _format_tool_event,
    )

    line = _format_tool_event(
        {
            "phase": "result",
            "tool_name": "mcpToolCall",
            "input": {"toolName": "search", "arguments": {"q": "hello"}},
        }
    )
    assert line.startswith("[tool mcp:search]")
    assert "hello" in line


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


def test_input_files_are_listed_as_paths_not_inlined(tmp_path: Path) -> None:
    """_render_prompt must validate that input_files exist on disk, list their
    paths in the prompt, and NOT inline their contents. Inlining used to blow
    up Codex / Claude context windows when a step declared many large input
    files. The agent should Read the files it actually needs."""
    write(tmp_path / "prompts/one.md", "do the thing")
    write(tmp_path / "data/evidence.md", "SECRET_INLINE_MARKER_xyz")
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
                    "input_files": [str(tmp_path / "data/evidence.md")],
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())

    prompt_text = results[0].output_text  # mock backend echoes prompt back as TEXT
    assert "do the thing" in prompt_text
    assert "# Input Files" in prompt_text
    assert "evidence.md" in prompt_text
    # The actual file content must NOT appear in the prompt.
    assert "SECRET_INLINE_MARKER_xyz" not in prompt_text


def test_missing_input_file_raises_before_agent_turn(tmp_path: Path) -> None:
    """If an input_file is declared but missing on disk, _render_prompt must
    raise FileNotFoundError before the agent is invoked — the failure must
    land on the failed step's agent message list (covered by the existing
    error-handling path)."""
    write(tmp_path / "prompts/one.md", "go")
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
                    "input_files": ["data/not_there.md"],
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except FileNotFoundError as exc:
        assert "not_there.md" in str(exc)
    else:
        raise AssertionError("missing input_file must raise FileNotFoundError")

    state = runner.agent_states.get("main-agent")
    assert state is not None
    assert state.status == "failed"
    assert "[error]" in state.output_tail


def test_agent_failure_is_not_masked_by_missing_required_files(tmp_path: Path) -> None:
    """Regression: when the agent itself fails (e.g. Codex
    contextWindowExceeded) the step's required output files are guaranteed
    missing — running _check_outputs would mask the real root cause with a
    misleading 'missing required files' error. The scheduler must surface the
    original agent failure instead."""
    write(tmp_path / "prompts/one.md", "trigger context window error")

    class FailingAgent:
        provider = "mock"
        context_id = "ctx-failing"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            yield AgentEvent.now(
                type=EventType.FAILED,
                provider="mock",
                context_id="ctx-failing",
                run_id="run-failing",
                payload={
                    "reason": "error",
                    "message": (
                        "Codex ran out of room in the model's context window. "
                        "Start a new thread."
                    ),
                },
            )

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
                    "outputs": {
                        # These files will NOT exist when the step ends because
                        # the agent failed before writing them.
                        "required_files": [
                            str(tmp_path / "out/report.md"),
                            str(tmp_path / "out/plan.md"),
                        ],
                    },
                }
            ],
        )
    )

    class FailingRunner(WorkflowRunner):
        async def _select_agent(self, node):
            agent = FailingAgent()
            self.agent_registry[self._agent_id(node)] = agent
            return agent

    runner = FailingRunner(spec, workspace=tmp_path)
    raised: Exception | None = None
    try:
        asyncio.run(runner.run())
    except Exception as exc:
        raised = exc

    assert raised is not None, "agent failure must propagate"
    msg = str(raised)
    # Must surface the real agent error, NOT the misleading output check.
    assert "context window" in msg.lower() or "ran out of room" in msg.lower(), (
        f"expected the real agent error to surface, got: {msg}"
    )
    assert "missing required files" not in msg.lower(), (
        f"the _check_outputs error must not mask the agent failure: {msg}"
    )

    state = runner.agent_states.get("main-agent")
    assert state is not None
    assert state.status == "failed"
    # The error block in the agent output should also reflect the real cause.
    assert "[error]" in state.output_tail
    assert "WorkflowOutputError" not in state.output_tail
    assert "context window" in state.output_tail.lower() or "ran out of room" in state.output_tail.lower()


def test_reasoning_events_surface_in_agent_output_tail_with_think_block(tmp_path: Path) -> None:
    """Codex reasoning deltas are TEXT events with payload kind='reasoning'.
    The scheduler must surface them in the agent output_tail wrapped with
    [think] / [/think] markers (so the TUI can render them distinctly), and
    must NOT mix them into StepResult.output_text (which feeds gates and
    downstream steps)."""
    write(tmp_path / "prompts/one.md", "answer the question")

    class ReasoningAgent:
        provider = "mock"
        context_id = "ctx-reasoning"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            def make(payload, evtype=EventType.TEXT):
                return AgentEvent.now(
                    type=evtype,
                    provider="mock",
                    context_id="ctx-reasoning",
                    run_id="run-r",
                    payload=payload,
                )

            yield make({"text": "thinking about ", "delta": True, "kind": "reasoning"})
            yield make({"text": "the problem", "delta": True, "kind": "reasoning"})
            yield make({"text": "FINAL_ANSWER_42", "delta": False})

    class ReasoningRunner(WorkflowRunner):
        async def _select_agent(self, node):
            agent = ReasoningAgent()
            self.agent_registry[self._agent_id(node)] = agent
            return agent

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

    runner = ReasoningRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())

    tail = runner.agent_states["main-agent"].output_tail
    # Both reasoning chunks present, wrapped with markers
    assert "[think]" in tail
    assert "[/think]" in tail
    assert "thinking about the problem" in tail
    # Final answer also in tail
    assert "FINAL_ANSWER_42" in tail
    # The think block precedes the final answer
    assert tail.index("[think]") < tail.index("FINAL_ANSWER_42")
    assert tail.index("[/think]") < tail.index("FINAL_ANSWER_42")
    # StepResult.output_text contains ONLY the final answer — reasoning is
    # display-only and must not pollute downstream consumers.
    assert results[0].output_text == "FINAL_ANSWER_42"


def test_tool_events_surface_as_one_line_in_agent_output_tail(tmp_path: Path) -> None:
    """EventType.TOOL events from the codex backend (commandExecution etc.)
    were previously only recorded to events.jsonl. Now they should also
    appear in the agent output_tail as a one-line summary."""
    write(tmp_path / "prompts/one.md", "do it")

    class ToolingAgent:
        provider = "mock"
        context_id = "ctx-tool"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            def make(payload, evtype=EventType.TEXT):
                return AgentEvent.now(
                    type=evtype,
                    provider="mock",
                    context_id="ctx-tool",
                    run_id="run-t",
                    payload=payload,
                )

            yield make({"text": "running command...", "delta": False})
            yield make(
                {
                    "phase": "result",
                    "tool_name": "commandExecution",
                    "input": {"command": "ls -la /tmp"},
                    "output": "...",
                },
                evtype=EventType.TOOL,
            )
            yield make({"text": "done.", "delta": False})

    class ToolingRunner(WorkflowRunner):
        async def _select_agent(self, node):
            agent = ToolingAgent()
            self.agent_registry[self._agent_id(node)] = agent
            return agent

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

    runner = ToolingRunner(spec, workspace=tmp_path)
    asyncio.run(runner.run())

    tail = runner.agent_states["main-agent"].output_tail
    assert "[tool commandExecution]" in tail
    assert "ls -la /tmp" in tail


def test_codex_backend_maps_reasoning_summary_delta_to_reasoning_text_event() -> None:
    """Unit test of CodexBackend._map_notification: reasoning delta methods
    must produce TEXT events with kind='reasoning'."""
    from gatewright.runtime.backends.codex import CodexBackend

    backend = CodexBackend()

    class FakeAgent:
        provider = "codex"
        context_id = "ctx"
        cwd = "/tmp"

    agent = FakeAgent()
    run_id = "turn-1"

    notification = {
        "method": "item/reasoning/summaryTextDelta",
        "payload": {
            "delta": "I should check the file first",
            "itemId": "item-1",
            "turnId": run_id,
        },
    }
    events = backend._map_notification(agent, run_id, notification)

    assert len(events) == 1
    ev = events[0]
    from gatewright.runtime import EventType

    assert ev.type == EventType.TEXT
    assert ev.payload["kind"] == "reasoning"
    assert ev.payload["text"] == "I should check the file first"
    assert ev.payload["delta"] is True


def test_codex_backend_maps_text_delta_reasoning_to_reasoning_event() -> None:
    """The full-reasoning channel (item/reasoning/textDelta) also maps to
    a TEXT event with kind='reasoning'."""
    from gatewright.runtime.backends.codex import CodexBackend
    from gatewright.runtime import EventType

    backend = CodexBackend()

    class FakeAgent:
        provider = "codex"
        context_id = "ctx"
        cwd = "/tmp"

    events = backend._map_notification(
        FakeAgent(),
        "turn-x",
        {
            "method": "item/reasoning/textDelta",
            "payload": {"delta": "step 2: compute", "itemId": "r1", "turnId": "turn-x"},
        },
    )

    assert len(events) == 1
    assert events[0].type == EventType.TEXT
    assert events[0].payload["kind"] == "reasoning"
    assert events[0].payload["text"] == "step 2: compute"


def test_codex_backend_summary_part_added_emits_reasoning_separator() -> None:
    """summaryPartAdded marks a new reasoning section. We emit a blank-line
    separator (with kind='reasoning') so successive parts do not run
    together."""
    from gatewright.runtime.backends.codex import CodexBackend
    from gatewright.runtime import EventType

    backend = CodexBackend()

    class FakeAgent:
        provider = "codex"
        context_id = "ctx"
        cwd = "/tmp"

    events = backend._map_notification(
        FakeAgent(),
        "turn-x",
        {
            "method": "item/reasoning/summaryPartAdded",
            "payload": {"itemId": "r1", "summaryIndex": 1, "turnId": "turn-x"},
        },
    )

    assert len(events) == 1
    assert events[0].type == EventType.TEXT
    assert events[0].payload["kind"] == "reasoning"
    assert events[0].payload["text"].strip() == ""


def test_step_recovery_success_appends_recovered_marker_to_agent_output(tmp_path: Path) -> None:
    """After a successful recovery, the agent's message list must include a
    visible `[recovered]` line directly after the prior `[error]` block.
    Without this marker the user has no clear in-line signal that the
    previously surfaced error has been resolved."""
    write(tmp_path / "prompts/body.md", "do the thing")

    class FlakyThenOkAgent:
        provider = "mock"
        context_id = "ctx-fl"

        def __init__(self, target_file: Path) -> None:
            self.target = target_file
            self.calls = 0

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            self.calls += 1
            if self.calls == 1:
                yield AgentEvent.now(
                    type=EventType.FAILED,
                    provider="mock",
                    context_id="ctx-fl",
                    run_id=f"r{self.calls}",
                    payload={"reason": "transient", "message": "boom"},
                )
                return
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.target.write_text("ok", encoding="utf-8")
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-fl",
                run_id=f"r{self.calls}",
                payload={"text": "done", "delta": False},
            )

    target = tmp_path / "out/required.md"
    agent = FlakyThenOkAgent(target)
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "step1",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "agent-1"},
                    "prompt_template": "prompts/body.md",
                    "outputs": {"required_files": [str(target)]},
                }
            ],
        )
    )

    class Runner(WorkflowRunner):
        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = agent
            return agent

    async def handler(plan, exc):  # noqa: ARG001
        return "retry"

    runner = Runner(spec, workspace=tmp_path, step_recovery_handler=handler)
    asyncio.run(runner.run())

    output = runner.agent_states["agent-1"].output_tail
    # Both the original [error] block AND the [recovered] marker must be
    # present, IN ORDER. Audit history preserved + resolution made visible.
    assert "[error]" in output
    assert "[recovered]" in output
    assert "outputs now present" in output
    assert output.index("[error]") < output.index("[recovered]"), (
        "[recovered] marker should appear AFTER the original [error] block, "
        "not before it"
    )
    # And the workflow's _last_failed_step record is cleared.
    assert "agent-1" not in runner._last_failed_step


def test_send_input_auto_resolves_failed_step_when_outputs_appear(tmp_path: Path) -> None:
    """The user's scenario: step aborts (recovery attempt also fails);
    workflow stops; user does an external action; user types into the
    operator input box; the agent writes the missing file in that turn.
    The orchestrator must auto-promote the step from failed to completed
    and surface a `[recovered]` marker so the user knows the prior
    `[error]` is resolved."""
    write(tmp_path / "prompts/body.md", "do the thing")
    target = tmp_path / "out/required.md"

    class AlwaysFailFirstThenOnDemandAgent:
        provider = "mock"
        context_id = "ctx-fa"

        def __init__(self) -> None:
            self.calls = 0
            self.write_on_next_run = False

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            self.calls += 1
            # First two calls (initial + open-message continuation): no file.
            if self.calls < 3 or not self.write_on_next_run:
                yield AgentEvent.now(
                    type=EventType.FAILED if self.calls == 1 else EventType.TEXT,
                    provider="mock",
                    context_id="ctx-fa",
                    run_id=f"r{self.calls}",
                    payload=(
                        {"reason": "x", "message": "still missing"}
                        if self.calls == 1
                        else {"text": "still waiting", "delta": False}
                    ),
                )
                return
            # Operator input turn: NOW write the file.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("ok", encoding="utf-8")
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-fa",
                run_id=f"r{self.calls}",
                payload={"text": "wrote it now", "delta": False},
            )

    agent = AlwaysFailFirstThenOnDemandAgent()
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "step1",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "agent-1"},
                    "prompt_template": "prompts/body.md",
                    "outputs": {"required_files": [str(target)]},
                }
            ],
        )
    )

    class Runner(WorkflowRunner):
        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = agent
            return agent

    async def handler(plan, exc):  # noqa: ARG001
        # Try retry once; it'll still fail; workflow then aborts.
        return "retry"

    runner = Runner(spec, workspace=tmp_path, step_recovery_handler=handler)
    try:
        asyncio.run(runner.run())
    except Exception:
        pass  # workflow aborted as expected

    # At this point: step is failed, _last_failed_step has the record,
    # output_tail has [error] but no [recovered] yet.
    assert "agent-1" in runner._last_failed_step
    state_before = runner.agent_states["agent-1"]
    assert "[error]" in state_before.output_tail
    assert "[recovered]" not in state_before.output_tail

    # Simulate: user does external action. Then types into operator input.
    # Tell the agent to write the file on its next turn.
    agent.write_on_next_run = True
    asyncio.run(runner.send_input("agent-1", "I've done the external action, continue"))

    # After send_input: orchestrator must have detected that the failed
    # step's required outputs are now present, promoted node state, and
    # appended a [recovered] marker.
    state_after = runner.agent_states["agent-1"]
    assert "[recovered]" in state_after.output_tail
    assert "via operator input" in state_after.output_tail
    assert "agent-1" not in runner._last_failed_step
    # node_state.json on disk should now say completed.
    import json as _json

    node_dir = runner.store.node_dir("step1")
    node_state = _json.loads((node_dir / "node_state.json").read_text(encoding="utf-8"))
    assert node_state["status"] == "completed"
    assert node_state["recovered"] == "operator_input"


def test_send_input_does_not_promote_when_outputs_still_missing(tmp_path: Path) -> None:
    """If the operator-input turn does NOT produce the missing outputs,
    the failed-step bookkeeping must be retained — the user can try again
    later (with another external action + another operator input)."""
    write(tmp_path / "prompts/body.md", "do the thing")
    target = tmp_path / "out/required.md"

    class NeverWritesAgent:
        provider = "mock"
        context_id = "ctx-nw"
        calls = 0

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            type(self).calls += 1
            if type(self).calls == 1:
                yield AgentEvent.now(
                    type=EventType.FAILED,
                    provider="mock",
                    context_id="ctx-nw",
                    run_id="r1",
                    payload={"reason": "x", "message": "missing"},
                )
                return
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-nw",
                run_id=f"r{type(self).calls}",
                payload={"text": "still no file", "delta": False},
            )

    agent = NeverWritesAgent()
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "step1",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "agent-1"},
                    "prompt_template": "prompts/body.md",
                    "outputs": {"required_files": [str(target)]},
                }
            ],
        )
    )

    class Runner(WorkflowRunner):
        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = agent
            return agent

    async def handler(plan, exc):  # noqa: ARG001
        return "retry"

    runner = Runner(spec, workspace=tmp_path, step_recovery_handler=handler)
    try:
        asyncio.run(runner.run())
    except Exception:
        pass

    # Operator types something but agent still doesn't write the file.
    asyncio.run(runner.send_input("agent-1", "tried again"))

    state = runner.agent_states["agent-1"]
    # No spurious [recovered] marker.
    assert "[recovered]" not in state.output_tail
    # Failed-step bookkeeping retained so the user can try again.
    assert "agent-1" in runner._last_failed_step
    import json as _json

    node_dir = runner.store.node_dir("step1")
    node_state = _json.loads((node_dir / "node_state.json").read_text(encoding="utf-8"))
    assert node_state["status"] == "failed"


def test_step_interaction_input_is_sent_to_same_agent_thread(tmp_path: Path) -> None:
    """New pause/error flow: the user reply from the open message is the next
    prompt sent to the same agent. It is not converted into a separate
    recovery command unless the input is empty."""
    write(tmp_path / "prompts/one.md", "do the thing")
    target_file = tmp_path / "out/required.md"

    class PausesThenContinuesAgent:
        provider = "mock"
        context_id = "ctx-direct"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run(self, request):
            from gatewright.runtime import AgentEvent, EventType

            self.prompts.append(request.prompt)
            if len(self.prompts) == 1:
                yield AgentEvent.now(
                    type=EventType.FAILED,
                    provider="mock",
                    context_id="ctx-direct",
                    run_id="r1",
                    payload={"reason": "pause", "message": "waiting for login"},
                )
                return
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text("ok", encoding="utf-8")
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-direct",
                run_id="r2",
                payload={"text": "continued", "delta": False},
            )

    agent = PausesThenContinuesAgent()
    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "agent-1"},
                    "prompt_template": "prompts/one.md",
                    "outputs": {"required_files": [str(target_file)]},
                }
            ],
        )
    )

    class Runner(WorkflowRunner):
        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = agent
            return agent

    async def interaction_handler(message: InteractionMessage) -> str:
        assert message.kind == "pause"
        assert message.state == "open"
        return "I finished login, continue now"

    runner = Runner(spec, workspace=tmp_path, step_interaction_handler=interaction_handler)
    results = asyncio.run(runner.run())

    assert [r.node_id for r in results] == ["one"]
    assert agent.prompts[1] == "I finished login, continue now"
    messages = runner.agent_states["agent-1"].messages
    assert any(message.kind == "error" for message in messages)
    pause = next(message for message in messages if message.kind == "pause")
    assert pause.state == "closed"
    assert pause.result == "I finished login, continue now"


def test_step_recovery_handler_retry_completes_step_and_continues_workflow(tmp_path: Path) -> None:
    """When a step fails and the recovery handler answers 'retry', the
    scheduler must send a continuation prompt to the same agent, re-check
    required outputs, and — if they're now present — mark the step as
    completed and proceed with the rest of the workflow."""
    write(tmp_path / "prompts/one.md", "do the thing")
    write(tmp_path / "prompts/two.md", "do the next thing")

    class FlakyAgent:
        provider = "mock"
        context_id = "ctx-flaky"
        provider_name = "mock"

        def __init__(self, target_file: Path) -> None:
            self.calls = 0
            self.target = target_file

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            self.calls += 1
            if self.calls == 1:
                # First call: agent "fails" (does NOT write the required file).
                yield AgentEvent.now(
                    type=EventType.FAILED,
                    provider="mock",
                    context_id="ctx-flaky",
                    run_id=f"r{self.calls}",
                    payload={"reason": "network", "message": "transient network drop"},
                )
                return
            # Retry: now writes the file the step requires.
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.target.write_text("ok", encoding="utf-8")
            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-flaky",
                run_id=f"r{self.calls}",
                payload={"text": "recovered", "delta": False},
            )

    class HappyAgent:
        provider = "mock"
        context_id = "ctx-happy"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            yield AgentEvent.now(
                type=EventType.TEXT,
                provider="mock",
                context_id="ctx-happy",
                run_id="r-happy",
                payload={"text": request.prompt, "delta": False},
            )

    target_file = tmp_path / "out/required.md"
    flaky_agent = FlakyAgent(target_file)
    happy_agent = HappyAgent()

    spec = workflow_from_dict(
        base_spec(
            tmp_path,
            [
                {
                    "id": "one",
                    "type": "step",
                    "agent": "main",
                    "context": {"mode": "reuse", "agent_id": "flaky-agent"},
                    "prompt_template": "prompts/one.md",
                    "outputs": {"required_files": [str(target_file)]},
                },
                {
                    "id": "two",
                    "type": "step",
                    "agent": "worker",
                    "context": {"mode": "create", "agent_id": "happy-agent"},
                    "prompt_template": "prompts/two.md",
                },
            ],
        )
    )

    class RecoveryRunner(WorkflowRunner):
        async def _select_agent(self, node):
            agent_id = self._agent_id(node)
            agent = flaky_agent if agent_id == "flaky-agent" else happy_agent
            self.agent_registry[agent_id] = agent
            return agent

    async def handler(plan, exc):  # noqa: ARG001
        return "retry"

    runner = RecoveryRunner(spec, workspace=tmp_path, step_recovery_handler=handler)
    results = asyncio.run(runner.run())

    # Step one was recovered (one initial turn + one retry turn), step two ran.
    assert flaky_agent.calls == 2
    assert [r.node_id for r in results] == ["one", "two"]
    assert all(r.status == "completed" for r in results)
    assert runner.agent_states["flaky-agent"].status == "idle"
    # The required output written by the recovery turn is present.
    assert target_file.exists()
    # No resume hint when the workflow finished successfully.
    assert runner.resume_hint is None


def test_step_recovery_abort_propagates_with_resume_hint(tmp_path: Path) -> None:
    """When the handler returns 'abort', the workflow stops as before, but
    `runner.resume_hint` must be populated so the TUI can display a
    --start-from command the user can copy."""
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')

    class AlwaysFailAgent:
        provider = "mock"
        context_id = "ctx-fail"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            yield AgentEvent.now(
                type=EventType.FAILED,
                provider="mock",
                context_id="ctx-fail",
                run_id="r",
                payload={"reason": "error", "message": "boom"},
            )

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
                            "context": {"mode": "reuse", "agent_id": "fail-agent"},
                            "prompt_template": "prompts/body.md",
                        }
                    ],
                    "gate": {
                        "id": "gate",
                        "type": "step",
                        "agent": "main",
                        "context": {"mode": "reuse", "agent_id": "fail-agent"},
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

    failing_agent = AlwaysFailAgent()

    class StubRunner(WorkflowRunner):
        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = failing_agent
            return failing_agent

    async def handler(plan, exc):  # noqa: ARG001
        return "abort"

    runner = StubRunner(
        spec,
        workspace=tmp_path,
        variables={"round": "4"},
        step_recovery_handler=handler,
    )

    raised: Exception | None = None
    try:
        asyncio.run(runner.run())
    except Exception as exc:
        raised = exc

    assert raised is not None
    assert runner.resume_hint is not None
    # The hint must include the nested --start-from form and the round.
    assert "--start-from loop/body" in runner.resume_hint
    assert "--var round=4" in runner.resume_hint


def test_step_recovery_retry_failure_still_aborts_with_resume_hint(tmp_path: Path) -> None:
    """If the retry turn itself fails (e.g. the agent fails again, or the
    outputs still are not present), the workflow aborts and a resume hint is
    set — i.e. one bad retry must not let the workflow continue."""
    write(tmp_path / "prompts/body.md", "body")
    write(tmp_path / "prompts/gate.md", '{"decision":"exit"}')

    class AlwaysFailAgent:
        provider = "mock"
        context_id = "ctx-perpetual"

        async def run(self, request):  # noqa: ARG002
            from gatewright.runtime import AgentEvent, EventType

            yield AgentEvent.now(
                type=EventType.FAILED,
                provider="mock",
                context_id="ctx-perpetual",
                run_id="r",
                payload={"reason": "error", "message": "still broken"},
            )

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
                            "context": {"mode": "reuse", "agent_id": "perpetual-agent"},
                            "prompt_template": "prompts/body.md",
                        }
                    ],
                    "gate": {
                        "id": "gate",
                        "type": "step",
                        "agent": "main",
                        "context": {"mode": "reuse", "agent_id": "perpetual-agent"},
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

    perp = AlwaysFailAgent()

    class StubRunner(WorkflowRunner):
        async def _select_agent(self, node):
            self.agent_registry[self._agent_id(node)] = perp
            return perp

    async def handler(plan, exc):  # noqa: ARG001
        return "retry"

    runner = StubRunner(
        spec,
        workspace=tmp_path,
        variables={"round": "7"},
        step_recovery_handler=handler,
    )

    try:
        asyncio.run(runner.run())
    except Exception:
        pass

    assert runner.resume_hint is not None
    assert "round=7" in runner.resume_hint


def test_build_resume_hint_top_level_node(tmp_path: Path) -> None:
    """For a top-level (non-loop) failure, the resume hint should reference
    just the top-level node id, with no '/round-' segment."""
    spec = workflow_from_dict(base_spec(tmp_path, []))
    runner = WorkflowRunner(spec, workspace=tmp_path)
    assert runner._build_resume_hint("initialize-company-context") == "--start-from initialize-company-context"


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
