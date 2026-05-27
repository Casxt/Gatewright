from __future__ import annotations

import json
import asyncio
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
    _workflow_text,
)
from gatewright.orchestrator.scheduler import (
    WorkflowError,
    WorkflowRunner,
    _prompt_excerpt,
    approve_all,
)
from gatewright.orchestrator.tui import AgentViewState
from gatewright.orchestrator.tui import agent_rows, agent_rows_from_state
from gatewright.orchestrator.workflow import load_workflow, workflow_from_dict


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


def test_prompt_excerpt_and_running_animation() -> None:
    prompt = "line one\n\nline two " + ("x" * 200)

    assert "\n" not in _prompt_excerpt(prompt)
    assert _prompt_excerpt(prompt).endswith("…")
    frames = [_animated_status("running", frame) for frame in range(4)]
    assert frames == ["-", "\\", "|", "/"]
    assert len({len(frame) for frame in frames}) == 1
    assert _animated_status("idle", 3) == "idle"
    assert _compact("abcdef", 4) == "abc…"


def test_decision_json_renders_as_decision_block() -> None:
    console = Console(record=True, width=80)
    block = _json_block('{"decision":"exit","reason":"done"}')

    assert block is not None
    console.print(block)
    rendered = console.export_text()
    assert "decision EXIT" in rendered
    assert "done" in rendered
    assert '{"decision"' not in rendered


def test_completed_agent_output_renders_markdown_but_running_output_is_plain_text() -> None:
    completed = _message_body_renderables(["# Title", "", "- item"], running=False)
    running = _message_body_renderables(["# Title", "", "- item"], running=True)

    assert any(isinstance(item, Markdown) for item in completed)
    assert not any(isinstance(item, Markdown) for item in running)
    assert any(isinstance(item, Text) and "# Title" in item.plain for item in running)


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
