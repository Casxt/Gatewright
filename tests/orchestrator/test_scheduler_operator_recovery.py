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
