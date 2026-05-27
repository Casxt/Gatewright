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
