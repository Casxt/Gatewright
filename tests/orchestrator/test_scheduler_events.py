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
