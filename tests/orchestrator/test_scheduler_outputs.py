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

def test_input_files_dict_form_lists_required_and_optional_present(tmp_path: Path) -> None:
    """input_files: {required_files, optional_files} — parallels outputs schema.
    Both kinds of paths get listed in the rendered prompt when present on disk."""
    write(tmp_path / "prompts/one.md", "do it")
    write(tmp_path / "data/req.md", "REQ_INLINE_MARKER")
    write(tmp_path / "data/opt.md", "OPT_INLINE_MARKER")
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
                    "input_files": {
                        "required_files": [str(tmp_path / "data/req.md")],
                        "optional_files": [str(tmp_path / "data/opt.md")],
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())

    prompt_text = results[0].output_text
    assert "do it" in prompt_text
    assert "# Input Files" in prompt_text
    assert "req.md" in prompt_text
    assert "opt.md" in prompt_text
    # File contents must still NOT be inlined.
    assert "REQ_INLINE_MARKER" not in prompt_text
    assert "OPT_INLINE_MARKER" not in prompt_text


def test_input_files_optional_absent_is_silently_skipped(tmp_path: Path) -> None:
    """An optional input that does not exist on disk must be silently skipped,
    NOT raise FileNotFoundError. This is the round-0 use case where the
    supplement-collection summary does not yet exist."""
    write(tmp_path / "prompts/one.md", "do it")
    write(tmp_path / "data/req.md", "real")
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
                    "input_files": {
                        "required_files": [str(tmp_path / "data/req.md")],
                        "optional_files": [
                            str(tmp_path / "data/does_not_exist.md"),
                            str(tmp_path / "data/also_missing.md"),
                        ],
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    # Must NOT raise — optional missing is fine.
    results = asyncio.run(runner.run())

    prompt_text = results[0].output_text
    assert "req.md" in prompt_text
    assert "does_not_exist.md" not in prompt_text
    assert "also_missing.md" not in prompt_text


def test_input_files_dict_required_missing_still_raises(tmp_path: Path) -> None:
    """In dict form, a missing required_file must still raise FileNotFoundError —
    optional semantics must not leak into the required list."""
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
                    "input_files": {
                        "required_files": [str(tmp_path / "data/missing_req.md")],
                        "optional_files": [],
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    try:
        asyncio.run(runner.run())
    except FileNotFoundError as exc:
        assert "missing_req.md" in str(exc)
    else:
        raise AssertionError("missing required input must raise FileNotFoundError")


def test_input_files_optional_only_renders_input_section(tmp_path: Path) -> None:
    """A step with only optional_files (no required) should still render the
    Input Files section when any optional file is present; if none are present
    the prompt should pass through unchanged."""
    write(tmp_path / "prompts/one.md", "go")
    write(tmp_path / "data/opt_present.md", "x")
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
                    "input_files": {
                        "optional_files": [
                            str(tmp_path / "data/opt_present.md"),
                            str(tmp_path / "data/opt_missing.md"),
                        ],
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    results = asyncio.run(runner.run())
    prompt_text = results[0].output_text
    assert "# Input Files" in prompt_text
    assert "opt_present.md" in prompt_text
    assert "opt_missing.md" not in prompt_text


def test_input_files_dict_unknown_key_raises_workflow_error(tmp_path: Path) -> None:
    """input_files dict only supports 'required_files' / 'optional_files'.
    A typo like 'optional_file' (singular) or 'inputs' must fail loudly."""
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
                    "input_files": {
                        "required_files": [],
                        "extras": ["nope"],
                    },
                }
            ],
        )
    )

    runner = WorkflowRunner(spec, workspace=tmp_path)
    raised: Exception | None = None
    try:
        asyncio.run(runner.run())
    except Exception as exc:
        raised = exc
    assert raised is not None
    assert "input_files dict" in str(raised) and "extras" in str(raised)


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
