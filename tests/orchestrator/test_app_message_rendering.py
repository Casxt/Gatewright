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


def test_prompt_excerpt_and_running_animation() -> None:
    prompt = "line one\n\nline two " + ("x" * 200)

    assert "\n" not in _prompt_excerpt(prompt)
    assert _prompt_excerpt(prompt).endswith("…")
    frames = [_animated_status("running", frame) for frame in range(4)]
    assert frames == ["-", "\\", "|", "/"]
    assert len({len(frame) for frame in frames}) == 1
    assert _animated_status("idle", 3) == "idle"
    assert _compact("abcdef", 4) == "abc…"

def test_completed_agent_output_renders_markdown_but_running_output_is_plain_text() -> None:
    completed = _message_body_renderables(["# Title", "", "- item"], running=False)
    running = _message_body_renderables(["# Title", "", "- item"], running=True)

    assert any(isinstance(item, Markdown) for item in completed)
    assert not any(isinstance(item, Markdown) for item in running)
    assert any(isinstance(item, Text) and "# Title" in item.plain for item in running)

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
