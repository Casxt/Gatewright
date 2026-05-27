from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from .scheduler import WorkflowRunner
from .tui import AgentViewState, InteractionMessage
from .workflow import StepPlan, StepResult


class OrchestratorTextualApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }

    #body {
        height: 1fr;
        background: transparent;
    }

    #workflow-pane {
        width: 28;
        padding: 0 1;
        border-right: solid $surface;
        background: transparent;
    }

    #output-pane {
        width: 1fr;
        padding: 0 1;
        border-right: solid $surface;
        background: transparent;
    }

    #agent-pane {
        width: 38;
        padding: 0 1;
        background: transparent;
    }

    .pane-title {
        color: $text-muted;
        height: 1;
        background: transparent;
    }

    .pane-body {
        height: 1fr;
        overflow: auto;
        background: transparent;
    }

    Static {
        background: transparent;
    }

    OptionList {
        background: transparent;
    }

    #output-rich {
        height: auto;
        background: transparent;
    }

    #output-scroll {
        height: 1fr;
        overflow-y: auto;
        background: transparent;
    }

    #operator-input {
        height: 3;
        margin-top: 1;
        background: transparent;
    }

    #status-line {
        /* `height: auto` so error / resume-hint text wraps onto multiple
           lines instead of being clipped at the right edge. Capped via
           max-height so a runaway message can't eat the output panel. */
        height: auto;
        max-height: 6;
        color: $text-muted;
        background: transparent;
    }
    """

    BINDINGS = [
        Binding("tab", "next_agent", "Next agent"),
        Binding("shift+tab", "cycle_mode", "Cycle mode"),
        Binding("ctrl+r", "toggle_reasoning", "Toggle [think]", priority=True),
        Binding("ctrl+t", "toggle_tools", "Toggle [tool]", priority=True),
        # F1 + `?` both open help. priority=True so they fire even when the
        # operator input widget has focus (which is most of the time).
        Binding("f1", "show_help", "Help", priority=True),
        Binding("question_mark", "show_help", "Help", show=False, priority=True),
        Binding("ctrl+y", "copy_output", "Copy messages"),
        Binding("ctrl+c", "request_exit", "Press twice to exit", show=False, priority=True),
        Binding("escape", "cancel_queued_input", "Cancel queued input", show=False, priority=True),
        Binding("up", "cancel_queued_input", "Cancel queued input", show=False, priority=True),
        Binding("q", "cancel", "Cancel"),
    ]

    def __init__(self, runner: WorkflowRunner, *, workflow_name: str, step_confirm: bool = False) -> None:
        super().__init__()
        self.runner = runner
        self.workflow_name = workflow_name
        self.step_confirm = step_confirm
        self.selected_agent_id: str | None = None
        self._option_to_agent_id: dict[str, str] = {}
        self._animation_frame = 0
        self._last_output_text = ""
        self._last_ctrl_c_at = 0.0
        self._pending_step_confirmation: StepPlan | None = None
        self._pending_step_confirmation_future: asyncio.Future[bool] | None = None
        self._pending_interaction: InteractionMessage | None = None
        self._pending_interaction_future: asyncio.Future[str] | None = None
        self.results: list[StepResult] = []
        self.error: str | None = None
        self._workflow_stop_reason: str | None = None
        self._workflow_resume_hint: str | None = None
        # Reasoning + tool blocks are noisy by default. Start collapsed; the
        # user can toggle each with Ctrl+R / Ctrl+T when they want detail.
        self._show_reasoning: bool = False
        self._show_tools: bool = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="workflow-pane"):
                yield Static("", id="workflow", classes="pane-body")
            with Vertical(id="output-pane"):
                with VerticalScroll(id="output-scroll", classes="pane-body"):
                    yield Static("", id="output-rich")
                yield Static("", id="status-line")
                yield Input(placeholder="send input to selected agent", id="operator-input")
            with Vertical(id="agent-pane"):
                yield OptionList(id="agents", classes="pane-body")

    async def on_mount(self) -> None:
        self.title = self.workflow_name
        self.runner.event_handler = self.handle_runner_event
        self.runner.step_interaction_handler = self.await_interaction_input
        self._set_step_mode(self.step_confirm)
        self.set_interval(0.5, self._tick_animation)
        self._refresh()
        self.run_worker(self._run_workflow(), name="workflow", exclusive=True)

    async def handle_runner_event(self, _: dict) -> None:
        self._refresh()

    async def _run_workflow(self) -> None:
        try:
            self.results = await self.runner.run()
        except Exception as exc:
            # The scheduler has already appended a `[error] ...` block to the
            # failed step's agent output, so the detail is visible by selecting
            # that agent in the right pane. Show a short toast and update the
            # status line — do NOT set self.error (which would replace the
            # entire output panel with a full-screen "Workflow failed:" block
            # and prevent the user from inspecting the failed agent).
            self._workflow_stop_reason = f"{type(exc).__name__}: {exc}"
            self._workflow_resume_hint = getattr(self.runner, "resume_hint", None)
            try:
                self.notify(
                    f"Workflow stopped: {self._workflow_stop_reason}",
                    severity="error",
                    timeout=15,
                )
            except Exception:
                pass
        self._refresh()

    def action_next_agent(self) -> None:
        ids = list(self._live_states())
        if not ids:
            return
        if self.selected_agent_id not in ids:
            self.selected_agent_id = ids[0]
        else:
            self.selected_agent_id = ids[(ids.index(self.selected_agent_id) + 1) % len(ids)]
        self._refresh()

    def action_cycle_mode(self) -> None:
        self._set_step_mode(not self.step_confirm)
        if not self.step_confirm and self._pending_step_confirmation_future is not None:
            self._resolve_step_confirmation("yes")
        self.notify(f"mode: {'step' if self.step_confirm else 'auto'}")
        self._refresh()

    def action_toggle_reasoning(self) -> None:
        self._show_reasoning = not self._show_reasoning
        self.notify(f"[think] blocks: {'expanded' if self._show_reasoning else 'collapsed'}")
        self._refresh()

    def action_show_help(self) -> None:
        # If the help screen is already on top, treat the keypress as a
        # second toggle (close). This way `?` / F1 toggle open and close
        # symmetrically — easier muscle memory than learning a separate
        # Esc-to-close.
        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
            return
        self.push_screen(HelpScreen())

    def action_toggle_tools(self) -> None:
        self._show_tools = not self._show_tools
        self.notify(f"[tool] blocks: {'expanded' if self._show_tools else 'collapsed'}")
        self._refresh()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "agents" or event.option.id is None:
            return
        agent_id = self._option_to_agent_id.get(event.option.id)
        if agent_id is None:
            return
        self.selected_agent_id = agent_id
        self._refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "operator-input":
            return
        text = event.value.strip()
        if self._pending_interaction_future is not None:
            event.input.clear()
            self._resolve_interaction_input(text)
            return
        if self._pending_step_confirmation_future is not None:
            event.input.clear()
            self._resolve_step_confirmation(text)
            return
        if not text:
            return
        agent_id = self.selected_agent_id
        if agent_id is None or agent_id not in self.runner.agent_registry:
            self.notify("no live agent selected", severity="warning")
            return
        event.input.clear()
        self.run_worker(self._send_operator_input(agent_id, text), name=f"operator-input-{agent_id}", exclusive=False)
        self._refresh()

    async def _send_operator_input(self, agent_id: str, text: str) -> None:
        try:
            result = await self.runner.send_input(agent_id, text)
        except Exception as exc:
            self.notify(str(exc), severity="error")
            return
        self.notify(f"{result} input for {agent_id}")
        self._refresh()

    async def action_cancel_queued_input(self) -> None:
        selected = self._selected_agent()
        if selected is None or not selected.queued_input_tail:
            return
        cancelled = await self.runner.cancel_queued_input(selected.agent_id)
        if cancelled:
            self.notify(f"cancelled {cancelled} queued input{'s' if cancelled != 1 else ''}")
        self._refresh()

    async def confirm_step(self, plan: StepPlan) -> bool:
        if plan.agent_id is not None:
            self.selected_agent_id = plan.agent_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_step_confirmation = plan
        self._pending_step_confirmation_future = future
        self._refresh()
        try:
            return await future
        finally:
            self._pending_step_confirmation = None
            self._pending_step_confirmation_future = None
            self._refresh()

    async def await_interaction_input(self, message: InteractionMessage) -> str:
        if message.agent_id:
            self.selected_agent_id = message.agent_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_interaction = message
        self._pending_interaction_future = future
        self._refresh()
        try:
            return await future
        finally:
            self._pending_interaction = None
            self._pending_interaction_future = None
            self._refresh()

    def _resolve_interaction_input(self, text: str) -> None:
        future = self._pending_interaction_future
        if future is None or future.done():
            return
        message = self._pending_interaction
        normalized = text.strip().lower()
        if message is not None and normalized in message.abort_tokens:
            future.set_result("abort")
        else:
            future.set_result(text)

    def _resolve_step_confirmation(self, text: str) -> None:
        future = self._pending_step_confirmation_future
        if future is None or future.done():
            return
        normalized = text.lower()
        accepted = normalized not in {"n", "no", "stop", "cancel", "q", "quit"}
        future.set_result(accepted)

    async def action_cancel(self) -> None:
        if self._pending_step_confirmation_future is not None and not self._pending_step_confirmation_future.done():
            self._pending_step_confirmation_future.set_result(False)
        if self._pending_interaction_future is not None and not self._pending_interaction_future.done():
            self._pending_interaction_future.set_result("abort")
        await self.runner.cancel()
        self.exit()

    async def action_request_exit(self) -> None:
        now = time.monotonic()
        if now - self._last_ctrl_c_at < 1.5:
            await self.action_cancel()
            return
        self._last_ctrl_c_at = now
        self.notify("press Ctrl+C again to exit")

    def action_copy_output(self) -> None:
        if self._last_output_text:
            self.copy_to_clipboard(self._last_output_text)
            self.notify("messages copied")

    def _tick_animation(self) -> None:
        if any(state.status == "running" for state in self._live_states().values()):
            self._animation_frame += 1
            self._refresh()

    def _refresh(self) -> None:
        selected = self._selected_agent()
        live_states = self._live_states()
        self.query_one("#workflow", Static).update(
            _workflow_renderable(
                self.workflow_name,
                self.runner.spec.workflow,
                live_states,
                self.runner.step_results_by_key,
            )
        )
        self._refresh_agent_list(live_states, selected.agent_id if selected else None)
        output_text = _output_text(selected, self.error)
        self._last_output_text = output_text
        self.query_one("#output-rich", Static).update(
            _output_renderable(
                output_text,
                selected,
                self.error,
                self._animation_frame,
                show_reasoning=self._show_reasoning,
                show_tools=self._show_tools,
            )
        )
        self.query_one("#status-line", Static).update(
            _status_line_renderable(
                selected,
                self._pending_step_confirmation,
                step_mode=self.step_confirm,
                animation_frame=self._animation_frame,
                workflow_stop_reason=self._workflow_stop_reason,
                workflow_resume_hint=self._workflow_resume_hint,
            )
        )
        input_widget = self.query_one("#operator-input", Input)
        if self._pending_interaction is not None:
            input_widget.placeholder = (
                "reply to continue this agent, or type a/abort to stop"
            )
        elif self._pending_step_confirmation is not None:
            input_widget.placeholder = "start step? Enter/yes to run, no to stop"
        else:
            input_widget.placeholder = "send input to selected agent"

    def _set_step_mode(self, enabled: bool) -> None:
        self.step_confirm = enabled
        self.runner.step_confirm_handler = self.confirm_step if enabled else None

    def _refresh_agent_list(self, states: dict[str, AgentViewState], selected_agent_id: str | None) -> None:
        agent_list = self.query_one("#agents", OptionList)
        self._option_to_agent_id = {}
        options: list[Option] = []
        selected_index: int | None = None
        for index, state in enumerate(states.values()):
            option_id = f"agent-{index}"
            self._option_to_agent_id[option_id] = state.agent_id
            if state.agent_id == selected_agent_id:
                selected_index = index
            options.append(Option(_agent_option_text(state, selected_agent_id, self._animation_frame), id=option_id))
        agent_list.clear_options()
        agent_list.add_options(options)
        if selected_index is not None:
            agent_list.highlighted = selected_index

    def _selected_agent(self) -> AgentViewState | None:
        states = self._live_states()
        if self.selected_agent_id in states:
            return states[self.selected_agent_id]
        for state in states.values():
            if state.status in {"running", "needs_decision"}:
                self.selected_agent_id = state.agent_id
                return state
        if states:
            first = next(iter(states.values()))
            self.selected_agent_id = first.agent_id
            return first
        return None

    def _live_states(self) -> dict[str, AgentViewState]:
        return _live_agent_states(self.runner.agent_states, self.runner.agent_registry.keys())


class HelpScreen(ModalScreen[None]):
    """Modal that lists every keyboard shortcut the user can use, grouped by
    when each one applies. F1, `?`, and Esc all close it."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-modal {
        width: 78;
        height: auto;
        max-height: 32;
        border: thick #88c0d0;
        background: #2e3440;
        padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("f1", "close", "Close"),
        Binding("question_mark", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(_help_renderable(), id="help-modal")

    def action_close(self) -> None:
        self.app.pop_screen()


def _help_renderable() -> Group:
    """Static help-screen content. Hand-curated rather than auto-derived from
    BINDINGS so each shortcut gets a one-sentence explanation of *when* to
    use it — important for the context-sensitive keys (Enter / r / a behave
    differently when an open message is pending)."""
    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Navigation",
            [
                ("Tab", "Cycle through agents in the left pane"),
                ("Shift+Tab", "Toggle auto / step-confirm mode"),
            ],
        ),
        (
            "Visibility (collapse / expand)",
            [
                ("Ctrl+R", "Toggle [think] (reasoning) blocks"),
                ("Ctrl+T", "Toggle [tool] call blocks"),
            ],
        ),
        (
            "Workflow control",
            [
                ("Ctrl+Y", "Copy current agent's message history to clipboard"),
                ("Ctrl+C ×2", "Exit (press twice within ~2s to confirm)"),
                ("q", "Cancel the workflow"),
            ],
        ),
        (
            "Operator input (bottom box, when no open message is pending)",
            [
                ("Enter", "Send the typed line to the selected agent"),
                ("Esc / ↑", "Cancel queued input for the selected agent"),
            ],
        ),
        (
            "When a step pauses with no outputs yet",
            [
                ("Enter", "Send the typed line to the same agent thread"),
                ("a / abort", "Stop — a --start-from resume command appears below"),
            ],
        ),
        (
            "When step-confirm is active",
            [
                ("Enter / yes", "Start the upcoming step"),
                ("no", "Stop the workflow before the step"),
            ],
        ),
        (
            "Help",
            [
                ("F1 / ?", "Open or close this help screen"),
                ("Esc", "Close this help screen"),
            ],
        ),
    ]

    parts: list[object] = [
        Text("TUI Flow Orchestrator — keyboard shortcuts", style="bold #88c0d0"),
    ]
    for title, rows in sections:
        parts.append(Text(""))
        parts.append(Text(title, style="bold #ebcb8b"))
        for key, desc in rows:
            line = Text()
            line.append(f"  {key:<13}  ", style="bold #a3be8c")
            line.append(desc, style="#d8dee9")
            parts.append(line)
    parts.append(Text(""))
    parts.append(
        Text(
            "Press F1, ?, Esc, or q to close.",
            style="dim italic",
        )
    )
    return Group(*parts)


def _workflow_text(workflow_name: str, states: dict[str, AgentViewState]) -> str:
    lines = [workflow_name, "", "Active nodes"]
    seen: set[str] = set()
    for state in states.values():
        if state.current_node and state.current_node not in seen:
            lines.append(f"- {state.current_node}")
            seen.add(state.current_node)
    return "\n".join(lines)


def _workflow_renderable(
    workflow_name: str,
    nodes: list[dict],
    states: dict[str, AgentViewState],
    completed: dict[str, StepResult],
) -> Group:
    active_nodes = {
        state.current_node
        for state in states.values()
        if state.current_node and state.status in {"running", "needs_decision"}
    }
    decision_nodes = {
        state.current_node
        for state in states.values()
        if state.current_node and state.status == "needs_decision"
    }
    rows = list(_workflow_rows(nodes, parent_key="", depth=0))

    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    table.add_row(Text(workflow_name, style="bold"))
    table.add_row(Text(""))
    table.add_row(Text("nodes", style="dim"))
    for row in rows:
        status = _workflow_status(row["node_key"], row["node_id"], active_nodes, decision_nodes, completed)
        table.add_row(_workflow_row_text(row, status), style=_workflow_row_style(status))

    if active_nodes:
        table.add_row(Text(""))
        table.add_row(Text("active", style="dim"))
        for node_key in sorted(active_nodes):
            table.add_row(Text(f"> {node_key}", style="bold"))
    return Group(table)


def _workflow_rows(nodes: list[dict], parent_key: str, depth: int) -> Iterable[dict[str, str | int]]:
    for node in nodes:
        node_id = str(node.get("id", "?"))
        node_type = str(node.get("type", "step"))
        node_key = f"{parent_key}/{node_id}".strip("/")
        yield {"node_id": node_id, "node_key": node_key, "node_type": node_type, "depth": depth}
        if node_type == "loop":
            yield from _workflow_rows(list(node.get("body", [])), node_key, depth + 1)
            gate = node.get("gate")
            if isinstance(gate, dict):
                yield from _workflow_rows([gate], node_key, depth + 1)
        elif node_type == "parallel":
            yield from _workflow_rows(list(node.get("children", [])), node_key, depth + 1)


def _workflow_status(
    node_key: str,
    node_id: str,
    active_nodes: set[str | None],
    decision_nodes: set[str | None],
    completed: dict[str, StepResult],
) -> str:
    if any(_node_matches(node_key, node_id, active) for active in decision_nodes):
        return "needs_decision"
    if any(_node_matches(node_key, node_id, active) for active in active_nodes):
        return "running"
    if node_key in completed or any(result.node_id == node_id for result in completed.values()):
        return "done"
    return "pending"


def _node_matches(node_key: str, node_id: str, active: str | None) -> bool:
    if not active:
        return False
    return active == node_key or active.endswith(f"/{node_key}") or active.rstrip("/").split("/")[-1] == node_id


def _workflow_row_text(row: dict[str, str | int], status: str) -> Text:
    markers = {
        "running": ">",
        "needs_decision": "?",
        "done": ".",
        "pending": "-",
    }
    styles = {
        "running": "bold white",
        "needs_decision": "bold yellow",
        "done": "dim",
        "pending": "#bdbdbd",
    }
    indent = "  " * int(row["depth"])
    node_type = str(row["node_type"])
    node_id = str(row["node_id"])
    suffix = "" if node_type == "step" else f" ({node_type})"
    return Text(f"{indent}{markers[status]} {node_id}{suffix}", style=styles[status])


def _workflow_row_style(status: str) -> str | None:
    if status == "running":
        return "on #1f4a66"
    if status == "needs_decision":
        return "on #5a481c"
    return None


def _live_agent_states(
    states: dict[str, AgentViewState],
    live_agent_ids: Iterable[str],
) -> dict[str, AgentViewState]:
    live_ids = set(live_agent_ids)
    return {agent_id: state for agent_id, state in states.items() if agent_id in live_ids}


def _agent_option_text(state: AgentViewState, selected_agent_id: str | None, animation_frame: int = 0) -> str:
    marker = ">" if state.agent_id == selected_agent_id else " "
    status = _agent_status_text(state.status, animation_frame)
    badges: list[str] = []
    if state.pending_decisions:
        badges.append(f"!{state.pending_decisions}")
    if state.queued_inputs:
        badges.append(f"+{state.queued_inputs}")
    badge_text = " " + " ".join(badges) if badges else ""
    agent_id = _compact(state.agent_id, 18)
    node = _compact(_node_label(state.current_node), 12)
    return f"{marker} {status} {agent_id:<18} {node}{badge_text}".rstrip()


def _agent_text(states: dict[str, AgentViewState], selected_agent_id: str | None) -> str:
    lines: list[str] = []
    for state in states.values():
        lines.append(_agent_option_text(state, selected_agent_id))
        lines.append("")
    return "\n".join(lines).rstrip()


def _output_text(state: AgentViewState | None, error: str | None) -> str:
    if error:
        return f"Workflow failed:\n{error}"
    if state is None:
        return "No agent selected."
    return "\n".join(
        _strip_trailing_empty(
            [
                f"{state.agent_id} [{state.status}]",
                f"context: {state.context_id or '-'}",
                "",
                state.output_tail or "(no output yet)",
            ]
        )
    )


def _output_renderable(
    text: str,
    state: AgentViewState | None,
    error: str | None,
    animation_frame: int = 0,
    *,
    show_reasoning: bool = False,
    show_tools: bool = False,
) -> Group:
    rendered: list[object] = []
    lines = text.splitlines()
    body_start = 0
    if error and lines:
        rendered.append(Text(lines[0], style="bold red"))
        body_start = 1
    elif state is not None:
        for line in lines[:2]:
            rendered.append(Text(line, style="dim"))
        body_start = min(2, len(lines))

    rendered.extend(
        _message_body_renderables(
            lines[body_start:],
            running=state is not None and state.status == "running",
            show_reasoning=show_reasoning,
            show_tools=show_tools,
        )
    )
    if state is not None and state.queued_input_tail:
        rendered.append(Text(""))
        for queued_input in state.queued_input_tail:
            rendered.append(_prompt_block("[input queued] pending (Esc/↑ cancels)", f"> {_compact_prompt(queued_input)}"))
    return Group(*rendered)


def _status_line_renderable(
    state: AgentViewState | None,
    pending_confirmation: StepPlan | None,
    *,
    step_mode: bool,
    animation_frame: int,
    workflow_stop_reason: str | None = None,
    workflow_resume_hint: str | None = None,
) -> Text:
    mode = "step" if step_mode else "auto"
    toggle = "Shift+Tab auto" if step_mode else "Shift+Tab step"
    if workflow_stop_reason is not None:
        reason = workflow_stop_reason.split("\n", 1)[0]
        hint = f" | resume: {workflow_resume_hint}" if workflow_resume_hint else ""
        return Text(
            f"⚠ workflow stopped: {reason} | select failed agent for details{hint} | {toggle}",
            style="bold #bf616a",
        )
    if pending_confirmation is not None:
        return Text(
            f"mode {mode} | start {pending_confirmation.node_key} | Enter/yes to run, no to stop | {toggle}",
            style="#d8dee9",
        )
    if state is not None and state.status == "running":
        prompt = f" | prompt: {_compact_prompt(state.current_prompt_excerpt or '', 90)}" if state.current_prompt_excerpt else ""
        return Text(f"mode {mode} | working{_dots(animation_frame)} {state.current_node or state.agent_id}{prompt} | {toggle}", style="dim")
    if state is not None and state.queued_input_tail:
        return Text(f"mode {mode} | queued input pending (Esc/↑ cancels) | {toggle}", style="dim")
    if state is not None and state.open_interaction() is not None:
        return Text(f"mode {mode} | open message: reply below, a/abort stops | {toggle}", style="dim")
    return Text(f"mode {mode} | {toggle}", style="dim")


def _message_body_renderables(
    lines: list[str],
    *,
    running: bool,
    show_reasoning: bool = True,
    show_tools: bool = True,
) -> list[object]:
    rendered: list[object] = []
    agent_buffer: list[str] = []
    reasoning_buffer: list[str] = []
    in_reasoning = False
    collapsed_tools = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("[step] ") or line.startswith("[input"):
            _flush_reasoning_buffer(rendered, reasoning_buffer, expanded=show_reasoning)
            in_reasoning = False
            _flush_collapsed_tools(rendered, collapsed_tools)
            collapsed_tools = 0
            _flush_agent_buffer(rendered, agent_buffer, running=False)
            if rendered and not _last_renderable_is_blank(rendered):
                rendered.append(Text(""))
            prompt_line = lines[index + 1] if index + 1 < len(lines) and lines[index + 1].startswith("> ") else ""
            rendered.append(_prompt_block(line, prompt_line))
            if prompt_line:
                index += 1
        elif line.startswith("[pause] ") or line.startswith("[error] ") or line.startswith("[recovered]"):
            _flush_reasoning_buffer(rendered, reasoning_buffer, expanded=show_reasoning)
            in_reasoning = False
            _flush_collapsed_tools(rendered, collapsed_tools)
            collapsed_tools = 0
            _flush_agent_buffer(rendered, agent_buffer, running=False)
            if rendered and not _last_renderable_is_blank(rendered):
                rendered.append(Text(""))
            block_lines = [line]
            next_index = index + 1
            while next_index < len(lines) and not (
                lines[next_index].startswith("[step] ")
                or lines[next_index].startswith("[input")
                or lines[next_index].startswith("[tool ")
                or lines[next_index] == "[think]"
                or lines[next_index].startswith("[pause] ")
                or lines[next_index].startswith("[error] ")
                or lines[next_index].startswith("[recovered]")
            ):
                block_lines.append(lines[next_index])
                next_index += 1
            rendered.append(_system_message_block(block_lines[0], "\n".join(block_lines[1:]).strip()))
            index = next_index - 1
        elif line.startswith("[tool "):
            if show_tools:
                # Expanded: each tool call as its own block, inline at the
                # right position. Breaking the agent text buffer is OK here
                # because the user opted in to see this detail.
                _flush_reasoning_buffer(rendered, reasoning_buffer, expanded=show_reasoning)
                in_reasoning = False
                _flush_collapsed_tools(rendered, collapsed_tools)
                collapsed_tools = 0
                _flush_agent_buffer(rendered, agent_buffer, running=False)
                if rendered and not _last_renderable_is_blank(rendered):
                    rendered.append(Text(""))
                rendered.append(_tool_block(line))
            else:
                # Collapsed: do NOT flush the agent buffer. The surrounding
                # text should render as a single continuous Markdown block;
                # flushing per tool call splits it into many small blocks and
                # Rich adds visible spacing between each, which looks awful
                # when there are dozens of tool calls per turn. Just count;
                # the aggregate counter is emitted once at end-of-output.
                collapsed_tools += 1
        elif line == "[think]":
            _flush_agent_buffer(rendered, agent_buffer, running=False)
            _flush_collapsed_tools(rendered, collapsed_tools)
            collapsed_tools = 0
            in_reasoning = True
        elif line == "[/think]":
            _flush_reasoning_buffer(rendered, reasoning_buffer, expanded=show_reasoning)
            in_reasoning = False
        elif in_reasoning:
            reasoning_buffer.append(line)
        else:
            agent_buffer.append(line)
        index += 1
    _flush_reasoning_buffer(rendered, reasoning_buffer, expanded=show_reasoning)
    _flush_collapsed_tools(rendered, collapsed_tools)
    _flush_agent_buffer(rendered, agent_buffer, running=running)
    return rendered


def _flush_reasoning_buffer(rendered: list[object], buffer: list[str], *, expanded: bool = True) -> None:
    if not buffer:
        return
    text = "\n".join(buffer).strip("\n")
    buffer.clear()
    if not text:
        return
    if expanded:
        rendered.append(_reasoning_block(text))
    else:
        line_count = text.count("\n") + 1
        char_count = len(text)
        rendered.append(_reasoning_block_collapsed(line_count, char_count))


def _flush_collapsed_tools(rendered: list[object], count: int) -> None:
    if count <= 0:
        return
    rendered.append(_tool_block_collapsed(count))


def _reasoning_block(text: str) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    table.add_row(Text("[think] (Ctrl+R to collapse)", style="bold #8fbcbb"), style="on #2e3440")
    for body_line in text.splitlines() or [""]:
        table.add_row(Text(body_line, style="italic #d8dee9"), style="on #2e3440")
    return table


def _reasoning_block_collapsed(line_count: int, char_count: int) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    table.add_row(
        Text(
            f"[think] {line_count} lines / {char_count} chars hidden — Ctrl+R to expand",
            style="dim italic #8fbcbb",
        ),
        style="on #2e3440",
    )
    return table


def _tool_block(line: str) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    table.add_row(Text(line, style="bold #ebcb8b"), style="on #3b4252")
    return table


def _tool_block_collapsed(count: int) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    table.add_row(
        Text(
            f"[tool] {count} call{'s' if count != 1 else ''} hidden — Ctrl+T to expand",
            style="dim italic #ebcb8b",
        ),
        style="on #3b4252",
    )
    return table


def _last_renderable_is_blank(rendered: list[object]) -> bool:
    last = rendered[-1]
    return isinstance(last, Text) and not last.plain.strip()


def _flush_agent_buffer(rendered: list[object], buffer: list[str], *, running: bool) -> None:
    if not buffer:
        return
    text = "\n".join(buffer).strip("\n")
    buffer.clear()
    if not text:
        # Buffer was all whitespace (often the blank lines that wrapped a
        # collapsed [tool] line). Drop it entirely instead of emitting a
        # stray blank line — those blank lines compound when there are
        # many tool calls and the whole panel ends up sparse.
        return
    json_renderable = _json_block(text)
    if json_renderable is not None:
        rendered.append(json_renderable)
        return
    if running:
        rendered.append(Text(text))
    else:
        rendered.append(Markdown(text, code_theme="monokai", hyperlinks=False))


def _prompt_block(meta_line: str, prompt_line: str) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
    table.add_row(Text(meta_line, style="bold #f0f0f0"), style="on #4a4a4a")
    if prompt_line:
        table.add_row(Text(prompt_line, style="#dddddd"), style="on #4a4a4a")
    return table


def _system_message_block(meta_line: str, body: str) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    style = "on #4c3743" if meta_line.startswith("[error]") else "on #3f4a37"
    if meta_line.startswith("[pause]"):
        style = "on #4a421f"
    table.add_row(Text(meta_line, style="bold #f0f0f0"), style=style)
    if body:
        for body_line in body.splitlines():
            table.add_row(Text(body_line, style="#f0f0f0"), style=style)
    return table


def _compact_prompt(prompt: str, limit: int = 120) -> str:
    normalized = " ".join(prompt.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _json_block(line: str) -> Table | None:
    stripped = line.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and "decision" in payload:
        return _decision_block(payload)
    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    for json_line in formatted.splitlines():
        table.add_row(Text(json_line, style="#d8dee9"), style="on #263238")
    return table


def _decision_block(payload: dict) -> Table:
    decision = str(payload.get("decision", "")).strip() or "unknown"
    reason = str(payload.get("reason", "")).strip()
    style = _decision_style(decision)
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    table.add_row(Text(f"decision {decision.upper()}", style="bold white"), style=style)
    if reason:
        table.add_row(Text(reason, style="white"), style=style)
    extras = {key: value for key, value in payload.items() if key not in {"decision", "reason"}}
    for key, value in extras.items():
        table.add_row(Text(f"{key}: {value}", style="#d8dee9"), style=style)
    return table


def _step_confirmation_block(plan: StepPlan) -> Table:
    table = Table.grid(expand=True, padding=(0, 0))
    table.add_column(ratio=1, overflow="fold")
    table.add_row(Text("start step", style="bold white"), style="on #5a481c")
    table.add_row(Text(plan.node_key, style="white"), style="on #5a481c")
    table.add_row(Text("Enter/yes to run, no to stop", style="#f0f0f0"), style="on #5a481c")
    return table


def _decision_style(decision: str) -> str:
    normalized = decision.lower()
    if normalized == "exit":
        return "on #24533a"
    if normalized == "continue":
        return "on #1f4a66"
    return "on #4a3f24"


def _dots(frame: int) -> str:
    # Fixed-width animation: width stays at 3 chars so the rest of the status
    # line does not jitter left/right as the dots animate.
    frames = ("   ", ".  ", ".. ", "...")
    return frames[frame % len(frames)]


def _strip_trailing_empty(lines: Iterable[str]) -> list[str]:
    result = list(lines)
    while result and not result[-1]:
        result.pop()
    return result


def _agent_status_text(status: str, frame: int) -> str:
    if status == "running":
        return _animated_status(status, frame)
    labels = {
        "idle": ".",
        "needs_decision": "?",
        "cancelled": "x",
        "failed": "!",
    }
    return labels.get(status, _compact(status, 1))


def _animated_status(status: str, frame: int) -> str:
    if status != "running":
        return status
    frames = ("-", "\\", "|", "/")
    return frames[frame % len(frames)]


def _node_label(node_key: str | None) -> str:
    if not node_key:
        return ""
    return node_key.rstrip("/").split("/")[-1]


def _compact(value: str | None, limit: int) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
