from __future__ import annotations

from dataclasses import dataclass, field

from .workflow import StepResult


DEFAULT_ABORT_TOKENS = frozenset({"a", "abort", "stop", "cancel", "q", "quit"})


@dataclass
class InteractionMessage:
    id: str
    agent_id: str
    kind: str
    text: str
    node_key: str | None = None
    state: str = "open"
    input_mode: str = "send_to_agent"
    abort_tokens: frozenset[str] = DEFAULT_ABORT_TOKENS
    result: str | None = None


@dataclass
class AgentViewState:
    agent_id: str
    provider: str
    current_node: str | None = None
    status: str = "idle"
    context_id: str | None = None
    queued_inputs: int = 0
    pending_decisions: int = 0
    output_tail: str = ""
    current_prompt_excerpt: str | None = None
    queued_input_tail: list[str] = field(default_factory=list)
    messages: list[InteractionMessage] = field(default_factory=list)

    def open_interaction(self) -> InteractionMessage | None:
        for message in reversed(self.messages):
            if message.state == "open":
                return message
        return None


@dataclass(frozen=True)
class AgentListRow:
    agent_id: str
    provider: str
    current_node: str | None
    status: str
    context_id: str | None
    queued_inputs: int = 0
    pending_decisions: int = 0


def agent_rows(results: list[StepResult]) -> list[AgentListRow]:
    latest: dict[str, StepResult] = {}
    for result in results:
        if result.agent_id:
            latest[result.agent_id] = result
    return [
        AgentListRow(
            agent_id=agent_id,
            provider=result.provider or "",
            current_node=result.node_id,
            status=result.status,
            context_id=result.context_id,
        )
        for agent_id, result in latest.items()
    ]


def agent_rows_from_state(states: dict[str, AgentViewState]) -> list[AgentListRow]:
    return [
        AgentListRow(
            agent_id=state.agent_id,
            provider=state.provider,
            current_node=state.current_node,
            status=state.status,
            context_id=state.context_id,
            queued_inputs=state.queued_inputs,
            pending_decisions=state.pending_decisions,
        )
        for state in states.values()
    ]
