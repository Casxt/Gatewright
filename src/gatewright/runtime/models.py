from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


ProviderName = Literal["codex", "claude", "mock"]


class AgentRuntimeError(RuntimeError):
    """Base exception for the agent runtime SDK."""


class AgentNotRunningError(AgentRuntimeError):
    """Raised when a request requires an active run but none exists."""


class PendingRequestNotFoundError(AgentRuntimeError):
    """Raised when resolving an unknown pending request."""


class AgentMode(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


class SandboxMode(StrEnum):
    WORKSPACE_WRITE = "workspace_write"
    GLOBAL_READ_WORKSPACE_WRITE = "global_read_workspace_write"


@dataclass(frozen=True)
class SandboxPolicy:
    mode: SandboxMode


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: AgentMode
    sandbox: SandboxPolicy
    tools: ToolPolicy = field(default_factory=ToolPolicy)


def default_policy(provider: ProviderName) -> ExecutionPolicy:
    if provider == "codex":
        return ExecutionPolicy(
            mode=AgentMode.AUTO,
            sandbox=SandboxPolicy(SandboxMode.GLOBAL_READ_WORKSPACE_WRITE),
        )
    return ExecutionPolicy(
        mode=AgentMode.AUTO,
        sandbox=SandboxPolicy(SandboxMode.WORKSPACE_WRITE),
    )


@dataclass
class AgentConfig:
    provider: ProviderName
    cwd: Path
    context_id: str | None = None
    policy: ExecutionPolicy | None = None


@dataclass(frozen=True)
class ForkAgentRequest:
    pass


@dataclass(frozen=True)
class RunRequest:
    prompt: str


class EventType(StrEnum):
    TEXT = "text"
    TOOL = "tool"
    NEEDS_DECISION = "needs_decision"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    provider: str
    context_id: str | None
    run_id: str | None
    timestamp: datetime
    payload: dict[str, Any]

    @classmethod
    def now(
        cls,
        *,
        type: EventType,
        provider: str,
        context_id: str | None,
        run_id: str | None,
        payload: dict[str, Any] | None = None,
    ) -> "AgentEvent":
        return cls(
            type=type,
            provider=provider,
            context_id=context_id,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc),
            payload=payload or {},
        )


class DecisionAction(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    ANSWER = "answer"


@dataclass(frozen=True)
class PendingRequest:
    id: str
    tool_name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class UserDecision:
    request_id: str
    action: DecisionAction
    input: dict[str, Any] | None = None
