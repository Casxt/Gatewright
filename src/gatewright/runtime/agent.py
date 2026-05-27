from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    AgentConfig,
    AgentEvent,
    ExecutionPolicy,
    ForkAgentRequest,
    ProviderName,
    RunRequest,
    UserDecision,
    default_policy,
)

if TYPE_CHECKING:
    from .backends.base import ProviderBackend


class Agent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        provider: ProviderName | None = None,
        context_id: str | None = None,
        cwd: Path | None = None,
        policy: ExecutionPolicy | None = None,
        backend: "ProviderBackend | None" = None,
    ) -> None:
        if config is None:
            if provider is None:
                raise TypeError("provider is required when config is not supplied")
            if cwd is None:
                raise TypeError("cwd is required when config is not supplied")
            config = AgentConfig(
                provider=provider,
                context_id=context_id,
                cwd=cwd,
                policy=policy,
            )

        self.provider: ProviderName = config.provider
        self.context_id: str | None = config.context_id
        self.cwd: Path = config.cwd
        self.policy: ExecutionPolicy = config.policy or default_policy(config.provider)
        self._backend = backend or get_backend(config.provider)

    async def run(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        async for event in self._backend.run(self, request):
            yield event

    async def resolve_request(self, decision: UserDecision) -> None:
        await self._backend.resolve_request(self, decision)

    async def interrupt(self, reason: str | None = None) -> None:
        await self._backend.interrupt(self, reason)

    async def fork(self, request: ForkAgentRequest | None = None) -> "Agent":
        return await self._backend.fork(self, request or ForkAgentRequest())


_BACKENDS: dict[str, "ProviderBackend"] = {}


def register_backend(backend: "ProviderBackend") -> None:
    _BACKENDS[backend.provider_name] = backend


def get_backend(provider: str) -> "ProviderBackend":
    if provider not in _BACKENDS:
        if provider == "mock":
            from .backends.mock import MockBackend

            register_backend(MockBackend())
        elif provider == "codex":
            from .backends.codex import CodexBackend

            register_backend(CodexBackend())
        elif provider == "claude":
            from .backends.claude import ClaudeBackend

            register_backend(ClaudeBackend())
    try:
        return _BACKENDS[provider]
    except KeyError as exc:
        raise ValueError(f"unknown provider: {provider}") from exc
