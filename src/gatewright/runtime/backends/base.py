from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..models import AgentEvent, ForkAgentRequest, RunRequest, UserDecision


class ProviderBackend(ABC):
    provider_name: str

    @abstractmethod
    async def run(self, agent: object, request: RunRequest) -> AsyncIterator[AgentEvent]:
        raise NotImplementedError

    @abstractmethod
    async def resolve_request(self, agent: object, decision: UserDecision) -> None:
        raise NotImplementedError

    @abstractmethod
    async def interrupt(self, agent: object, reason: str | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    async def fork(self, agent: object, request: ForkAgentRequest) -> object:
        raise NotImplementedError
