from __future__ import annotations

from dataclasses import is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump(by_alias=True))
    if is_dataclass(value):
        return to_plain(asdict(value))
    if hasattr(value, "__dict__"):
        return to_plain(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        )
    return repr(value)


def get_field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def event(
    agent: object,
    event_type,
    run_id: str | None,
    payload: dict[str, Any] | None = None,
):
    from ..models import AgentEvent

    return AgentEvent.now(
        type=event_type,
        provider=getattr(agent, "provider"),
        context_id=getattr(agent, "context_id", None),
        run_id=run_id,
        payload=payload or {},
    )
