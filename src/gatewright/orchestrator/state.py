from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


class RunStateStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.nodes_dir = run_dir / "nodes"
        self.events_path = run_dir / "events.jsonl"
        self.run_state_path = run_dir / "run_state.json"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)

    def write_run_state(self, payload: dict[str, Any]) -> None:
        data = {"updated_at": utc_now(), **payload}
        self.run_state_path.write_text(
            json.dumps(to_jsonable(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_node_state(self, node_key: str, payload: dict[str, Any]) -> None:
        node_dir = self.node_dir(node_key)
        node_dir.mkdir(parents=True, exist_ok=True)
        data = {"updated_at": utc_now(), **payload}
        (node_dir / "node_state.json").write_text(
            json.dumps(to_jsonable(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_prompt(self, node_key: str, prompt: str) -> None:
        node_dir = self.node_dir(node_key)
        node_dir.mkdir(parents=True, exist_ok=True)
        (node_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    def append_event(self, node_key: str, event: dict[str, Any]) -> None:
        record = {"time": utc_now(), "node_key": node_key, **event}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")

    def node_dir(self, node_key: str) -> Path:
        return self.nodes_dir / node_key.replace("/", "__").replace(":", "_")
