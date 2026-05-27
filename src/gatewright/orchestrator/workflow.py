from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    name: str
    provider: str
    policy: str = "default"


@dataclass(frozen=True)
class WorkflowSpec:
    version: int
    name: str
    runtime: dict[str, Any]
    agents: dict[str, AgentSpec]
    variables: dict[str, Any]
    workflow: list[dict[str, Any]]
    description: str | None = None


def load_workflow(path) -> WorkflowSpec:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"workflow file must contain a mapping: {path}")
        return workflow_from_dict(data)

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load workflow YAML files") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"workflow file must contain a mapping: {path}")
    return workflow_from_dict(data)


def workflow_from_dict(data: dict[str, Any]) -> WorkflowSpec:
    agents = {
        name: AgentSpec(
            name=name,
            provider=str(raw["provider"]),
            policy=str(raw.get("policy", "default")),
        )
        for name, raw in data.get("agents", {}).items()
    }
    return WorkflowSpec(
        version=int(data.get("version", 1)),
        name=str(data["name"]),
        description=data.get("description"),
        runtime=dict(data.get("runtime", {})),
        agents=agents,
        variables=dict(data.get("variables", {})),
        workflow=list(data.get("workflow", [])),
    )


def expand(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        result = value
        for _ in range(max(1, len(variables) + 1)):
            expanded = re.sub(
                r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda match: str(variables.get(match.group(1), match.group(0))),
                result,
            )
            if expanded == result:
                return result
            result = expanded
        return result
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def resolve_variables(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    variables = {**raw, **overrides}
    for _ in range(max(1, len(variables) + 1)):
        changed = False
        for key, value in list(variables.items()):
            expanded = expand(value, variables)
            if expanded != value:
                variables[key] = expanded
                changed = True
        if not changed:
            break
    return variables


@dataclass
class StepPlan:
    node_id: str
    node_key: str
    agent_id: str | None = None
    provider: str | None = None


@dataclass
class StepResult:
    node_id: str
    node_key: str
    status: str
    agent_id: str | None = None
    provider: str | None = None
    context_id: str | None = None
    output_text: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
