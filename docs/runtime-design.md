# Agent Runtime SDK Design

## Purpose

Provide one local, open-sourceable interface over Claude Agent SDK and Codex SDK.

The abstraction is deliberately small. It does not try to normalize every provider feature. It keeps only the features needed to wrap a live local provider agent that can run with network access, fork provider context, accept user input, and cancel work.

This design is constrained by the implementability review in [IMPLEMENTABILITY_REVIEW.md](IMPLEMENTABILITY_REVIEW.md). The preferred adapters are:

- Claude: Claude Agent SDK streaming input mode.
- Codex: official Codex Python SDK (`codex_app_server`) over its app-server thread API.

Command-line adapters are out of scope. The runtime must not depend on `codex exec`, `claude -p`, terminal sessions, or subprocess control.

## Non-goals

- No investment research workflow logic.
- No provider-specific prompt engineering in the core layer.
- No hidden fallback from one provider to another unless the caller asks for it.
- No attempt to make Claude and Codex feature-identical.
- No command-line provider adapters.
- No subprocess-based cancellation or resume.
- No cross-process recovery in the MVP. Callers are expected to hold live `Agent` objects in memory.

## Core Concepts

### Live Agent

The SDK is a provider wrapper. It does not own workflow ids, workflow state, trace directories, or UI recovery policy.

The SDK exposes a bound `Agent` object. The caller constructs an agent with a provider and, optionally, a provider-native context id:

```python
agent = Agent(provider="codex", context_id=codex_thread_id, cwd=Path.cwd(), policy=policy)
agent = Agent(provider="claude", context_id=claude_session_id, cwd=Path.cwd(), policy=policy)
```

If `context_id` is `None`, the adapter creates a fresh provider context on the first run. Once the provider returns a thread/session id, the live `Agent.context_id` is updated.

`context_id` is the current provider-native conversation/session id:

- Codex: thread id
- Claude / Claude Code: session id

It is not a workflow id and not a persistence record.

`Agent(provider=..., context_id=...)` means "continue this existing provider
conversation/session". It does not fork.

Public API:

```python
class Agent(Protocol):
    provider: Literal["codex", "claude"]
    context_id: str | None
    cwd: Path
    policy: ExecutionPolicy

    async def run(self, request: RunRequest) -> AsyncIterator[AgentEvent]: ...
    async def resolve_request(self, decision: UserDecision) -> None: ...
    async def interrupt(self, reason: str | None = None) -> None: ...
    async def fork(self, request: ForkAgentRequest) -> Agent: ...
```

The MVP assumes this object stays live in the caller process. If the process is killed, the caller creates new agent objects according to its own policy.

### Provider Adapter

Each provider implements the same narrow backend protocol. This is adapter-facing, not the main orchestration API.

The public `Agent` delegates to the selected backend internally. Callers should not need an `AgentRuntime` object for the MVP.

Naming rule:

- `Agent.run()` follows Codex `thread.run()` and maps to Claude `query()`.
- `Agent.resolve_request()` answers a pending permission request or question.
- `Agent.interrupt()` cancels the active run.
- `Agent.fork()` creates a new bound agent with the same conversation context but
  a different provider conversation/session id.

```python
class ProviderBackend(Protocol):
    provider_name: str

    async def run(self, agent: Agent, request: RunRequest) -> AsyncIterator[AgentEvent]: ...
    async def resolve_request(self, agent: Agent, decision: UserDecision) -> None: ...
    async def interrupt(self, agent: Agent, reason: str | None = None) -> None: ...
    async def fork(self, agent: Agent, request: ForkAgentRequest) -> Agent: ...
```

Agent construction is intentionally boring:

```python
@dataclass
class AgentConfig:
    provider: Literal["codex", "claude"]
    context_id: str | None
    cwd: Path
    policy: ExecutionPolicy | None = None

@dataclass
class ForkAgentRequest:
    pass
```

Fork semantics:

| Provider | Fork behavior |
| --- | --- |
| Codex | Calls SDK `thread_fork(parent_thread_id)` and returns an agent bound to the new forked thread id. The parent must already have `context_id`; otherwise there is no context to fork. |
| Claude / Claude Code | Creates a child agent with `context_id=None`, stores the parent session id as the fork source, and starts the first child run with `resume=<parent_session_id>, fork_session=True`. The child receives a new session id from Claude and updates `child.context_id`. |

The child must never share the same `context_id` as the parent after fork. It
inherits context content, not identity.

Two out-of-band actions can happen while a run is active:

- answer a permission request or clarifying question
- interrupt the active run

Provider mapping:

| SDK method | Codex SDK / app-server | Claude Agent SDK |
| --- | --- | --- |
| `resolve_request(decision)` | JSON-RPC response to `item/commandExecution/requestApproval`, `item/fileChange/requestApproval`, or `tool/requestUserInput` | return from `canUseTool`; `AskUserQuestion` is also handled through `canUseTool` |
| `interrupt(reason)` | `turn/interrupt` | abort / cancel the active streaming query through the SDK client |

Active steering is intentionally out of scope for the MVP. Callers that receive user input during a run should queue it and call `agent.run()` after the current run completes.

Both methods return `None`. If the action cannot be accepted, the adapter raises an SDK exception or the active event stream emits `FAILED`.

Adapters:

| Adapter | Responsibility |
| --- | --- |
| `CodexBackend` | Wraps the official Codex Python SDK app-server client. Supports turn events, approvals, interruption, and thread fork. |
| `ClaudeBackend` | Wraps Claude Agent SDK streaming input / client API. Supports sessions, permission callbacks, clarifying questions, and interruption. |

The core package depends only on `Agent` and `ProviderBackend`, not on provider SDK classes.

### Input Model

For the MVP, the SDK accepts one text prompt per run. File contents, skill text, mentions, and other workflow material must be resolved by the caller before invoking the SDK.

Do not model `FileInput`, `SkillInput`, or `MentionInput` in the SDK. Those are caller-side prompt assembly concepts, not provider-agent concepts shared by Claude and Codex.

Images and richer multimodal inputs can be added later as a separate explicit extension after the text-only wrapper is stable.

### RunRequest

```python
@dataclass
class RunRequest:
    prompt: str
```

`prompt` maps directly to Codex `thread.run()` / app-server `turn/start.input` text and Claude `query()` input. This keeps the SDK neutral and pushes all prompt composition to callers.

`cwd` and `policy` belong to `AgentConfig`, not `RunRequest`.

- Codex app-server does allow `cwd`, `approvalPolicy`, and sandbox policy on `turn/start`, but the MVP treats them as fixed properties of the live `Agent` to avoid per-turn policy drift.
- Claude options are also naturally bound to the active query/client setup.
- If a caller needs a different cwd, mode, sandbox, or tool policy, create or fork a separate `Agent`.

Do not put workflow output checks, metadata, response schemas, or timeouts in `RunRequest`. Required files, markers, domain validators, tracing labels, structured report contracts, and scheduling timeouts belong to callers above this SDK.

If a caller wants a run timeout, it should wrap the async task and call `agent.interrupt(reason="timeout")` when the deadline expires.

### ExecutionPolicy

`ExecutionPolicy` groups provider execution policy and is stored on `AgentConfig`.

```python
@dataclass
class ExecutionPolicy:
    mode: AgentMode
    sandbox: SandboxPolicy
    tools: ToolPolicy
```

### AgentMode

```python
class AgentMode(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"
```

| Mode | Meaning |
| --- | --- |
| `MANUAL` | Provider may ask for approval and stop for user input. |
| `AUTO` | Provider should proceed automatically when it can, but must still surface permission or confirmation requests that the provider requires. |

For Codex, both modes keep approval requests enabled via `approvalPolicy="on-request"` so the SDK can surface provider approval requests as `NEEDS_DECISION`. `AUTO` is not mapped to `approvalPolicy="never"`.

If a provider cannot fully honor a mode, the adapter must emit `FAILED` with reason `capability_degraded` and state which field was degraded.

### SandboxPolicy

```python
class SandboxMode(StrEnum):
    WORKSPACE_WRITE = "workspace_write"
    GLOBAL_READ_WORKSPACE_WRITE = "global_read_workspace_write"

@dataclass
class SandboxPolicy:
    mode: SandboxMode
```

`GLOBAL_READ_WORKSPACE_WRITE` is the intended Codex default: read broadly, write only in the workspace. The adapter must not map it to unrestricted write access.

### Network Behavior

Network is not modeled as an SDK policy in the MVP.

- Codex backend default should enable network when the app-server / SDK sandbox configuration supports it.
- Claude / Claude Code backend follows its configured environment and permission mode.
- Proxy environment variables are process-level caller configuration, not SDK fields.

### Provider Defaults

```python
def default_policy(provider: Literal["codex", "claude"]) -> ExecutionPolicy:
    ...
```

Default mapping:

| Provider | Default policy |
| --- | --- |
| Codex | `mode=AUTO`, `sandbox=GLOBAL_READ_WORKSPACE_WRITE`, network enabled by backend default when supported, no tool allow/deny lists. |
| Claude / Claude Code | `mode=AUTO`, auto-mode-equivalent permission setup, network follows configured environment, no tool allow/deny lists. |

Do not map defaults to Codex unrestricted write access, Claude `bypassPermissions`, or any provider mode that removes meaningful safety boundaries.

### ToolPolicy

```python
@dataclass
class ToolPolicy:
    allowed_tools: list[str]
    denied_tools: list[str]
```

Provider mapping:

- Codex: map to SDK/app-server tool allow/deny options when available; command-specific approvals still surface as `PendingRequest`.
- Claude: map to `allowedTools` and `disallowedTools`.

Do not model command prefixes, network hosts, filesystem roots, or persisted approval rules in the MVP SDK. Those are provider-specific security policy layers or caller-side policy.

## Event Model

Adapters stream normalized events.

```python
class EventType(StrEnum):
    TEXT = "text"
    TOOL = "tool"
    NEEDS_DECISION = "needs_decision"
    COMPLETED = "completed"
    FAILED = "failed"
```

Every event has:

```python
@dataclass
class AgentEvent:
    type: EventType
    provider: str
    context_id: str | None
    run_id: str | None
    timestamp: datetime
    payload: dict[str, Any]
```

The SDK streams events. It does not own a trace directory. Callers that need durable traces should write these events to their own storage.

Event payload conventions:

| Event | Payload |
| --- | --- |
| `TEXT` | `{"text": "...", "delta": true}` for streaming text or `{"text": "...", "delta": false}` for a full block. |
| `TOOL` | `{"phase": "call" | "result", "tool_name": "...", "input": {...}, "output": ...}`. File changes are represented as tool events, not a separate event type. |
| `NEEDS_DECISION` | `{"request": PendingRequest}`. |
| `COMPLETED` | Provider final metadata, if any. |
| `FAILED` | `{"reason": "cancelled" | "cancel_timeout" | "error" | "capability_degraded", "message": "..."}`. |

## Permission and Confirmation Semantics

Permission prompts and clarifying questions are first-class runtime events. They are not normal user chat messages.

The adapter detects provider approval / question requests while `run()` is streaming provider events.

Flow:

```text
adapter.run()
  -> provider asks for approval or user input
  -> adapter yields AgentEvent(type=NEEDS_DECISION, payload=PendingRequest)
  -> caller renders it and collects the user's decision
  -> caller calls agent.resolve_request(UserDecision(...))
  -> provider continues the same active run
```

Provider detection:

| Provider | How the adapter detects it | How the adapter keeps it pending |
| --- | --- | --- |
| Codex app-server | A server-initiated JSON-RPC request such as `item/commandExecution/requestApproval`, `item/fileChange/requestApproval`, or `tool/requestUserInput` arrives while the turn is active. | Store the JSON-RPC request id plus `threadId`, `turnId`, and item/request id. Do not answer the JSON-RPC request until `resolve_request()` is called. |
| Claude Agent SDK | `canUseTool` is invoked for a tool permission request or `AskUserQuestion` call while `query()` / streaming input is active. | Keep the callback suspended until `resolve_request()` supplies the approval, denial, modified input, or answer. |

```python
class DecisionAction(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    ANSWER = "answer"

@dataclass
class PendingRequest:
    id: str
    tool_name: str
    input: dict[str, Any]

@dataclass
class UserDecision:
    request_id: str
    action: DecisionAction
    input: dict[str, Any] | None = None
```

`PendingRequest.id` is SDK-local. Provider callback/future objects, JSON-RPC ids, `threadId`, `turnId`, `itemId`, native payloads, UI text, available buttons, and provider-specific persistence suggestions stay inside the adapter or caller. The public request only carries the tool name and input needed to render a confirmation or question.

For Claude `canUseTool`, `APPROVE` maps to allow, `DENY` maps to deny, and `input` can carry updated input. For `AskUserQuestion` or Codex `tool/requestUserInput`, use `ANSWER` with `input` carrying the answer payload.

Adapter mapping:

| Runtime event | Codex app-server | Claude Agent SDK |
| --- | --- | --- |
| approval | command/file/MCP approval requests | `canUseTool(tool_name, input, context)` for tools such as `Bash`, `Write`, `Edit` |
| question | `tool/requestUserInput` when question-shaped | `canUseTool("AskUserQuestion", input, context)` |

When a decision is pending, the adapter must keep the provider turn paused through the SDK/app-server protocol. If a provider API cannot pause and later answer the decision, that adapter does not satisfy the runtime contract.

## User Input Semantics

Provider mapping:

| Input path | Codex SDK / app-server | Claude Agent SDK |
| --- | --- | --- |
| New run input | `thread.run()` / `turn/start` on an idle thread | next `query()` |

The SDK does not implement an after-stop input queue. A caller that wants queued follow-up input should call `agent.run()` after the current run completes.

## Cancellation Semantics

`agent.interrupt(...)` must be best-effort and idempotent.

Required behavior:

- emit `FAILED` with reason `cancelled` when the provider confirms cancellation or interruption
- emit `FAILED` with reason `cancel_timeout` if cancellation cannot be confirmed
- keep provider context readable when the provider exposes it
- never delete provider context automatically

Provider mapping:

- Codex app-server: `turn/interrupt` for an active turn.
- Claude Agent SDK: cancel the active query / abort the streaming input flow through the SDK client.

Callers are responsible for cascading cancellation across multiple agents.

## Backend Requirements

Streaming events, permission requests, cancellation, and fork are core SDK requirements. A backend that cannot stream events, surface permission requests, cancel active work, or fork context should not be accepted as an MVP backend.

## Structure Review

Current shape after pruning:

| Structure | Keep / change | Reason |
| --- | --- | --- |
| `Agent` | Keep, bound live/proxy object | This is what callers use for `run()`, `resolve_request()`, `interrupt()`, and `fork()`. It owns provider state. |
| `RunRequest` | Keep minimal | Contains only prompt text. No cwd, policy, timeout, metadata, response schema, attachments, or workflow output checks. |
| `ExecutionPolicy` | Keep on `AgentConfig` | Execution policy is fixed for the live agent. Create or fork a different agent when policy changes. |
| `PendingRequest` | Keep, minimal | It is the approval/question request shown by callers. Provider callback/future objects and native request ids stay internal. |

## Public API Shape

```python
agent = Agent(
    AgentConfig(
        provider="codex",
        context_id=None,
        cwd=Path.cwd(),
        policy=default_policy("codex"),
    ),
)

events = agent.run(
    RunRequest(
        prompt=prompt_text,
    ),
)

async for event in events:
    handle(event)
```

## Directory Plan

```text
src/gatewright/runtime/
  README.md
  DESIGN.md
  runtime/
    __init__.py
    agent.py
    models.py
    backends/
      base.py
      mock.py
      codex.py
      claude.py
  tests/
```

Implementation can start with `models.py` and one adapter. Workflow orchestrators should depend only on this package's public API.

## Open-source Boundary

The open-sourceable part is:

- normalized runtime models
- provider adapter interface
- cancellation semantics
- SDK/app-server examples

Keep private workflow prompts, company-analysis logic, local database paths, and Xueqiu-specific rules out of this package.

## First Implementation Milestones

1. Define models.
2. Implement a mock adapter for deterministic tests.
3. Implement one real provider adapter.
4. Add fork tests.
5. Add cancellation tests.
6. Implement the second provider adapter.
7. Stabilize the public API before publishing.
