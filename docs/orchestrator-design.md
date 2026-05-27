# Gatewright Design

This file describes the current implementation so future agents can debug it.
For workflow authoring and commands, read [README.md](README.md).

## Design Goal

The orchestrator is a small workflow runner over `gatewright.runtime.Agent`.

It owns:

- workflow graph execution
- loop and parallel scheduling
- logical agent registry
- live agent state for the TUI
- queued operator input
- provider permission decision routing
- run-state files
- cancellation of active agents

It does not own:

- Claude or Codex SDK details
- provider session persistence after process death
- domain quality judgment
- automatic repair logic
- durable replay of killed provider agents

The first target workflow is the company-analysis `02-06` loop, but the runner
is generic. A loop is always body steps plus an agent-backed gate step.

## Package Layout

```text
src/gatewright/orchestrator/
  README.md                     usage and workflow authoring
  DESIGN.md                     implementation design and bug-fix guide
  examples/                     runnable YAML workflows and prompts
  tests/test_scheduler.py       unit tests for scheduler and render helpers
  gatewright.orchestrator/
    workflow.py                 dataclasses, YAML load, variable expansion
    scheduler.py                WorkflowRunner and execution semantics
    app.py                      Textual application
    tui.py                      TUI state dataclasses and agent list helpers
    state.py                    run-state file writer
    runner.py                   CLI entrypoint
```

## Main Objects

### `WorkflowSpec`

Defined in `workflow.py`.

It is the parsed YAML/JSON workflow:

- `version`
- `name`
- `runtime`
- `agents`
- `variables`
- `workflow`
- optional `description`

`load_workflow(path)` loads YAML or JSON. YAML requires PyYAML.

### `AgentSpec`

One named provider entry:

```python
AgentSpec(name="main", provider="codex", policy="default")
```

`policy` is parsed but not used by the MVP scheduler.

### `StepPlan`

The pre-run confirmation object:

```python
StepPlan(
    node_id="write-final",
    node_key="write-final",
    agent_id="codex-real-writer",
    provider="codex",
)
```

`--step-confirm` handlers receive `StepPlan` before the step creates or reuses
an SDK `Agent`. If the handler returns false, the step does not start.

### `StepResult`

The completed step record:

```python
StepResult(
    node_id="write-final",
    node_key="write-final",
    status="completed",
    agent_id="codex-real-writer",
    provider="codex",
    context_id="...",
    output_text="...",
    error=None,
)
```

The scheduler stores results by both `node_id` and full `node_key`.
`fork_from_step` currently looks up by `node_id`, so duplicate step ids across
different branches can be ambiguous. Prefer unique step ids when a later step
uses `from_step`.

## Layer Boundary

```text
WorkflowRunner
  creates and holds Agent(provider=..., cwd=workspace)
  calls agent.run(RunRequest(prompt=...))
  calls agent.resolve_request(decision)
  calls agent.interrupt(reason)
  calls agent.fork()

gatewright.runtime
  maps provider events and context ids
  implements Claude/Codex provider behavior
```

The orchestrator never imports provider SDKs directly.

## Execution Flow

Top-level `WorkflowRunner.run()`:

1. Write `run_state.json` with status `running`.
2. Apply `--start-from` filtering if configured.
3. Run each top-level node.
4. Collect `StepResult` objects.
5. Write final `run_state.json` as `completed` or `failed`.

Node dispatch:

```text
_run_node
  step     -> _run_step
  loop     -> _run_loop
  parallel -> _run_parallel
```

### Step Flow

`_run_step(node, parent_key)`:

1. Build `node_key` from `parent_key` and `node.id`.
2. Expand variables in the node.
3. Compute logical `agent_id`.
4. Send `StepPlan` to `_confirm_step`.
5. Select or create the SDK `Agent`.
6. Render prompt from `prompt_template` and `input_files`.
7. Write `nodes/{node_key}/prompt.md`.
8. Write `node_state.json` as `running`.
9. Mark the agent as `running` in TUI state.
10. Run one provider turn through `_run_agent_turn`.
11. If queued input exists, run follow-up turns on the same live agent.
12. Build and remember `StepResult`.
13. Validate `outputs.required_files`.
14. Write final `node_state.json`.
15. Mark agent `idle`, or `failed` and raise.

Important: step confirmation happens before agent selection. A denied step
does not create an agent, write a prompt, or call the provider.

### Agent Turn Flow

`_run_agent_turn(...)`:

1. Append a prompt preview to the selected agent message history.
2. Iterate `agent.run(RunRequest(prompt=prompt))`.
3. For `TEXT`, append text to step output and TUI message history.
4. For `NEEDS_DECISION`, either:
   - call `decision_handler(agent, event)` and pass the decision to
     `agent.resolve_request(...)`, or
   - raise `WorkflowNeedsDecision` if there is no handler.
5. For `FAILED`, return failed status and error.

The TUI receives events through `event_handler` and refreshes from
`runner.agent_states`.

### Queued Operator Input

If `send_input(agent_id, text)` is called while the agent is active:

1. The text is appended to `runner.queued_inputs[agent_id]`.
2. The TUI state records it in `queued_input_tail`.
3. The active provider turn is not interrupted.
4. After the active turn completes, the scheduler sends a follow-up prompt:

```text
User follow-up input after the previous turn stopped:

{queued text}
```

If `send_input` is called for an idle known agent, the runner starts an
operator turn immediately using the existing live SDK `Agent`.

### Loop Flow

`_run_loop(node, parent_key)`:

1. Initialize `round_variable`, default `round`, to `0` if absent.
2. Read `max_loop_rounds`, default `5`, as a non-negative absolute round cap.
3. If `round >= max_loop_rounds`, stop before the body/gate and ask the
   operator to confirm whether post-loop steps may run. `0` therefore skips
   the loop while still requiring confirmation.
4. Set `previous_round` to `round - 1`.
5. Run each `body` child under node key `loop-id/round-N`.
6. Run `gate` as a normal step.
7. Parse the gate output JSON.
8. If decision is `exit`, break.
9. If decision is `continue`, increment `round` and repeat.
10. Any other decision raises `WorkflowError`.

The loop controller does not inspect domain files to decide exit. The gate is
an agent step and the structured output is the control signal.

Gate parsing:

- `decision.parser` must be `json`.
- The scheduler first tries to parse the whole output as JSON.
- If that fails, it extracts the first `{...}` block with a DOTALL regex.
- It compares configured paths like `decision` with `continue_when` and
  `exit_when`.

### Parallel Flow

`_run_parallel(node, parent_key)`:

1. Create a semaphore using `max_concurrency`.
2. Run children with `asyncio.gather`.
3. With `failure_policy: fail_fast`, child exceptions propagate.
4. With `failure_policy: collect_failures`, exceptions are collected and the
   parallel node writes `completed_with_failures`.
5. Returned child `StepResult` objects are flattened.

Step confirmations inside parallel children are serialized by
`_step_confirm_lock`, so the TUI does not show multiple simultaneous step
confirmation prompts.

## Agent Registry and Context

`WorkflowRunner.agent_registry` maps logical `agent_id` to live SDK `Agent`.

Context modes:

| Mode | Implementation |
| --- | --- |
| `create` | `Agent(provider=spec.provider, cwd=workspace)` and bind to `agent_id`. |
| `reuse` | Return existing `agent_registry[agent_id]`; create if missing. |
| `fork_from_step` | Look up `step_results[from_step].agent_id`, then call parent `fork()`. |
| `fork_from_agent` | Look up `agent_registry[from_agent]`, then call parent `fork()`. |

Fork requires the same provider as the target `agent` spec. Cross-provider
context transfer must be file-backed through `input_files` or domain artifacts.

`context_id` is read from the SDK `Agent` and written to state files, but it is
not used to reconstruct killed provider sessions in the MVP.

## Run State Files

`RunStateStore` writes:

```text
{run_dir}/
  run_state.json
  events.jsonl
  nodes/
    {node_key}/
      node_state.json
      prompt.md
```

Node directory names are sanitized:

```python
node_key.replace("/", "__").replace(":", "_")
```

`run_state.json` records:

- `run_id`
- `workflow`
- `status`
- `variables`
- `start_from`
- `updated_at`

`node_state.json` records:

- `node_id`
- `node_key`
- `status`
- `agent_id`
- `provider`
- `context_id`
- `round`
- optional `error`
- `updated_at`

`events.jsonl` records provider events:

- `time`
- `node_key`
- `event_type`
- `agent_id`
- `provider`
- `context_id`
- `run_id`
- `payload`

These files are trace/debug artifacts. They are not a full recovery log for
provider agent objects.

## TUI Design

Implemented in `app.py` with Textual and Rich.

Layout:

```text
left workflow pane | center selected agent messages | right live agent list
bottom input line
```

Current behavior:

- The workflow pane highlights running or decision-needed nodes.
- The right pane lists live logical agents only.
- Selecting an agent changes the message history shown in the center pane.
- Prompt previews are rendered before model output.
- Completed model output is rendered as Markdown.
- Running model output stays plain text.
- Gate JSON decisions render as a special decision block instead of raw JSON.
- The status line above the input shows mode, pending step confirmation, or the running step.
- `Shift+Tab` toggles auto mode and step mode.
- `Ctrl+Y` copies current messages as plain text.
- `Ctrl+C` twice within 1.5 seconds cancels and exits.
- `q` cancels and exits.

The TUI is a view/controller over `WorkflowRunner`. Switching selected agents
must not affect scheduling.

## Permission and Approval Handling

Provider-level permission requests arrive as `EventType.NEEDS_DECISION`.

Flow:

```text
agent event -> _run_agent_turn -> decision_handler -> agent.resolve_request
```

Headless CLI:

- `--auto-approve` sets `decision_handler=approve_all`.
- Without a handler, a provider permission request raises
  `WorkflowNeedsDecision`.

Live TUI:

- The current MVP wires step confirmation through the input box.
- Provider approval UI is still minimal. The scheduler hook exists, but richer
  approval rendering can be added in `app.py`.

## Start From

`--start-from NODE_ID` is implemented in `_workflow_from_start()`.

Rules:

- Only top-level workflow node ids are supported.
- If the id is found at top level, earlier top-level nodes are skipped.
- If the id exists only inside a loop or parallel node, the runner raises
  `WorkflowError`.
- Context reuse is the caller's responsibility. The MVP does not reconstruct
  skipped agents from prior run-state files.

## Step Confirmation

`--step-confirm` pauses before each step.

CLI:

- `confirm_step_cli(plan)` asks `start step: {node_key}. Run? [Y/n]`.
- `n`, `no`, `stop`, `cancel`, `q`, `quit` deny.

TUI:

- `OrchestratorTextualApp.confirm_step(plan)` stores a pending `StepPlan`.
- The input box accepts Enter/yes to run, or no/stop/cancel/q/quit to deny.
- Denial raises `WorkflowError("workflow stopped before step: ...")`.

Step confirmation is deliberately pre-run. Post-step confirmation was removed
because confirming after the final step has no useful scheduling effect.

## Output Validation

Only `outputs.required_files` is currently implemented.

After a step completes, `_check_outputs(expanded_node)` verifies every listed
file exists. Missing files raise `WorkflowOutputError`.

There is no automatic repair loop yet. If needed, add it after `_check_outputs`
and before writing final node status, but keep it explicit and tested.

## Cancellation

`WorkflowRunner.cancel()`:

1. Sets `cancel_requested = True`.
2. Iterates active agents.
3. Calls `agent.interrupt("workflow cancelled")`.
4. Marks their TUI state as `cancelled`.

The runner checks `_raise_if_cancelled()` before scheduling nodes and while
streaming provider events.

## Known Sharp Edges

- `run_when.expression` uses `eval` with empty builtins. It is still code-like;
  keep workflow files trusted and expressions simple.
- `fork_from_step` uses `step_results[node_id]`, not full `node_key`.
  Duplicate step ids can cause the last completed one to win.
- `--start-from` does not support nested loop or parallel nodes.
- `collect_failures` marks the parallel node completed with failures, but the
  top-level run can still complete if callers do not inspect the node state.
- The TUI displays only live agents in the right pane. Historical agents from a
  killed process are not reloadable.
- Provider approvals beyond auto-approve need more UI work.
- The MVP does not persist `workflow.yaml` into run directories.

## Debugging Checklist

When a workflow behaves incorrectly:

1. Check `run_state.json` for overall status, variables, and `start_from`.
2. Check `nodes/*/prompt.md` to verify the exact prompt sent to the agent.
3. Check `nodes/*/node_state.json` for `agent_id`, `context_id`, status, and
   missing error text.
4. Check `events.jsonl` for provider event sequence.
5. If a loop did not exit, inspect the gate step output and decision parser.
6. If a fork failed, verify the source step completed in the same process and
   used the same provider.
7. If queued input is missing, check whether the target agent was active or
   already idle when `send_input` was called.

## Test Coverage

Run:

```bash
python3 -m pytest tests/orchestrator -q
```

Current tests cover:

- YAML loading
- step state and context recording
- queued input during active runs
- input after workflow completion
- prompt/output rendering helpers
- reused agent turn boundaries
- live agent list filtering
- Markdown rendering for completed output
- decision block rendering
- loop gate continue/exit
- parallel fan-out
- context fork from step and agent
- required-file validation
- provider approval auto handling
- start-from top-level behavior
- step confirmation before step start
- cancellation of active agents

Add tests before changing scheduler semantics. Most bugs are easier to catch at
the scheduler level without running the Textual app.

## Company 02-06 Mapping

The intended company-analysis mapping is:

```text
StepNode kickoff
LoopNode research-loop
  StepNode plan round
  ParallelNode collection fan-out
  StepNode update model / draft / review
  StepNode quality gate
StepNode final report or handoff
```

The quality gate remains an agent call. It should output JSON with
`decision: continue` or `decision: exit`. Domain artifacts such as
`00-loop_state.md`, `quality_gate.md`, `next_round_plan.md`, `report_draft.md`,
`scenario_model.md`, and `calc_table.csv` are still source-of-truth files in
the company-analysis tree. The orchestrator should coordinate their production
and validation, not replace them with chat history.
