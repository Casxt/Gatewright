# Gatewright — Examples

Runnable example workflows organized by what they demonstrate. Most use the
`codex` provider; one uses `claude`; a third mixes both. Every example is
self-contained — its prompts live next to the YAML.

## Running an example

```bash
# Live TUI (recommended for the error-handling examples — open pause/error
# messages are easiest to handle in the TUI).
python -m gatewright.orchestrator.runner \
  --live --auto-approve \
  examples/01-basic/codex-single-step.yaml

# Plain stdout (good for CI / piping):
python -m gatewright.orchestrator.runner \
  --no-live \
  examples/01-basic/codex-single-step.yaml
```

Pass `--var KEY=VALUE` after a `--` to override workflow variables, e.g.
`-- --var round=2`.

## Directory layout

| Path                      | Purpose                                                          |
| ------------------------- | ---------------------------------------------------------------- |
| `01-basic/`               | Single step, linear pipeline. Start here.                        |
| `02-loops-and-parallel/`  | Iterative loops with JSON gates; parallel fanout with fork.      |
| `03-multi-provider/`      | Mixing `codex` + `claude` in one workflow.                       |
| `04-error-handling/`      | Pause/error messages, missing inputs, mid-loop resume.           |
| `05-human-in-the-loop/`   | Agent waits for user (external action OR step-confirm review).   |
| `_testing/`               | Mock-only fixtures used by `tests/`. Not interesting to read.    |
| `_archive/`               | Historical prototypes, kept for reference. Do not run.           |

## Catalog

### `01-basic/codex-single-step.yaml`
One step, one required output. The minimum viable Codex workflow. Use to
verify your `codex` provider is wired up.

### `01-basic/codex-linear-pipeline.yaml`
Two-step linear pipeline. Step 1 writes a draft; step 2 declares the draft
as an `input_file`, validates it exists, and writes the final version.
Shows reuse-mode context across steps (single Codex thread spans both
turns).

### `01-basic/claude-single-step.yaml`
Same shape as `codex-single-step.yaml` but with `provider: claude`. Use to
smoke-test the Claude backend.

### `02-loops-and-parallel/codex-gated-loop.yaml`
Iterative refinement loop. Each round writes a draft; a JSON gate decides
`continue` or `exit`. Demonstrates `{round_dir}` lazy expansion (re-resolved
each iteration) and gate decision parsing.

### `02-loops-and-parallel/codex-parallel-fanout.yaml`
Coordinator → 3 parallel workers (forked from coordinator's thread) →
synthesis. Demonstrates `parallel` node with `fail_fast` and
`fork_from_step` context.

### `03-multi-provider/codex-and-claude-pipeline.yaml`
Step 1: Codex generates a Python module. Step 2: Claude (separate thread)
reviews it. Useful pattern when you want each provider doing what it is
best at.

### `04-error-handling/codex-recoverable-failure.yaml`
The agent intentionally aborts halfway through writing required outputs.
The TUI appends an open pause/error message to that agent's message list.
Reply in the input box to continue the same Codex thread, or type `a` /
`abort` to stop and resume later.

### `04-error-handling/codex-missing-input.yaml`
Declares an `input_files` path that does not exist on disk. The orchestrator
raises `FileNotFoundError` before invoking Codex. Verifies the early-fail
path produces a clean error surface (no LLM turn wasted, error appended to
the failing step's agent message list).

### `05-human-in-the-loop/codex-wait-for-user-confirmation.yaml`
The agent itself pauses mid-step until the user completes an external
action (scan a QR code, solve a captcha, sign in elsewhere). Same pattern
the `xueqiu-opinion-collection` skill uses for WAF / login.

Mechanism: the agent ends its turn without writing the step's required
output files. The scheduler appends an open pause message to the same agent.
The user does the external action, types a short instruction such as
`done, continue`, and the SAME codex thread resumes (its context is intact)
and finishes the step. Typing `a` / `abort` stops the workflow.

```bash
python -m gatewright.orchestrator.runner \
  --live --auto-approve \
  examples/05-human-in-the-loop/codex-wait-for-user-confirmation.yaml
# In another terminal, once the TUI shows the pause message:
#   touch /tmp/tui-flow-confirm/session.token
# Then back in the TUI, type: done, continue
```

### `05-human-in-the-loop/codex-plan-then-execute.yaml`
Plan → review → execute → review → verify pipeline for a small Codex coding
task. The workflow does not pause automatically — pausing is opted-in with
`--step-confirm` (or by toggling step mode in the TUI with `Shift+Tab`).
Between steps the TUI prompts `start step <name>? Enter/yes to run, no to
stop`, giving you a chance to `cat` the prior step's artifact before
authorizing the next turn.

```bash
python -m gatewright.orchestrator.runner \
  --live --auto-approve --step-confirm \
  examples/05-human-in-the-loop/codex-plan-then-execute.yaml
```

Useful when you want to keep Codex on a leash for sensitive operations
(refactors, migrations, anything destructive).

### `04-error-handling/codex-resume-mid-loop.yaml`
Three-step loop body. Reference for the nested resume form:

```bash
runner.py codex-resume-mid-loop.yaml \
  --start-from refine-loop/summarize -- --var round=0
```

Skips `prepare` and `analyze` on the entry iteration, runs `summarize` and
the gate. Subsequent rounds run the full body.

## Feature coverage matrix

| Example                              | step | loop | parallel | input_files | output_files | gate JSON | run_when | fork | recovery | step_confirm |
| ------------------------------------ | :--: | :--: | :------: | :---------: | :----------: | :-------: | :------: | :--: | :------: | :----------: |
| codex-single-step                    | ✓   |      |          |             | ✓           |           |          |      |          |              |
| codex-linear-pipeline                | ✓   |      |          | ✓          | ✓           |           |          |      |          |              |
| claude-single-step                   | ✓   |      |          |             | ✓           |           |          |      |          |              |
| codex-gated-loop                     | ✓   | ✓   |          | ✓          | ✓           | ✓        |          |      |          |              |
| codex-parallel-fanout                | ✓   |      | ✓       | ✓          | ✓           |           |          | ✓   |          |              |
| codex-and-claude-pipeline            | ✓   |      |          | ✓          | ✓           |           |          |      |          |              |
| codex-recoverable-failure            | ✓   |      |          |             | ✓           |           |          |      | ✓       |              |
| codex-missing-input                  | ✓   |      |          | ✓          | ✓           |           |          |      |          |              |
| codex-resume-mid-loop                | ✓   | ✓   |          | ✓          | ✓           | ✓        |          |      |          |              |
| codex-plan-then-execute              | ✓   |      |          | ✓          | ✓           |           |          |      |          | ✓           |
| codex-wait-for-user-confirmation     | ✓   |      |          |             | ✓           |           |          |      | ✓       |              |
| _testing/complex-feature-flow        | ✓   | ✓   | ✓       | ✓          | ✓           | ✓        | ✓       | ✓   |          |              |

## Notes

- Examples that need to write files use `/tmp/tui-flow-...` so they do not
  pollute the repo. Clean up with `rm -rf /tmp/tui-flow-*` between runs if
  you want a fresh slate.
- The `_testing/complex-feature-flow.yaml` is the most feature-dense
  example — it covers every feature in the matrix — but it depends on
  pre-staged fixture inputs and the mock backend, so it is for unit-test
  use only, not as a runnable demo.
