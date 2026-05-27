# Gatewright

Loop-gated orchestration for live agent threads.

Gatewright runs YAML-defined agent workflows where a loop body is checked by
an explicit gate before the workflow continues, exits, retries, or forks. The
runtime layer wraps live Codex, Claude, or mock provider threads; the
orchestrator owns workflow state, loop scheduling, output checks, human
approval, and the optional terminal UI.

## Layout

```text
src/gatewright/runtime/       provider-neutral live agent runtime
src/gatewright/orchestrator/  YAML workflow runner, loop/gate scheduler, TUI
examples/                    runnable workflow examples
tests/                       unit tests for runtime and orchestrator behavior
```

## Environment Requirements

Gatewright is an orchestration layer over local provider agents. Installing the
Python SDK packages is not enough for real `codex` or `claude` workflows: the
machine must also have working, logged-in, subscribed provider commands.

Required baseline:

```bash
python --version   # Python 3.11+
```

Install the Python packages Gatewright needs at runtime:

```bash
python -m pip install PyYAML textual
```

Use the repo-local entrypoint while the project is still early:

```bash
./gatewright.sh --help
```

For Codex workflows:

```bash
codex --version
```

The `codex` command must be usable in the current shell, logged in, and backed
by an active Codex subscription.

Install the Codex Python SDK from the matching local Codex source checkout:

```bash
CODEX_VERSION="$(codex --version | awk '{print $2}')"
CODEX_TAG="rust-v${CODEX_VERSION}"

git clone https://github.com/openai/codex.git codex_src \
  --branch "${CODEX_TAG}" \
  --depth 1

python -m pip install -e codex_src/sdk/python
```

For Claude Code workflows:

```bash
claude --version   # or: cc --version, if your local alias is cc
```

The Claude Code command must be usable in the current shell, logged in, and
backed by an active Claude Code subscription. Gatewright's Claude backend uses
Anthropic's official Claude Agent SDK package:

```bash
python -m pip install claude-agent-sdk
```

After installing provider SDKs, verify imports:

```bash
python -c "import codex_app_server; print('codex sdk ok')"
python -c "import claude_agent_sdk; print('claude sdk ok')"
```

Mock workflows do not require Codex or Claude credentials.

## Run Tests

Tests also need `pytest`:

```bash
python -m pip install pytest
```

```bash
PYTHONPATH=src python -m pytest -q
```

## Run A Mock Loop

```bash
./gatewright.sh examples/_testing/mock-gated-loop.yaml --workspace . --auto-approve
```

Detailed design notes live in [docs/runtime-design.md](docs/runtime-design.md)
and [docs/orchestrator-design.md](docs/orchestrator-design.md).

## Core Idea

```text
loop body -> gate -> continue | exit | retry | fork
```

The project is deliberately file-backed: prompts, node states, run states, and
required output checks live on disk so long workflows do not depend on a single
chat context staying small forever.
