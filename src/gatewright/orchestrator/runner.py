from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .app import OrchestratorTextualApp
from .scheduler import WorkflowRunner, approve_all
from .tui import InteractionMessage
from .workflow import StepPlan
from .workflow import load_workflow


async def confirm_step_cli(plan: StepPlan) -> bool:
    prompt = f"start step: {plan.node_key}. Run? [Y/n] "
    try:
        answer = await asyncio.to_thread(input, prompt)
    except EOFError:
        return False
    return answer.strip().lower() not in {"n", "no", "stop", "cancel", "q", "quit"}


async def interaction_cli(message: InteractionMessage) -> str:
    if message.input_mode == "confirm":
        print(message.text)
        try:
            return await asyncio.to_thread(input, "Continue after this loop? [y/N] ")
        except EOFError:
            return "abort"
    return "abort"


async def run_cli(args: argparse.Namespace) -> None:
    spec = load_workflow(Path(args.workflow))
    variables = dict(item.split("=", 1) for item in args.var)
    live_enabled = args.live and sys.stdout.isatty()

    runner = WorkflowRunner(
        spec,
        variables=variables,
        workspace=Path(args.workspace).resolve(),
        run_dir=Path(args.run_dir).resolve() if args.run_dir else None,
        decision_handler=approve_all if args.auto_approve else None,
        start_from=args.start_from,
        step_confirm_handler=confirm_step_cli if args.step_confirm and not live_enabled else None,
        step_interaction_handler=interaction_cli if not live_enabled else None,
    )
    if live_enabled:
        await OrchestratorTextualApp(runner, workflow_name=spec.name, step_confirm=args.step_confirm).run_async()
        return

    results = await runner.run()
    print(f"run_id: {runner.run_id}")
    print(f"run_dir: {runner.store.run_dir}")
    for result in results:
        print(f"{result.status}: {result.node_key} agent={result.agent_id} context={result.context_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Gatewright loop-gated agent workflow.")
    parser.add_argument("workflow")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--run-dir")
    parser.add_argument("--auto-approve", action="store_true")
    parser.add_argument("--live", action="store_true", help="Render a minimal live terminal dashboard.")
    parser.add_argument("--start-from", help="Start from a top-level workflow node id. Nested loop/parallel nodes are not supported yet.")
    parser.add_argument("--step-confirm", action="store_true", help="Start in step mode. Default is auto mode; live TUI can toggle with Shift+Tab.")
    parser.add_argument("--var", action="append", default=[], help="Workflow variable, KEY=VALUE")
    asyncio.run(run_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
