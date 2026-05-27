from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from gatewright.runtime import (
    Agent,
    AgentNotRunningError,
    AgentEvent,
    DecisionAction,
    EventType,
    RunRequest,
    UserDecision,
)

from .state import RunStateStore
from .tui import AgentViewState, InteractionMessage
from .workflow import StepPlan, StepResult, WorkflowSpec, expand


DecisionHandler = Callable[[Agent, AgentEvent], Awaitable[UserDecision]]
EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
StepConfirmHandler = Callable[[StepPlan], Awaitable[bool]]
# Returns the user's reply for an open pause/error message. Normal text is
# sent to the same live agent; a/abort stops the workflow.
StepInteractionHandler = Callable[[InteractionMessage], Awaitable[str]]
# Deprecated compatibility hook for older callers.
StepRecoveryHandler = Callable[[StepPlan, BaseException], Awaitable[str]]


OUTPUT_TAIL_LIMIT = 400_000


DEFAULT_CONTINUE_PROMPT = (
    "Your previous turn ended without producing the required output files. "
    "Continue from the same context, re-check what is now available, and "
    "finish writing every required output file for this step."
)


class WorkflowError(RuntimeError):
    pass


class WorkflowNeedsDecision(WorkflowError):
    pass


class WorkflowOutputError(WorkflowError):
    pass


class WorkflowRunner:
    def __init__(
        self,
        spec: WorkflowSpec,
        *,
        variables: dict[str, Any] | None = None,
        workspace: Path | None = None,
        run_dir: Path | None = None,
        decision_handler: DecisionHandler | None = None,
        event_handler: EventHandler | None = None,
        start_from: str | None = None,
        step_confirm_handler: StepConfirmHandler | None = None,
        step_interaction_handler: StepInteractionHandler | None = None,
        step_recovery_handler: StepRecoveryHandler | None = None,
    ) -> None:
        self.spec = spec
        # Keep spec.variables as raw templates (some depend on {round} and
        # change per loop iteration). `expand()` is multi-pass so derived
        # paths like round_dir = "{company_dir}/round-{round}" resolve
        # lazily at node-expansion time, automatically reflecting whatever
        # `round` currently is. Eagerly resolving here would freeze them at
        # the entry round.
        self.variables: dict[str, Any] = {**spec.variables, **(variables or {})}
        default_workspace = spec.runtime.get("default_workspace")
        self.workspace = Path(workspace or default_workspace or Path.cwd()).resolve()
        trace_root = Path(spec.runtime.get("trace_root", ".gatewright/runs"))
        if not trace_root.is_absolute():
            trace_root = self.workspace / trace_root
        self.run_id = (run_dir or trace_root / f"run-{uuid4().hex[:12]}").name
        self.store = RunStateStore(Path(run_dir) if run_dir else trace_root / self.run_id)
        self.decision_handler = decision_handler
        self.event_handler = event_handler
        self.start_from = start_from
        # When start_from is "loop_id/step_id", the loop must skip body steps
        # before step_id on its FIRST iteration. Subsequent iterations run the
        # full body. Plain top-level start_from leaves these as None.
        self._start_loop_id: str | None = None
        self._start_step_in_loop: str | None = None
        self.step_confirm_handler = step_confirm_handler
        self.step_interaction_handler = step_interaction_handler
        # Deprecated compatibility hook for older tests / callers.
        self.step_recovery_handler = step_recovery_handler
        # Filled when a step stops WITHOUT recovery so the TUI / CLI can
        # display a precise resume command. Cleared on each new run.
        self.resume_hint: str | None = None
        # Per-agent record of the most recent step that stopped without
        # resolution. Used to auto-detect "operator typed input post-abort,
        # the agent wrote the missing files" recoveries and update node
        # state accordingly — otherwise the workflow stays "stopped" even
        # after the user effectively unstuck the step via the input box.
        # Cleared as soon as a recovery (open message OR operator-input) succeeds.
        self._last_failed_step: dict[str, tuple[str, str, dict[str, Any]]] = {}
        self.agent_registry: dict[str, Agent] = {}
        self.agent_states: dict[str, AgentViewState] = {}
        self.active_agents: dict[str, Agent] = {}
        self.queued_inputs: dict[str, list[str]] = {}
        self.cancel_requested = False
        self.step_results: dict[str, StepResult] = {}
        self.step_results_by_key: dict[str, StepResult] = {}
        self._step_confirm_lock = asyncio.Lock()

    def queue_input(self, agent_id: str, text: str) -> None:
        self.queued_inputs.setdefault(agent_id, []).append(text)
        state = self.agent_states.get(agent_id)
        if state is not None:
            state.queued_inputs += 1
            state.queued_input_tail.append(text)

    async def cancel_queued_input(self, agent_id: str) -> int:
        queued = self.queued_inputs.pop(agent_id, [])
        state = self.agent_states.get(agent_id)
        count = len(queued)
        if state is not None:
            count = max(count, state.queued_inputs, len(state.queued_input_tail))
            await self._set_agent_state(
                agent_id,
                queued_inputs=0,
                clear_queued_input_tail=True,
            )
        return count

    async def send_input(self, agent_id: str, text: str) -> str:
        if agent_id in self.active_agents:
            self.queue_input(agent_id, text)
            await self._emit_event({"kind": "agent_state", "agent": self.agent_states.get(agent_id)})
            return "queued"
        try:
            agent = self.agent_registry[agent_id]
        except KeyError as exc:
            raise WorkflowError(f"unknown live agent: {agent_id}") from exc

        node_key = f"operator/{agent_id}"
        await self._set_agent_state(
            agent_id,
            provider=agent.provider,
            current_node=node_key,
            status="running",
            context_id=agent.context_id,
        )
        self.active_agents[agent_id] = agent
        output: list[str] = []
        status = "completed"
        error: str | None = None
        try:
            status, error = await self._run_agent_turn(agent, agent_id, node_key, text, output, display_label="input")
            while status == "completed" and self.queued_inputs.get(agent_id):
                queued = self.queued_inputs.pop(agent_id)
                await self._set_agent_state(agent_id, status="running", queued_inputs=0, clear_queued_input_tail=True)
                follow_up = "\n\n".join(["User follow-up input after the previous turn stopped:", *queued])
                status, error = await self._run_agent_turn(
                    agent,
                    agent_id,
                    node_key,
                    follow_up,
                    output,
                    display_label="input",
                )
        finally:
            self.active_agents.pop(agent_id, None)

        if status == "failed":
            await self._set_agent_state(agent_id, status="failed", context_id=agent.context_id)
            raise WorkflowError(error or f"agent input failed: {agent_id}")
        # If this agent has a step that previously stopped, check whether
        # the operator-driven turn just produced the missing required
        # outputs. If yes, promote the step in node_state.json and surface
        # a [recovered] marker so the user sees that the prior [error]
        # block has been resolved (audit-history-friendly: we don't delete
        # the error, we annotate that it was addressed).
        await self._auto_resolve_failed_step_if_outputs_present(agent_id)
        await self._set_agent_state(agent_id, status="idle", current_node=node_key, context_id=agent.context_id)
        return "sent"

    async def _auto_resolve_failed_step_if_outputs_present(self, agent_id: str) -> None:
        last = self._last_failed_step.get(agent_id)
        if last is None:
            return
        failed_node_id, failed_node_key, expanded_node = last
        try:
            self._check_outputs(expanded_node)
        except Exception:
            # Outputs still missing — nothing changed, leave the failure
            # record in place so a future operator turn can still resolve it.
            return
        self._last_failed_step.pop(agent_id, None)
        agent = self.agent_registry.get(agent_id)
        await self._append_agent_output(
            agent_id,
            f"\n[recovered] step {failed_node_key} ✓ outputs now present "
            f"(via operator input) — node state promoted to completed\n",
        )
        self.store.write_node_state(
            failed_node_key,
            {
                "node_id": failed_node_id,
                "node_key": failed_node_key,
                "status": "completed",
                "agent_id": agent_id,
                "provider": getattr(agent, "provider", None) if agent else None,
                "context_id": getattr(agent, "context_id", None) if agent else None,
                "round": self.variables.get("round"),
                "recovered": "operator_input",
            },
        )

    async def cancel(self) -> None:
        self.cancel_requested = True
        for agent_id, agent in list(self.active_agents.items()):
            try:
                await agent.interrupt("workflow cancelled")
            except AgentNotRunningError:
                pass
            await self._set_agent_state(agent_id, status="cancelled")

    async def run(self) -> list[StepResult]:
        self.resume_hint = None
        self.store.write_run_state(
            {
                "run_id": self.run_id,
                "workflow": self.spec.name,
                "status": "running",
                "variables": self.variables,
                "start_from": self.start_from,
            }
        )
        results: list[StepResult] = []
        try:
            workflow = self._workflow_from_start()
            for node in workflow:
                self._raise_if_cancelled()
                result = await self._run_node(node, parent_key="")
                if isinstance(result, list):
                    results.extend(result)
                elif result is not None:
                    results.append(result)
            self.store.write_run_state(
                {
                    "run_id": self.run_id,
                    "workflow": self.spec.name,
                    "status": "completed",
                    "variables": self.variables,
                    "start_from": self.start_from,
                }
            )
            return results
        except Exception:
            self.store.write_run_state(
                {
                    "run_id": self.run_id,
                    "workflow": self.spec.name,
                    "status": "failed",
                    "variables": self.variables,
                    "start_from": self.start_from,
                }
            )
            raise

    def _workflow_from_start(self) -> list[dict[str, Any]]:
        if not self.start_from:
            return self.spec.workflow
        # Nested form: "<loop_id>/<step_id>" — enter the loop and skip body
        # steps before <step_id> on the first iteration. The loop's gate
        # still runs. Subsequent iterations run the full body.
        if "/" in self.start_from:
            loop_id, _, step_id = self.start_from.partition("/")
            for index, node in enumerate(self.spec.workflow):
                if str(node.get("id")) != loop_id:
                    continue
                if node.get("type") != "loop":
                    raise WorkflowError(
                        f"--start-from {self.start_from!r}: {loop_id!r} is not a loop node"
                    )
                body_ids = [str(child.get("id")) for child in node.get("body", [])]
                gate = node.get("gate") or {}
                if step_id not in body_ids and str(gate.get("id")) != step_id:
                    raise WorkflowError(
                        f"--start-from {self.start_from!r}: unknown step {step_id!r} in loop {loop_id!r}"
                    )
                self._start_loop_id = loop_id
                self._start_step_in_loop = step_id
                return self.spec.workflow[index:]
            raise WorkflowError(f"unknown loop node: {loop_id}")
        for index, node in enumerate(self.spec.workflow):
            if str(node.get("id")) == self.start_from:
                return self.spec.workflow[index:]
        if self._contains_nested_node_id(self.spec.workflow, self.start_from):
            raise WorkflowError(
                "--start-from for a nested step requires the explicit "
                "'<loop_id>/<step_id>' form (e.g. company-research-loop/conclude-and-plan)"
            )
        raise WorkflowError(f"unknown top-level start node: {self.start_from}")

    def _contains_nested_node_id(self, nodes: list[dict[str, Any]], node_id: str) -> bool:
        for node in nodes:
            children: list[dict[str, Any]] = []
            if node.get("type") == "loop":
                children.extend(list(node.get("body", [])))
                gate = node.get("gate")
                if isinstance(gate, dict):
                    children.append(gate)
            elif node.get("type") == "parallel":
                children.extend(list(node.get("children", [])))
            if any(str(child.get("id")) == node_id for child in children):
                return True
            if children and self._contains_nested_node_id(children, node_id):
                return True
        return False

    async def _run_node(self, node: dict[str, Any], parent_key: str) -> StepResult | list[StepResult] | None:
        self._raise_if_cancelled()
        if not self._should_run(node):
            return None
        node_type = node.get("type", "step")
        if node_type == "step":
            return await self._run_step(node, parent_key)
        if node_type == "loop":
            return await self._run_loop(node, parent_key)
        if node_type == "parallel":
            return await self._run_parallel(node, parent_key)
        raise WorkflowError(f"unknown node type: {node_type}")

    async def _run_step(self, node: dict[str, Any], parent_key: str) -> StepResult:
        node_id = str(node["id"])
        node_key = f"{parent_key}/{node_id}".strip("/")
        expanded_node = expand(node, self.variables)
        agent_id = self._agent_id(expanded_node)
        agent_spec_name = str(expanded_node["agent"])
        provider = self.spec.agents[agent_spec_name].provider if agent_spec_name in self.spec.agents else None

        # Confirm step BEFORE creating any agent state — declining the step
        # should leave runner.agent_states untouched (a test contract).
        await self._confirm_step(StepPlan(node_id=node_id, node_key=node_key, agent_id=agent_id, provider=provider))

        # Now create a minimal agent state up front so any error in remaining
        # pre-turn setup (select / render_prompt) lands as a visible message
        # instead of crashing the TUI before the agent surface exists.
        await self._set_agent_state(
            agent_id,
            provider=provider,
            current_node=node_key,
            status="running",
        )

        agent: Any = None
        try:
            agent = await self._select_agent(expanded_node)
            prompt = self._render_prompt(expanded_node)
            self.store.write_prompt(node_key, prompt)
            self.store.write_node_state(
                node_key,
                {
                    "node_id": node_id,
                    "node_key": node_key,
                    "status": "running",
                    "agent_id": agent_id,
                    "provider": agent.provider,
                    "context_id": agent.context_id,
                    "round": self.variables.get("round"),
                },
            )
            await self._set_agent_state(
                agent_id,
                provider=agent.provider,
                current_node=node_key,
                status="running",
                context_id=agent.context_id,
            )
            self.active_agents[agent_id] = agent

            output: list[str] = []
            status = "completed"
            error: str | None = None
            try:
                status, error = await self._run_agent_turn(agent, agent_id, node_key, prompt, output)
                while status == "completed" and self.queued_inputs.get(agent_id):
                    queued = self.queued_inputs.pop(agent_id)
                    await self._set_agent_state(agent_id, status="running", queued_inputs=0, clear_queued_input_tail=True)
                    follow_up = "\n\n".join(["User follow-up input after the previous turn stopped:", *queued])
                    status, error = await self._run_agent_turn(agent, agent_id, node_key, follow_up, output)
            finally:
                self.active_agents.pop(agent_id, None)

            result = StepResult(
                node_id=node_id,
                node_key=node_key,
                status=status,
                agent_id=agent_id,
                provider=agent.provider,
                context_id=agent.context_id,
                output_text="".join(output),
                error=error,
            )
            self._remember_result(result)
            # Only run the output-file check on success. When the agent turn
            # itself failed (e.g. Codex contextWindowExceeded), the files it
            # was supposed to write are guaranteed missing — running the check
            # would mask the real root cause with a misleading
            # "missing required files" error.
            if status == "completed":
                self._check_outputs(expanded_node)
            self.store.write_node_state(
                node_key,
                {
                    "node_id": node_id,
                    "node_key": node_key,
                    "status": result.status,
                    "agent_id": result.agent_id,
                    "provider": result.provider,
                    "context_id": result.context_id,
                    "round": self.variables.get("round"),
                    "error": result.error,
                },
            )
            if status == "failed":
                await self._set_agent_state(agent_id, status="failed", context_id=agent.context_id)
                raise WorkflowError(error or f"step failed: {node_id}")
            await self._set_agent_state(agent_id, status="idle", current_node=node_key, context_id=agent.context_id)
            return result
        except Exception as exc:
            # Don't overwrite intentional cancellation state — cancel() already
            # set agent status to "cancelled" and the test contract relies on
            # it. Detect by message; the only signal we have at this layer.
            if isinstance(exc, WorkflowError) and "workflow cancelled" in str(exc):
                self.active_agents.pop(agent_id, None)
                raise
            # Surface the error as a message on the failed step's agent rather
            # than letting it propagate and blank the TUI.
            await self._record_step_error(node_id, node_key, agent_id, agent, exc)
            # Open a pause/error interaction as a visible agent message. The
            # input box response is normally sent straight back to this same
            # agent. Only a small abort vocabulary stops the workflow.
            if agent is not None and (
                self.step_interaction_handler is not None
                or self.step_recovery_handler is not None
            ):
                plan = StepPlan(
                    node_id=node_id,
                    node_key=node_key,
                    agent_id=agent_id,
                    provider=getattr(agent, "provider", None),
                )
                choice = await self._await_step_interaction(plan, exc)
                if not self._is_abort_input(choice):
                    prompt = choice.strip() or DEFAULT_CONTINUE_PROMPT
                    recovered = await self._continue_failed_step(
                        node_id,
                        node_key,
                        agent_id,
                        agent,
                        expanded_node,
                        prompt,
                    )
                    if recovered is not None:
                        return recovered
            # Remember this step's identity so a post-stop operator-input
            # turn can auto-resolve it when the missing outputs eventually
            # land on disk. Without this the workflow stays "stopped" even
            # after the user effectively unstuck the agent.
            self._last_failed_step[agent_id] = (node_id, node_key, expanded_node)
            self.resume_hint = self._build_resume_hint(node_key)
            raise

    async def _continue_failed_step(
        self,
        node_id: str,
        node_key: str,
        agent_id: str,
        agent: Any,
        expanded_node: dict[str, Any],
        prompt: str,
    ) -> StepResult | None:
        """Try to finish a failed step using the operator's message as the
        next prompt to the same live agent."""
        if agent is None or agent_id not in self.agent_registry:
            return None
        await self._set_agent_state(
            agent_id,
            provider=getattr(agent, "provider", None),
            current_node=node_key,
            status="running",
            context_id=getattr(agent, "context_id", None),
        )
        self.active_agents[agent_id] = agent
        output: list[str] = []
        try:
            status, error = await self._run_agent_turn(
                agent,
                agent_id,
                node_key,
                prompt,
                output,
                display_label="input",
            )
        finally:
            self.active_agents.pop(agent_id, None)
        if status != "completed":
            return None
        try:
            self._check_outputs(expanded_node)
        except Exception as check_exc:
            await self._append_agent_output(
                agent_id,
                f"\n[pause] {node_key}\nOutputs are still missing: {check_exc}\n"
                "Type another instruction below to continue, or a/abort to stop.\n",
            )
            return None
        result = StepResult(
            node_id=node_id,
            node_key=node_key,
            status="completed",
            agent_id=agent_id,
            provider=getattr(agent, "provider", None),
            context_id=getattr(agent, "context_id", None),
            output_text="".join(output),
            error=None,
        )
        self._remember_result(result)
        self.store.write_node_state(
            node_key,
            {
                "node_id": node_id,
                "node_key": node_key,
                "status": "completed",
                "agent_id": agent_id,
                "provider": getattr(agent, "provider", None),
                "context_id": getattr(agent, "context_id", None),
                "round": self.variables.get("round"),
                "recovered": True,
            },
        )
        # Clear bookkeeping for "this agent has a pending failed step that
        # operator-input can still resolve" — the open message input has just
        # resolved it.
        self._last_failed_step.pop(agent_id, None)
        # Make the success visible right after the prior [error] block so
        # the user does not have to mentally cross-reference the [error]
        # marker against the agent's idle state to know the situation has
        # been resolved.
        await self._append_agent_output(
            agent_id,
            f"\n[recovered] step {node_key} ✓ outputs now present — workflow continuing\n",
        )
        await self._set_agent_state(
            agent_id,
            status="idle",
            current_node=node_key,
            context_id=getattr(agent, "context_id", None),
        )
        return result

    async def _await_step_interaction(self, plan: StepPlan, exc: BaseException) -> str:
        message = await self._open_step_interaction(plan, exc)
        try:
            if self.step_interaction_handler is not None:
                result = await self.step_interaction_handler(message)
            elif self.step_recovery_handler is not None:
                legacy = await self.step_recovery_handler(plan, exc)
                result = DEFAULT_CONTINUE_PROMPT if str(legacy).lower() == "retry" else str(legacy)
            else:
                result = "abort"
        except Exception:
            result = "abort"
        message.result = result
        message.state = "aborted" if self._is_abort_input(result) else "closed"
        await self._emit_event({"kind": "interaction_state", "message": message})
        return result

    async def _open_step_interaction(self, plan: StepPlan, exc: BaseException) -> InteractionMessage:
        error_type = type(exc).__name__
        error_msg = str(exc) or repr(exc)
        text = (
            f"{plan.node_key} stopped before required outputs were verified.\n"
            f"{error_type}: {error_msg}\n\n"
            "Reply in the input box to continue this same agent thread. "
            "Type a or abort to stop the workflow."
        )
        message = InteractionMessage(
            id=f"interaction-{uuid4().hex[:12]}",
            agent_id=plan.agent_id or "",
            node_key=plan.node_key,
            kind="pause",
            text=text,
        )
        await self._append_interaction_message(message)
        return message

    def _is_abort_input(self, text: str) -> bool:
        return text.strip().lower() in {"a", "abort", "stop", "cancel", "q", "quit"}

    def _build_resume_hint(self, node_key: str) -> str:
        """Compose a one-line CLI snippet the user can copy to resume from
        the failed step. The TUI displays this on the workflow-stopped screen
        so the user does not have to compute round / step paths by hand."""
        parts = node_key.split("/")
        if len(parts) >= 3 and parts[1].startswith("round-"):
            loop_id, round_part, step_id = parts[0], parts[1], parts[-1]
            round_num = round_part.split("-", 1)[1]
            return (
                f"--start-from {loop_id}/{step_id} -- --var round={round_num}"
            )
        return f"--start-from {parts[0]}"

    async def _record_step_error(
        self,
        node_id: str,
        node_key: str,
        agent_id: str,
        agent: Any,
        exc: BaseException,
    ) -> None:
        """Append a human-readable error block to the agent's message list and
        mark the agent / node state as failed. Best-effort: never raises."""
        try:
            error_type = type(exc).__name__
            error_msg = str(exc) or repr(exc)
            block = (
                f"\n\n[error] {node_key}\n"
                f"{error_type}: {error_msg}\n"
            )
            # _append_agent_output requires the agent_state to exist; we created
            # it at the top of _run_step, so this should always succeed.
            if agent_id in self.agent_states:
                self.agent_states[agent_id].messages.append(
                    InteractionMessage(
                        id=f"error-{uuid4().hex[:12]}",
                        agent_id=agent_id,
                        node_key=node_key,
                        kind="error",
                        text=f"{error_type}: {error_msg}",
                        state="closed",
                        input_mode="none",
                    )
                )
                await self._append_agent_output(agent_id, block)
            context_id = getattr(agent, "context_id", None)
            await self._set_agent_state(
                agent_id,
                status="failed",
                context_id=context_id,
            )
            self.store.write_node_state(
                node_key,
                {
                    "node_id": node_id,
                    "node_key": node_key,
                    "status": "failed",
                    "agent_id": agent_id,
                    "provider": getattr(agent, "provider", None),
                    "context_id": context_id,
                    "round": self.variables.get("round"),
                    "error": f"{error_type}: {error_msg}",
                },
            )
            self.active_agents.pop(agent_id, None)
        except Exception:
            # Swallow secondary errors so the original exception still surfaces.
            pass

    async def _confirm_step(self, plan: StepPlan) -> None:
        if self.step_confirm_handler is None:
            return
        async with self._step_confirm_lock:
            accepted = await self.step_confirm_handler(plan)
        if not accepted:
            raise WorkflowError(f"workflow stopped before step: {plan.node_key}")

    async def _run_loop(self, node: dict[str, Any], parent_key: str) -> list[StepResult]:
        node_id = str(node["id"])
        node_key = f"{parent_key}/{node_id}".strip("/")
        round_var = str(node.get("round_variable", "round"))
        self.variables.setdefault(round_var, 0)
        # Coerce to int so run_when expressions like `round > 0` work even when the
        # value came in as a string via --var on the CLI.
        try:
            self.variables[round_var] = int(self.variables[round_var])
        except (TypeError, ValueError) as exc:
            raise WorkflowError(
                f"loop variable {round_var!r} must be an integer; got {self.variables[round_var]!r}"
            ) from exc
        results: list[StepResult] = []
        first_iteration = True
        while True:
            self._raise_if_cancelled()
            current_round = int(self.variables[round_var])
            self.variables[round_var] = current_round
            self.variables["previous_round"] = current_round - 1
            round_key = f"{node_key}/round-{current_round}"
            self.store.write_node_state(
                round_key,
                {
                    "node_id": node_id,
                    "node_key": round_key,
                    "status": "running",
                    "round": current_round,
                },
            )
            body = list(node.get("body", []))
            # Honor --start-from <loop_id>/<step_id>: on the first iteration
            # only, skip body steps before the requested step. The gate still
            # runs. Cleared after this iteration so subsequent rounds are
            # complete.
            skip_until: str | None = None
            if (
                first_iteration
                and self._start_loop_id == node_id
                and self._start_step_in_loop is not None
            ):
                skip_until = self._start_step_in_loop
            for child in body:
                self._raise_if_cancelled()
                if skip_until is not None:
                    if str(child.get("id")) != skip_until:
                        continue
                    skip_until = None
                child_result = await self._run_node(child, round_key)
                if isinstance(child_result, list):
                    results.extend(child_result)
                elif child_result is not None:
                    results.append(child_result)

            gate = dict(node["gate"])
            gate_result = await self._run_step(gate, round_key)
            results.append(gate_result)
            decision = self._parse_gate_decision(gate, gate_result.output_text)
            self.store.write_node_state(
                round_key,
                {
                    "node_id": node_id,
                    "node_key": round_key,
                    "status": "completed",
                    "round": current_round,
                    "decision": decision,
                },
            )
            if decision == "exit":
                break
            if decision != "continue":
                raise WorkflowError(f"unknown loop decision: {decision}")
            self.variables[round_var] = current_round + 1
            first_iteration = False
        return results

    async def _run_parallel(self, node: dict[str, Any], parent_key: str) -> list[StepResult]:
        node_id = str(node["id"])
        node_key = f"{parent_key}/{node_id}".strip("/")
        children = list(node.get("children", []))
        max_concurrency = int(node.get("max_concurrency", len(children) or 1))
        failure_policy = node.get("failure_policy", "fail_fast")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_child(child: dict[str, Any]) -> StepResult | list[StepResult] | Exception | None:
            async with semaphore:
                self._raise_if_cancelled()
                try:
                    return await self._run_node(child, node_key)
                except Exception as exc:
                    if failure_policy == "collect_failures":
                        return exc
                    raise

        raw_results = await asyncio.gather(*(run_child(child) for child in children))
        results: list[StepResult] = []
        failures: list[str] = []
        for item in raw_results:
            if isinstance(item, Exception):
                failures.append(str(item))
            elif isinstance(item, list):
                results.extend(item)
            elif item is not None:
                results.append(item)
        self.store.write_node_state(
            node_key,
            {
                "node_id": node_id,
                "node_key": node_key,
                "status": "completed_with_failures" if failures else "completed",
                "failures": failures,
            },
        )
        return results

    async def _run_agent_turn(
        self,
        agent: Agent,
        agent_id: str,
        node_key: str,
        prompt: str,
        output: list[str],
        *,
        display_label: str = "step",
    ) -> tuple[str, str | None]:
        await self._prepare_turn_display(agent_id, node_key, output, prompt, label=display_label)
        status = "completed"
        error: str | None = None
        # Reasoning chunks stream in, mixed with regular text. We wrap a
        # contiguous run of reasoning chunks with `[think]` / `[/think]`
        # markers so the TUI can render them visually distinct from the
        # final answer, without having to keep its own state machine.
        in_reasoning = False

        async def close_reasoning_if_open() -> None:
            nonlocal in_reasoning
            if in_reasoning:
                await self._append_agent_output(agent_id, "\n[/think]\n")
                in_reasoning = False

        async for event in agent.run(RunRequest(prompt=prompt)):
            self._raise_if_cancelled()
            await self._record_agent_event(node_key, agent_id, event)
            if event.type == EventType.TEXT:
                text = str(event.payload.get("text", ""))
                kind = event.payload.get("kind")
                if kind == "reasoning":
                    if not in_reasoning:
                        await self._append_agent_output(agent_id, "\n[think]\n")
                        in_reasoning = True
                    # Reasoning is for display only — do NOT add to `output`
                    # (which becomes StepResult.output_text and feeds gate
                    # decisions / downstream steps).
                    await self._append_agent_output(agent_id, text)
                else:
                    await close_reasoning_if_open()
                    output.append(text)
                    await self._append_agent_output(agent_id, text)
            elif event.type == EventType.TOOL:
                await close_reasoning_if_open()
                # Tool calls were previously only recorded to events.jsonl.
                # Surface a one-line summary so the operator can see what the
                # agent is doing without having to open the trace.
                summary = _format_tool_event(event.payload)
                if summary:
                    await self._append_agent_output(agent_id, f"\n{summary}\n")
            elif event.type == EventType.NEEDS_DECISION:
                await close_reasoning_if_open()
                await self._set_agent_state(agent_id, status="needs_decision", pending_delta=1)
                if self.decision_handler is None:
                    status = "needs_decision"
                    error = "agent needs a user decision"
                    raise WorkflowNeedsDecision(error)
                await agent.resolve_request(await self.decision_handler(agent, event))
                await self._set_agent_state(agent_id, status="running", pending_delta=-1)
            elif event.type == EventType.FAILED:
                await close_reasoning_if_open()
                status = "failed"
                error = str(event.payload.get("message") or event.payload.get("reason") or "agent failed")
        await close_reasoning_if_open()
        return status, error

    async def _select_agent(self, node: dict[str, Any]) -> Agent:
        agent_name = str(node["agent"])
        try:
            spec = self.spec.agents[agent_name]
        except KeyError as exc:
            raise WorkflowError(f"unknown agent spec: {agent_name}") from exc

        context = node.get("context") or {}
        mode = context.get("mode", "create")
        agent_id = self._agent_id(node)
        if mode == "reuse":
            if agent_id not in self.agent_registry:
                self.agent_registry[agent_id] = Agent(provider=spec.provider, cwd=self.workspace)
            return self.agent_registry[agent_id]
        if mode == "create":
            agent = Agent(provider=spec.provider, cwd=self.workspace)
            self.agent_registry[agent_id] = agent
            return agent
        if mode == "fork_from_step":
            source = str(context["from_step"])
            source_result = self.step_results.get(source)
            if source_result is None or source_result.agent_id is None:
                raise WorkflowError(f"cannot fork from missing step: {source}")
            parent = self.agent_registry[source_result.agent_id]
            if parent.provider != spec.provider:
                raise WorkflowError("fork_from_step requires the same provider")
            agent = await parent.fork()
            self.agent_registry[agent_id] = agent
            return agent
        if mode == "fork_from_agent":
            source_agent_id = str(context["from_agent"])
            parent = self.agent_registry[source_agent_id]
            if parent.provider != spec.provider:
                raise WorkflowError("fork_from_agent requires the same provider")
            agent = await parent.fork()
            self.agent_registry[agent_id] = agent
            return agent
        raise WorkflowError(f"unknown context mode: {mode}")

    def _agent_id(self, node: dict[str, Any]) -> str:
        context = node.get("context") or {}
        return str(context.get("agent_id") or node["agent"])

    def _render_prompt(self, node: dict[str, Any]) -> str:
        prompt_path = self._workspace_path(str(node["prompt_template"]))
        prompt = expand(prompt_path.read_text(encoding="utf-8"), self.variables)
        input_files = node.get("input_files") or []
        if not input_files:
            return prompt
        # Validate that every declared input exists on disk so misconfigured
        # paths fail fast (same guarantee as the old inline behaviour) — but
        # do NOT inline the file contents. Codex / Claude context windows are
        # the bottleneck; agents should Read the files they actually need.
        missing: list[str] = []
        resolved: list[Path] = []
        for raw_path in input_files:
            path = self._workspace_path(str(raw_path))
            if not path.exists():
                missing.append(str(path))
            else:
                resolved.append(path)
        if missing:
            raise FileNotFoundError(
                "input_files declared but not found on disk: " + ", ".join(missing)
            )
        lines = [
            prompt.rstrip(),
            "",
            "",
            "# Input Files (paths only — read each file on demand, not inlined)",
            "",
        ]
        for path in resolved:
            lines.append(f"- {path}")
        return "\n".join(lines) + "\n"

    def _check_outputs(self, node: dict[str, Any]) -> None:
        outputs = node.get("outputs") or {}
        missing: list[str] = []
        for raw_path in outputs.get("required_files", []):
            path = self._workspace_path(str(raw_path))
            if not path.exists():
                missing.append(str(path))
        if missing:
            raise WorkflowOutputError("missing required files: " + ", ".join(missing))

    def _parse_gate_decision(self, node: dict[str, Any], output_text: str) -> str:
        decision_config = node.get("decision") or {}
        if decision_config.get("parser") != "json":
            raise WorkflowError("only json gate decisions are supported in the MVP")
        payload = self._parse_json_object(output_text)
        for key in ("exit_when", "continue_when"):
            rule = decision_config.get(key)
            if rule and self._get_path(payload, str(rule["path"])) == rule.get("equals"):
                return "exit" if key == "exit_when" else "continue"
        raise WorkflowError(f"gate output did not match any decision rule: {payload}")

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise WorkflowError("gate output did not contain a JSON object")
            return json.loads(match.group(0))

    def _get_path(self, payload: dict[str, Any], path: str) -> Any:
        value: Any = payload
        for part in path.split("."):
            value = value[part]
        return value

    def _should_run(self, node: dict[str, Any]) -> bool:
        run_when = node.get("run_when")
        if not run_when:
            return True
        expression = run_when.get("expression")
        if not expression:
            return True
        return bool(eval(expression, {"__builtins__": {}}, dict(self.variables)))

    def _workspace_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.workspace / candidate

    def _remember_result(self, result: StepResult) -> None:
        self.step_results[result.node_id] = result
        self.step_results_by_key[result.node_key] = result

    def _raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise WorkflowError("workflow cancelled")

    async def _record_agent_event(self, node_key: str, agent_id: str, event: AgentEvent) -> None:
        record = {
            "event_type": event.type,
            "agent_id": agent_id,
            "provider": event.provider,
            "context_id": event.context_id,
            "run_id": event.run_id,
            "payload": event.payload,
        }
        self.store.append_event(node_key, record)
        await self._emit_event({"kind": "agent_event", "node_key": node_key, **record})

    async def _set_agent_state(
        self,
        agent_id: str,
        *,
        provider: str | None = None,
        current_node: str | None = None,
        status: str | None = None,
        context_id: str | None = None,
        pending_delta: int = 0,
        queued_inputs: int | None = None,
        clear_queued_input_tail: bool = False,
    ) -> None:
        state = self.agent_states.get(agent_id)
        if state is None:
            state = AgentViewState(agent_id=agent_id, provider=provider or "")
            self.agent_states[agent_id] = state
        if provider is not None:
            state.provider = provider
        if current_node is not None:
            state.current_node = current_node
        if status is not None:
            state.status = status
        if context_id is not None:
            state.context_id = context_id
        if pending_delta:
            state.pending_decisions = max(0, state.pending_decisions + pending_delta)
        if queued_inputs is not None:
            state.queued_inputs = queued_inputs
        if clear_queued_input_tail:
            state.queued_input_tail.clear()
        await self._emit_event({"kind": "agent_state", "agent": state})

    async def _append_agent_output(self, agent_id: str, text: str) -> None:
        state = self.agent_states[agent_id]
        state.output_tail = (state.output_tail + text)[-OUTPUT_TAIL_LIMIT:]
        await self._emit_event({"kind": "agent_output", "agent_id": agent_id, "text": text})

    async def _append_interaction_message(self, message: InteractionMessage) -> None:
        state = self.agent_states[message.agent_id]
        state.messages.append(message)
        await self._append_agent_output(
            message.agent_id,
            f"\n[pause] {message.node_key or message.agent_id}\n{message.text}\n",
        )

    async def _prepare_turn_display(
        self,
        agent_id: str,
        node_key: str,
        output: list[str],
        prompt: str,
        *,
        label: str = "step",
    ) -> None:
        state = self.agent_states.get(agent_id)
        if output:
            if not output[-1].endswith(("\n", "\r")):
                output.append("\n\n")
        if state is None:
            return
        state.current_prompt_excerpt = _prompt_excerpt(prompt)
        if state is not None and state.output_tail and not state.output_tail.endswith(("\n", "\r")):
            await self._append_agent_output(agent_id, "\n\n")
        await self._append_agent_output(agent_id, f"[{label}] {node_key}\n> {state.current_prompt_excerpt}\n\n")

    async def _emit_event(self, event: dict[str, Any]) -> None:
        if self.event_handler is None:
            return
        result = self.event_handler(event)
        if inspect.isawaitable(result):
            await result


async def approve_all(agent: Agent, event: AgentEvent) -> UserDecision:
    request = event.payload["request"]
    return UserDecision(
        request_id=request.id,
        action=DecisionAction.APPROVE,
        input=getattr(request, "input", None),
    )


def _prompt_excerpt(prompt: str, limit: int = 120) -> str:
    normalized = " ".join(prompt.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _format_tool_event(payload: dict[str, Any], limit: int = 160) -> str:
    """Render a single TOOL event payload as one human-readable line for the
    agent output_tail. Best-effort: must handle commandExecution / fileChange /
    mcpToolCall / unknown shapes without raising."""
    if not isinstance(payload, dict):
        return ""
    tool_name = str(payload.get("tool_name") or "tool")
    raw_input = payload.get("input") or {}
    detail: str = ""
    if isinstance(raw_input, dict):
        if tool_name == "commandExecution" or "command" in raw_input:
            command = raw_input.get("command")
            if isinstance(command, list):
                detail = " ".join(str(part) for part in command)
            elif command is not None:
                detail = str(command)
        elif tool_name == "fileChange" or "changes" in raw_input or "path" in raw_input:
            changes = raw_input.get("changes")
            if isinstance(changes, list):
                paths = [str(c.get("path") or c.get("file") or "") for c in changes if isinstance(c, dict)]
                detail = ", ".join(p for p in paths if p)
            else:
                detail = str(raw_input.get("path") or "")
        elif tool_name == "mcpToolCall" or "toolName" in raw_input or "arguments" in raw_input:
            inner_name = raw_input.get("toolName") or raw_input.get("tool_name")
            if inner_name:
                tool_name = f"mcp:{inner_name}"
            args = raw_input.get("arguments") or raw_input.get("input") or {}
            detail = str(args)
        else:
            detail = str(raw_input)
    elif raw_input:
        detail = str(raw_input)
    detail = " ".join(detail.split())
    if len(detail) > limit:
        detail = detail[: limit - 1] + "…"
    phase = payload.get("phase")
    suffix = "" if phase in (None, "result", "call") else f" ({phase})"
    return f"[tool {tool_name}{suffix}] {detail}".rstrip()
