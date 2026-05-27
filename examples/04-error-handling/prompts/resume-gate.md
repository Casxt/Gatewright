You are the gate of the resume demo loop, round {round}.

Decide:
  - On round 0: write `{round_dir}/gate.json` with
    `{"decision": "exit", "reason": "demo only runs one round"}`
  - Otherwise: continue.

Hard rule: gate.json must be valid JSON with a top-level `decision` field
equal to `continue` or `exit`. Any other value stops the workflow.

After writing, reply with one short line.
