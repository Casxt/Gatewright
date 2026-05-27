You are the refiner agent. This is round {round} (0-indexed) of an iterative
refinement loop.

Task:
1. Create the directory `{round_dir}` if it does not exist.
2. Write `{round_dir}/draft.md`. The draft topic is "A simple Python utility
   for parsing semantic version strings".
3. On round 0, write a minimal first draft (~3 short sentences).
4. On rounds > 0, read the prior round's draft from
   `{output_dir}/round-{previous_round}/draft.md` and produce an improved
   version that addresses the gate's prior critique (the gate.json from the
   prior round is at `{output_dir}/round-{previous_round}/gate.json`).
5. After writing, reply with one short line summarizing what changed this
   round.
