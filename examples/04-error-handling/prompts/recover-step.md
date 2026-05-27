You are exercising the orchestrator's message-driven pause/error flow.

This step has three required outputs:
  - `{part_a_file}`
  - `{part_b_file}`
  - `{part_c_file}`

ON THIS FIRST TURN (the "flaky" attempt):
1. Create the directory `{output_dir}`.
2. Write ONLY `{part_a_file}` with a sentence saying "Part A produced by the
   first attempt".
3. Then STOP — do not write the other two files, do not write a summary.
   Reply with literally:

     "Simulated network drop after part A — leaving the rest unfinished."

   This forces the orchestrator to detect missing required files and append
   an open pause/error message for the user.

WHEN THE USER REPLIES TO CONTINUE (your next turn, conversation context is
intact):
1. Recognize that part_a is already on disk; do NOT overwrite it.
2. Write the two remaining files:
   - `{part_b_file}` — one sentence "Part B written by the continuation turn".
   - `{part_c_file}` — one sentence "Part C written by the continuation turn".
3. Reply with one short line confirming both new paths.
