You are the assistant for a plan-then-execute Codex workflow demo.

Task: write a PLAN (no code yet) for a small Python script that:
  - Reads a CSV file path from the first CLI argument.
  - Reads a column name from the second CLI argument.
  - Sums all numeric values in that column and prints the total to stdout.
  - Handles three error cases gracefully (missing file, missing column,
    non-numeric cell).

Output:
1. Create the directory `{output_dir}` if needed.
2. Write the plan to `{plan_file}` as Markdown with these sections:
     ## Inputs
     ## Outputs
     ## Steps
     ## Error handling
     ## Open questions for the user
   Keep each section to 2-4 bullets.
3. After writing, reply with one short line confirming the path.

Important: do NOT write `{script_file}` in this step. The user will review
the plan before authorizing the execute step.
