You are still the same Codex thread. The user has reviewed your script at
`{script_file}` and authorized verification.

Task:
1. Create a tiny test CSV at `{output_dir}/sample.csv` with three rows
   covering: a normal numeric column, a header that matches a known
   column name, and at least one row demonstrating an error case the
   script is supposed to handle (e.g. a non-numeric cell).
2. Run the script against the sample (`python {script_file} <csv>
   <column>`). Capture stdout AND stderr. If codex's sandbox prompts you
   for command approval, accept.
3. Write `{verification_file}` containing:
     ## Test inputs
     ## Observed output
     ## Pass / fail per error case from the plan
     ## Verdict
4. Reply with one short line: pass, partial, or fail.

If the script crashes unexpectedly, write the failure mode to
verification.md anyway and end with `Verdict: fail` — do NOT silently
patch the script in this step.
