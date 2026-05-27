You are running one step of a three-step refine loop body, round {round}.
Your agent_id tells you which step you are (resume-worker-round-N — but the
node id is one of prepare / analyze / summarize). Figure out which step by
checking which file you are supposed to write — it's the one in
`outputs.required_files` for this node.

The scheduler passes the step's required output via the prompt template
substitution machinery — write whichever of the following files matches the
current node:

  - `{round_dir}/prepare.md`   (if node id is "prepare")
  - `{round_dir}/analyze.md`   (if node id is "analyze")
  - `{round_dir}/summary.md`   (if node id is "summarize")

Each file should be a short Markdown note (~3 sentences) about a fictional
refactor task. The point of this example is to exercise `--start-from
refine-loop/summarize -- --var round=0` resume behavior, not realistic
content.

After writing the file, reply with one short line confirming the path.
