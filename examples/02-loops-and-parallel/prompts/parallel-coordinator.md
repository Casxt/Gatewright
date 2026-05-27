You are the coordinator agent of a parallel research fanout.

Task:
1. Create the directory `{output_dir}` if it does not exist.
2. Write `{output_dir}/brief.md` containing a short brief (~5 bullets) for
   a research topic: "Comparing pydantic vs dataclasses vs attrs for a
   server-side config schema". Include three angles to investigate:
   performance, correctness/runtime validation, and ergonomics.
3. After writing, reply with one short line. The three forked workers will
   each read this brief and produce a focused report on one angle.
