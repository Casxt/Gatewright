You are the generator agent (Codex provider).

Task:
1. Create the directory `{output_dir}` if it does not exist.
2. Write `{module_file}` containing a small, self-contained Python module
   exposing:
     - `parse(version: str) -> tuple[int, int, int, str | None, str | None]`
       returning (major, minor, patch, pre_release, build_metadata).
     - A `class SemVer` dataclass with a `from_string(cls, version)`
       classmethod and a `__lt__` for comparison.
     - At least one module-level docstring.
3. Keep the module under ~80 lines. No external imports beyond
   `dataclasses` and `re`.
4. After writing, reply with one short line confirming the path.
