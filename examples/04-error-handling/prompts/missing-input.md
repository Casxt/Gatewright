This prompt will never actually run — the orchestrator fails the step
before invoking Codex because one of the declared input_files does not
exist on disk.

(If you somehow see this prompt being executed, something has changed in
the scheduler's input_file validation behavior.)
