You are the writer agent for a single-step Codex workflow demo.

Task:
1. Create the directory `{output_dir}` if it does not exist.
2. Write the file `{note_file}` with this exact Markdown content:

   ```markdown
   # Hello from Codex

   - Step name: write-note
   - Provider: codex
   - This file proves the single-step workflow ran end-to-end.
   ```

3. After the file is written, reply with a single short line confirming
   the path you wrote.
