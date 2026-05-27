You are the writer agent for a Claude single-step workflow demo.

Task:
1. Create the directory `{output_dir}` if it does not exist.
2. Write the file `{note_file}` with this exact Markdown content:

   ```markdown
   # Hello from Claude

   - Step name: write-note
   - Provider: claude
   - This file proves the Claude backend can drive a single workflow step.
   ```

3. After the file is written, reply with one short line confirming
   the path you wrote.
