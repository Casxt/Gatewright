You are the collector agent. You need to "log into" a fictional service
that is gated by a session token. The token's presence is the signal that
the user has completed login.

Token gate:
  - Required file:  `{token_file}`
  - This file ONLY exists once the user has done the manual login action
    (in the real xueqiu skill, the equivalent is the user scanning a QR
    code in the browser to authenticate).

Procedure:

1. Check whether `{token_file}` already exists.

2. If it does NOT exist:
     a. Tell the user, in 1–2 short sentences, that they must perform the
        manual login step. Use this exact line so it's easy to scan:

            ACTION REQUIRED: Please create the session token by running:

              touch {token_file}

            in another terminal, then return to this TUI and type
            "done, continue" in the operator input.

     b. END THIS TURN. Do NOT busy-wait. Do NOT write `{session_file}`.
        Do NOT keep polling — that would burn tokens. The orchestrator
        will surface an open pause message that
        the user will resolve once they have done the manual step.

3. If `{token_file}` DOES exist (either this is your first turn and the
   user pre-staged it, or this is the continuation turn after the user
   completed the manual step):
     a. Read the token file (it can be empty — its existence is the
        signal).
     b. Write `{session_file}` containing a brief Markdown report:

            # Mock session summary

            - Token file: <token_file path>
            - Logged in: yes
            - Notes: confirmed by user-driven action

     c. Reply with one short line confirming the path you wrote.

Important: on the second invocation of this step (when the orchestrator
re-prompts you after the user's confirmation), your conversation context
is intact — you remember asking the user to create the file. Re-check
existence and proceed with step 3.
