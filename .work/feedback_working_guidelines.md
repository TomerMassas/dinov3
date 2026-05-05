---
name: Working Guidelines
description: Core collaboration rules — no unsolicited code changes, explain before acting, stay concise and focused
type: feedback
---

- Reading/exploring files does NOT require permission — just do it
- Never make code changes without Tomer's explicit permission
- Explain what you plan to change and why before doing it
- Keep sessions focused — don't go on tangents
- Verification = local script runs + code review (remote VM via PyCharm)
- Be concise, no fluff
- Prefer hardcoded constants over argparse CLI — Tomer runs scripts directly as `python3 script.py`
- When asking for approval on a proposed change, offer 4 explicit options: **(a) approve / go**, **(b) reject**, **(c) comment / suggest changes**, **(d) discuss further**. Don't end with just "say go" — Tomer wants to push back inline with specific edits without having to reject and re-explain.
- When proposing data artifacts tied to a specific model/version (embeddings, clusters, distances, etc.), link the filename to the model identifier via an explicit dict (`{model_id: filename}`) — never an inline ternary. Reason: avoids losing track of which file came from which backbone as more versions accumulate.
- When committing, stage **all** modified + untracked files (use `git add -A`). Don't pick subsets and don't ask which files to include — Tomer wants a single commit covering everything in the working tree at commit time. Reason: avoids fragmented commits and the back-and-forth of selecting per-file. The only exception is if a sensitive file (.env, credentials) appears unexpectedly — flag and pause rather than committing it.
