---
name: Fix all sister scripts, not just the active one
description: When a bug is found in a shared pattern, patch every sibling script even if labeled legacy — prevents silent footguns on revisits
type: feedback
originSessionId: 422bdd19-0c5e-4d70-969d-9a8b1a50fcd1
---
When a bug is found in one script of a family (e.g. `train_dino_grad_accum.py` + `train_dino.py` + `train_playground.py`), fix it in **all** of them — not just the one currently in active use.

**Why:** Tomer may revisit a legacy script months later and forget it was broken. A "we're not using that anymore" script left uncorrected is a footgun. Prefer a small shared helper module over duplicating a fix three times.

**How to apply:** When proposing a code-level fix, enumerate all scripts that share the buggy pattern (grep the repo) and fix all of them. If the fix is more than ~5 lines, extract it into a shared module (e.g. `train_pictime/<helper>.py`) and import from each script.
