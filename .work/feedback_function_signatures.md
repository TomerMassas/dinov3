---
name: Function signature & call line wrapping
description: Multi-line `def` signatures AND function/method calls — first arg on opening-paren line, args column-aligned, closing `)` column-aligned with opening `(`
type: feedback
---

**Rule:** When writing or editing a function **definition** OR a function/method **call**:

- **One line** if the full expression fits within 120 chars
  (project line length from `pyproject.toml`; matches PyCharm default).
- **Otherwise multi-line** in this exact form:
  ```python
  def func(arg1,
           arg2,
           arg3,
          ):
  ```
  ```python
  result = func(arg1,
                arg2,
                arg3,
               )
  ```
  - First arg sits immediately after the opening `(` on the same line.
  - Subsequent args are column-aligned under the first arg
    (column of `(` + 1).
  - Closing `)` is column-aligned with the opening `(` (NOT column 0),
    then `:` (or `-> Type:`) for definitions, or whatever follows for calls
    (e.g. `.method()` chain, assignment continuation).

**Why:** Tomer's preference; matches PyCharm's "horizontal" continuation
style. Visual alignment of args + closing paren makes signatures and calls
scannable at a glance, and keeps definitions and call sites visually
consistent.

**How to apply:**
- All `def` (regular functions, methods, `__init__`, decorated). Decorators
  stay on their own lines above the `def`.
- All function/method **calls** including constructors (`Foo(...)`),
  `obj.method(...)`, library calls (`torch.stack(...)`,
  `transforms.Compose(...)`), `wandb.init(...)`, etc.
- Not lambdas (single-line by nature).
- If the one-line form fits within 120 chars, keep it one line; don't
  multi-line a short call/signature.
- Edge cases that still wrap as one signature/call: `*args`, `**kwargs`,
  defaults with paren-bearing values like `def f(x=foo(1, 2))`, type
  annotations like `def f(x: List[Tuple[int, ...]] = None) -> Dict[str, Any]:`,
  nested call args. The rule is about visual layout — apply the same
  wrapping logic.
- Common pitfall: do NOT put the closing `)` at column 0 (or any column NOT
  aligned with `(`). It must vertically align with the opening `(`. Examples:
  ```python
      def __init__(self,
                   x,
                   y,
                  ):

      result = some_function(arg1,
                             arg2,
                            )

      run = wandb.init(project=project,
                       entity=entity,
                       name=run_name,
                      )
  ```
