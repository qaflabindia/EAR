"""Goal -- a completion condition that turns a single reasoning pass into
controlled, bounded iteration.

A Goal is Policy's mirror image. A `Policy` is a gate the cycle must not
cross ("the loan must not exceed $75,000"); a `Goal` is a finish line the
runtime iterates toward ("a final grade A-E and an approve/decline
decision are set"). Where a violated Policy *stops* a cycle, an unmet Goal
*re-enters* it -- feeding each cycle's decision forward -- until the Goal
is met, a blocker is hit, or the `max_cycles` safety cap is reached.

Like Policy, a Goal is judged in natural language by an LLM when a
ModelBinding is active (the primary path), and falls back to a safe,
deterministic `fallback_expression` when none is -- so goal-driven
iteration is fully usable and testable with no LLM configured at all. The
`max_cycles` cap means a Goal can never run away: iteration is always
bounded, which is what makes this safe for an enterprise runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Goal:
    """A Goal is a plain-English completion condition for a reasoning loop.

    `statement` is judged by an LLM against the running decision when a
    binding is active. `fallback_expression` (a short boolean expression
    over the intent's context and the special `decision` variable, safely
    evaluated -- never `eval`/`exec`) keeps the Goal enforceable offline.
    `max_cycles` hard-caps how many times the runtime may re-enter the
    cycle chasing this Goal."""

    statement: str = ""
    fallback_expression: str = ""
    max_cycles: int = 5
