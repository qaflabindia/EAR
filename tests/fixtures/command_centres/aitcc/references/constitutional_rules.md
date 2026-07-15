# AITCC Constitutional Rules

Immutable IT-operations rules, ranked by constitutional priority.

## CR-IT-01 -- No production change without approval

Rank: 1
Verdict: DEFER
Fallback: not production_change
Applies to: runtime

A change that touches production requires human approval before it ships;
the request parks for that verdict rather than deploying on an agent's say.

## CR-IT-02 -- Least privilege on every access grant

Rank: 2
Verdict: HALT
Fallback: not (access_grant and privilege_exceeds_task)
Applies to: runtime

An access grant may confer only the privilege the task needs. Granting
standing or excess privilege is a breach, not convenience.

## CR-IT-03 -- Honour the change-freeze

Rank: 3
Verdict: HALT
Fallback: not (production_change and change_freeze_active)
Applies to: runtime

No production change proceeds during an active change-freeze window; the
freeze is a control, not a suggestion.
