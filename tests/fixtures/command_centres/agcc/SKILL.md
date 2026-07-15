---
name: Agentic Governance Command Centre
slug: agcc
plane: governance
---

# Agentic Governance Command Centre (AGCC)

The execution-governance command centre. Every agent-initiated,
human-initiated, or self-generated intent clears AGCC before it runs: it
scores the action, simulates its blast radius, arbitrates competing
verdicts, and records the outcome to the audit trail before execution
proceeds.

AGCC's constitution (`references/constitutional_rules.md`) is the set of
rules no score, urgency, instruction, or agent verdict can override. In the
Enterprise AGI binding those rules compile to enforceable EAR policies at
runtime scope, so the choke point AGCC describes *is* `Governor.govern`.

## Triggers

- Any intent whose action is irreversible or affects money, rights, safety,
  or employment.
- Any proposed mutation of policy or governance state.
- Any action taken under declared high or critical urgency.

## State

- `authority_envelopes.json` -- the certified capability envelopes agents hold.
- `trust_scores.json` -- current trust score and standing per agent.
- `audit_trail.jsonl` -- the append-only ledger of governed decisions.
