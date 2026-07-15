---
name: Adversarial Testing Command Centre
slug: atc
plane: governance
---

# Adversarial Testing Command Centre (ATC)

The red team on the intent path. A flagged intent -- one declared high
stakes, or requested for adversarial review, or driven by an agent on
probation -- takes an adversarial pass before it executes: ATC argues the
strongest case against the action, argues the defense, and returns a
verdict of uphold, escalate, or overturn. It is deliberation when
triggered, never a tax on every cycle.

## Triggers

- Any intent whose context declares high stakes or an irreversible action.
- Any explicit request for adversarial review.
- Any action by an agent AECC has placed on probation.

## State

- `review_log.json` -- the record of adversarial passes and their verdicts.
- `audit_trail.jsonl` -- the append-only ledger of adversarial reviews.
