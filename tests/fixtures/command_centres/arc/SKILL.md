---
name: Agentic Reasoning Command Centre
slug: arc
plane: cognitive
---

# Agentic Reasoning Command Centre (ARC)

The auditor of the runtime's own reasoning. Where AKC governs what enters
knowledge, ARC watches how the runtime reasons: it scans deliberation and
decision on the output side for biased premises and unsupported assumptions,
records what it finds as advisories, and escalates to execution governance
when a one-off becomes a pattern. It informs; it does not block.

## Triggers

- Any completed deliberation or decision on the reasoning trail.
- Any run of flagged reasoning that suggests a systematic bias.

## State

- `patterns.json` -- known reasoning failure patterns and their thresholds.
- `audit_trail.jsonl` -- the append-only ledger of epistemic advisories.
