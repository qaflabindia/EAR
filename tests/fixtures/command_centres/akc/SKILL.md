---
name: Agentic Knowledge Command Centre
slug: akc
plane: cognitive
---

# Agentic Knowledge Command Centre (AKC)

The governed door into the organization's knowledge. Nothing enters the
knowledge base by being written to a file: every claim is validated for form
and source, scored epistemically, and checked for contradiction before it is
admitted -- and a superseded or withdrawn claim is retired, not left to
mislead. Reason from what is admitted as if it were true; admit only what
earns that standing.

## Triggers

- Any claim, document, or fact proposed for the knowledge base.
- Any claim that contradicts what is already held.
- Any retirement or supersession of an existing passage.

## State

- `sources.json` -- the registry of trusted knowledge sources and tiers.
- `audit_trail.jsonl` -- the append-only ledger of admissions and retirements.
