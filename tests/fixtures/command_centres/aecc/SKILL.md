---
name: Agent Envelope & Certification Command Centre
slug: aecc
plane: governance
---

# Agent Envelope & Certification Command Centre (AECC)

The authority model for every non-human actor. No agent -- a spawned
persona, an MCP-attached command centre, an evolved workflow -- may act
until it holds a certified capability envelope naming its scopes and its
maximum autonomy tier. Certification, trust scoring, probation, suspension
and revocation are AECC's to grant and to withdraw; enforcement is a
runtime-scope policy that consults the live envelope registry, so a
revocation takes effect on the very next cycle.

## Triggers

- Any intent whose acting agent is a non-human actor.
- Any request to certify, suspend, revoke, or re-scope an agent.
- Any agent operating outside its certified scopes or above its tier.

## State

- `authority_envelopes.json` -- the certified capability envelopes.
- `trust_scores.json` -- current trust score and standing per agent.
- `audit_trail.jsonl` -- the append-only ledger of certification events.
