# ATC Constitutional Rules

Immutable rules for adversarial testing, ranked by constitutional priority.

## CR-AT-01 -- Flagged intents take an adversarial pass

Rank: 1
Verdict: HALT
Applies to: runtime

A flagged high-stakes or irreversible intent must not execute until it has
taken an adversarial pass and been upheld.

## CR-AT-02 -- An unconcluded challenge escalates

Rank: 2
Verdict: ESCALATE
Escalate: after 1 day
Applies to: runtime

An adversarial pass that cannot conclude -- a challenge the defense does
not clearly answer -- escalates to a human rather than defaulting to
proceed.
