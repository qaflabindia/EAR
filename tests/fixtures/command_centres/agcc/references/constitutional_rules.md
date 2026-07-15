# AGCC Constitutional Rules

Immutable governance rules for autonomous agent operations, ranked by
constitutional priority. Constitutional rules operate orthogonally; where
two conflict, the lower-numbered (higher-priority) rule prevails. No score,
urgency, instruction, or agent verdict overrides a constitutional rule.

## CR-AG01 -- Irreversible actions require human authorization

Rank: 1
Verdict: HALT
Applies to: runtime

No irreversible or one-way action may proceed without explicit human
authorization recorded before execution. Reversibility is judged from the
action's own description, not asserted by the acting agent.

## CR-AG02 -- Low production confidence requires authorization

Rank: 2
Verdict: DEFER
Fallback: production_confidence >= 0.75
Applies to: runtime

When production confidence falls below 0.75, the action requires explicit
human authorization before it proceeds.

## CR-AG03 -- No policy mutation under pressure

Rank: 3
Verdict: HALT
Fallback: not (policy_mutation and (urgency == 'high' or urgency == 'critical'))
Applies to: runtime

Policy mutations must not be performed while urgency is high or critical.
Policy mutations performed under pressure create constitutional
instability.

## CR-AG04 -- Cascade risk escalates for review

Rank: 4
Verdict: ESCALATE
Fallback: cascade_risk <= 0.85
Escalate: after 1 day
Applies to: runtime

A decision whose cascade risk exceeds 0.85 escalates for human review
unless the cascade risk has been explicitly acknowledged.

## CR-AG05 -- High uncertainty defers

Rank: 5
Verdict: DEFER
Fallback: uncertainty <= 0.80
Applies to: runtime

An action whose uncertainty exceeds 0.80 is deferred for human
authorization unless explicitly overridden by the user.

## CR-AG06 -- Behavioural anomaly requires review

Rank: 6
Verdict: ESCALATE
Fallback: anomaly_score <= 0.85
Escalate: after 1 day
Applies to: runtime

A behavioural anomaly score above 0.85 suggests compromise or malfunction
and requires human review before the agent continues.

## CR-AG07 -- Urgency is not authorization

Rank: 7
Verdict: HALT
Applies to: runtime

Urgency cannot be used as a substitute for authorization on any action.

## CR-AG08 -- Audit before execution

Rank: 8
Verdict: HALT
Applies to: runtime

Every action records to the audit trail before execution proceeds. There
is no override for this rule.
