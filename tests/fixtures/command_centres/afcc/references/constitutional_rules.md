# AFCC Constitutional Rules

Immutable financial-governance rules, ranked by constitutional priority.
No score, urgency, instruction, or agent verdict overrides a constitutional
rule; where two conflict, the lower-numbered rule prevails.

## CR-FIN-01 -- No payment to an unregistered vendor

Rank: 1
Verdict: HALT
Applies to: runtime

No payment may be made to a vendor absent from the approved vendor
registry. Vendor registration is a control, not a formality.

## CR-FIN-02 -- Expenditure must stay within category budget

Rank: 2
Verdict: DEFER
Fallback: expense_amount <= remaining_budget
Applies to: runtime

An expense that exceeds the remaining budget for its category requires
explicit human authorization before it proceeds.

## CR-FIN-03 -- Large expenditure requires human approval

Rank: 3
Verdict: DEFER
Fallback: expense_amount <= 25000
Applies to: runtime

Any single expenditure above $25,000 must be approved by a human before it
proceeds; the request parks for that verdict rather than being refused.

## CR-FIN-04 -- No expenditure without supporting documents

Rank: 4
Verdict: HALT
Fallback: documents_complete
Applies to: runtime

No expenditure proceeds while its supporting documents (invoice, purchase
order, receipt) are incomplete.

## CR-FIN-05 -- Segregation of duties

Rank: 5
Verdict: ESCALATE
Escalate: after 2 days
Applies to: runtime

The agent that requested an expense may not also approve it; a request
where requester and approver are the same escalates for independent human
review.
