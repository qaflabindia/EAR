# ALGCC Constitutional Rules

Immutable logistics rules, ranked by constitutional priority.

## CR-LG-01 -- Hazardous materials follow the safety code

Rank: 1
Verdict: HALT
Fallback: not (hazardous and not hazmat_certified_route)
Applies to: runtime

A hazardous-materials shipment may move only on a certified, compliant
route. A non-compliant hazmat movement is halted, never expedited.

## CR-LG-02 -- High-value shipments require approval

Rank: 2
Verdict: DEFER
Fallback: shipment_value <= 50000
Applies to: runtime

A shipment whose value exceeds $50,000 requires human approval before
dispatch; the request parks for that verdict.

## CR-LG-03 -- Inventory integrity is never overridden

Rank: 3
Verdict: ESCALATE
Escalate: after 1 day
Applies to: runtime

A dispatch that would drive recorded inventory negative escalates for
human reconciliation rather than proceeding on a broken count.
