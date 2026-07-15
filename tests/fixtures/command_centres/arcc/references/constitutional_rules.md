# ARCC Constitutional Rules

Immutable customer-relationship rules, ranked by constitutional priority.

## CR-RE-01 -- No promise beyond policy

Rank: 1
Verdict: HALT
Applies to: runtime

No commitment may be made to a customer that the business's own policy does
not permit. A promise the company cannot keep is a breach, not a
concession.

## CR-RE-02 -- Discounts stay within the approved limit

Rank: 2
Verdict: DEFER
Fallback: discount_pct <= 20
Applies to: runtime

A discount above 20 percent requires human approval before it is offered;
the request parks for that verdict.

## CR-RE-03 -- Customer data honours consent

Rank: 3
Verdict: HALT
Fallback: not (marketing_use and not marketing_consent)
Applies to: runtime

A customer's personal data is used only within the consent they gave;
marketing use without consent is a breach.
