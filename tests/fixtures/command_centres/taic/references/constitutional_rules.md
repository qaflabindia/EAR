# TAIC Constitutional Rules

Immutable talent-acquisition rules, ranked by constitutional priority.

## CR-TA-01 -- Screen only on job-relevant evidence

Rank: 1
Verdict: HALT
Applies to: runtime

A candidate may be judged only on evidence relevant to the role. Screening
on age, gender, ethnicity, or any other protected attribute is a breach,
never a signal.

## CR-TA-02 -- Offers stay within the approved band

Rank: 2
Verdict: DEFER
Fallback: offer_amount <= band_maximum
Applies to: runtime

An offer above the approved compensation band for the role requires human
approval before it is extended; the request parks for that verdict.

## CR-TA-03 -- Candidate data is retained only as long as lawful

Rank: 3
Verdict: HALT
Applies to: runtime

A candidate's personal data is used only for the hiring decision at hand
and retained only within the lawful window; use or retention beyond that is
a breach.
