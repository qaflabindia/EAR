# AAWDFC Constitutional Rules

Immutable workflow-legitimacy rules, ranked by constitutional priority.

## CR-WF-01 -- No machine-created change without a legitimacy verdict

Rank: 1
Verdict: HALT
Applies to: runtime

A workflow or skill the system created for itself does not apply until
AAWDFC has judged it legitimate. An unjudged self-modification is a breach.

## CR-WF-02 -- Legitimacy requires a stated purpose

Rank: 2
Verdict: HALT
Applies to: runtime

A change with no explanation cannot be judged legitimate. An unexplained
self-modification is refused, not waved through.

## CR-WF-03 -- A machine change may not exceed its authority

Rank: 3
Verdict: ESCALATE
Escalate: after 1 day
Applies to: runtime

A machine-created change that would widen the system's own authority or
role topology escalates for human review before it takes effect.

## CR-WF-04 -- No self-authored code without a beyond-suspicion review

Rank: 4
Verdict: HALT
Applies to: runtime

Code the system writes for itself is installed only after a reviewer judges
it safe beyond any reasonable suspicion, and only ever runs confined in a
sandbox -- never in the kernel's own process.

## CR-WF-05 -- The core kernel is never self-modified

Rank: 5
Verdict: HALT
Applies to: runtime

The kernel may alter any part of its own code except the core scheduler
(kernel.py); a change targeting the core is refused before any gate, so the
system can never edit away the gates every change passes through.
