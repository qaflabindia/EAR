"""Identity -- who is calling, and which Tenant they may act as.

`Tenant` (`ear/tenant.py`) is the org boundary a Runtime instance belongs
to; its own docstring explicitly defers "who a caller is, and which
Tenant they may act as" to this module. `Claim` is that answer: a
caller's verified subject plus the org_id(s) they are authorized to act
as. Authentication -- how a caller proved who they are -- stays the
Server's bearer-token guard's concern; this module is authorization only,
given a caller already identified, which org(s) they may touch.

A Claim is checked at the two places work actually reaches a Tenant's
data:

    Runtime.reason(intent, claim=claim)   -- refuses before the cycle
                                              starts if the Claim may not
                                              act as runtime.tenant.org_id
    Kernel.submit(..., claim=claim)       -- carried on the Task, checked
                                              at dispatch time so scheduled
                                              work gets the same boundary

Both raise/report `TenantBoundaryViolation`, a `PermissionError` like
every other refusal in EAR (`ApprovalRequired`, `SandboxViolation`, a
violated Policy) -- so a Kernel task lands `blocked`, not `failed`, and a
direct `reason()` call is refused the same way a violated Policy is.

No `claim` supplied is not a violation -- the same "off unless declared"
posture as Tenant itself: a Runtime never authored with a tenant.md, and
never called with a Claim, behaves exactly as it always has.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class TenantBoundaryViolation(PermissionError):
    """A Claim tried to act as an org it is not authorized for."""


@dataclass
class Claim:
    """A caller's verified identity plus the org_id(s) they may act as."""

    subject: str
    org_ids: tuple[str, ...] = field(default_factory=tuple)

    def may_act_as(self, org_id: str) -> bool:
        """Whether this Claim is authorized to act within `org_id`."""
        return org_id in self.org_ids

    def require(self, org_id: str) -> None:
        """Raise `TenantBoundaryViolation` if this Claim may not act as
        `org_id` -- the enforcement call `Runtime.reason` and `Kernel`
        dispatch make at their respective boundaries."""
        if not self.may_act_as(org_id):
            authorized = ", ".join(self.org_ids) or "none"
            raise TenantBoundaryViolation(
                f"Claim '{self.subject}' may not act as org '{org_id}' (authorized for: {authorized})"
            )
