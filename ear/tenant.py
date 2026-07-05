"""Tenant -- the org a stack belongs to, stacked in `tenant.md`.

One markdown file, written the same natural-language way as every other
stacked file, declares which org this Runtime instance belongs to and the
fiscal year its workday-notation schedules (see `schedule.md`, `workday.py`)
resolve `q`/`h`/`y`/`a` occurrences against:

    ## Acme Capital

    Org id: org_acme_prod
    Fiscal year start: 2026-04-01
    Fiscal year end: 2027-03-31
    Timezone: Asia/Kolkata
    Secret env var: EAR_ACME_SECRET

`tenant.md` is optional -- a stack that declares none gets the default
tenant (`org_id="default"`, no fiscal year, so `q`/`h`/`y`/`a` notation
falls back to the calendar year containing the date being resolved). This
is the same "off unless declared" posture as `memory.md`'s Catalogue Store
and Sandbox sections.

`org_id` is never something a *request* supplies -- it is read once, from
the stack's own file, at load time, and stamped onto every catalogue
object and Kernel task the loaded Runtime produces. Boundary, not
authentication: Tenant carries no notion of who a caller is, only which
org's data this Runtime instance's Kernel/Store activity belongs to. Who a
caller is, and which Tenant they may act as, is `ear.identity.Claim`'s
concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

DEFAULT_ORG_ID = "default"


@dataclass
class Tenant:
    """The org this stack was loaded for."""

    org_id: str = DEFAULT_ORG_ID
    name: str = ""
    fiscal_year_start: Optional[date] = None
    fiscal_year_end: Optional[date] = None
    timezone: Optional[str] = None
    secret_env_var: Optional[str] = None

    def fiscal_year_bounds(self, today: Optional[date] = None) -> tuple[date, date]:
        """The fiscal year window used for workday-notation resolution.

        A declared `fiscal_year_start`/`fiscal_year_end` is used as-is (the
        author edits `tenant.md` when the fiscal year rolls, the same
        reload-every-load posture as the rest of the stack). Undeclared
        falls back to the calendar year containing `today`."""
        if self.fiscal_year_start is not None and self.fiscal_year_end is not None:
            return self.fiscal_year_start, self.fiscal_year_end
        today = today or date.today()
        return date(today.year, 1, 1), date(today.year, 12, 31)
