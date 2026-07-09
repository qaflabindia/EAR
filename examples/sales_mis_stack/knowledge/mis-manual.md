# Sales MIS Manual

The reference material behind the four-step MIS cycle. There is no fixed
script here for any step -- the Reasoner writes and runs its own code inside
the sandbox each cycle, and cites the section it relied on. These sections
are the rules that code must follow, not the code itself.

## Section 1 -- Load

Raw source data arrives as a daily sales workbook -- in production, extracted
from ERP, CRM and banking APIs into that one file, dropped into the sandbox's
`uploads/` directory. Loading is not just an open-and-read: a small sanity
check runs immediately, so a schema change or a corrupt extract is caught
before anything downstream trusts the file.

## Section 2 -- Sanity Check

Re-read the workbook with dynamic, relative cell ranges -- locate the header
row and the last data row by inspecting the sheet itself rather than assuming
a fixed row count, because the source grows one row per transaction every
day. Check schema (every expected column present), count (row total and
calendar coverage), nulls (excluding the workbook's own formula columns,
which carry no cached value until Excel opens them -- recompute their
achievement-percent values from the raw counts instead) and duplicates (same
date, branch, product, channel and RM). Anything found is named in the
anomaly report, never silently dropped. Stage the clean result to a file the
next step reads.

## Section 3 -- Slice & Dice

Read the clean staged dataset and fill the dashboard workbook: the KPI strip,
the monthly trend, the asset/liability and channel-mix boxes, product
performance, branch performance, the RM leaderboard and the top-10 branches --
plus the five charts bound to those ranges. Read the product, branch and RM
counts from the data itself, not from how many rows the template happened to
ship with. Where a section's row count is genuinely data-driven (products,
branches, RMs), resize it -- insert or delete rows, rewrite its chart ranges
and its total row to match. Where a section's size is structural rather than
data-driven (twelve calendar months, the Asset/Liability split, the three
channels, the top-10 cap), fill what's present and leave the rest at zero
rather than reshaping the sheet. Prefer writing the totals the reconciliation
step will check as plain computed numbers rather than Excel formulas a
headless read can't evaluate.

## Section 4 -- Dashboard Validation

Recompute every headline total independently from the staged dataset and
compare it to what actually landed in the completed dashboard, sheet by
sheet: KPIs, monthly trend, product, branch and RM leaderboard sums must all
tie back within half a percent. Record a pass or fail per check in the
validation log -- never an average verdict masking one bad sheet. Only a
dashboard whose log shows every sheet passed reaches Business and Leadership.
