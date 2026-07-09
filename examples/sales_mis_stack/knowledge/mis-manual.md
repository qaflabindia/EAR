# Sales MIS Manual

The reference material behind the four-step MIS cycle. There is no fixed
script shipped for any step -- the Reasoner writes and runs its own code
inside the sandbox each cycle, and cites the section it relied on. These
sections are the rules that code must follow, not the code itself. Each
checking and generation step has a canonical script name in the sandbox
workspace -- `validate_data.py` (Section 2), `generate.py` (Section 3),
`validate_dashboard.py` (Section 4). When the named script is absent, the
Reasoner authors it; when present, it reruns it. Every script prints the
concrete evidence of what it did -- rows read, sheets filled, checks passed
or failed -- so a tool result is always verifiable fact, never a bare exit
code.

## Section 1 -- Load Data

Suppliers: ERP, CRM and APIs. Raw source data arrives as
`daily_bank_sales_data_2025.xlsx` -- in production, extracted from the source
systems into that one workbook and dropped into the sandbox's `uploads/`
directory. Loading is not just an open-and-read: extract and load the data,
then run a small sanity check immediately, so a schema change or a corrupt
extract is caught before anything downstream trusts the file. Outputs are the
staged dataset and an anomaly report when anomalies are found; Step 2 is the
customer.

## Section 2 -- Sanity Check

Suppliers: Step 1. Inputs: loaded data and the canonical script
`workspace/validate_data.py`. If the script is absent, author it first.
Re-read the staged data with dynamic, relative cell ranges -- locate the
header row and the last data row by inspecting the sheet itself rather than
assuming a fixed row count, because the source grows one row per transaction
every day. Check schema (every expected column present), count (row total and
calendar coverage), nulls (excluding the workbook's own formula columns,
which carry no cached value until Excel opens them -- recompute their
achievement-percent values from the raw counts instead) and duplicates (same
date, branch, product, channel and RM). Anything found is named in the
anomaly report, never silently dropped. Output the clean data file that Step
3 reads.

## Section 3 -- Slice & Dice

Suppliers: Step 2. Inputs: clean data,
`daily_bank_sales_dashboard_2025.xlsx` and the canonical script
`workspace/generate.py`. If the script is absent, author it first. The
script must be dynamic (source data and dashboard fill regions may grow or
shrink vertically) and failsafe (anything it cannot fill is named loudly,
never skipped silently). Read the clean staged dataset and fill every
dashboard sheet according to its context: the KPI strip, the monthly trend,
the asset/liability and channel-mix boxes, product performance, branch
performance, the RM leaderboard and the top-10 branches -- plus the five
charts bound to those ranges. Preserve cell formats while filling. Read the
product, branch and RM counts from the data itself, not from how many rows
the template happened to ship with. Where a section's row count is genuinely
data-driven (products, branches, RMs), resize it -- insert or delete rows,
rewrite its chart ranges and its total row to match. Where a section's size
is structural rather than data-driven (twelve calendar months, the
Asset/Liability split, the three channels, the top-10 cap), fill what's
present and leave the rest at zero rather than reshaping the sheet. Prefer
writing the totals the reconciliation step will check as plain computed
numbers rather than Excel formulas a headless read can't evaluate. Output
the Completed Sales Dashboard for Step 4.

## Section 4 -- Dashboard Validation

Suppliers: Step 3. Inputs: Completed Sales Dashboard.xlsx and the canonical
script `workspace/validate_dashboard.py`. If the script is absent, author it
first. Recompute every headline total independently from the source or clean
dataset and compare it to what actually landed in the completed dashboard,
sheet by sheet: KPIs, monthly trend, product, branch and RM leaderboard sums
must all tie back within half a percent. Record a pass or fail per sheet and
per check in the validation log -- never an average verdict masking one bad
sheet. Only a dashboard whose log shows every sheet passed is saved as
Validated dashboard.xlsx and sent to Business and Leadership.
