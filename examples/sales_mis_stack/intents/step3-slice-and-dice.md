# Step 3 -- Slice & Dice

Parsed from `knowledge/mis-manual.md` Section 3 -- Slice & Dice.

Do step 3 of the Sales MIS Workflow this cycle only. Use Step 2's clean data, the dashboard template `uploads/daily_bank_sales_dashboard_2025.xlsx` and the canonical script `workspace/generate.py`; if that script is absent, author it first. The script must be dynamic and failsafe: source data and dashboard fill regions may grow or shrink vertically, and any sheet or region that cannot be filled must be named loudly rather than skipped silently. Fill every dashboard sheet according to its context -- KPI strip, monthly trend, asset/liability and channel-mix boxes, product performance, branch performance, RM leaderboard, top-10 branches and the five charts bound to those ranges. Preserve cell formats, resize genuinely data-driven sections, update chart ranges and total rows, and write reconciliation totals as computed numbers when headless formula evaluation would be unreliable. Stop there: Step 4 is the customer.

## Context

- input (workspace/clean_daily_sales.csv): verified present
- input (uploads/daily_bank_sales_dashboard_2025.xlsx): verified present
- script (workspace/generate.py): create if absent
- output_expected: workspace/Completed Sales Dashboard.xlsx

Every output must be written to its exact output_expected path above -- the next step is gated on that literal filename, so a different name, however reasonable, is a failed handoff.
