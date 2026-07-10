# Step 3 -- Slice & Dice

Parsed from `knowledge/mis-manual.md` Section 3 -- Slice & Dice.

Do step 3, Slice & Dice, of the Sales MIS Workflow this cycle only. Run the canonical script `workspace/generate.py`; if it is absent, author it first. Follow the manual section below as the source of truth for suppliers, inputs, process, outputs and customers.

Manual section:

Suppliers: Step 2. Inputs: clean data, `daily_bank_sales_dashboard_2025.xlsx` and the canonical script `workspace/generate.py`. If the script is absent, author it first. The script must be dynamic (source data and dashboard fill regions may grow or shrink vertically) and failsafe (anything it cannot fill is named loudly, never skipped silently). Read the clean staged dataset and fill every dashboard sheet according to its context: the KPI strip, the monthly trend, the asset/liability and channel-mix boxes, product performance, branch performance, the RM leaderboard and the top-10 branches -- plus the five charts bound to those ranges. Preserve cell formats while filling. Read the product, branch and RM counts from the data itself, not from how many rows the template happened to ship with. Where a section's row count is genuinely data-driven (products, branches, RMs), resize it -- insert or delete rows, rewrite its chart ranges and its total row to match. Where a section's size is structural rather than data-driven (twelve calendar months, the Asset/Liability split, the three channels, the top-10 cap), fill what's present and leave the rest at zero rather than reshaping the sheet. Prefer writing the totals the reconciliation step will check as plain computed numbers rather than Excel formulas a headless read can't evaluate. Output the Completed Sales Dashboard for Step 4.

Stop there: Step 4 is the customer.

## Context

- input (workspace/clean_daily_sales.csv): verified present, 128,991 bytes
- input (uploads/daily_bank_sales_dashboard_2025.xlsx): verified present, 39,387 bytes
- script (workspace/generate.py): create if absent
- output_expected: workspace/Completed Sales Dashboard.xlsx

Every output must be written to its exact output_expected path above -- the next step is gated on that literal filename, so a different name, however reasonable, is a failed handoff.
