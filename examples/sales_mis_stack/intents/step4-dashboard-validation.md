# Step 4 -- Dashboard Validation

Parsed from `knowledge/mis-manual.md` Section 4 -- Dashboard Validation.

Do step 4 of the Sales MIS Workflow this cycle only. Use Step 3's completed dashboard and the canonical script `workspace/validate_dashboard.py`; if that script is absent, author it first. Recompute every headline total independently from the source or clean data and compare it with what landed in the completed dashboard, sheet by sheet. Reconcile KPIs, monthly trend, product, branch and RM leaderboard sums within the manual's tolerance. Write a validation log with pass or fail per sheet and per check, never one averaged verdict. If every sheet passes, save the final workbook as `workspace/Validated dashboard.xlsx` for Business and Leadership.

## Context

- input (workspace/Completed Sales Dashboard.xlsx): verified present
- input (workspace/clean_daily_sales.csv): verified present
- script (workspace/validate_dashboard.py): create if absent
- output_expected: workspace/Validated dashboard.xlsx
- output_expected: workspace/validation_log.md

Every output must be written to its exact output_expected path above -- Business and Leadership are gated on those literal filenames, so a different name, however reasonable, is a failed handoff.
