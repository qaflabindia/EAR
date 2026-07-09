# Step 2 -- Sanity Check

Parsed from `knowledge/mis-manual.md` Section 2 -- Sanity Check.

Do step 2 of the Sales MIS Workflow this cycle only. Use Step 1's loaded data and run the canonical script `workspace/validate_data.py`; if that script is absent, author it first. The script must read the staged data with dynamic, relative cell ranges by locating the header row and last data row from the sheet itself, never from fixed coordinates or a fixed row count. Validate schema, count, nulls and duplicates, recomputing achievement-percent values from raw counts when formula caches are empty. Name any anomaly found, write the clean data file that Step 3 reads, and state every check's result with observed counts. Stop there: Step 3 is the customer.

## Context

- input (workspace/staged_daily_sales.csv): verified present
- script (workspace/validate_data.py): create if absent
- output_expected: workspace/clean_daily_sales.csv

Every output must be written to its exact output_expected path above -- the next step is gated on that literal filename, so a different name, however reasonable, is a failed handoff.
