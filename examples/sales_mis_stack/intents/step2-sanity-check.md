# Step 2 -- Sanity Check

Parsed from `knowledge/mis-manual.md` Section 2 -- Sanity Check.

Do step 2, Sanity Check, of the Sales MIS Workflow this cycle only. Run the canonical script `workspace/validate_data.py`; if it is absent, author it first. Follow the manual section below as the source of truth for suppliers, inputs, process, outputs and customers.

Manual section:

Suppliers: Step 1. Inputs: loaded data and the canonical script `workspace/validate_data.py`. If the script is absent, author it first. Re-read the staged data with dynamic, relative cell ranges -- locate the header row and the last data row by inspecting the sheet itself rather than assuming a fixed row count, because the source grows one row per transaction every day. Check schema (every expected column present), count (row total and calendar coverage), nulls (excluding the workbook's own formula columns, which carry no cached value until Excel opens them -- recompute their achievement-percent values from the raw counts instead) and duplicates (same date, branch, product, channel and RM). Anything found is named in the anomaly report, never silently dropped. Output the clean data file that Step 3 reads.

Stop there: Step 3 is the customer.

## Context

- input (workspace/staged_daily_sales.csv): verified present, 146,707 bytes
- script (workspace/validate_data.py): create if absent
- output_expected: workspace/clean_daily_sales.csv

Every output must be written to its exact output_expected path above -- the next step is gated on that literal filename, so a different name, however reasonable, is a failed handoff.
