# Step 1 -- Load Data

Parsed from `knowledge/mis-manual.md` Section 1 -- Load Data.

Do step 1, Load Data, of the Sales MIS Workflow this cycle only. Follow the manual section below as the source of truth for suppliers, inputs, process, outputs and customers. Also request these conditional outputs when the manual calls for them: `workspace/anomaly_report.md`.

Manual section:

Suppliers: ERP, CRM and APIs. Raw source data arrives as `daily_bank_sales_data_2025.xlsx` -- in production, extracted from the source systems into that one workbook and dropped into the sandbox's `uploads/` directory. Loading is not just an open-and-read: extract and load the data, then run a small sanity check immediately, so a schema change or a corrupt extract is caught before anything downstream trusts the file. Outputs are the staged dataset and an anomaly report when anomalies are found; Step 2 is the customer.

Stop there: Step 2 is the customer.

## Context

- input (uploads/daily_bank_sales_data_2025.xlsx): verified present, 98,272 bytes
- output_expected: workspace/staged_daily_sales.csv
- output_expected_if_applicable: workspace/anomaly_report.md

Every output must be written to its exact output_expected path above -- the next step is gated on that literal filename, so a different name, however reasonable, is a failed handoff.
