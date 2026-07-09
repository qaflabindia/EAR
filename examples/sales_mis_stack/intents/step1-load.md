# Step 1 -- Load Data

Parsed from `knowledge/mis-manual.md` Section 1 -- Load Data.

Do step 1 of the Sales MIS Workflow this cycle only. Suppliers are ERP, CRM and APIs, and their raw source data is staged as `uploads/daily_bank_sales_data_2025.xlsx`. Extract and load the workbook, then run the small sanity check described in the manual before downstream steps trust the file. Write the staged dataset, and write an anomaly report when any anomaly is found. Report the sheets, rows and columns actually read from the workbook -- observed facts only, never assumed counts. Stop there: Step 2 is the customer.

## Context

- input (uploads/daily_bank_sales_data_2025.xlsx): verified present
- output_expected: workspace/staged_daily_sales.csv
- output_expected_if_anomaly: workspace/anomaly_report.md

Every required output must be written to its exact output path above -- the next step is gated on that literal filename, so a different name, however reasonable, is a failed handoff.
