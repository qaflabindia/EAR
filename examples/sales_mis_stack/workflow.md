# Workflows

## Sales MIS Workflow

1. Load Data. Suppliers: ERP, CRM and APIs. Inputs: raw source data in `daily_bank_sales_data_2025.xlsx`. Process: extract and load the workbook, then run a small sanity check before downstream steps trust it. Outputs: staged dataset and an anomaly report when any anomaly is found. Customers: Step 2. (Sales MIS Guru)
2. Sanity Check. Suppliers: Step 1. Inputs: loaded data and `validate_data.py`. Process: validate schema, count, nulls and duplicates by running `validate_data.py`; if the script is absent, create it. The script must discover each sheet's active read range dynamically with relative cell references, not fixed coordinates or row counts. Outputs: clean data. Customers: Step 3. (Sales MIS Guru)
3. Slice & Dice. Suppliers: Step 2. Inputs: clean data, `daily_bank_sales_dashboard_2025.xlsx` and `generate.py`. Process: slice and dice according to the dashboard sheets, updating every sheet with the right data fill regions, cell formats and charts by running `generate.py`; if the script is absent, create it. The script must handle source data and dashboard fill-in ranges that grow or shrink vertically and must fail safely by naming any sheet or region it cannot fill. Outputs: Completed Sales Dashboard. Customers: Step 4. (Dashboard Analyst)
4. Dashboard Validation. Suppliers: Step 3. Inputs: Completed Sales Dashboard.xlsx and `validate_dashboard.py`. Process: reconcile sums and counts between the completed dashboard and the source data sheet by sheet; if the script is absent, create it. Outputs: Validated dashboard.xlsx as the final workbook, plus a validation log that captures pass or fail sheet by sheet. Customers: Business and Leadership. (MIS Controller)

### Deliverable

The cycle's outcome as structured facts, extracted from the prose and judged
against these meanings before delivery. A cycle scoped to one step of the
workflow reports `pending` for any field only a later step can know -- that
is conformant; a hedge or an invented number is not:

- status: exactly one of validated, blocked or pending -- validated only
  when this cycle's own step completed with its outputs verified on disk
- anomalies found: the anomaly count the sanity check reported, zero when
  clean, pending before the sanity check has run
- reconciliation gap: the widest percentage gap any validated check found,
  pending before the dashboard validation step has run
- headline kpis: total accounts, total amount and both achievement
  percentages, pending before they have been computed from the data
