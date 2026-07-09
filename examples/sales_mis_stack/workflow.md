# Workflows

## Sales MIS Workflow

1. Load the raw source data -- ERP, CRM and API extracts staged as the daily
   sales workbook -- and run the small sanity check that flags anomalies as
   they land. (Sales MIS Guru)
2. Sanity-check the loaded data against schema, count, null and duplicate
   rules, producing the clean staged dataset the rest of the cycle runs on.
   (Sales MIS Guru)
3. Slice and dice the clean data into the daily sales dashboard -- every
   sheet, fill region, format and chart -- understanding the context each one
   serves rather than mechanically copying cells. (Dashboard Analyst)
4. Validate the completed dashboard by reconciling its sums and counts back to
   the source data, sheet by sheet, and write the delivery note for Business
   and Leadership. (MIS Controller)

### Deliverable

The cycle's outcome as structured facts, extracted from the prose and judged
against these meanings before delivery:

- status: exactly one of validated or blocked
- anomalies found: the anomaly count the sanity check reported, zero when clean
- reconciliation gap: the widest percentage gap any validated check found
- headline kpis: total accounts, total amount and both achievement percentages
