# Sales MIS Guru Runtime

Workflows stacked into processes, and processes stacked into the runtime this
file's title names. Each heading names a process; the prose is the description
the Discoverer reasons over; the `Workflows:` line stacks workflows from
workflow.md by name.

## Complete Sales MIS Process

Runs the daily bank sales MIS end to end: loading the raw source workbook,
`daily_bank_sales_data_2025.xlsx`, sanity-checking it into a clean staged
dataset with `validate_data.py`, slicing and dicing that data into
`daily_bank_sales_dashboard_2025.xlsx` with `generate.py`, and validating
the completed dashboard with `validate_dashboard.py` before the final
validated workbook and sheet-by-sheet validation log reach Business and
Leadership.

Workflows: Sales MIS Workflow
