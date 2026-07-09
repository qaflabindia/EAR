# Skills

## read_excel

A workbook is not a document -- it is several sheets of ranges, formats,
merges, tables and charts, and reading one naively is how numbers get
invented. Every script that reads a workbook follows these rules:

- Enumerate every sheet (`wb.sheetnames`) and state which ones were read;
  never assume the first sheet is the only one.
- Locate each sheet's real data range dynamically: find the header row by
  scanning for its expected labels and the last populated row by walking
  cell values downward -- `max_row`/`max_column` and the sheet's declared
  dimensions can overstate the range (formatting padding) and must never be
  trusted as the data boundary. Ranges grow and shrink between runs; use
  relative references anchored to what the sheet actually contains.
- Formulas versus values: a load with `data_only=False` shows formula text;
  `data_only=True` shows the last value Excel cached -- which is `None` for
  a workbook no one opened in Excel. Never report a formula cell's value
  from the cached side without checking; recompute derived columns from the
  raw inputs instead.
- Merged cells: only the top-left anchor of a merged range carries the
  value; the rest read `None`. Consult `ws.merged_cells.ranges` before
  concluding a cell is empty.
- Structured regions: tables (`ws.tables`) and defined names
  (`wb.defined_names`) mark ranges the workbook itself declares; prefer
  them over guessed coordinates when present.
- Formats carry meaning: a cell's `number_format` distinguishes a percent
  from a count from currency -- read it when the sheet is a formatted
  dashboard, not just the raw value.
- Charts: each chart on `ws._charts` binds to cell ranges through its
  series references; those references say which cells feed the picture.

## write_excel

Writing a workbook means keeping every ingredient consistent -- data,
formats, merges, references and charts -- not just poking values into
cells. Every script that writes a workbook follows these rules:

- Never overwrite the template: load it, fill it, save to the output path,
  and then *re-open the saved file and read back* the key cells just
  written -- a write only counts when the read-back confirms it landed.
- Dynamic fill regions: when the data carries more or fewer rows than the
  template region, resize by inserting or deleting whole rows inside the
  region -- so sections below shift intact -- and then update every
  reference that pointed at the moved range: chart series references,
  total-row formulas, defined names. A resized region with stale
  references is a silently wrong dashboard.
- Merged cells: write through the merge's top-left anchor only; writing
  into a covered cell raises or vanishes. Preserve existing merges rather
  than unmerging around them.
- Formats: a newly inserted row copies its section's cell formats
  (number_format, font, fill, borders) from an existing template row of
  that section, so the dashboard stays a dashboard; set `number_format` to
  match the column's meaning when writing fresh cells.
- Totals the next step must reconcile are written as plain computed
  numbers, never as formulas -- a headless read cannot evaluate a formula,
  and validation would compare against nothing.
- Charts already bound to template ranges keep working when the ranges are
  filled in place; when a range moved or grew, rewrite the chart's series
  references to the new extent and say so in the output.
- State in the script's printed output exactly which sheets and ranges were
  written, with row counts -- the evidence the validation step checks.

## load_source_data

The raw daily sales workbook is `uploads/daily_bank_sales_data_2025.xlsx`,
fed by ERP, CRM and APIs. Use `list_files` to confirm it is there, then write
and run whatever short Python it takes (`write_file` a script, `run_shell` to
execute it) to open the workbook, locate its header row and its last
populated data row dynamically -- by inspecting the sheet itself, never by
assuming a fixed row count -- and run the small sanity check the mis manual's
Section 1 describes. Stage the loaded result and write the anomaly report
next to it when anomalies are found. Report how many rows and which columns
were actually read -- numbers observed from the file, never assumed -- before
sanity-checking proceeds in earnest.

## run_sanity_check

Run `workspace/validate_data.py`; when it does not exist yet, author it
first -- the validation code the mis manual's Section 2 describes: schema
(every required column present), count (row total and calendar coverage),
nulls (skip the workbook's own formula-derived achievement-percent columns,
which carry no cached value until Excel opens them) and duplicates (same
date, branch, product, channel and RM). The script must read the data range
dynamically -- locate the header row and the last populated row by
inspecting the data itself, relative references only, never a fixed row
count, because the source grows every day. Write the clean result -- with
achievement percentages recomputed from the raw counts -- to the file the
next step reads. State the findings in plain terms with the counts each
check observed: clean, or every anomaly named.

## slice_and_dice

Run `workspace/generate.py`; when it does not exist yet, author it first --
the code the mis manual's Section 3 describes: read the clean dataset and
fill `uploads/daily_bank_sales_dashboard_2025.xlsx` across all dashboard
sheets -- KPIs, monthly trend,
asset/liability and channel-mix boxes, product performance, branch
performance, RM leaderboard, top-10 branches -- and the five charts bound
to those ranges. The script must be dynamic: read the product, branch and
RM counts from the data itself, and resize a section -- insert or delete
rows, rewrite its chart ranges and its total row -- when the day's data
carries more or fewer of any of them than the template shipped with, since
fill regions grow and shrink vertically as the source does. It must be
failsafe: any sheet or region it cannot fill is named loudly in its output,
never skipped silently. Preserve the workbook's cell formats while filling
values, formulas, total rows and charts. Confirm which sheets were touched,
with how many rows each, and name any section that had to be resized. Output
the Completed Sales Dashboard.

## validate_completed_dashboard

Run `workspace/validate_dashboard.py`; when it does not exist yet, author
it first -- the reconciliation code the mis manual's Section 4 describes:
recompute every headline total independently from the clean dataset and
compare it, sheet by sheet, to what actually landed in the completed
dashboard. Write the validation log recording a pass or fail per sheet and
per check -- never one averaged verdict. When every sheet passes, save the
final workbook as Validated dashboard.xlsx. State the overall status in the
first sentence, then name every failing check and its reconciliation gap.

## write_delivery_note

Draft a short note to Business and Leadership stating whether this cycle's
dashboard is validated and ready, the headline KPIs (total accounts, total
amount, achievement against target), and -- only when validation failed --
which reconciliation broke and what must be fixed before redelivery.
