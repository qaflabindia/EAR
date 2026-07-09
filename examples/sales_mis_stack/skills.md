# Skills

## load_source_data

The raw daily sales workbook the intent names sits in the sandbox's `uploads/`
directory. Use `list_files` to confirm it is there, then write and run
whatever short Python it takes (`write_file` a script, `run_shell` to execute
it) to open the workbook, locate its header row and its last populated data
row dynamically -- by inspecting the sheet itself, never by assuming a fixed
row count -- and run the small sanity check the mis manual's Section 1
describes. Report what was found before sanity-checking proceeds in earnest.

## run_sanity_check

Write and run the validation code the mis manual's Section 2 describes:
schema (every required column present), count (row total and calendar
coverage), nulls (skip the workbook's own formula-derived achievement-percent
columns, which carry no cached value until Excel opens them) and duplicates
(same date, branch, product, channel and RM). Stage the clean result -- with
achievement percentages recomputed from the raw counts -- to a file in the
sandbox workspace the next step reads, and write the anomaly report next to
it. State the findings in plain terms: clean, or every anomaly named.

## slice_and_dice

Write and run the code the mis manual's Section 3 describes: read the staged
clean dataset and fill the dashboard template's sheets -- KPIs, monthly
trend, asset/liability and channel-mix boxes, product performance, branch
performance, RM leaderboard, top-10 branches -- and the five charts bound to
those ranges. Read the product, branch and RM counts from the data itself;
resize a section -- insert or delete rows, rewrite its chart ranges and its
total row -- when the day's data carries more or fewer of any of them than
the template shipped with. Confirm which sheets were touched and name any
section that had to be resized.

## validate_completed_dashboard

Write and run the reconciliation code the mis manual's Section 4 describes:
recompute every headline total independently from the staged clean dataset
and compare it, sheet by sheet, to what actually landed in the completed
dashboard. Write the validation log recording a pass or fail per check --
never one averaged verdict. State the overall status in the first sentence,
then name every failing check and its reconciliation gap.

## write_delivery_note

Draft a short note to Business and Leadership stating whether this cycle's
dashboard is validated and ready, the headline KPIs (total accounts, total
amount, achievement against target), and -- only when validation failed --
which reconciliation broke and what must be fixed before redelivery.
