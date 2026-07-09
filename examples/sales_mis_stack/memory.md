# Memory & Strategy

The runtime's operating strategy, declared in plain English. Every setting
below is read out of this prose -- nothing is hardcoded in Python.

## Context History

Keep the 30 most recent cycles verbatim; compress anything older into
summaries so the reasoning context stays bounded as history grows.

## Cross-Session Data

Persist memory, experience and learned adaptations to `.ear/session.md` so a
new session picks up exactly where the last one left off.

## Subagent Spawning

Allow spawning up to 3 subagents, each scoped to a single persona.

## Model Selection

Reason with anthropic/claude-opus-4-8, reading the credential from
ANTHROPIC_API_KEY. Allow up to 8000 tokens per reply -- a tool-loop turn
that writes a whole openpyxl script needs room for the full source, not
just a short judgment. When the credential is absent from the environment,
the runtime stays on its deterministic fallback.

## Auxiliary Model

Reason with anthropic/claude-sonnet-5, reading the credential from
ANTHROPIC_API_KEY, for two mechanical jobs on top of the always-on
deterministic compressor (`ear.caveman`): squeezing a tool result further,
and consolidating everything gathered so far into one checkpoint every 3
tool calls, so key facts stay retained rather than diluting across a
lengthening list of compressed entries. Absolute rules for both: no
fabrication -- never state a fact not literally in the source; no
shallowness -- never drop a fact that changes what the next turn should do;
no fluff; no sloppiness -- an ambiguous shorter sentence is wrong, not
concise; no context loss or distortion of any number, path, name, or
outcome. Declaring no Auxiliary Model at all would leave the deterministic
pass as the only compression, exactly as safe, just less aggressive.

## Sandbox

Each runtime runs in an isolated workspace under `.ear/box`, seeded under
`uploads/` with the raw source workbook
`daily_bank_sales_data_2025.xlsx` and the dashboard template
`daily_bank_sales_dashboard_2025.xlsx`. Shell commands time out after 90
seconds. Expose file and shell tools, so the Reasoner can read the
workbooks, write whatever code a step needs, and run it itself -- no step
script is shipped or handed to it; every script that does the loading,
sanity-checking, slicing and dicing, or reconciliation is authored by the
model, at the time it's needed, guided by the skills below and the mis
manual's rules. The manual fixes only the *names*: the model authors
`workspace/validate_data.py`, `workspace/generate.py` and
`workspace/validate_dashboard.py` itself when they are absent, and reruns
them when present.

## Toolsets

Every basic toolset is enabled:

- Internet Access: enabled
- Internet Search: enabled
- Read Documents: enabled
- Write Documents: enabled
- Code Executor: enabled
- Browser Automation: enabled
- Terminal: enabled
- Email Sender: enabled
- MCP Connector: enabled
- Environment Admin: enabled

## Reasoning Audit Trail

Log every reasoning step -- each policy judgment with its rationale, process
discovery, the deliberation with the full stacked prompt material, and the
explanation -- to `.ear/reasoning.md`, append-only across sessions, so the
trail can be reviewed and the stacked prompts optimised.

## Knowledge

The reference material the Librarian may consult and cite while running the
MIS cycle; sources resolve relative to this stack directory.

- mis manual: `knowledge/mis-manual.md`

## Skills Discovery

Rank processes by reading their descriptions against the intent, most
relevant first, and prefer a single best-fit process over a broad sweep.

## Ontological Settings

- staged dataset: the clean, schema-validated copy of the raw sales workbook
  that every downstream step reads, never the raw file itself
- reconciliation gap: the absolute percentage difference between a dashboard
  total and the same total recomputed from the staged dataset
- status: exactly one of validated, blocked or pending, never a hedge --
  validated means this cycle's step completed with outputs verified on
  disk; pending means a later step owns the answer
