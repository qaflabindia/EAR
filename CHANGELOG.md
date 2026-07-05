# Changelog

All notable changes to EAR are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
has not yet made a versioned release, so entries accumulate under
`[Unreleased]` until the first tagged version.

## [Unreleased]

### Added
- CI: a GitHub Actions workflow (`.github/workflows/ci.yml`) running the
  test suite on Python 3.10-3.12, plus a dedicated bare-environment job
  that installs EAR with zero third-party packages and asserts none crept
  in before running the suite against that install.
- `LICENSE`: the MIT license text, matching the license already declared
  in `pyproject.toml`.

### Fixed
- MCP client: a timed-out call used to leave its reader thread alive,
  racing later calls' own reader threads for the same stdout bytes and
  occasionally stealing a later call's response. Replaced with one
  persistent reader thread per connection, dispatching responses by
  JSON-RPC id.
- Retention: a declared retention window ("keep 90 days") only rotated
  the trail when the Journey runner's batch pass crossed it; a plain
  `Runtime.reason()` cycle silently ignored it. `apply_retention()` now
  runs from both `Runtime.reason` (every cycle) and `Journeys.run_all`
  (as a batch-pass backstop).
- Secrets hygiene: `KubeConfig.token`, `ModelBinding.api_key` and
  `LM.api_key` are now `field(repr=False)`, so a stray `print`/`repr`/
  traceback dump can no longer leak a credential.
- Documentation drift: the server's bearer-token guard was documented as
  exempting `/health`; the code (and its test) actually requires the
  token there too. Docs corrected to match the code.

## [0.1.0] - initial native parity milestone

The first complete pass of the native-parity plan
(`docs/IMPLEMENTATION_PLAN.md`, phases N1-N4), plus the runtime substrate
it stands on. Highlights:

### Core authoring & runtime
- Markdown-authored stacks: skills, personas, workflows, processes,
  policies and memory, stacked and loaded by `Loader` with zero required
  Python.
- The full reasoning pipeline (`Runtime.reason`): initialize, discover,
  select, delegate, compose, schedule, govern, deliberate, decide,
  validate, remember, research, explain, audit, store, learn, adapt.
- `Judgment`/`Section`: native structured prompting over markdown, no
  JSON-schema library.
- `ModelBinding` and `LM`: a dependency-free LLM client speaking to
  Anthropic natively and any OpenAI-compatible provider over `api_base`.

### N1 - reasoning & optimization depth (parity: DSPy + GEPA)
- LM retries with backoff, on the record; per-judgment tokens and
  latency; prose-declared `Pricing` turning usage into dollars.
- `Optimizer.search`: iterative, evaluation-graded instruction search
  with worked-example demos and persisted, loader-applied instructions.

### N2 - evaluation & knowledge depth (parity: LangSmith/Phoenix, LlamaIndex)
- `Examiner`: report history with regression diffs, rubric-scored
  criteria, and A/B stack comparison via judged preference.
- `Knowledge`: native BM25 narrowing, a persisted gist index, and URL
  knowledge sources fetched over the native HTTPS client.

### N3 - execution depth (parity: LangGraph, Temporal runtime, AutoGen)
- `Journey`: prose-authored routing with revisit budgets, leg retry
  policies, a batch runner with deadline escalation, and event documents.
- `Panel`: judged next-speaker selection, early consensus, and tools
  usable inside a persona's turn.

### N4 - governance & connectivity depth (parity: Temporal governance, LangChain reach)
- Approver allow-lists and tool-scoped policies.
- A hash-chained, tamper-evident reasoning trail with `verify()`.
- Retention rotation and a native usage ledger.
- `McpClient`: a native MCP client (stdio JSON-RPC from the standard
  library) binding a declared server's tools into the cycle.

### Beyond the original plan
- `Router`: an omni-route, provider-agnostic `ModelBinding` with routing
  strategies, fallback and circuit-breaking across 250+ providers.
- `GoalKeeper`: session goals with typed blockers and a bounded
  autonomous continuation loop.
- `Sandbox`: a confined per-instance workspace and governed command
  runner, nesting per subagent.
- `Delegator`/sub-agent seams: declarative sub-agents (Delegator,
  Synthesizer) over the Sandbox.
- `Kernel` and `Server`: an OS-style scheduler (run-or-sleep idle loop)
  and a stdlib HTTP control plane over it.
- `KubeProvider` (`ear.k8s`): running instances as Kubernetes Jobs/
  CronJobs, spoken natively over the REST API.
- `Dashboard` and `Monitor`: a self-contained HTML runtime board and a
  live ANSI-truecolor TUI of the fleet.
