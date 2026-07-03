"""ReasoningLog -- the audit trail of every reasoning step the runtime
takes, kept so the LLM's judgment is reviewable and the stacked prompts
can be optimised against what the model actually reasoned over.

Every judgment-laden stage writes one record per judgment:

    intent        the cycle opened, with the intent and its context
    policy        each Policy judgment, with the judge's rationale
    discovery     which processes were found relevant, and from what catalogue
    deliberation  the Reasoner's decision, with the full stacked capabilities
                  block and memory context it reasoned with -- the exact
                  prompt material an author reviews to optimise skills.md
    explanation   the Explainer's prose and the evidence it rested on

Records carry which model produced them ("deterministic-fallback" when no
ModelBinding was active), so offline and live cycles are distinguishable in
the same trail. Blocked cycles are logged too -- a Policy violation is
exactly what an auditor wants to see, not a gap in the record.

Declared in `memory.md` (a Reasoning Audit Trail section naming a path);
the Runtime flushes new records to that file after every cycle,
append-only, so the trail also accumulates across sessions. The file's
extension picks the codec: `.md` (the system-native default) appends
readable markdown, one `## Cycle` section per cycle with every free-text
value blockquoted so it can never be mistaken for structure; any other
extension appends JSONL for machine pipelines.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .section import quote

_CYCLE_HEADING = re.compile(r"^## Cycle (\d+)", re.MULTILINE)
_CHAIN_MD = re.compile(r"<!-- chain: ([0-9a-f]+) -->")

# The chain's seed: the previous-hash the first record links against, so a
# trail with even one record has a fixed, verifiable starting point.
GENESIS = "ear-genesis"


def _link(previous: str, payload: str) -> str:
    """One link of the tamper-evident chain: the SHA-256 (stdlib) of the
    previous link and this record's persisted payload. Editing any byte of
    any record breaks its own link and every link after it."""
    return hashlib.sha256((previous + "\n" + payload).encode("utf-8")).hexdigest()


def model_name(model_binding: Any) -> str:
    """The name a record attributes its judgment to: the bound model when
    one is active, the deterministic fallback otherwise."""
    if model_binding is not None and getattr(model_binding, "lm", None) is not None:
        return model_binding.model_id
    return "deterministic-fallback"


def calls_so_far(lm: Any) -> int:
    """Where an LM's call history stands now -- the start mark for
    attributing a stage's usage to its record."""
    history = getattr(lm, "history", None)
    return len(history) if history is not None else 0


def usage_since(lm: Any, start: int) -> Optional[dict[str, int]]:
    """The tokens, latency and retries the LM spent since `start`, summed
    -- or None when no call actually happened, so a fallback judgment is
    never billed for a model it didn't use."""
    history = getattr(lm, "history", None) or []
    calls = history[start:]
    if not calls:
        return None
    usage = {"input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "retries": 0}
    for call in calls:
        if not isinstance(call, dict):
            continue
        tokens = call.get("usage") or {}
        usage["input_tokens"] += int(tokens.get("prompt_tokens") or 0)
        usage["output_tokens"] += int(tokens.get("completion_tokens") or 0)
        usage["latency_ms"] += int(call.get("latency_ms") or 0)
        usage["retries"] += int(call.get("retries") or 0)
    return usage


@dataclass
class ReasoningRecord:
    """One logged judgment: which cycle and stage, what the stage reasoned
    over (`inputs`), what it concluded (`output`), why (`rationale`), and
    which model concluded it."""

    cycle: int
    stage: str
    inputs: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    rationale: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def render(self, width: int = 160) -> str:
        line = f"[{self.stage}] -> {_clip(self.output, width)}"
        if self.model:
            line += f"  ({self.model})"
        if self.input_tokens or self.output_tokens:
            line += f"  [{self.input_tokens}+{self.output_tokens} tok, {self.latency_ms} ms]"
        if self.rationale:
            line += f"\n    why: {_clip(self.rationale, width)}"
        return line

    def to_json(self) -> str:
        return json.dumps(
            {
                "cycle": self.cycle,
                "stage": self.stage,
                "timestamp": self.timestamp.isoformat(),
                "model": self.model,
                "inputs": self.inputs,
                "output": self.output,
                "rationale": self.rationale,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_ms": self.latency_ms,
            },
            default=str,
        )

    def to_markdown(self) -> str:
        """This record as a markdown block: one-line inputs as bullets,
        multi-line values (the stacked capabilities, a long decision) as
        blockquotes under their own label."""
        header = f"### {self.stage}"
        if self.output:
            header += f" -- {_clip(self.output, 80)}"
        if self.model:
            header += f"  ({self.model})"
        lines = [header, ""]
        simple = {key: value for key, value in self.inputs.items() if "\n" not in str(value)}
        multiline = {key: value for key, value in self.inputs.items() if "\n" in str(value)}
        if simple:
            lines += [f"- {key}: {value}" for key, value in simple.items()] + [""]
        for key, value in multiline.items():
            if str(value).strip():
                lines += [f"{key.capitalize()}:", quote(value), ""]
        if self.rationale:
            lines += ["Why:", quote(self.rationale), ""]
        if "\n" in self.output or len(self.output) > 80:
            lines += ["Output:", quote(self.output), ""]
        if self.input_tokens or self.output_tokens:
            lines += [f"Spent: {self.input_tokens}+{self.output_tokens} tokens, {self.latency_ms} ms", ""]
        return "\n".join(lines)


@dataclass
class ReasoningLog:
    """The runtime's reasoning audit trail: an ordered list of
    ReasoningRecords, grouped by cycle, flushed append-only to the trail
    file at `path` and fanned out to any attached `exporters`.

    An exporter is anything with `export(record)` (and optionally
    `flush()`) -- a native protocol, so shipping the trail to any external
    system is a few lines of your own code, never a dependency of EAR's.
    The file on disk stays the canonical record: an exporter that raises
    never breaks a cycle, its failure is kept visible in `export_errors`
    instead."""

    path: str = ""
    records: list[ReasoningRecord] = field(default_factory=list)
    cycle: int = 0
    flushed: int = 0
    flushed_cycle: Optional[int] = None
    exporters: list[Any] = field(default_factory=list)
    export_errors: list[str] = field(default_factory=list)
    # The tip of the tamper-evident hash chain -- the hash of the last
    # record flushed to `path`, which the next record links against. Seeded
    # at GENESIS and continued across sessions by `resume`.
    chain_tip: str = GENESIS

    def resume(self) -> int:
        """Continue cycle numbering from an existing trail file, so a new
        session's cycles never repeat numbers inside the same audit trail.
        A missing or unreadable file leaves the counter untouched."""
        if not self.path or not os.path.exists(self.path):
            return self.cycle
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return self.cycle
        numbers: list[int] = []
        markers: list[str] = []
        if self.path.endswith(".md"):
            numbers = [int(number) for number in _CYCLE_HEADING.findall(text)]
            markers = _CHAIN_MD.findall(text)
        else:
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                numbers.append(int(record.get("cycle", 0)))
                if record.get("chain"):
                    markers.append(str(record["chain"]))
        self.cycle = max(numbers, default=self.cycle)
        # Continue the chain from where the file ends, so a resumed session
        # links its first new record to the last persisted one.
        if markers:
            self.chain_tip = markers[-1]
        return self.cycle

    def begin_cycle(self, intent: Any) -> int:
        """Open a new cycle in the trail, recording the intent that
        started it."""
        self.cycle += 1
        self.record(
            stage="intent",
            inputs={"context": dict(getattr(intent, "context", {}) or {})},
            output=str(intent),
        )
        return self.cycle

    def record(
        self,
        stage: str,
        inputs: Optional[dict[str, Any]] = None,
        output: Any = "",
        rationale: str = "",
        model: str = "",
        usage: Optional[dict[str, int]] = None,
    ) -> ReasoningRecord:
        entry = ReasoningRecord(
            cycle=self.cycle,
            stage=stage,
            inputs=dict(inputs or {}),
            output=str(output),
            rationale=str(rationale),
            model=model,
            input_tokens=int((usage or {}).get("input_tokens") or 0),
            output_tokens=int((usage or {}).get("output_tokens") or 0),
            latency_ms=int((usage or {}).get("latency_ms") or 0),
        )
        self.records.append(entry)
        return entry

    def for_stage(self, stage: str) -> list[ReasoningRecord]:
        return [record for record in self.records if record.stage == stage]

    def for_cycle(self, cycle: int) -> list[ReasoningRecord]:
        return [record for record in self.records if record.cycle == cycle]

    def render(self, cycle: Optional[int] = None, width: int = 160) -> str:
        """The trail as readable text, one cycle per block -- the skim
        view; full inputs (the stacked capabilities, the judged context)
        stay on the records and in the JSONL file."""
        records = self.records if cycle is None else self.for_cycle(cycle)
        lines: list[str] = []
        seen_cycle: Optional[int] = None
        for record in records:
            if record.cycle != seen_cycle:
                seen_cycle = record.cycle
                lines.append(f"=== Cycle {record.cycle} ({record.timestamp:%Y-%m-%d %H:%M:%S}) ===")
            lines.append(record.render(width=width))
        return "\n".join(lines) if lines else "No reasoning recorded yet."

    def flush(self) -> Optional[str]:
        """Write records not yet flushed to the trail file at `path`
        (markdown when the path ends in `.md`, JSONL otherwise) and fan
        the same records out to every attached exporter. With no path and
        no exporters the trail stays in memory only."""
        pending = self.records[self.flushed :]
        if not pending or (not self.path and not self.exporters):
            return None
        if self.path:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            as_markdown = self.path.endswith(".md")
            with open(self.path, "a", encoding="utf-8") as handle:
                for record in pending:
                    if as_markdown:
                        if record.cycle != self.flushed_cycle:
                            self.flushed_cycle = record.cycle
                            handle.write(
                                f"\n## Cycle {record.cycle} -- {record.timestamp:%Y-%m-%d %H:%M:%S} UTC\n\n"
                            )
                        payload = record.to_markdown()
                        self.chain_tip = _link(self.chain_tip, payload)
                        # The chain hash rides an HTML comment: invisible in
                        # a rendered view, and never colliding with content.
                        handle.write(payload + f"\n<!-- chain: {self.chain_tip} -->\n")
                    else:
                        payload = record.to_json()
                        self.chain_tip = _link(self.chain_tip, payload)
                        obj = json.loads(payload)
                        obj["chain"] = self.chain_tip
                        handle.write(json.dumps(obj, default=str) + "\n")
        for exporter in self.exporters:
            try:
                for record in pending:
                    exporter.export(record)
                finish = getattr(exporter, "flush", None)
                if callable(finish):
                    finish()
            except Exception as error:  # noqa: BLE001 -- an exporter must never break a cycle
                self.export_errors.append(f"{type(exporter).__name__}: {error}")
                del self.export_errors[:-20]
        self.flushed = len(self.records)
        return self.path or None

    @classmethod
    def from_trail(cls, path: str) -> "ReasoningLog":
        """Reconstruct a log from a persisted JSONL trail -- lossless, so a
        dashboard or ledger can be built from a finished run on disk. (The
        markdown codec is a human view and not fully reconstructable; JSONL
        is the machine record, and this reads it back exactly.)"""
        log = cls(path=path)
        if not path or not os.path.exists(path) or path.endswith(".md"):
            return log
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                stamp = data.get("timestamp")
                try:
                    when = datetime.fromisoformat(stamp) if stamp else datetime.now(timezone.utc)
                except (TypeError, ValueError):
                    when = datetime.now(timezone.utc)
                log.records.append(
                    ReasoningRecord(
                        cycle=int(data.get("cycle", 0)),
                        stage=str(data.get("stage", "")),
                        inputs=dict(data.get("inputs") or {}),
                        output=str(data.get("output", "")),
                        rationale=str(data.get("rationale", "")),
                        model=str(data.get("model", "")),
                        input_tokens=int(data.get("input_tokens") or 0),
                        output_tokens=int(data.get("output_tokens") or 0),
                        latency_ms=int(data.get("latency_ms") or 0),
                        timestamp=when,
                    )
                )
        log.cycle = max((record.cycle for record in log.records), default=0)
        log.flushed = len(log.records)
        return log

    @staticmethod
    def verify(path: str) -> tuple[bool, str]:
        """Prove a persisted trail unbroken, or name the first broken link.
        Recomputes the hash chain over the file's own bytes -- so any
        edit, insertion or deletion of a record surfaces as the exact
        record where the chain first fails to reproduce."""
        if not path or not os.path.exists(path):
            return False, f"no trail file at {path!r}"
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        links = _read_chain_md(text) if path.endswith(".md") else _read_chain_jsonl(text)
        if not links:
            return False, "no chain records found -- this trail carries no integrity hashes"
        previous = GENESIS
        for index, (label, payload, stored) in enumerate(links, start=1):
            expected = _link(previous, payload)
            if stored != expected:
                return False, f"broken chain at record {index} ({label}) -- the trail was altered here or earlier"
            previous = stored
        return True, f"chain intact over {len(links)} records"

    def usage_report(self, strategy: Any = None) -> str:
        """The operational ledger, rendered from the trail: one row per
        cycle -- model calls, tokens, dollars (when Pricing is declared),
        latency and tool calls -- with totals. A markdown document, the
        same as every other artifact."""
        cycles = sorted({record.cycle for record in self.records})
        header = [
            "# Usage Report",
            "",
            "| Cycle | Model calls | In+Out tokens | Cost | Latency (ms) | Tool calls |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        totals = {"calls": 0, "in": 0, "out": 0, "latency": 0, "tools": 0, "cost": 0.0}
        priced = False
        for cycle in cycles:
            records = self.for_cycle(cycle)
            billed = [r for r in records if r.input_tokens or r.output_tokens]
            in_tokens = sum(r.input_tokens for r in records)
            out_tokens = sum(r.output_tokens for r in records)
            latency = sum(r.latency_ms for r in records)
            tools = sum(1 for r in records if r.stage == "tool")
            dollars = strategy.dollars(in_tokens, out_tokens) if strategy is not None else None
            totals["calls"] += len(billed)
            totals["in"] += in_tokens
            totals["out"] += out_tokens
            totals["latency"] += latency
            totals["tools"] += tools
            if dollars is not None:
                totals["cost"] += dollars
                priced = True
            cost_cell = f"${dollars:.6f}" if dollars is not None else "—"
            header.append(
                f"| {cycle} | {len(billed)} | {in_tokens}+{out_tokens} | {cost_cell} | {latency} | {tools} |"
            )
        total_cost = f"${totals['cost']:.6f}" if priced else "—"
        header.append(
            f"| **total** | **{totals['calls']}** | **{totals['in']}+{totals['out']}** | "
            f"**{total_cost}** | **{totals['latency']}** | **{totals['tools']}** |"
        )
        return "\n".join(header) + "\n"

    def rotate(self, retention_days: float, now: Optional[datetime] = None) -> int:
        """Retention as rotation, never silent deletion: cycles whose
        records are all older than the window are replaced by a single
        `retention` note recording how many were rotated out, and the
        trail file is rewritten (re-chained from GENESIS over the
        survivors). Returns how many records were rotated. A window that
        covers everything rotates nothing."""
        if not retention_days or not self.records:
            return 0
        moment = now or datetime.now(timezone.utc)
        cutoff = moment.timestamp() - retention_days * 86400
        expired_cycles = sorted(
            cycle
            for cycle in {record.cycle for record in self.records}
            if all(record.timestamp.timestamp() < cutoff for record in self.for_cycle(cycle))
        )
        if not expired_cycles:
            return 0
        rotated = [record for record in self.records if record.cycle in set(expired_cycles)]
        survivors = [record for record in self.records if record.cycle not in set(expired_cycles)]
        note = ReasoningRecord(
            cycle=survivors[0].cycle if survivors else self.cycle,
            stage="retention",
            inputs={
                "retention_days": retention_days,
                "rotated_cycles": expired_cycles,
                "rotated_records": len(rotated),
            },
            output=f"rotated {len(rotated)} records across {len(expired_cycles)} cycles older than {retention_days} days",
            rationale="retention is rotation, not deletion -- this note stands in for what was rotated out",
        )
        self.records = [note] + survivors
        self._rewrite()
        return len(rotated)

    def _rewrite(self) -> None:
        """Rewrite the whole trail file from the current records, re-chained
        from GENESIS. Used by rotation, the one operation that is not
        append-only -- and it stays honest by leaving the chain verifiable."""
        self.chain_tip = GENESIS
        self.flushed = 0
        self.flushed_cycle = None
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        self.flush()

    def __len__(self) -> int:
        return len(self.records)


def _read_chain_md(text: str) -> list[tuple[str, str, str]]:
    """Split a markdown trail into (label, payload, stored-hash) triples --
    each record's block text exactly as persisted, and the chain hash from
    its trailing comment. The payload is the bytes the hash was taken over,
    so verification reproduces it without reparsing the record."""
    links: list[tuple[str, str, str]] = []
    block: Optional[list[str]] = None
    for line in text.split("\n"):
        marker = _CHAIN_MD.search(line)
        if line.startswith("### "):
            block = [line]
        elif marker is not None and block is not None:
            label = block[0][4:].split(" -- ", 1)[0].strip() if block else "?"
            links.append((label, "\n".join(block), marker.group(1)))
            block = None
        elif block is not None:
            block.append(line)
    return links


def _read_chain_jsonl(text: str) -> list[tuple[str, str, str]]:
    """The same triples from a JSONL trail: the record re-serialized as the
    canonical payload that was hashed (the object minus its `chain` field),
    and the stored hash."""
    links: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or "chain" not in record:
            continue
        stored = str(record.pop("chain"))
        links.append((str(record.get("stage", "?")), json.dumps(record, default=str), stored))
    return links


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."
