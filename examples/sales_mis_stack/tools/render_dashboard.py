#!/usr/bin/env python3
"""One-off viewer: reconstruct ReasoningRecords from an existing .ear/reasoning.md
trail (already-spent tokens, no new LLM calls) and render EAR's own Dashboard
HTML from them. Not part of the EAR package or the sales_mis_stack -- purely
a way to look at the trail we already paid for, the same way `to_markdown()`
already writes it, just read backwards with the same section parser EAR
itself ships (ear.section.parse_document / labelled_blocks)."""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
STACK = TOOLS_DIR.parent  # examples/sales_mis_stack
REPO_ROOT = STACK.parent.parent  # tools/ -> sales_mis_stack/ -> examples/ -> repo root
sys.path.insert(0, str(REPO_ROOT))

from ear.reasoning_log import ReasoningLog, ReasoningRecord  # noqa: E402
from ear.section import labelled_blocks, parse_document  # noqa: E402

_CYCLE_HEADING = re.compile(r"^## Cycle (\d+)", re.MULTILINE)
_STAGE_HEADING = re.compile(r"^(?P<stage>.+?)(?: -- (?P<clipped>.*))?$")
_MODEL_SUFFIX = re.compile(r"\s{2}\(([^)]+)\)\s*$")
_SPENT = re.compile(r"Spent:\s*(\d+)\+(\d+)\s*tokens,\s*(\d+)\s*ms")


def parse_trail(text: str) -> list[ReasoningRecord]:
    records = []
    cycle_splits = list(_CYCLE_HEADING.finditer(text))
    for index, match in enumerate(cycle_splits):
        cycle_num = int(match.group(1))
        start = match.end()
        end = cycle_splits[index + 1].start() if index + 1 < len(cycle_splits) else len(text)
        block = text[start:end]
        doc = parse_document("# x\n" + block)  # dummy title so ### becomes sections
        for section in doc.sections:
            heading = section.name
            model = ""
            model_match = _MODEL_SUFFIX.search(heading)
            if model_match:
                model = model_match.group(1)
                heading = heading[: model_match.start()]
            stage_match = _STAGE_HEADING.match(heading)
            stage = stage_match.group("stage").strip() if stage_match else heading.strip()
            clipped = (stage_match.group("clipped") or "").strip() if stage_match else ""

            blocks = labelled_blocks(section.lines)
            rationale = blocks.pop("why", "")
            output = blocks.pop("output", clipped)

            input_tokens = output_tokens = latency_ms = 0
            for line in section.lines:
                spent = _SPENT.search(line)
                if spent:
                    input_tokens, output_tokens, latency_ms = map(int, spent.groups())

            inputs: dict = {}
            for line in section.lines:
                bullet = re.match(r"^-\s+([^:]+):\s*(.*)$", line.strip())
                if bullet:
                    inputs[bullet.group(1).strip()] = bullet.group(2).strip()
            for key, value in blocks.items():
                inputs[key] = value

            records.append(ReasoningRecord(
                cycle=cycle_num, stage=stage, inputs=inputs, output=output, rationale=rationale,
                model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                latency_ms=latency_ms, timestamp=datetime.now(timezone.utc),
            ))
    return records


def main():
    trail_path = Path(sys.argv[1]) if len(sys.argv) > 1 else STACK / ".ear" / "reasoning.md"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else STACK / ".ear" / "dashboard.html"
    text = trail_path.read_text(encoding="utf-8")
    records = parse_trail(text)
    print(f"Reconstructed {len(records)} records across {max((r.cycle for r in records), default=0)} cycles")

    log = ReasoningLog(path=None)
    log.records = records
    log.cycle = max((r.cycle for r in records), default=0)

    from ear.dashboard import Dashboard
    html = Dashboard().render(log, title="Sales MIS Guru -- reconstructed trail")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
