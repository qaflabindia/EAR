"""Dashboard -- a visual runtime dashboard, the way TensorBoard is for
training runs, rendered natively to one self-contained HTML file.

EAR's ReasoningLog is the single source of truth: every judgment, on the
record. The Dashboard is a *view* of that record, never a second
instrumentation path -- it reads the log and renders it, so what you see
is exactly what the trail holds. And, like every other artifact in this
package, the view is a plain file on disk: one HTML document with its CSS,
its charts (inline SVG) and its interactions (a few lines of inline
script) embedded, no CDN, no framework, no build step, no dependency. It
opens in any browser, works offline, and never phones home.

The mapping to TensorBoard is direct: a **cycle** is a step, the
**scalars** are the tokens, latency and dollars each cycle spent, and the
trail file is the run. On top of the scalars the dashboard shows what a
training board cannot: the governance story (which policies passed,
blocked or parked), the tool calls, and every stage's reasoning, each
expandable to the inputs and rationale the model actually worked with.

Three ways in, all zero-dependency:

    Dashboard().write(runtime, "dashboard.html")   # a snapshot to disk
    html = Dashboard().render(runtime)             # the HTML as a string
    serve(runtime, port=8000)                       # a live http.server view

`render`/`write` take a Runtime or a ReasoningLog; `serve` additionally
takes a JSONL trail path (rebuilt losslessly via `ReasoningLog.from_trail`)
and re-renders on every request, so a long-running stack's board refreshes
itself -- the closest thing to `tensorboard --logdir` the standard library
allows.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .reasoning_log import ReasoningLog, ReasoningRecord

# Stage -> visual category, so kindred stages read as one colour family.
# The categories, not the exact stage names, carry meaning to the eye.
_CATEGORY = {
    "intent": "flow",
    "policy": "govern",
    "approval": "govern",
    "escalation": "govern",
    "retention": "govern",
    "discovery": "reason",
    "selection": "reason",
    "scheduling": "reason",
    "delegation": "reason",
    "deliberation": "reason",
    "decision": "reason",
    "conversation": "reason",
    "routing": "flow",
    "retry": "flow",
    "event": "flow",
    "retrieval": "know",
    "indexing": "know",
    "recall": "know",
    "tool": "tool",
    "explanation": "reflect",
    "audit": "reflect",
    "evaluation": "reflect",
    "comparison": "reflect",
    "adaptation": "learn",
    "usage": "meter",
}

# Outcome words that colour a stage chip regardless of its category: a
# block is red wherever it happens, a pass green.
_BAD = ("violated", "blocked", "failed", "rejected", "refused", "error")
_WARN = ("pending", "escalated", "ungraded")
_GOOD = ("complies", "passed", "approved", "decided", "conformant", "intact")


@dataclass
class Dashboard:
    """Renders a Runtime or ReasoningLog to a self-contained HTML page."""

    def write(self, source: Any, path: Union[str, Path], title: Optional[str] = None) -> str:
        html_text = self.render(source, title=title)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html_text, encoding="utf-8")
        return html_text

    def render(self, source: Any, title: Optional[str] = None) -> str:
        log, strategy, name = _resolve(source)
        title = title or f"{name} — Runtime Dashboard"
        records = log.records
        cycles = sorted({record.cycle for record in records})
        rows = [_cycle_row(log, strategy, cycle) for cycle in cycles]
        totals = _totals(rows)
        integrity = _integrity(log)

        parts = [
            _head(title),
            _header(name, totals, integrity, len(cycles)),
            _scalars_section(rows),
            _distribution_section(records),
            _governance_section(records),
            _tools_section(records),
            _cycles_section(log, cycles),
            _footer(),
        ]
        return _PAGE.replace("{{TITLE}}", html.escape(title)).replace("{{BODY}}", "\n".join(parts))


# -- serving -----------------------------------------------------------------


def serve(source: Any, port: int = 8000, host: str = "127.0.0.1") -> None:  # pragma: no cover - blocking loop
    """Serve a live dashboard over the standard-library HTTP server,
    re-rendering on every request so a running stack's board refreshes.
    `source` may be a Runtime, a ReasoningLog, or a path to a JSONL trail
    (reloaded each request). Ctrl-C to stop. Zero dependencies -- this is
    `http.server`, nothing more."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    dashboard = Dashboard()

    def current_html() -> str:
        origin = ReasoningLog.from_trail(str(source)) if _is_trail_path(source) else source
        return dashboard.render(origin)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = current_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            return  # keep the terminal quiet; the trail is the record

    server = HTTPServer((host, port), Handler)
    print(f"EAR dashboard live at http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def _is_trail_path(source: Any) -> bool:
    return isinstance(source, (str, Path)) and str(source).endswith((".jsonl", ".json"))


# -- data shaping ------------------------------------------------------------


def _resolve(source: Any) -> tuple[ReasoningLog, Any, str]:
    """A (log, strategy, name) triple from a Runtime, a ReasoningLog, or a
    JSONL trail path."""
    if _is_trail_path(source):
        return ReasoningLog.from_trail(str(source)), None, Path(str(source)).stem
    log = getattr(source, "reasoning_log", None)
    if log is not None:
        return log, getattr(source, "strategy", None), getattr(source, "name", "Runtime")
    if isinstance(source, ReasoningLog):
        return source, None, "Runtime"
    raise TypeError("Dashboard source must be a Runtime, a ReasoningLog, or a JSONL trail path")


def _cycle_row(log: ReasoningLog, strategy: Any, cycle: int) -> dict:
    records = log.for_cycle(cycle)
    in_tokens = sum(record.input_tokens for record in records)
    out_tokens = sum(record.output_tokens for record in records)
    latency = sum(record.latency_ms for record in records)
    calls = sum(1 for record in records if record.input_tokens or record.output_tokens)
    tools = sum(1 for record in records if record.stage == "tool")
    dollars = strategy.dollars(in_tokens, out_tokens) if strategy is not None else None
    intent = next((record.output for record in records if record.stage == "intent"), "")
    blocked = any(_verdict(record) == "bad" for record in records)
    return {
        "cycle": cycle,
        "intent": intent,
        "in": in_tokens,
        "out": out_tokens,
        "tokens": in_tokens + out_tokens,
        "latency": latency,
        "calls": calls,
        "tools": tools,
        "dollars": dollars,
        "blocked": blocked,
    }


def _totals(rows: list[dict]) -> dict:
    priced = [row["dollars"] for row in rows if row["dollars"] is not None]
    return {
        "cycles": len(rows),
        "calls": sum(row["calls"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
        "latency": sum(row["latency"] for row in rows),
        "tools": sum(row["tools"] for row in rows),
        "dollars": sum(priced) if priced else None,
    }


def _integrity(log: ReasoningLog) -> Optional[tuple[bool, str]]:
    path = getattr(log, "path", "")
    if path and Path(path).exists():
        return ReasoningLog.verify(path)
    return None


def _verdict(record: ReasoningRecord) -> str:
    text = record.output.lower()
    if any(word in text for word in _BAD):
        return "bad"
    if any(word in text for word in _WARN):
        return "warn"
    if any(word in text for word in _GOOD):
        return "good"
    return "neutral"


# -- HTML pieces -------------------------------------------------------------


def _header(name: str, totals: dict, integrity: Optional[tuple[bool, str]], cycles: int) -> str:
    dollars = f"${totals['dollars']:.4f}" if totals["dollars"] is not None else "—"
    tiles = [
        ("Cycles", str(cycles)),
        ("Model calls", str(totals["calls"])),
        ("Tokens", f"{totals['tokens']:,}"),
        ("Cost", dollars),
        ("Latency", f"{totals['latency']:,} ms"),
        ("Tool calls", str(totals["tools"])),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tile-v">{html.escape(value)}</div>'
        f'<div class="tile-k">{html.escape(label)}</div></div>'
        for label, value in tiles
    )
    if integrity is None:
        badge = '<span class="badge badge-mute">trail not persisted</span>'
    elif integrity[0]:
        badge = f'<span class="badge badge-good">✓ {html.escape(integrity[1])}</span>'
    else:
        badge = f'<span class="badge badge-bad">✗ {html.escape(integrity[1])}</span>'
    return (
        f'<header><div class="hrow"><h1>{html.escape(name)}</h1>{badge}</div>'
        f'<div class="tiles">{tile_html}</div></header>'
    )


def _scalars_section(rows: list[dict]) -> str:
    if not rows:
        return ""
    labels = [f"#{row['cycle']}" for row in rows]
    charts = [
        _bar_chart("Tokens per cycle", labels, [row["tokens"] for row in rows], "tok", "reason"),
        _bar_chart("Latency per cycle", labels, [row["latency"] for row in rows], "ms", "flow"),
    ]
    if any(row["dollars"] is not None for row in rows):
        charts.append(
            _bar_chart(
                "Cost per cycle",
                labels,
                [round((row["dollars"] or 0) * 1_000_000) for row in rows],
                "µ$",
                "meter",
            )
        )
    return _panel("Scalars", '<div class="charts">' + "".join(charts) + "</div>")


def _distribution_section(records: list[ReasoningRecord]) -> str:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.stage] = counts.get(record.stage, 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    labels = [stage for stage, _ in ordered]
    values = [count for _, count in ordered]
    cats = [_CATEGORY.get(stage, "neutral") for stage in labels]
    return _panel("Stage frequency", _bar_chart("", labels, values, "", None, per_bar_cat=cats, horizontal=True))


def _governance_section(records: list[ReasoningRecord]) -> str:
    govern = [r for r in records if r.stage in ("policy", "approval", "escalation")]
    if not govern:
        return ""
    rows = []
    for record in govern:
        verdict = _verdict(record)
        policy = html.escape(str(record.inputs.get("policy", record.stage)))
        rows.append(
            f'<tr class="v-{verdict}"><td>{policy}</td><td>{html.escape(record.stage)}</td>'
            f'<td>{html.escape(_clip(record.output, 90))}</td>'
            f'<td class="muted">{html.escape(_clip(record.rationale, 120))}</td></tr>'
        )
    table = (
        '<table class="grid"><thead><tr><th>Policy</th><th>Stage</th><th>Outcome</th>'
        f'<th>Rationale</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )
    return _panel("Governance", table)


def _tools_section(records: list[ReasoningRecord]) -> str:
    tools = [r for r in records if r.stage == "tool"]
    if not tools:
        return ""
    rows = []
    for record in tools:
        name = html.escape(str(record.inputs.get("tool", "tool")))
        verdict = _verdict(record)
        rows.append(
            f'<tr class="v-{verdict}"><td>c{record.cycle}</td><td>{name}</td>'
            f'<td>{html.escape(_clip(record.output, 110))}</td></tr>'
        )
    table = (
        '<table class="grid"><thead><tr><th>Cycle</th><th>Tool</th><th>Result</th></tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table>'
    )
    return _panel(f"Tool calls ({len(tools)})", table)


def _cycles_section(log: ReasoningLog, cycles: list[int]) -> str:
    blocks = []
    for cycle in cycles:
        records = log.for_cycle(cycle)
        intent = next((r.output for r in records if r.stage == "intent"), f"Cycle {cycle}")
        chips = "".join(_stage_chip(record) for record in records)
        details = "".join(_record_detail(record) for record in records)
        blocks.append(
            f'<details class="cycle"><summary><span class="cyc-n">#{cycle}</span>'
            f'<span class="cyc-t">{html.escape(_clip(intent, 90))}</span>'
            f'<span class="chips">{chips}</span></summary>'
            f'<div class="records">{details}</div></details>'
        )
    return _panel("Cycles", "".join(blocks))


def _stage_chip(record: ReasoningRecord) -> str:
    category = _CATEGORY.get(record.stage, "neutral")
    verdict = _verdict(record)
    mark = {"bad": "●", "warn": "◐", "good": "○"}.get(verdict, "")
    return f'<span class="chip cat-{category} v-{verdict}" title="{html.escape(record.stage)}">{mark}{html.escape(record.stage)}</span>'


def _record_detail(record: ReasoningRecord) -> str:
    spent = ""
    if record.input_tokens or record.output_tokens:
        spent = f'<span class="spent">{record.input_tokens}+{record.output_tokens} tok · {record.latency_ms} ms</span>'
    model = f'<span class="model">{html.escape(record.model)}</span>' if record.model else ""
    output = html.escape(_clip(record.output, 400)) or "<em>—</em>"
    why = f'<div class="why">{html.escape(_clip(record.rationale, 400))}</div>' if record.rationale else ""
    simple = {
        key: value
        for key, value in record.inputs.items()
        if "\n" not in str(value) and len(str(value)) < 120
    }
    meta = "".join(
        f'<span class="kv"><b>{html.escape(str(key))}</b> {html.escape(_clip(str(value), 80))}</span>'
        for key, value in simple.items()
    )
    return (
        f'<div class="rec"><div class="rec-h"><span class="rec-s cat-{_CATEGORY.get(record.stage, "neutral")}">'
        f'{html.escape(record.stage)}</span>{model}{spent}</div>'
        f'<div class="rec-o">{output}</div>{why}'
        f'<div class="rec-m">{meta}</div></div>'
    )


def _bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    unit: str,
    category: Optional[str],
    per_bar_cat: Optional[list[str]] = None,
    horizontal: bool = False,
) -> str:
    peak = max(values) if values and max(values) > 0 else 1
    if horizontal:
        bar_h, gap, label_w = 22, 8, 130
        height = len(values) * (bar_h + gap) + 8
        svg_rows = []
        for index, (label, value) in enumerate(zip(labels, values)):
            y = index * (bar_h + gap) + 4
            width = round(value / peak * (1000 - label_w - 90))
            cat = (per_bar_cat[index] if per_bar_cat else category) or "neutral"
            svg_rows.append(
                f'<text x="{label_w - 8}" y="{y + bar_h - 6}" class="axis" text-anchor="end">{html.escape(label)}</text>'
                f'<rect x="{label_w}" y="{y}" width="{max(width, 2)}" height="{bar_h}" rx="3" class="bar cat-{cat}"></rect>'
                f'<text x="{label_w + max(width, 2) + 8}" y="{y + bar_h - 6}" class="val">{value:,}</text>'
            )
        body = "".join(svg_rows)
        svg = f'<svg viewBox="0 0 1000 {height}" class="chart" preserveAspectRatio="xMidYMid meet">{body}</svg>'
    else:
        count = len(values)
        slot = 1000 / max(count, 1)
        bar_w = min(slot * 0.7, 64)
        height = 220
        floor = height - 26
        svg_cols = []
        for index, (label, value) in enumerate(zip(labels, values)):
            bar_height = round(value / peak * (floor - 20))
            x = index * slot + (slot - bar_w) / 2
            y = floor - bar_height
            cat = (per_bar_cat[index] if per_bar_cat else category) or "neutral"
            svg_cols.append(
                f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{max(bar_height, 2)}" rx="3" class="bar cat-{cat}"></rect>'
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 5}" class="val" text-anchor="middle">{value:,}</text>'
                f'<text x="{x + bar_w / 2:.1f}" y="{height - 8}" class="axis" text-anchor="middle">{html.escape(label)}</text>'
            )
        body = "".join(svg_cols)
        svg = f'<svg viewBox="0 0 1000 {height}" class="chart" preserveAspectRatio="xMidYMid meet">{body}</svg>'
    caption = f'<div class="chart-t">{html.escape(title)}{f" · {unit}" if unit else ""}</div>' if title or unit else ""
    return f'<figure class="chart-fig">{caption}{svg}</figure>'


def _panel(title: str, body: str) -> str:
    return f'<section class="panel"><h2>{html.escape(title)}</h2>{body}</section>'


def _footer() -> str:
    return (
        '<footer>Rendered by EAR — a view of the reasoning trail, the canonical record. '
        "Zero dependencies; this page is self-contained.</footer>"
    )


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _head(title: str) -> str:
    return ""  # the title rides the page template; kept as a seam


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root{
  --bg:#f7f8fa; --card:#ffffff; --ink:#1c2230; --muted:#6b7280; --line:#e6e8ec;
  --reason:#4c6ef5; --flow:#12b886; --govern:#e8590c; --know:#0ca678;
  --tool:#7048e8; --reflect:#1098ad; --learn:#2f9e44; --meter:#f08c00; --neutral:#868e96;
  --good:#2f9e44; --warn:#f08c00; --bad:#e03131;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0f1115; --card:#171a21; --ink:#e6e8ec; --muted:#9aa2ad; --line:#262b34; }
}
:root[data-theme="dark"]{ --bg:#0f1115; --card:#171a21; --ink:#e6e8ec; --muted:#9aa2ad; --line:#262b34; }
:root[data-theme="light"]{ --bg:#f7f8fa; --card:#ffffff; --ink:#1c2230; --muted:#6b7280; --line:#e6e8ec; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
header,.panel,footer{max-width:1080px;margin:0 auto;padding:0 20px}
header{padding-top:28px}
.hrow{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:22px;margin:0;font-weight:650}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  margin:30px 0 12px;font-weight:650}
.badge{font-size:12px;padding:4px 10px;border-radius:999px;font-weight:600}
.badge-good{background:rgba(47,158,68,.14);color:var(--good)}
.badge-bad{background:rgba(224,49,49,.14);color:var(--bad)}
.badge-mute{background:var(--line);color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile-v{font-size:22px;font-weight:680;letter-spacing:-.01em}
.tile-k{font-size:12px;color:var(--muted);margin-top:2px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.chart-fig{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px 6px;overflow-x:auto}
.chart-t{font-size:13px;font-weight:600;margin-bottom:6px}
.chart{width:100%;height:auto}
.bar{opacity:.92}
.val{fill:var(--muted);font-size:12px}
.axis{fill:var(--muted);font-size:12px}
.cat-reason{fill:var(--reason);color:var(--reason)} .cat-flow{fill:var(--flow);color:var(--flow)}
.cat-govern{fill:var(--govern);color:var(--govern)} .cat-know{fill:var(--know);color:var(--know)}
.cat-tool{fill:var(--tool);color:var(--tool)} .cat-reflect{fill:var(--reflect);color:var(--reflect)}
.cat-learn{fill:var(--learn);color:var(--learn)} .cat-meter{fill:var(--meter);color:var(--meter)}
.cat-neutral{fill:var(--neutral);color:var(--neutral)}
.grid{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card);
  border:1px solid var(--line);border-radius:12px;overflow:hidden}
.grid th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);padding:10px 12px;border-bottom:1px solid var(--line)}
.grid td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.grid tr:last-child td{border-bottom:none}
.muted{color:var(--muted)}
tr.v-bad td:first-child{box-shadow:inset 3px 0 var(--bad)}
tr.v-warn td:first-child{box-shadow:inset 3px 0 var(--warn)}
tr.v-good td:first-child{box-shadow:inset 3px 0 var(--good)}
.cycle{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:10px;padding:2px 6px}
.cycle summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:12px;
  padding:12px 10px;flex-wrap:wrap}
.cycle summary::-webkit-details-marker{display:none}
.cyc-n{font-weight:700;color:var(--muted);min-width:34px}
.cyc-t{font-weight:560;flex:1;min-width:180px}
.chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{font-size:11px;padding:2px 8px;border-radius:6px;border:1px solid var(--line);
  background:var(--bg);font-weight:560}
.chip.v-bad{border-color:var(--bad);color:var(--bad)}
.chip.v-warn{border-color:var(--warn);color:var(--warn)}
.records{padding:6px 10px 14px}
.rec{border-top:1px solid var(--line);padding:12px 2px}
.rec-h{display:flex;align-items:center;gap:10px;margin-bottom:5px}
.rec-s{font-weight:650;font-size:12.5px}
.model{font-size:11.5px;color:var(--muted)}
.spent{font-size:11.5px;color:var(--muted);margin-left:auto}
.rec-o{font-size:13.5px}
.why{font-size:12.5px;color:var(--muted);margin-top:4px}
.why::before{content:"why: ";font-weight:600}
.rec-m{display:flex;flex-wrap:wrap;gap:6px 12px;margin-top:6px}
.kv{font-size:11.5px;color:var(--muted)} .kv b{color:var(--ink);font-weight:600}
footer{color:var(--muted);font-size:12px;padding:26px 20px 40px;text-align:center}
</style></head>
<body>{{BODY}}</body></html>
"""
