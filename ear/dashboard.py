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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    "summarize": "tool",
    "checkpoint": "tool",
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

    def render(self, source: Any, title: Optional[str] = None, refresh: int = 0) -> str:
        log, strategy, name = _resolve(source)
        title = title or f"{name} — Runtime Dashboard"
        body = _runtime_body(log, strategy, name) + _footer()
        return _page(title, body, refresh)

    def render_gantt(
        self, source: Any, title: Optional[str] = None, refresh: int = 0, now: Optional[Any] = None
    ) -> str:
        """A Gantt view: every process laid out on a wall-clock time axis,
        coloured by status, with a live 'now' marker — the timeline of what
        ran when, and how each run ended. `source` is a Runtime/ReasoningLog
        (one lane per cycle) or a fleet (one lane per runtime, cycles as
        bars). Pass `refresh` (seconds) to make the page tick itself when
        served — the autonomous loop that keeps it live."""
        moment = _now(now)
        if _is_fleet(source):
            fleet = _fleet_sources(source)
            tracks = _gantt_tracks_fleet(fleet, moment)
            title = title or "Fleet — Live Gantt"
            head = _fleet_header([_fleet_summary(n, lg, st, moment) for n, lg, st in fleet])
        else:
            log, strategy, name = _resolve(source)
            tracks = _gantt_tracks_single(log, moment)
            title = title or f"{name} — Live Gantt"
            head = _header(name, _totals([_cycle_row(log, strategy, c) for c in sorted({r.cycle for r in log.records})]),
                           _integrity(log), len({r.cycle for r in log.records}))
        body = head + _gantt_section(tracks, moment) + _gantt_legend() + _footer()
        return _page(title, body, refresh)

    def write_gantt(
        self, source: Any, path: Union[str, Path], title: Optional[str] = None, refresh: int = 0
    ) -> str:
        html_text = self.render_gantt(source, title=title, refresh=refresh)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html_text, encoding="utf-8")
        return html_text

    # -- fleet: many runtimes at once -----------------------------------------

    def render_fleet(
        self, sources: Any, title: str = "Fleet — Runtime Dashboard", refresh: int = 0, now: Optional[Any] = None
    ) -> str:
        """Render a whole fleet on one page: a live Gantt across every
        runtime, an overview of each one's health and progress, cross-run
        comparison charts, and each runtime's full board a click away.
        `sources` is a dict {name: runtime}, a list of runtimes (or
        (name, runtime) pairs), or a directory of JSONL trails (one run per
        file, discovered and rebuilt from disk). Pass `refresh` to make it
        tick itself when served."""
        moment = _now(now)
        fleet = _fleet_sources(sources)
        if not fleet:
            body = _panel("Fleet", '<p class="muted">No runtimes to show yet.</p>') + _footer()
            return _page(title, body, refresh)
        summaries = [_fleet_summary(name, log, strategy, moment) for name, log, strategy in fleet]
        parts = [
            _fleet_header(summaries),
            _gantt_section(_gantt_tracks_fleet(fleet, moment), moment),
            _fleet_comparison(summaries),
            _fleet_runs(fleet, summaries),
            _footer(),
        ]
        return _page(title, "\n".join(parts), refresh)

    def write_fleet(self, sources: Any, path: Union[str, Path], title: Optional[str] = None) -> str:
        html_text = self.render_fleet(sources, title=title or "Fleet — Runtime Dashboard")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html_text, encoding="utf-8")
        return html_text


# -- serving -----------------------------------------------------------------


def create_server(
    source: Any,
    port: int = 8000,
    host: str = "127.0.0.1",
    refresh: int = 3,
    gantt: bool = False,
    status_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Build the live dashboard's HTTPServer, wired and bound, but not yet
    serving -- `serve()` below is a thin `serve_forever()` wrapper around
    this for the common case; a caller that wants to run the loop in its
    own thread (so its own main thread can keep running after a task
    finishes, and stop the server only when told to) uses this directly.

    Three routes: `GET /` re-renders on every request and tells the browser
    to tick itself every `refresh` seconds, so the page stays live without
    the server ever pushing anything. `GET /download/<name>` streams a file
    out of the sandbox's `outputs/` directory -- confined by
    `Sandbox.resolve`, so a crafted path can never escape it. `POST
    /shutdown` stops the server (and only the server -- nothing under
    `uploads/`, `workspace/` or `outputs/` is touched) from a page button,
    so a live run stays reachable exactly until a human says otherwise
    rather than dying the instant the driving script's cycles finish.

    `source` may be a Runtime, a ReasoningLog, a JSONL trail path, a
    directory of trails, or a {name: runtime} fleet; a path is reloaded
    from disk each tick, so a separate process writing the trail is watched
    live. `gantt=True` serves the Gantt timeline. `status_path` names a
    driver-owned status document (the Sales MIS step board, say): its text
    is re-read from disk on every request and rendered as the page's first
    panel with its last-written clock time -- so a run's own truth layer,
    not just EAR's reasoning trail, is what greets the reader. Zero
    dependencies -- this is `http.server`, nothing more."""
    import urllib.parse
    from http.server import BaseHTTPRequestHandler, HTTPServer

    dashboard = Dashboard()
    is_fleet = _is_fleet(source)

    def current_html() -> str:
        origin = source
        if not is_fleet and _is_trail_path(source):
            origin = ReasoningLog.from_trail(str(source))
        if gantt:
            page = dashboard.render_gantt(origin, refresh=refresh)
        elif is_fleet:
            page = dashboard.render_fleet(source, refresh=refresh)
        else:
            page = dashboard.render(origin, refresh=refresh)
        # Live-only controls spliced in just before </body> -- render()/
        # write() (the static snapshot path) stay plain, since a file on
        # disk has no server behind /download or /shutdown to answer to.
        if not is_fleet:
            extra = (
                _artifacts_section(getattr(source, "sandbox", None))
                + _outputs_section(_outputs_dir(source))
                + _shutdown_control()
            )
            page = page.replace("</body>", extra + "</body>")
        if status_path is not None:
            page = page.replace("<body>", "<body>" + _status_section(status_path), 1)
        return page

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.startswith("/download/"):
                self._serve_download(self.path[len("/download/"):])
                return
            body = current_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path == "/shutdown":
                body = (
                    "<!doctype html><meta charset='utf-8'><title>Dashboard stopped</title>"
                    "<body style='font:15px sans-serif;padding:40px;text-align:center'>"
                    "<h1>Dashboard stopped.</h1>"
                    "<p>Files in outputs/ and workspace/ are untouched.</p></body>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                # shutdown() deadlocks if called from the thread running
                # serve_forever() -- which is this request's own thread in
                # the single-threaded HTTPServer -- so it runs on a
                # one-shot thread instead.
                import threading

                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            self.send_response(404)
            self.end_headers()

        def _serve_download(self, encoded_name: str) -> None:
            # Reuse Sandbox.resolve's own tested confinement (refuses an
            # absolute path or a '..' escape) rather than re-deriving path
            # safety here -- one rule for what "inside outputs/" means.
            sandbox = getattr(source, "sandbox", None)
            name = urllib.parse.unquote(encoded_name)
            target = None
            if sandbox is not None:
                try:
                    candidate = sandbox.resolve(f"outputs/{name}")
                except Exception:  # noqa: BLE001 - SandboxViolation or a bad path both mean 404
                    candidate = None
                if candidate is not None:
                    target = candidate
            if target is None or not target.is_file():
                self.send_response(404)
                self.end_headers()
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            return  # keep the terminal quiet; the trail is the record

    server = HTTPServer((host, port), Handler)
    return server


def serve(
    source: Any, port: int = 8000, host: str = "127.0.0.1", refresh: int = 3, gantt: bool = False
) -> None:  # pragma: no cover - blocking loop
    """Serve a live dashboard, blocking, until Ctrl-C or a page's Shut Down
    button stops it -- see `create_server` for the routes and what each
    does. This is the simple case: build the server and run it in the
    calling thread. A caller that needs its own thread to keep running
    after the server stops (so it can do something once the dashboard is
    closed, rather than the process just ending) should call
    `create_server` directly and drive `serve_forever()`/`shutdown()` itself."""
    server = create_server(source, port=port, host=host, refresh=refresh, gantt=gantt)
    print(f"EAR dashboard live at http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
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


def _fleet_sources(sources: Any) -> list[tuple[str, ReasoningLog, Any]]:
    """Normalize any way of naming a fleet into (name, log, strategy)
    triples: a {name: runtime} dict, a list of runtimes or (name, runtime)
    pairs, or a directory of JSONL trails (one run per file)."""
    if isinstance(sources, dict):
        items: list[tuple[Optional[str], Any]] = list(sources.items())
    elif isinstance(sources, (str, Path)) and Path(str(sources)).is_dir():
        directory = Path(str(sources))
        files = sorted(list(directory.glob("*.jsonl")) + list(directory.glob("*.json")))
        items = [(path.stem, str(path)) for path in files]
    elif isinstance(sources, (list, tuple)):
        items = []
        for element in sources:
            if isinstance(element, (list, tuple)) and len(element) == 2 and isinstance(element[0], str):
                items.append((element[0], element[1]))
            else:
                items.append((None, element))
    else:
        items = [(None, sources)]
    fleet: list[tuple[str, ReasoningLog, Any]] = []
    for name, src in items:
        log, strategy, derived = _resolve(src)
        fleet.append((name or derived, log, strategy))
    return fleet


def _runtime_body(log: ReasoningLog, strategy: Any, name: str) -> str:
    """One runtime's dashboard body -- header tiles and every panel, minus
    the page wrapper -- so a single board and a fleet drill-down share the
    exact same view."""
    records = log.records
    cycles = sorted({record.cycle for record in records})
    rows = [_cycle_row(log, strategy, cycle) for cycle in cycles]
    totals = _totals(rows)
    integrity = _integrity(log)
    return "\n".join(
        [
            _header(name, totals, integrity, len(cycles)),
            _gantt_section(_gantt_tracks_single(log, _now(None)), _now(None)),
            _scalars_section(rows),
            _distribution_section(records),
            _governance_section(records),
            _tools_section(records),
            _cycles_section(log, cycles),
        ]
    )


def _fleet_summary(name: str, log: ReasoningLog, strategy: Any, now: Optional[Any] = None) -> dict:
    """One runtime's health and progress, distilled for the fleet overview."""
    records = log.records
    cycles = sorted({record.cycle for record in records})
    rows = [_cycle_row(log, strategy, cycle) for cycle in cycles]
    totals = _totals(rows)
    integrity = _integrity(log)
    blocked = sum(1 for row in rows if row["blocked"])
    pending = sum(
        1
        for record in records
        if record.stage in ("approval", "escalation")
        and ("pending" in record.output.lower() or "escalated" in record.output.lower())
    )
    # A cycle "fails" only through a governance stop -- a journey leg
    # exhausting its retry budget -- not because deliberation prose used
    # the word; the retry stage's output is a controlled vocabulary.
    failed = sum(1 for record in records if record.stage == "retry" and "exhausted" in record.output.lower())
    export_errors = list(getattr(log, "export_errors", []) or [])
    last = max((record.timestamp for record in records), default=None)
    status, reason = _classify(integrity, export_errors, failed, pending, blocked)
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "cycles": len(cycles),
        "calls": totals["calls"],
        "tokens": totals["tokens"],
        "latency": totals["latency"],
        "dollars": totals["dollars"],
        "blocked": blocked,
        "pending": pending,
        "failed": failed,
        "integrity": integrity,
        "last": last,
        "freshness": _freshness(last, _now(now)),
        "spark": [row["tokens"] for row in rows] or [0],
        "body": _runtime_body(log, strategy, name),
    }


# How recently a runtime must have acted to read as live. A tick or two of
# the served refresh; beyond the stale mark, a quiet runtime is flagged as
# possibly dead/hung rather than merely idle. Wall-clock heuristics, not
# judgment.
FRESH_ACTIVE_SECONDS = 15
FRESH_STALE_SECONDS = 3600


def _now(now: Optional[Any]) -> Any:
    return now if now is not None else datetime.now(timezone.utc)


def _freshness(last: Optional[Any], now: Any) -> str:
    """active / idle / stale from the age of the last recorded activity --
    the heartbeat that tells a live runtime from a quiet or dead one."""
    if last is None:
        return "idle"
    try:
        age = (now - last).total_seconds()
    except TypeError:
        return "idle"
    if age <= FRESH_ACTIVE_SECONDS:
        return "active"
    if age >= FRESH_STALE_SECONDS:
        return "stale"
    return "idle"


# -- the Gantt: processes on a wall-clock timeline ---------------------------


def _cycle_span(log: ReasoningLog, cycle: int) -> dict:
    """One cycle as a Gantt bar: when it started and ended (record
    timestamps), how it ended (status), and what it was."""
    records = log.for_cycle(cycle)
    stamps = [record.timestamp for record in records]
    start, end = (min(stamps), max(stamps)) if stamps else (None, None)
    status = _cycle_status(records)
    intent = next((record.output for record in records if record.stage == "intent"), None)
    # A cycle with no intent (load-time work like gist indexing) is labelled
    # by what it did, not "cycle 0".
    what = intent if intent else (f"{records[0].stage} (setup)" if records else f"cycle {cycle}")
    latency = sum(record.latency_ms for record in records)
    return {
        "start": start,
        "end": end,
        "status": status,
        "label": f"#{cycle}",
        "tip": f"#{cycle}: {_clip(what, 70)} · {latency} ms",
    }


def _gantt_tracks_single(log: ReasoningLog, now: Any) -> list[dict]:
    """One lane per cycle -- a staircase of runs down the page, time across
    it -- for a single runtime."""
    tracks = []
    for cycle in sorted({record.cycle for record in log.records}):
        bar = _cycle_span(log, cycle)
        tracks.append(
            {
                "label": _clip(bar["tip"].split(" · ")[0], 42),
                "health": bar["status"],
                "fresh": _freshness(bar["end"], now),
                "bars": [bar],
            }
        )
    return tracks


def _gantt_tracks_fleet(fleet: list, now: Any) -> list[dict]:
    """One lane per runtime, its cycles as bars along the shared axis -- the
    fleet's whole timeline at a glance."""
    tracks = []
    for name, log, _strategy in fleet:
        bars = [_cycle_span(log, cycle) for cycle in sorted({record.cycle for record in log.records})]
        bars = [bar for bar in bars if bar["start"] is not None]
        summary_status, _reason = _classify(
            _integrity(log),
            list(getattr(log, "export_errors", []) or []),
            sum(1 for b in bars if b["status"] == "bad"),
            sum(1 for b in bars if b["status"] == "warn"),
            0,
        )
        last = max((bar["end"] for bar in bars), default=None)
        tracks.append(
            {
                "label": name,
                "health": {"healthy": "good", "attention": "warn", "broken": "bad"}[summary_status],
                "fresh": _freshness(last, now),
                "bars": bars,
            }
        )
    return tracks


def _gantt_section(tracks: list[dict], now: Any, title: str = "Live Gantt — progression & status") -> str:
    bars = [bar for track in tracks for bar in track["bars"] if bar["start"] is not None]
    if not bars:
        return _panel(title, '<p class="muted">No activity on the timeline yet.</p>')
    t0 = min(bar["start"] for bar in bars)
    t1 = max([bar["end"] for bar in bars] + [now])
    span = max((t1 - t0).total_seconds(), 1.0)
    label_w, right_pad, row_h, gap, axis_h = 190, 24, 24, 7, 34
    plot_w = 1000 - label_w - right_pad
    height = len(tracks) * (row_h + gap) + axis_h

    def x_of(moment: Any) -> float:
        return label_w + (moment - t0).total_seconds() / span * plot_w

    grid, rows = [], []
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        gx = label_w + fraction * plot_w
        tick_time = t0 + (t1 - t0) * fraction
        grid.append(f'<line x1="{gx:.1f}" y1="0" x2="{gx:.1f}" y2="{height - axis_h}" class="gline"/>')
        grid.append(
            f'<text x="{gx:.1f}" y="{height - 12}" class="axis" text-anchor="middle">'
            f'{tick_time.strftime("%H:%M:%S")}</text>'
        )
    health_dot = {"good": "hgood", "warn": "hwarn", "bad": "hbad"}
    for index, track in enumerate(tracks):
        y = index * (row_h + gap)
        dot = health_dot[track["health"]]
        rows.append(
            f'<circle cx="9" cy="{y + row_h / 2:.1f}" r="5" class="cat-{dot}"/>'
            f'<text x="22" y="{y + row_h - 7:.1f}" class="glabel">{html.escape(_clip(track["label"], 26))}</text>'
        )
        if track.get("fresh") == "active":
            rows.append(f'<circle cx="{label_w - 12}" cy="{y + row_h / 2:.1f}" r="4" class="pulse"/>')
        for bar in track["bars"]:
            if bar["start"] is None:
                continue
            x0 = x_of(bar["start"])
            x1_ = x_of(bar["end"])
            width = max(x1_ - x0, 3)
            rows.append(
                f'<rect x="{x0:.1f}" y="{y + 3:.1f}" width="{width:.1f}" height="{row_h - 6}" rx="3" '
                f'class="gbar {bar["status"]}"><title>{html.escape(bar["tip"])}</title></rect>'
            )
    now_x = x_of(now)
    now_line = (
        f'<line x1="{now_x:.1f}" y1="0" x2="{now_x:.1f}" y2="{height - axis_h}" class="nowline"/>'
        f'<text x="{now_x:.1f}" y="10" class="nowlabel" text-anchor="end">now</text>'
        if t0 <= now <= t1
        else ""
    )
    svg = (
        f'<svg viewBox="0 0 1000 {height}" class="gantt" preserveAspectRatio="xMinYMin meet">'
        f'{"".join(grid)}{"".join(rows)}{now_line}</svg>'
    )
    return _panel(title, f'<div class="gantt-wrap">{svg}</div>')


def _gantt_legend() -> str:
    items = [
        ("gbar good", "decided / healthy"),
        ("gbar warn", "pending / awaiting"),
        ("gbar bad", "blocked / failed"),
    ]
    swatches = "".join(
        f'<span class="leg"><span class="sw {cls}"></span>{html.escape(text)}</span>' for cls, text in items
    )
    swatches += '<span class="leg"><span class="sw pulse-sw"></span>active now</span>'
    return _panel("Legend", f'<div class="legend">{swatches}</div>')


def _is_fleet(source: Any) -> bool:
    return isinstance(source, (dict, list, tuple)) or (
        isinstance(source, (str, Path)) and Path(str(source)).is_dir()
    )


def _page(title: str, body: str, refresh: int = 0) -> str:
    meta = f'<meta http-equiv="refresh" content="{int(refresh)}">' if refresh and refresh > 0 else ""
    return (
        _PAGE.replace("{{TITLE}}", html.escape(title))
        .replace("{{REFRESH}}", meta)
        .replace("{{BODY}}", body)
    )


def _classify(
    integrity: Optional[tuple[bool, str]], export_errors: list, failed: int, pending: int, blocked: int
) -> tuple[str, str]:
    """A runtime's health in one word. A broken chain is the only hard
    fault; failures and exporter errors need attention; a pending approval
    is waiting on a human; policy blocks are governance working and stay
    healthy (surfaced as a count, not a fault)."""
    if integrity is not None and not integrity[0]:
        return "broken", "audit trail chain is broken"
    if failed:
        return "attention", f"{failed} failed cycle(s)"
    if export_errors:
        return "attention", f"{len(export_errors)} exporter error(s)"
    if pending:
        return "attention", f"{pending} awaiting approval / escalated"
    if blocked:
        return "healthy", f"{blocked} policy block(s) — governance working"
    return "healthy", "all clear"


_HEALTH_CAT = {"healthy": "hgood", "attention": "hwarn", "broken": "hbad"}
_STATUS_RANK = {"healthy": 0, "attention": 1, "broken": 2}


def _cycle_row(log: ReasoningLog, strategy: Any, cycle: int) -> dict:
    records = log.for_cycle(cycle)
    in_tokens = sum(record.input_tokens for record in records)
    out_tokens = sum(record.output_tokens for record in records)
    latency = sum(record.latency_ms for record in records)
    calls = sum(1 for record in records if record.input_tokens or record.output_tokens)
    tools = sum(1 for record in records if record.stage == "tool")
    dollars = strategy.dollars(in_tokens, out_tokens) if strategy is not None else None
    intent = next((record.output for record in records if record.stage == "intent"), "")
    blocked = _cycle_status(records) == "bad"
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


# Stages whose outputs are a controlled vocabulary (complies / VIOLATED /
# PENDING / approved / FAILED / passed …). A cycle's health is read only
# from these -- never from free-text deliberation, explanation or audit
# prose, where words like "error" or "failed" appear innocently and would
# paint a healthy cycle red.
_STATUS_STAGES = ("policy", "approval", "escalation", "retry", "evaluation")


def _cycle_status(records: list) -> str:
    """A cycle's health from its governance stages alone: bad when a policy
    blocked, a leg exhausted its retries, or an evaluation failed; warn when
    an approval is pending or a journey escalated; good otherwise. A loan
    *declined* on the merits is a sound decision, not ill health -- so a
    decline reads good, only a governance stop reads bad."""
    relevant = [record for record in records if record.stage in _STATUS_STAGES]
    if any(_verdict(record) == "bad" for record in relevant):
        return "bad"
    if any(_verdict(record) == "warn" for record in relevant):
        return "warn"
    return "good"


# -- HTML pieces -------------------------------------------------------------


def _fleet_header(summaries: list[dict]) -> str:
    counts = {"healthy": 0, "attention": 0, "broken": 0}
    for summary in summaries:
        counts[summary["status"]] += 1
    priced = [s["dollars"] for s in summaries if s["dollars"] is not None]
    dollars = f"${sum(priced):.4f}" if priced else "—"
    worst = max(summaries, key=lambda s: _STATUS_RANK[s["status"]])["status"]
    tiles = [
        ("Runtimes", str(len(summaries)), "neutral"),
        ("Healthy", str(counts["healthy"]), "hgood"),
        ("Attention", str(counts["attention"]), "hwarn"),
        ("Broken", str(counts["broken"]), "hbad"),
        ("Cycles", f"{sum(s['cycles'] for s in summaries):,}", "neutral"),
        ("Tokens", f"{sum(s['tokens'] for s in summaries):,}", "neutral"),
        ("Cost", dollars, "neutral"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tile-v cat-{cat}">{html.escape(value)}</div>'
        f'<div class="tile-k">{html.escape(label)}</div></div>'
        for label, value, cat in tiles
    )
    badge_class = {"healthy": "badge-good", "attention": "badge-warn", "broken": "badge-bad"}[worst]
    badge_text = {"healthy": "all runtimes healthy", "attention": "needs attention", "broken": "trail integrity broken"}[worst]
    badge = f'<span class="badge {badge_class}">{html.escape(badge_text)}</span>'
    return (
        f'<header><div class="hrow"><h1>Fleet</h1>{badge}</div>'
        f'<div class="tiles">{tile_html}</div></header>'
    )


def _fleet_comparison(summaries: list[dict]) -> str:
    if len(summaries) < 2:
        return ""
    labels = [s["name"] for s in summaries]
    cats = [_HEALTH_CAT[s["status"]] for s in summaries]
    charts = [
        _bar_chart("Cycles per runtime", labels, [s["cycles"] for s in summaries], "", None, per_bar_cat=cats, horizontal=True),
        _bar_chart("Tokens per runtime", labels, [s["tokens"] for s in summaries], "tok", None, per_bar_cat=cats, horizontal=True),
    ]
    if any(s["dollars"] is not None for s in summaries):
        charts.append(
            _bar_chart(
                "Cost per runtime",
                labels,
                [round((s["dollars"] or 0) * 1_000_000) for s in summaries],
                "µ$",
                None,
                per_bar_cat=cats,
                horizontal=True,
            )
        )
    return _panel("Compare runtimes", '<div class="charts">' + "".join(charts) + "</div>")


def _fleet_runs(fleet: list, summaries: list[dict]) -> str:
    cards = []
    for summary in summaries:
        cards.append(
            f'<details class="run" data-key="run-{html.escape(summary["name"])}">'
            f'<summary>{_run_card(summary)}</summary>'
            f'<div class="run-body">{summary["body"]}</div></details>'
        )
    return _panel("Runtimes", "".join(cards))


def _run_card(summary: dict) -> str:
    status = summary["status"]
    dollars = f"${summary['dollars']:.4f}" if summary["dollars"] is not None else "—"
    last = summary["last"].strftime("%Y-%m-%d %H:%M UTC") if summary["last"] else "no activity"
    stats = (
        f'<span><b>{summary["cycles"]}</b> cycles</span>'
        f'<span><b>{summary["calls"]}</b> calls</span>'
        f'<span><b>{summary["tokens"]:,}</b> tok</span>'
        f'<span><b>{dollars}</b></span>'
        f'<span><b>{summary["latency"]:,}</b> ms</span>'
    )
    flags = []
    fresh = summary.get("freshness", "idle")
    fresh_class = {"active": "info fresh-active", "idle": "info", "stale": "warn"}[fresh]
    flags.append(f'<span class="flag {fresh_class}">{html.escape(fresh)}</span>')
    if summary["failed"]:
        flags.append(f'<span class="flag bad">{summary["failed"]} failed</span>')
    if summary["pending"]:
        flags.append(f'<span class="flag warn">{summary["pending"]} pending</span>')
    if summary["blocked"]:
        flags.append(f'<span class="flag info">{summary["blocked"]} blocked</span>')
    if summary["integrity"] is not None:
        ok = summary["integrity"][0]
        flags.append(
            f'<span class="flag {"info" if ok else "bad"}">{"✓ chain" if ok else "✗ chain"}</span>'
        )
    return (
        f'<div class="run-name"><span class="dot {status}" title="{html.escape(summary["reason"])}"></span>'
        f'{html.escape(summary["name"])}</div>'
        f'<div class="run-mid"><div class="run-stats">{stats}</div>'
        f'<div class="flags">{"".join(flags)}</div>'
        f'<div class="run-reason">{html.escape(summary["reason"])} · last {html.escape(last)}</div></div>'
        f'<div class="run-spark">{_sparkline(summary["spark"], _HEALTH_CAT[status])}</div>'
    )


def _sparkline(values: list[float], cat: str) -> str:
    values = values or [0]
    peak = max(values) if max(values) > 0 else 1
    width, height = 170, 30
    if len(values) == 1:
        y = height - 3 - (values[0] / peak) * (height - 6)
        points = f"0,{y:.1f} {width},{y:.1f}"
    else:
        step = width / (len(values) - 1)
        points = " ".join(
            f"{index * step:.1f},{height - 3 - (value / peak) * (height - 6):.1f}"
            for index, value in enumerate(values)
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark cat-{cat}" preserveAspectRatio="none">'
        f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )


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
        ok = verdict != "bad"
        status = f'<span class="tool-status {"good" if ok else "bad"}">{"✓" if ok else "✗"}</span>'
        rows.append(
            f'<tr class="v-{verdict}"><td>c{record.cycle}</td><td>{status}</td><td>{name}</td>'
            f'<td>{html.escape(_clip(record.output, 110))}</td></tr>'
        )
    table = (
        '<table class="grid"><thead><tr><th>Cycle</th><th></th><th>Tool</th><th>Result</th></tr>'
        f'</thead><tbody>{"".join(rows)}</tbody></table>'
    )
    return _panel(f"Tool calls ({len(tools)})", table)


def _cycles_section(log: ReasoningLog, cycles: list[int]) -> str:
    blocks = []
    for cycle in cycles:
        records = log.for_cycle(cycle)
        intent = next((r.output for r in records if r.stage == "intent"), f"Cycle {cycle}")
        chips = "".join(_stage_chip(record) for record in records)
        details = "".join(_record_detail(record, index) for index, record in enumerate(records))
        blocks.append(
            f'<details class="cycle" data-key="cycle-{cycle}"><summary><span class="cyc-n">#{cycle}</span>'
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


_DETAIL_TEXT_CAP = 20_000  # a safety valve against a pathological script's output, not a normal-case limit


def _capped(text: str, cap: int = _DETAIL_TEXT_CAP) -> str:
    text = str(text)
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n…[{len(text) - cap} more characters -- see the full .ear/reasoning.md trail]"


def _record_detail(record: ReasoningRecord, index: int = 0) -> str:
    """One stage's full detail inside an expanded cycle: what the model
    reasoned with (`deliberation`'s multi-line inputs -- capabilities,
    memory, strategy, knowledge -- were previously dropped entirely by a
    'short values only' filter; a click on a step showed none of the
    actual reasoning material), its full response (previously clipped to
    400 characters), and -- for a `tool` stage -- an explicit success/error
    badge rather than only a colour on the row."""
    spent = ""
    if record.input_tokens or record.output_tokens:
        spent = f'<span class="spent">{record.input_tokens}+{record.output_tokens} tok · {record.latency_ms} ms</span>'
    model = f'<span class="model">{html.escape(record.model)}</span>' if record.model else ""
    verdict = _verdict(record)
    status = ""
    if record.stage == "tool":
        ok = verdict != "bad"
        status = f'<span class="tool-status {"good" if ok else "bad"}">{"✓ success" if ok else "✗ error"}</span>'
    output = html.escape(_capped(record.output)) or "<em>—</em>"
    why = f'<div class="why">{html.escape(_capped(record.rationale))}</div>' if record.rationale else ""
    simple = {
        key: value
        for key, value in record.inputs.items()
        if "\n" not in str(value) and len(str(value)) < 120
    }
    # Everything the "simple" filter above excluded -- the actual reasoning
    # material for a deliberation record (capabilities, memory, strategy,
    # knowledge) or any other multi-line input -- gets its own expandable
    # field instead of being silently dropped from the dashboard.
    full_fields = {key: value for key, value in record.inputs.items() if key not in simple and str(value).strip()}
    meta = "".join(
        f'<span class="kv"><b>{html.escape(str(key))}</b> {html.escape(_clip(str(value), 80))}</span>'
        for key, value in simple.items()
    )
    fields_html = "".join(
        f'<details class="field" data-key="c{record.cycle}-r{index}-{html.escape(str(key))}">'
        f'<summary>{html.escape(str(key))}</summary>'
        f'<pre>{html.escape(_capped(value if isinstance(value, str) else repr(value)))}</pre></details>'
        for key, value in full_fields.items()
    )
    return (
        f'<div class="rec"><div class="rec-h"><span class="rec-s cat-{_CATEGORY.get(record.stage, "neutral")}">'
        f'{html.escape(record.stage)}</span>{status}{model}{spent}</div>'
        f'<div class="rec-o">{output}</div>{why}'
        f'<div class="rec-m">{meta}</div>{fields_html}</div>'
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


# -- live-only controls (serve() splices these in; render()/write() stay plain,
# since a static snapshot has no server behind /download or /shutdown) --------


def _outputs_dir(source: Any) -> Optional[Path]:
    """The sandbox's `outputs/` directory for `source` (a live Runtime), or
    None when `source` carries no sandbox -- a bare ReasoningLog, a JSONL
    trail path, a fleet. No outputs section is shown in that case."""
    sandbox = getattr(source, "sandbox", None)
    if sandbox is None:
        return None
    try:
        return sandbox.resolve("outputs")
    except Exception:  # noqa: BLE001 - a confinement violation just means no listing
        return None


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - unreachable, satisfies linters


def _list_outputs(outputs_dir: Optional[Path]) -> list[tuple[str, int]]:
    if outputs_dir is None or not outputs_dir.is_dir():
        return []
    files = []
    for path in sorted(outputs_dir.rglob("*")):
        if path.is_file():
            files.append((str(path.relative_to(outputs_dir)), path.stat().st_size))
    return files


_ARTIFACT_AREAS = (
    ("uploads", "inputs staged"),
    ("workspace", "work in progress"),
    ("outputs", "outputs produced"),
)


def _artifacts_section(sandbox: Any) -> str:
    """The sandbox's ground truth, one line per real file: what was staged
    into `uploads/`, what the run has actually written to `workspace/`, and
    what landed in `outputs/` -- each with its size and last-modified clock
    time, read from the filesystem at render time. This panel is the
    difference between a dashboard that *narrates* progress and one that
    *proves* it: a step can only claim its input was read or its artifact
    produced if the file is sitting right here."""
    if sandbox is None:
        return ""
    columns = []
    for area, caption in _ARTIFACT_AREAS:
        try:
            base = sandbox.resolve(area)
        except Exception:  # noqa: BLE001 - a confinement violation just means no listing
            continue
        rows = []
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                stamp = time.strftime("%H:%M:%S", time.localtime(stat.st_mtime))
                rows.append(
                    f'<div class="output-row"><span>{html.escape(str(path.relative_to(base)))}</span>'
                    f'<span class="muted">{_human_size(stat.st_size)} · {stamp}</span></div>'
                )
        body = "".join(rows) if rows else '<div class="output-row"><span class="muted">(empty)</span></div>'
        columns.append(
            f'<div class="artifact-area"><div class="artifact-h">{area}/ <span class="muted">— {caption}</span></div>'
            f'<div class="outputs-list">{body}</div></div>'
        )
    if not columns:
        return ""
    return _panel("Sandbox artifacts — verified on disk", '<div class="artifacts">' + "".join(columns) + "</div>")


def _outputs_section(outputs_dir: Optional[Path]) -> str:
    """A live-refreshing panel listing every file under the sandbox's
    `outputs/` as a download link -- appears the instant a step writes a
    real output (the completed dashboard, a validation log), no restart or
    re-render trigger needed since the page already ticks itself."""
    import urllib.parse

    files = _list_outputs(outputs_dir)
    if not files:
        body = '<p class="muted">No outputs yet — still running.</p>'
    else:
        rows = "".join(
            f'<div class="output-row"><a href="/download/{urllib.parse.quote(name)}" download>'
            f"{html.escape(name)}</a><span class=\"muted\">{_human_size(size)}</span></div>"
            for name, size in files
        )
        body = f'<div class="outputs-list">{rows}</div>'
    return _panel("Outputs", body)


def _status_section(status_path: Union[str, Path]) -> str:
    """The driver's own status document (a step board), rendered as the
    page's first panel and re-read from disk on every request. `# ` names
    the panel; each `## ` line becomes a step row with a health dot read
    from the board's own words (verified good, missing/gated bad, not
    reached quiet); the detail lines under it stay verbatim. The board's
    last-written clock time rides the heading, so the reader always knows
    how fresh the truth on screen is."""
    path = Path(status_path)
    title = "Run status"
    if not path.exists():
        return _panel(title, '<p class="muted">No status board written yet.</p>')
    stamp = time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime))
    blocks: list[str] = []
    detail: list[str] = []

    def flush() -> None:
        if detail:
            blocks.append(f'<pre class="status-detail">{html.escape(chr(10).join(detail))}</pre>')
            detail.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            text = line[3:].strip()
            lowered = text.lower()
            if "missing" in lowered or "gated" in lowered:
                dot = "broken"
            elif "not reached" in lowered:
                dot = "quiet"
            else:
                dot = "healthy"
            blocks.append(
                f'<div class="status-step"><span class="dot {dot}"></span>{html.escape(text)}</div>'
            )
        elif line.strip():
            detail.append(line)
    flush()
    return _panel(f"{title} — updated {stamp}", "".join(blocks))


def _shutdown_control() -> str:
    """A form (no JS required beyond a confirm() dialog) posting to
    /shutdown -- stops the HTTP server only. Files under uploads/,
    workspace/ and outputs/ are never touched; the sandbox is not
    ephemeral and this control has no path to delete anything."""
    return (
        '<div class="controls">'
        '<form method="POST" action="/shutdown" '
        "onsubmit=\"return confirm('Shut down the live dashboard? "
        "Files in outputs/ and workspace/ are not affected.')\">"
        '<button type="submit" class="btn btn-danger">⏻ Shut Down Dashboard</button>'
        "</form></div>"
    )


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _head(title: str) -> str:
    return ""  # the title rides the page template; kept as a seam


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{{REFRESH}}
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
.badge-warn{background:rgba(240,140,0,.14);color:var(--warn)}
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
.cat-hgood{fill:var(--good);color:var(--good)} .cat-hwarn{fill:var(--warn);color:var(--warn)}
.cat-hbad{fill:var(--bad);color:var(--bad)}
.gantt-wrap{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;overflow-x:auto}
.gantt{width:100%;min-width:640px;height:auto}
.gline{stroke:var(--line);stroke-width:1}
.glabel{fill:var(--ink);font-size:12px}
.gbar{opacity:.95}
.gbar.good{fill:var(--good)} .gbar.warn{fill:var(--warn)} .gbar.bad{fill:var(--bad)} .gbar.neutral{fill:var(--neutral)}
.nowline{stroke:var(--reason);stroke-width:1.5;stroke-dasharray:3 3}
.nowlabel{fill:var(--reason);font-size:11px;font-weight:600}
.pulse{fill:var(--good)}
.pulse{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.3}50%{opacity:1}}
.legend{display:flex;flex-wrap:wrap;gap:16px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px}
.leg{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--muted)}
.sw{width:14px;height:14px;border-radius:4px;display:inline-block}
.sw.good{background:var(--good)} .sw.warn{background:var(--warn)} .sw.bad{background:var(--bad)}
.sw.pulse-sw{background:var(--good);animation:pulse 1.6s ease-in-out infinite}
.flag.fresh-active{background:rgba(47,158,68,.16);color:var(--good)}
.run{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:10px}
.run>summary{list-style:none;cursor:pointer;padding:14px 16px;display:grid;
  grid-template-columns:minmax(150px,1.1fr) 2fr auto;gap:16px;align-items:center}
.run>summary::-webkit-details-marker{display:none}
.run-name{font-weight:650;display:flex;align-items:center;gap:9px}
.dot{width:11px;height:11px;border-radius:50%;flex:none}
.dot.healthy{background:var(--good)} .dot.attention{background:var(--warn)} .dot.broken{background:var(--bad)}
.dot.quiet{background:var(--neutral)}
.status-step{display:flex;align-items:center;gap:9px;font-weight:640;margin:12px 0 6px}
.status-detail{font-size:12.5px;color:var(--muted);background:var(--card);border:1px solid var(--line);
  border-radius:10px;padding:10px 14px;margin:0;white-space:pre-wrap;word-break:break-word;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.run-mid{min-width:0}
.run-stats{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12.5px}
.run-stats b{color:var(--ink);font-weight:650}
.flags{display:flex;gap:6px;flex-wrap:wrap;margin-top:5px}
.flag{font-size:11px;padding:2px 8px;border-radius:6px;font-weight:600}
.flag.warn{background:rgba(240,140,0,.14);color:var(--warn)}
.flag.bad{background:rgba(224,49,49,.14);color:var(--bad)}
.flag.info{background:var(--line);color:var(--muted)}
.run-reason{font-size:11.5px;color:var(--muted);margin-top:5px}
.run-spark{width:170px}
.spark{width:170px;height:30px;display:block}
.run-body{padding:2px 16px 14px;border-top:1px solid var(--line)}
.run-body header{padding-top:14px}
@media(max-width:640px){.run>summary{grid-template-columns:1fr}.run-spark{width:100%}.spark{width:100%}}
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
.rec-o{font-size:13.5px;white-space:pre-wrap;word-break:break-word;max-height:420px;overflow-y:auto}
.why{font-size:12.5px;color:var(--muted);margin-top:4px;white-space:pre-wrap}
.why::before{content:"why: ";font-weight:600}
.rec-m{display:flex;flex-wrap:wrap;gap:6px 12px;margin-top:6px}
.kv{font-size:11.5px;color:var(--muted)} .kv b{color:var(--ink);font-weight:600}
.tool-status{font-size:11px;font-weight:700;padding:2px 9px;border-radius:999px}
.tool-status.good{background:rgba(47,158,68,.14);color:var(--good)}
.tool-status.bad{background:rgba(224,49,49,.14);color:var(--bad)}
.field{margin-top:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.field summary{cursor:pointer;padding:6px 10px;font-size:11.5px;font-weight:650;
  color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.field pre{margin:0;padding:10px 12px;border-top:1px solid var(--line);font-size:12.5px;
  white-space:pre-wrap;word-break:break-word;max-height:480px;overflow-y:auto;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
footer{color:var(--muted);font-size:12px;padding:26px 20px 40px;text-align:center}
.artifacts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.artifact-h{font-size:12.5px;font-weight:650;margin-bottom:6px}
.outputs-list{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.output-row{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:11px 16px;border-bottom:1px solid var(--line)}
.output-row:last-child{border-bottom:none}
.output-row a{color:var(--reason);text-decoration:none;font-weight:560;word-break:break-all}
.output-row a:hover{text-decoration:underline}
.btn{font:inherit;font-size:13px;font-weight:650;padding:9px 16px;border-radius:9px;
  border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.btn:hover{border-color:var(--muted)}
.btn-danger{border-color:rgba(224,49,49,.35);color:var(--bad)}
.btn-danger:hover{background:rgba(224,49,49,.08);border-color:var(--bad)}
.controls{max-width:1080px;margin:0 auto;padding:0 20px}
.stopped-banner{max-width:1080px;margin:14px auto 0;padding:12px 16px;border-radius:10px;
  background:rgba(224,49,49,.1);border:1px solid rgba(224,49,49,.3);color:var(--bad);font-weight:600}
</style></head>
<body>{{BODY}}
<script>
/* Expanded-state cache: the page reloads itself every few seconds (and the
   server restarts between step runs), which would collapse every <details>
   the reader had opened. Each <details> carries a stable data-key; the set
   of open keys lives in localStorage and is re-applied on every load, so
   what the reader opened stays open across refreshes and restarts. */
(function () {
  var STORE = "ear-dash-open:" + location.pathname;
  function keyOf(el) {
    var parts = [], d = el;
    while (d) {
      if (!d.dataset.key) return null;
      parts.unshift(d.dataset.key);
      d = d.parentElement ? d.parentElement.closest("details[data-key]") : null;
    }
    return parts.join("/");
  }
  function load() {
    try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch (e) { return {}; }
  }
  function save(state) {
    try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {}
  }
  var state = load();
  document.querySelectorAll("details[data-key]").forEach(function (d) {
    var k = keyOf(d);
    if (k && state[k]) d.open = true;
  });
  /* toggle does not bubble; listen in the capture phase to hear them all */
  document.addEventListener("toggle", function (event) {
    var d = event.target;
    if (!d || d.tagName !== "DETAILS") return;
    var k = keyOf(d);
    if (!k) return;
    var current = load();
    if (d.open) current[k] = 1; else delete current[k];
    save(current);
  }, true);
})();
</script>
</body></html>
"""
