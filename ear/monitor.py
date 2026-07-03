"""Monitor -- the factory assembly line: a live, premium terminal view of
a whole fleet of runtime instances, rendered from the reasoning trail with
zero dependencies.

Where the Dashboard is the board you open in a browser, the Monitor is the
wall of screens in the control room. Each runtime instance is an assembly
lane; the cycle flowing down it lights up the pipeline stations it passed
-- govern, discover, deliberate, decide, audit -- and a conveyor of recent
outcomes streams past, coloured by health. A gradient banner, a running
clock, KPI tiles, braille sparklines, a pulse on every live instance, and
a spinner that never stops: the picture updates itself on a tick, so an
operator watches the fleet breathe rather than refreshing a page.

It is drawn with nothing but ANSI escape codes and Unicode box/block
glyphs from the standard library -- truecolor where the terminal supports
it, and a plain file's worth of Python. `render_frame(...)` returns one
frame as a string (so it is testable without a terminal); `run(...)` drives
the live loop over the alternate screen buffer, restoring the terminal on
exit. The data is the same fleet the Dashboard reads -- one source of truth
for health and progress, two ways to look at it.

    from ear import Monitor
    Monitor().run({"lending": rt_a, "mortgage": rt_b})   # live
    Monitor().run("trails/")                              # a directory of trails
    frame = Monitor().render_frame(fleet, frame=0)        # one still

Run it as a module against a directory of JSONL trails:

    python -m ear.monitor trails/
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .dashboard import _CATEGORY, _fleet_sources, _fleet_summary

# -- ANSI (truecolor) --------------------------------------------------------

_ESC = "\x1b["
RESET = _ESC + "0m"
BOLD = _ESC + "1m"
DIM = _ESC + "2m"
_HOME = _ESC + "H"
_CLEAR = _ESC + "2J"
_CLEAR_TAIL = _ESC + "J"
_HIDE = _ESC + "?25l"
_SHOW = _ESC + "?25h"
_ALT = _ESC + "?1049h"
_UNALT = _ESC + "?1049l"


def _fg(rgb: tuple) -> str:
    return f"{_ESC}38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _bg(rgb: tuple) -> str:
    return f"{_ESC}48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


# A tasteful dark control-room palette.
INK = (232, 234, 240)
MUTE = (120, 128, 142)
FAINT = (74, 80, 92)
PANEL = (22, 25, 33)
LINE = (40, 45, 56)
ACCENT = (108, 122, 255)
CYAN = (56, 214, 226)
GOOD = (60, 208, 130)
WARN = (240, 176, 64)
BAD = (240, 84, 84)

# Health -> colour and glyph.
_HEALTH = {
    "healthy": (GOOD, "●"),
    "attention": (WARN, "◐"),
    "broken": (BAD, "✗"),
}

# Category -> colour, so the assembly stations read as colour families that
# match the HTML dashboard's stage colouring.
_CAT_RGB = {
    "govern": (240, 140, 60),
    "reason": (108, 130, 255),
    "flow": (60, 200, 150),
    "know": (30, 200, 170),
    "tool": (150, 110, 255),
    "reflect": (40, 190, 210),
    "learn": (70, 190, 90),
    "meter": (240, 160, 40),
    "neutral": (120, 128, 142),
}

# The curated assembly line: the pipeline stations a cycle flows through,
# each mapped to the trail stages that light it up. Order is the order of
# work, left to right.
_STATIONS = [
    ("GOV", ("policy", "approval", "escalation")),
    ("DIS", ("discovery",)),
    ("SEL", ("selection",)),
    ("SCH", ("scheduling",)),
    ("DEL", ("delegation",)),
    ("RES", ("retrieval", "indexing")),
    ("TOL", ("tool",)),
    ("DLB", ("deliberation", "conversation")),
    ("EXP", ("explanation",)),
    ("AUD", ("audit", "contract")),
    ("LRN", ("adaptation",)),
]

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BLOCKS = " ▁▂▃▄▅▆▇█"
_PULSE = "◦∙●∙"


@dataclass
class Monitor:
    """Renders and drives the live assembly-line view of a fleet."""

    max_lanes: int = 14

    # -- one frame (testable without a terminal) ------------------------------

    def render_frame(
        self, source: Any, width: Optional[int] = None, frame: int = 0, now: Optional[datetime] = None
    ) -> str:
        moment = now or datetime.now(timezone.utc)
        width = max(72, width or shutil.get_terminal_size((100, 40)).columns)
        fleet = _fleet_sources(source)
        lanes = [_lane(name, log, strategy, moment) for name, log, strategy in fleet]
        lines: list[str] = []
        lines += _banner(lanes, frame, moment, width)
        lines.append("")
        if not lanes:
            lines.append(_pad("  " + _fg(MUTE) + "no runtime instances on the line yet" + RESET, width))
        for index, lane in enumerate(lanes[: self.max_lanes]):
            lines += _render_lane(lane, frame, width, index)
        if len(lanes) > self.max_lanes:
            lines.append(_pad(f"  {_fg(MUTE)}… and {len(lanes) - self.max_lanes} more lanes{RESET}", width))
        lines.append("")
        lines.append(_footer(lanes, frame, moment, width))
        return "\n".join(_pad(line, width) for line in lines)

    # -- the live loop --------------------------------------------------------

    def run(self, source: Any, fps: float = 6.0, once: bool = False) -> None:  # pragma: no cover - blocking
        """Drive the live view until Ctrl-C. `source` is a fleet the same
        way the Dashboard takes one -- a dict of runtimes, a list, or a
        directory of JSONL trails (reloaded each frame, so a separate
        process writing the trail is watched live)."""
        import sys

        interval = 1.0 / max(fps, 0.5)
        out = sys.stdout
        out.write(_ALT + _HIDE + _CLEAR)
        out.flush()
        frame = 0
        try:
            while True:
                canvas = self.render_frame(source, frame=frame)
                out.write(_HOME + canvas + _CLEAR_TAIL)
                out.flush()
                if once:
                    break
                time.sleep(interval)
                frame += 1
        except KeyboardInterrupt:
            pass
        finally:
            out.write(RESET + _SHOW + _UNALT)
            out.flush()


# -- lane data ---------------------------------------------------------------


@dataclass
class _Lane:
    name: str
    status: str
    freshness: str
    cycles: int
    tokens: int
    dollars: Optional[float]
    latency: int
    spark: list
    stations: set
    recent: list  # recent cycle statuses, oldest -> newest


def _lane(name: str, log: Any, strategy: Any, moment: datetime) -> _Lane:
    from .dashboard import _cycle_status

    summary = _fleet_summary(name, log, strategy, moment)
    cycles = sorted({record.cycle for record in log.records})
    latest = cycles[-1] if cycles else None
    stations = {record.stage for record in log.for_cycle(latest)} if latest is not None else set()
    recent = [_cycle_status(log.for_cycle(cycle)) for cycle in cycles[-24:]]
    return _Lane(
        name=name,
        status=summary["status"],
        freshness=summary["freshness"],
        cycles=summary["cycles"],
        tokens=summary["tokens"],
        dollars=summary["dollars"],
        latency=summary["latency"],
        spark=summary["spark"],
        stations=stations,
        recent=recent,
    )


# -- rendering pieces --------------------------------------------------------


def _banner(lanes: list, frame: int, moment: datetime, width: int) -> list:
    title = "EAR · ASSEMBLY LINE"
    shimmered = _gradient(title, frame)
    spin = _fg(CYAN) + _SPINNER[frame % len(_SPINNER)] + RESET
    clock = _fg(MUTE) + moment.strftime("%H:%M:%S UTC") + RESET
    live = _fg(GOOD) + BOLD + "▶ LIVE" + RESET
    top = f"{_fg(LINE)}╭{'─' * (width - 2)}╮{RESET}"
    header = _row(width, f"  {BOLD}{shimmered}{RESET}", f"{live} {spin}  {clock}  ")

    counts = {"healthy": 0, "attention": 0, "broken": 0}
    for lane in lanes:
        counts[lane.status] += 1
    active = sum(1 for lane in lanes if lane.freshness == "active")
    tiles = [
        _tile("INSTANCES", str(len(lanes)), INK),
        _tile("LIVE", str(active), CYAN),
        _tile("● HEALTHY", str(counts["healthy"]), GOOD),
        _tile("◐ ATTN", str(counts["attention"]), WARN),
        _tile("✗ BROKEN", str(counts["broken"]), BAD),
    ]
    kpi1 = _row(width, "  " + "   ".join(tiles), "")
    totals = [
        _tile("CYCLES", _num(sum(l.cycles for l in lanes)), INK),
        _tile("TOKENS", _num(sum(l.tokens for l in lanes)), ACCENT),
        _tile("COST", _cost(sum((l.dollars or 0) for l in lanes), any(l.dollars is not None for l in lanes)), GOOD),
        _tile("LATENCY", _dur(sum(l.latency for l in lanes)), CYAN),
    ]
    kpi2 = _row(width, "  " + "   ".join(totals), "")
    bottom = f"{_fg(LINE)}╰{'─' * (width - 2)}╯{RESET}"
    return [top, header, kpi1, kpi2, bottom]


def _render_lane(lane: _Lane, frame: int, width: int, index: int) -> list:
    colour, glyph = _HEALTH[lane.status]
    dot = _pulse(colour, frame, index) if lane.freshness == "active" else _fg(colour) + glyph + RESET
    name = _fg(INK) + BOLD + _fit(lane.name, 18) + RESET
    line1 = f"  {dot} {name}  {_stations(lane, frame)}"

    spark = _fg(_lerp(FAINT, colour, 0.9)) + _sparkline(lane.spark) + RESET
    conveyor = _conveyor(lane.recent, frame)
    stats = (
        f"{_fg(MUTE)}{lane.cycles:>4} cyc{RESET}   "
        f"{_fg(ACCENT)}{_num(lane.tokens):>7} tok{RESET}   "
        f"{_fg(GOOD)}{_cost(lane.dollars or 0, lane.dollars is not None):>9}{RESET}   "
        f"{_fg(CYAN)}{_dur(lane.latency):>7}{RESET}"
    )
    line2 = f"      {conveyor}  {spark}  {stats}"
    return [line1, line2, ""]


def _stations(lane: _Lane, frame: int) -> str:
    """The pipeline stations, each lit in its category colour when the
    latest cycle reached it, dim when it did not -- with a scanning
    highlight that sweeps the line so the lane looks alive."""
    lit_indices = [i for i, (_, stages) in enumerate(_STATIONS) if lane.stations.intersection(stages)]
    sweep = lit_indices[frame % len(lit_indices)] if lit_indices else -1
    cells = []
    for position, (code, stages) in enumerate(_STATIONS):
        reached = bool(lane.stations.intersection(stages))
        rgb = _CAT_RGB.get(_CATEGORY.get(stages[0], "neutral"), MUTE)
        if position == sweep:
            cells.append(_bg(_lerp(rgb, (0, 0, 0), 0.2)) + _fg(INK) + BOLD + code + RESET)
        elif reached:
            cells.append(_fg(rgb) + BOLD + code + RESET)
        else:
            cells.append(_fg(FAINT) + code + RESET)
    return (_fg(LINE) + "·" + RESET).join(cells)


def _conveyor(recent: list, frame: int) -> str:
    """Recent cycle outcomes as a belt of coloured blocks, a bright cell
    sweeping across so the belt appears to run."""
    if not recent:
        return _fg(FAINT) + "▕" + "░" * 10 + "▏" + RESET
    belt = recent[-14:]
    sweep = frame % len(belt)
    out = [_fg(_lerp(FAINT, ACCENT, 0.6)) + "▕" + RESET]
    for position, status in enumerate(belt):
        rgb = {"good": GOOD, "warn": WARN, "bad": BAD}.get(status, MUTE)
        glyph = "█"
        if position == sweep:
            out.append(_fg(_lerp(rgb, INK, 0.6)) + BOLD + glyph + RESET)
        else:
            out.append(_fg(rgb) + glyph + RESET)
    out.append(_fg(_lerp(FAINT, ACCENT, 0.6)) + "▏" + RESET)
    return "".join(out)


def _footer(lanes: list, frame: int, moment: datetime, width: int) -> str:
    healthy = sum(1 for l in lanes if l.status == "healthy")
    rate = int(100 * healthy / len(lanes)) if lanes else 100
    stream = _fg(CYAN) + _SPINNER[(frame + 5) % len(_SPINNER)] + RESET
    left = f"  {_fg(_health_rate_rgb(rate))}{rate}% healthy{RESET} {_fg(FAINT)}·{RESET} {_fg(MUTE)}watching {len(lanes)} lanes{RESET}"
    right = f"{_fg(MUTE)}the trail is the record{RESET} {stream}  "
    body = _row(width - 2, left, right)
    return f"{_fg(LINE)}╰{RESET}{body}{_fg(LINE)}╯{RESET}"


# -- small helpers -----------------------------------------------------------


def _tile(label: str, value: str, rgb: tuple) -> str:
    return f"{_fg(rgb)}{BOLD}{value}{RESET} {_fg(MUTE)}{label}{RESET}"


def _gradient(text: str, frame: int) -> str:
    """A shimmering gradient across the title, phase-shifted per frame."""
    out = []
    span = max(len(text), 1)
    for index, char in enumerate(text):
        t = ((index + frame) % span) / span
        out.append(_fg(_lerp(ACCENT, CYAN, 0.5 + 0.5 * _tri(t))) + char)
    return "".join(out) + RESET


def _pulse(rgb: tuple, frame: int, index: int) -> str:
    glyph = _PULSE[(frame + index) % len(_PULSE)]
    brightness = 0.55 + 0.45 * _tri(((frame + index) % 8) / 8)
    return _fg(_lerp((30, 34, 42), rgb, brightness)) + glyph + RESET


def _sparkline(values: list) -> str:
    values = [v for v in values if v is not None] or [0]
    peak = max(values) or 1
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, round(v / peak * (len(_BLOCKS) - 1)))] for v in values[-16:])


def _row(width: int, left: str, right: str) -> str:
    gap = width - _visible(left) - _visible(right)
    return left + " " * max(gap, 1) + right


def _pad(line: str, width: int) -> str:
    pad = width - _visible(line)
    return line + " " * pad if pad > 0 else line


def _fit(text: str, size: int) -> str:
    return text if len(text) <= size else text[: size - 1] + "…"


def _visible(text: str) -> int:
    """Printable width -- ANSI sequences and zero-width glyphs excluded."""
    count = 0
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            end = text.find("m", i)
            i = end + 1 if end != -1 else i + 1
            continue
        count += 1
        i += 1
    return count


def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


def _tri(t: float) -> float:
    """A 0->1->0 triangle wave for smooth shimmer/pulse."""
    return 1 - abs((t % 1.0) * 2 - 1)


def _num(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _cost(value: float, priced: bool) -> str:
    return f"${value:.2f}" if priced else "—"


def _dur(ms: int) -> str:
    if ms >= 60_000:
        return f"{ms / 60_000:.1f}m"
    if ms >= 1_000:
        return f"{ms / 1_000:.1f}s"
    return f"{ms}ms"


def _health_rate_rgb(rate: int) -> tuple:
    if rate >= 90:
        return GOOD
    if rate >= 60:
        return WARN
    return BAD


def _main() -> None:  # pragma: no cover - CLI entry
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m ear.monitor <trail-dir | trail.jsonl>")
        raise SystemExit(2)
    Monitor().run(sys.argv[1])


if __name__ == "__main__":  # pragma: no cover
    _main()
