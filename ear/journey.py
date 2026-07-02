"""Journey -- durable, resumable, step-wise execution of the stack,
native to the runtime: what workflow platforms provide as checkpointed
graphs, EAR provides as a markdown file.

A Journey walks the authored stack one step at a time for one intent.
Each leg is a **full governed cycle** on the runtime -- the Governor
(approval gates included), knowledge, tools, the trail and memory all
apply per leg -- carrying the overall intent and the earlier legs'
conclusions forward. After every leg the journey's state is written to
its markdown file, so a crash mid-journey loses at most the leg in
flight: a fresh runtime resumes exactly where the record ends. The state
file is the same natural language as everything else -- inspectable,
diffable, parsed back by the same Section codec.

Governance shapes the walk: a hard block ends the journey (`BLOCKED`), an
approval gate parks it (`PENDING APPROVAL`) until `run` is called again
with the human's `Approval`; a completed journey is settled -- running it
again replays nothing. And because the stack is the source of truth, a
journey refuses to resume over a stack whose steps no longer match the
legs it already walked: silently continuing a different plan would forge
the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from .approval import ApprovalRequired
from .intent import Intent
from .section import Section, coerce, labelled_blocks, normalize, parse_document, quote, unquote

COMPLETED = "completed"
IN_PROGRESS = "in progress"
BLOCKED = "BLOCKED"
PENDING = "PENDING APPROVAL"


@dataclass
class Leg:
    """One walked step of a journey: where, what, and how it ended."""

    number: int
    workflow: str
    instruction: str
    decision: str = ""
    status: str = IN_PROGRESS


@dataclass
class Journey:
    """A durable walk of the authored stack, one governed cycle per leg,
    its state a markdown file."""

    path: Union[str, Path]
    intent_text: str = ""
    context: dict = field(default_factory=dict)
    legs: list[Leg] = field(default_factory=list)
    status: str = IN_PROGRESS

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def decision(self) -> str:
        return self.legs[-1].decision if self.legs else ""

    def run(self, runtime: Any, intent: Optional[Intent] = None, approval: Any = None) -> str:
        """Walk the remaining legs. On the first run an `intent` is
        required; a resumed journey reads its own record. Returns the
        journey's status; the final decision is `journey.decision`."""
        authored = self._authored_steps(runtime)
        if self.path.exists():
            self._load()
            self._verify_record_matches(authored)
        elif intent is not None:
            self.intent_text, self.context = intent.text, dict(intent.context)
            # The record exists before the first leg is walked, so even a
            # crash on leg one leaves a resumable journey behind.
            self._save()
        else:
            raise ValueError(f"Journey '{self.path}' has no record yet -- the first run needs an intent")

        if self.status == COMPLETED:
            return self.status
        if self.status == BLOCKED:
            return self.status
        if self.status == PENDING and approval is None:
            return self.status

        # A parked leg is retried with the human's verdict; decided legs
        # are settled and never replayed.
        self.legs = [leg for leg in self.legs if leg.status == "decided"]
        for number in range(len(self.legs), len(authored)):
            workflow, step = authored[number]
            leg = Leg(number=number + 1, workflow=workflow.name, instruction=step.instruction)
            leg_approval, approval = approval, None  # the verdict applies to one leg only
            try:
                leg.decision = str(runtime.reason(self._leg_intent(step), approval=leg_approval))
                leg.status = "decided"
                self.legs.append(leg)
                self.status = IN_PROGRESS
                self._save()
            except ApprovalRequired as parked:
                leg.decision, leg.status = str(parked), PENDING
                self.legs.append(leg)
                self.status = PENDING
                self._save()
                return self.status
            except PermissionError as blocked:
                leg.decision, leg.status = str(blocked), BLOCKED
                self.legs.append(leg)
                self.status = BLOCKED
                self._save()
                return self.status
        self.status = COMPLETED
        self._save()
        return self.status

    # -- walking ---------------------------------------------------------------

    @staticmethod
    def _authored_steps(runtime: Any) -> list[tuple]:
        authored = [
            (workflow, step)
            for process in getattr(runtime, "processes", [])
            for workflow in process.workflows
            for step in workflow.steps
        ]
        if not authored:
            raise ValueError(f"Runtime '{runtime.name}' has no workflow steps for a journey to walk")
        return authored

    def _leg_intent(self, step: Any) -> Intent:
        text = step.instruction
        if self.intent_text:
            text += f"\n\nThe overall intent this step serves: {self.intent_text}"
        decided = [leg for leg in self.legs if leg.status == "decided"]
        if decided:
            transcript = "\n".join(f"- {leg.instruction} -> {leg.decision}" for leg in decided)
            text += f"\n\nEarlier legs concluded:\n{transcript}"
        return Intent(text=text, context=dict(self.context))

    def _verify_record_matches(self, authored: list[tuple]) -> None:
        for leg in self.legs:
            if leg.number > len(authored):
                raise ValueError(
                    f"Journey '{self.path}' has walked {leg.number} legs but the stack now authors "
                    f"only {len(authored)} steps -- the record and the stack no longer match"
                )
            _, step = authored[leg.number - 1]
            if leg.instruction != step.instruction:
                raise ValueError(
                    f"Journey '{self.path}' leg {leg.number} walked '{leg.instruction}' but the stack "
                    f"now authors '{step.instruction}' there -- resuming over a changed plan would "
                    "forge the record; start a new journey instead"
                )

    # -- the markdown record ----------------------------------------------------

    def _save(self) -> None:
        title = self.intent_text.partition("\n")[0].strip()
        lines = [
            f"# Journey -- {title}",
            "",
            f"Status: {self.status}",
            f"Legs walked: {len(self.legs)}",
            "",
            "## Intent",
            "",
            quote(self.intent_text),
        ]
        if self.context:
            lines += ["", "Context:", ""]
            lines += [f"- {key}: {value}" for key, value in self.context.items()]
        for leg in self.legs:
            lines += [
                "",
                f"## Leg {leg.number} -- {leg.workflow} ({leg.status})",
                "",
                f"Instruction: {leg.instruction}",
                "",
                "Decision:",
                quote(leg.decision),
            ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        all_lines = text.replace("\r\n", "\n").split("\n")
        header = Section(name="journey", lines=all_lines).body(field_keys=("status",))
        self.status = header.field("status") or IN_PROGRESS

        self.legs = []
        for section in parse_document(text).sections:
            key = normalize(section.name)
            if key == "intent":
                self.intent_text = unquote(section.lines)
                self.context = {}
                for bullet in section.body().bullets:
                    name, separator, value = bullet.partition(": ")
                    if not separator:
                        name, separator, value = bullet.partition(":")
                    if separator:
                        self.context[name.strip()] = coerce(value)
            elif key.startswith("leg"):
                self.legs.append(self._leg_from_section(section))

    @staticmethod
    def _leg_from_section(section: Section) -> Leg:
        # "Leg 3 -- Underwriting Workflow (decided)"
        head, _, status = section.name.rpartition(" (")
        number_part, _, workflow = head.partition(" -- ")
        number_text = "".join(ch for ch in number_part if ch.isdigit())
        body = section.body(field_keys=("instruction",))
        return Leg(
            number=int(number_text or 0),
            workflow=workflow.strip(),
            instruction=body.field("instruction"),
            decision=labelled_blocks(section.lines).get("decision", ""),
            status=status.rstrip(")").strip() or IN_PROGRESS,
        )