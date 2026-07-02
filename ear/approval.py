"""Approval -- a human's verdict on a cycle that governance parked.

A policy authored with `Approval: required` in policy.md converts its hard
block into a parkable gate: when such a policy is violated, the cycle is
not refused -- it raises `ApprovalRequired` and waits for a human. The
approval itself is a markdown document, like everything else at the
boundary:

    # Approval -- Underwrite a $60,000 loan

    Verdict: approved
    Approver: lakshminarasimhan.santhanam@gigkri.com

    > Reviewed the exception; the collateral covers it.

Dropped as `approval.md` beside `intent.md` (or `approvals/<name>.md` for
`intents/<name>.md`), the next `Exchange.run` releases the parked cycle:
an approved verdict lets the Governor pass the gate -- on the record, with
the approver's name and note -- and a rejected verdict blocks it exactly
like any violation. A verdict outside the vocabulary is None: excluded,
never guessed, and the Exchange refuses it loudly rather than leaving the
cycle silently parked.

The judgment of *whether* the gate triggers stays the model's (the policy
statement is judged like any other); the decision to *waive* it belongs
only to a human; and code enforces both -- the same split as everywhere
else in this runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .section import Section, labelled_blocks, normalize, quote, unquote

_APPROVE_WORDS = {"approved", "approve", "granted", "yes", "proceed", "accepted"}
_REJECT_WORDS = {"rejected", "reject", "denied", "declined", "refused", "no"}


class ApprovalRequired(PermissionError):
    """Raised when a cycle parks on approval-required policies instead of
    proceeding or blocking. A PermissionError, so any handler that treats
    governance stops as refusals keeps working; carries the policies the
    approval must resolve."""

    def __init__(self, policies: list) -> None:
        self.policies = list(policies)
        names = ", ".join(policy.name for policy in self.policies)
        super().__init__(f"Approval required: {names}")


@dataclass
class Approval:
    """One human verdict: approved or rejected (None when unreadable),
    by whom, and why."""

    verdict: Optional[bool] = None
    approver: str = ""
    note: str = ""

    @classmethod
    def from_markdown(cls, markdown: str) -> "Approval":
        """Read an approval document. The whole document is read as one
        section, so `Verdict:`/`Approver:` lines work wherever the human
        wrote them, and any blockquote is the note."""
        lines = markdown.replace("\r\n", "\n").split("\n")
        body = Section(name="approval", lines=lines).body(field_keys=("verdict", "approver"))
        blocks = labelled_blocks(lines)
        return cls(
            verdict=cls._read_verdict(body.field("verdict")),
            approver=body.field("approver"),
            note=blocks.get("note", "") or unquote(lines),
        )

    def to_markdown(self, title: str = "") -> str:
        lines = [f"# Approval -- {title}" if title else "# Approval", ""]
        lines.append(f"Verdict: {'approved' if self.verdict else 'rejected'}")
        if self.approver:
            lines.append(f"Approver: {self.approver}")
        if self.note:
            lines += ["", quote(self.note)]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _read_verdict(value: str) -> Optional[bool]:
        words = normalize(value).split()
        if not words:
            return None
        if words[0] in _APPROVE_WORDS:
            return True
        if words[0] in _REJECT_WORDS:
            return False
        return None