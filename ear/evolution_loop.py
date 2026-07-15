"""The evolution loop -- ALCC learns, AAWDFC judges legitimacy, and the
existing evolution gate applies (framework architecture §7).

EAR already gates self-modification: `Runtime.enable_evolution(policy)` and
the `Evolver`, which walks every proposed change through the
`EvolutionPolicy`'s fences (kind allowed, explanation present, human approval
where required, a sandbox, a rollback, an evaluation). Phase 4 closes the
loop constitutionally by adding the two cognitive-plane steps that come
*before* the gate:

1. **ALCC turns Experience into candidates.** `LearningLoop.candidates`
   reads the runtime's distilled `Adaptation`s and proposes each as an
   `EvolutionChange` -- a candidate the runtime might make to itself.
   Proposing is not applying: a candidate is just a suggestion until the
   gates pass it.

2. **AAWDFC judges legitimacy.** Before any machine-created change is
   applied, `LegitimacyGate.judge` decides whether it is *fit to exist* --
   explained, constitutionally compatible, and structurally coherent. The
   `Evolver` consults `runtime.legitimacy_gate` before it runs `apply`, so an
   illegitimate change is refused on the record, never applied.

3. **AGCC verdicts still gate application.** Nothing here bypasses the
   `Evolver`: a change that `needs_approval` follows the DEFER/approval path;
   the rest apply under the same requirements as any other change. The system
   may improve itself; it may not do so outside the gate everything else
   passes through.

Reason-first above a deterministic floor throughout: with a model bound the
model judges legitimacy (`JudgeWorkflowLegitimacy`); offline a structural
floor stands in and says so. Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .evolution import EvolutionChange
from .reasoning_log import model_name


@dataclass
class LegitimacyVerdict:
    """AAWDFC's verdict on a machine-created change: fit to exist, or not,
    and why."""

    legitimate: bool
    reason: str
    basis: str = ""


@dataclass
class LegitimacyGate:
    """AAWDFC -- workflow legitimacy and synthesis governance. Judges whether
    a machine-created change is fit to exist before the Evolver applies it.

    The floor is absolute: a change with no explanation is never legitimate
    (an unexplained self-modification cannot be reviewed). Above the floor,
    legitimacy -- constitutional fit and structural coherence -- is a
    judgment: the model decides when one is bound, a permissive structural
    check stands in offline."""

    def judge(self, change: EvolutionChange, runtime: Any = None, model_binding: Any = None) -> LegitimacyVerdict:
        # The floor: an unexplained change cannot be judged legitimate.
        if not change.explanation.strip():
            verdict = LegitimacyVerdict(
                False,
                "no explanation -- a machine-created change with no stated purpose cannot be judged legitimate",
                basis="floor",
            )
            self._record(runtime, change, verdict, model_binding)
            return verdict

        binding = model_binding if model_binding is not None else getattr(runtime, "model_binding", None)
        lm = getattr(binding, "lm", None) if binding is not None else None
        if binding is not None:
            binding.activate()
            lm = getattr(binding, "lm", None)

        if lm is not None:
            verdict = self._judge(lm, change, runtime)
        else:
            # Offline: the floor already passed (there is an explanation), so
            # the structural check accepts, labelled a fallback -- a legitimacy
            # ruling the model did not make is not dressed up as one.
            verdict = LegitimacyVerdict(
                True,
                "explained and structurally acceptable to the offline floor (no model to judge constitutional fit)",
                basis="deterministic floor (no model)",
            )
        self._record(runtime, change, verdict, model_binding)
        return verdict

    def _judge(self, lm: Any, change: EvolutionChange, runtime: Any) -> LegitimacyVerdict:
        from .signatures import JudgeWorkflowLegitimacy

        constitution = self._constitution_summary(runtime)
        result = JudgeWorkflowLegitimacy.run(
            lm,
            kind=change.kind,
            name=change.name or "(unnamed)",
            description=change.description or "(none)",
            explanation=change.explanation,
            constitution=constitution,
        )
        return LegitimacyVerdict(
            legitimate=bool(getattr(result, "legitimate", False)),
            reason=str(getattr(result, "rationale", "") or "judged by the model"),
            basis="judged by the model",
        )

    @staticmethod
    def _constitution_summary(runtime: Any) -> str:
        policies = getattr(runtime, "policies", []) if runtime is not None else []
        names = "; ".join(getattr(policy, "name", "") for policy in policies[:20])
        return names or "(no runtime-scope constitution attached)"

    @staticmethod
    def _record(runtime: Any, change: EvolutionChange, verdict: LegitimacyVerdict, binding: Any) -> None:
        log = getattr(runtime, "reasoning_log", None)
        if log is None:
            return
        log.record(
            stage="legitimacy",
            inputs={"kind": change.kind, "name": change.name, "basis": verdict.basis},
            output=("LEGITIMATE" if verdict.legitimate else "ILLEGITIMATE") + f" -- {change.label()}",
            rationale=verdict.reason,
            model=model_name(binding) if verdict.basis == "judged by the model" else "",
        )


@dataclass
class LearningLoop:
    """ALCC -- turns Experience into candidate self-improvements. Reads the
    runtime's distilled `Adaptation`s and proposes each as an
    `EvolutionChange` the runtime *might* make to itself. Proposing is not
    applying; every candidate still walks the AAWDFC legitimacy gate and the
    Evolver's fences before anything changes."""

    kind: str = "skill_prompt"

    def candidates(self, runtime: Any) -> list[EvolutionChange]:
        """Candidate changes distilled from the runtime's adaptations. Each
        standing impression becomes one proposed change, carrying its
        provenance as the required explanation."""
        bank = getattr(runtime, "adaptations", None)
        impressions = list(getattr(bank, "impressions", []) or [])
        changes: list[EvolutionChange] = []
        for index, adaptation in enumerate(impressions):
            insight = getattr(adaptation, "insight", "").strip()
            if not insight:
                continue
            changes.append(
                EvolutionChange(
                    kind=self.kind,
                    name=f"learned-improvement-{index + 1}",
                    description=f"Fold a learned adaptation into the stack: {insight[:120]}",
                    explanation=f"ALCC distilled this from repeated Experience: {insight}",
                    payload={"insight": insight, "origin": "alcc"},
                )
            )
        return changes
