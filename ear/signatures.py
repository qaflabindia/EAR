"""DSPy signatures -- the natural-language prompts that back every
judgment-laden reasoning step in this package. Nothing here hardcodes a
decision: each signature only declares the inputs an LLM is given and the
outputs it must produce, so the actual judgment happens at runtime against
whichever ModelBinding (Claude, or any other LiteLLM-supported provider)
is active.

Used sparingly and deliberately: only the steps that genuinely require
judgment (policy compliance, process relevance, the core decision, plan
explanation, history summarization, experience distillation) are backed by
a signature here. Structural/mechanical steps (selecting among already
discovered candidates, composing workflows, scheduling a plan) stay plain
Python, because there is no judgment call to make there.
"""

from __future__ import annotations

import dspy


class JudgePolicyCompliance(dspy.Signature):
    """Decide whether the given context complies with a written policy
    statement, the way a careful compliance reviewer would: read the policy
    in plain English, check the context against it, and explain your
    reasoning in one sentence."""

    policy_statement: str = dspy.InputField(desc="The policy, written in plain English")
    context: dict = dspy.InputField(desc="The intent's context values relevant to the policy")
    complies: bool = dspy.OutputField(desc="True if the context satisfies the policy, False if it violates it")
    rationale: str = dspy.OutputField(desc="One sentence explaining the judgment")


class DiscoverRelevantProcesses(dspy.Signature):
    """Identify which of the runtime's registered processes are relevant to
    handling the given intent, most relevant first."""

    intent_text: str = dspy.InputField()
    available_processes: str = dspy.InputField(desc="One 'name: description' pair per line")
    relevant_process_names: list[str] = dspy.OutputField(desc="Names of the relevant processes, most relevant first")


class SelectProcesses(dspy.Signature):
    """Choose which of the candidate processes should actually run for the
    given intent, most relevant first. Select every process genuinely
    needed to serve the intent and omit the rest."""

    intent_text: str = dspy.InputField()
    candidate_processes: str = dspy.InputField(desc="One 'name: description' pair per line")
    selected_process_names: list[str] = dspy.OutputField(desc="Names of the processes to run, most relevant first")


class ScheduleWorkflows(dspy.Signature):
    """Order the composed plan's workflows into the best execution order
    for the given intent -- prerequisites and information-producing
    workflows first. Keep every workflow: ordering is the only judgment
    being made here, never omission."""

    intent_text: str = dspy.InputField()
    workflows: str = dspy.InputField(desc="One 'name: step summary' pair per line, in current order")
    ordered_workflow_names: list[str] = dspy.OutputField(desc="Every workflow name, in execution order")


class DelegateSteps(dspy.Signature):
    """Assign each undelegated workflow step to the persona best suited to
    carry it out, judged from the step's instruction against each persona's
    standing instructions and stacked skills."""

    steps: str = dspy.InputField(desc="One 'number: instruction' line per undelegated step")
    personas: str = dspy.InputField(desc="One 'name: instructions and skills' line per available persona")
    assignments: list[str] = dspy.OutputField(desc="One 'number: persona name' entry per step")


class ReasonAboutIntent(dspy.Signature):
    """Resolve an intent into a final, concrete decision given its context.

    Reason *as* the assembled capabilities: the persona instructions and the
    stacked skill prompts describe who is acting and how -- follow them when
    reaching the decision."""

    intent: str = dspy.InputField(desc="The natural-language intent to resolve")
    context: dict = dspy.InputField(desc="Structured context relevant to the intent")
    capabilities: str = dspy.InputField(
        desc="The stacked personas and skill prompts the runtime composed for this "
        "intent -- the standing instructions and capabilities to reason with"
    )
    decision: str = dspy.OutputField(desc="The concrete decision reached, with a brief justification")


class RecallRelevantMemory(dspy.Signature):
    """From the runtime's remembered history, recall what is genuinely
    relevant to the intent at hand -- prior decisions, amounts and
    outcomes that should inform this cycle -- and leave the rest behind.
    Recall facts as they were recorded; never invent or embellish them."""

    intent_text: str = dspy.InputField()
    history: str = dspy.InputField(desc="The full remembered context window")
    relevant_context: str = dspy.OutputField(desc="Only the remembered facts relevant to this intent, verbatim where possible")


class AuditEvidence(dspy.Signature):
    """Inspect a decision's evidence the way an internal auditor would:
    check the decision against its basis, the policies checked and the
    plan, and say whether the evidence supports the decision, naming any
    gap or inconsistency plainly."""

    decision: str = dspy.InputField()
    evidence: str = dspy.InputField(desc="The basis, policies checked, plan and recalled memory behind the decision")
    assessment: str = dspy.OutputField(desc="One or two sentences: supported or not, and any gap found")


class ExplainDecision(dspy.Signature):
    """Write a short, human-readable explanation of why a decision was
    reached, given the evidentiary basis for it."""

    basis: str = dspy.InputField(desc="The evidentiary basis the decision rests on")
    decision: str = dspy.InputField()
    explanation: str = dspy.OutputField(desc="One or two plain-English sentences")


class SummarizeHistory(dspy.Signature):
    """Summarize execution history into a short paragraph, preserving any
    decisions, amounts and outcomes that later reasoning might need."""

    history: str = dspy.InputField()
    summary: str = dspy.OutputField()


class DistillInsight(dspy.Signature):
    """State one durable lesson, in one sentence, from aggregated execution
    experience that should bias future decisions."""

    experience_summary: str = dspy.InputField()
    insight: str = dspy.OutputField()
