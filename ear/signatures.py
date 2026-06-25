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


class ReasonAboutIntent(dspy.Signature):
    """Resolve an intent into a final, concrete decision given its context."""

    intent: str = dspy.InputField(desc="The natural-language intent to resolve")
    context: dict = dspy.InputField(desc="Structured context relevant to the intent")
    decision: str = dspy.OutputField(desc="The concrete decision reached, with a brief justification")


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
