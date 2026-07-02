"""signatures -- the natural-language reasoning tasks behind every
judgment-laden stage, declared natively (see `ear/judgment.py`), with no
third-party framework underneath.

Each `Judgment` names the inputs the model is given and the outputs it must
return; the judgment itself happens at runtime against whichever `LM` the
active `ModelBinding` built. Nothing here hardcodes a decision -- these are
prompts, and the model decides.

Used deliberately, only where genuine judgment lives: policy compliance,
process relevance, selection, scheduling, delegation, the core decision,
recall, audit, explanation, history summarisation, experience distillation,
knowledge relevance, contract conformance, decision-quality grading, and
panel deliberation. Mechanical stages (composing, validating) stay plain
Python.
"""

from __future__ import annotations

from .judgment import Field, Judgment

JudgePolicyCompliance = Judgment(
    instruction=(
        "Decide whether the given context complies with a written policy "
        "statement, the way a careful compliance reviewer would: read the "
        "policy in plain English, check the context against it, and explain "
        "your reasoning in one sentence."
    ),
    inputs=[
        Field("policy_statement", "The policy, written in plain English"),
        Field("context", "The intent's context values relevant to the policy"),
    ],
    outputs=[
        Field("complies", "True if the context satisfies the policy, False if it violates it", "bool"),
        Field("rationale", "One sentence explaining the judgment", "text"),
    ],
)

DiscoverRelevantProcesses = Judgment(
    instruction=(
        "Identify which of the runtime's registered processes are relevant "
        "to handling the given intent, most relevant first."
    ),
    inputs=[
        Field("intent_text"),
        Field("available_processes", "One 'name: description' pair per line"),
    ],
    outputs=[Field("relevant_process_names", "Names of the relevant processes, most relevant first", "list")],
)

SelectProcesses = Judgment(
    instruction=(
        "Choose which of the candidate processes should actually run for the "
        "given intent, most relevant first. Select every process genuinely "
        "needed to serve the intent and omit the rest."
    ),
    inputs=[
        Field("intent_text"),
        Field("candidate_processes", "One 'name: description' pair per line"),
    ],
    outputs=[Field("selected_process_names", "Names of the processes to run, most relevant first", "list")],
)

ScheduleWorkflows = Judgment(
    instruction=(
        "Order the composed plan's workflows into the best execution order "
        "for the given intent -- prerequisites and information-producing "
        "workflows first. Keep every workflow: ordering is the only judgment "
        "being made here, never omission."
    ),
    inputs=[
        Field("intent_text"),
        Field("workflows", "One 'name: step summary' pair per line, in current order"),
    ],
    outputs=[Field("ordered_workflow_names", "Every workflow name, in execution order", "list")],
)

DelegateSteps = Judgment(
    instruction=(
        "Assign each undelegated workflow step to the persona best suited to "
        "carry it out, judged from the step's instruction against each "
        "persona's standing instructions and stacked skills."
    ),
    inputs=[
        Field("steps", "One 'number: instruction' line per undelegated step"),
        Field("personas", "One 'name: instructions and skills' line per available persona"),
    ],
    outputs=[Field("assignments", "One 'number: persona name' item per step", "list")],
)

JudgeContractConformance = Judgment(
    instruction=(
        "Decide whether delivered data honors a contract, the way a careful "
        "reviewer would: read each field's authored meaning, check the "
        "delivered value against it -- a value the meaning does not support, "
        "an evasive non-answer, or a hedge where the meaning demands one of a "
        "set, all fail -- and explain the judgment in one sentence."
    ),
    inputs=[
        Field("contract", "One '- name: meaning' line per declared field"),
        Field("data", "One '- name: value' line per delivered field"),
    ],
    outputs=[
        Field("conforms", "True only if every delivered value honors its field's meaning", "bool"),
        Field("rationale", "One sentence explaining the judgment", "text"),
    ],
)

JudgeDecisionQuality = Judgment(
    instruction=(
        "Grade a runtime's decision against what an evaluator said should "
        "happen, the way a careful reviewer would: read the expectation in "
        "plain English, compare the actual outcome (including a blocked "
        "refusal, which can itself be the expected outcome) against it, and "
        "give a verdict with a one-sentence reason."
    ),
    inputs=[
        Field("expected", "What the evaluator said should happen, in plain English"),
        Field("actual", "The outcome the runtime actually reached, with any structured data"),
    ],
    outputs=[
        Field("passed", "True only if the actual outcome satisfies the expectation", "bool"),
        Field("rationale", "One sentence explaining the verdict", "text"),
    ],
)

SpeakInPanel = Judgment(
    instruction=(
        "Speak one turn in a panel deliberation, entirely as the given "
        "persona: follow its standing instructions and stacked skills, engage "
        "concretely with what earlier speakers said (agree, challenge, or add "
        "-- never merely restate), honor the authored deliberation pattern, "
        "and keep the turn to a few sentences."
    ),
    inputs=[
        Field("intent_text", "What the panel is deliberating"),
        Field("persona", "Who is speaking: instructions and stacked skills"),
        Field("pattern", "The authored deliberation pattern, in plain English"),
        Field("transcript", "The turns so far, speakers bracketed"),
    ],
    outputs=[Field("statement", "This persona's turn, a few sentences", "text")],
)

SynthesizePanel = Judgment(
    instruction=(
        "Conclude a panel deliberation into one decision: weigh what each "
        "speaker established, resolve disagreements the way the authored "
        "pattern directs, and state the single concrete outcome with the "
        "decisive reasoning -- never a hedge between positions."
    ),
    inputs=[
        Field("intent_text", "What the panel deliberated"),
        Field("pattern", "The authored deliberation pattern, in plain English"),
        Field("transcript", "The full deliberation, speakers bracketed"),
    ],
    outputs=[Field("decision", "The panel's single concluded decision, with its reasoning", "text")],
)

RecallRelevantMemory = Judgment(
    instruction=(
        "From the runtime's remembered history, recall what is genuinely "
        "relevant to the intent at hand -- prior decisions, amounts and "
        "outcomes that should inform this cycle -- and leave the rest behind. "
        "Recall facts as they were recorded; never invent or embellish them."
    ),
    inputs=[
        Field("intent_text"),
        Field("history", "The full remembered context window"),
    ],
    outputs=[Field("relevant_context", "Only the remembered facts relevant to this intent", "text")],
)

AuditEvidence = Judgment(
    instruction=(
        "Inspect a decision's evidence the way an internal auditor would: "
        "check the decision against its basis, the policies checked and the "
        "plan, and say whether the evidence supports the decision, naming any "
        "gap or inconsistency plainly."
    ),
    inputs=[
        Field("decision"),
        Field("evidence", "The basis, policies checked, plan and recalled memory behind the decision"),
    ],
    outputs=[Field("assessment", "One or two sentences: supported or not, and any gap found", "text")],
)

SelectRelevantPassages = Judgment(
    instruction=(
        "From the numbered knowledge passages, choose the ones a careful "
        "analyst would actually consult for this intent. Choosing none is a "
        "valid judgment; never refer to a passage that is not in the list."
    ),
    inputs=[
        Field("intent_text"),
        Field("passages", "Numbered passages, each headed by its [source]"),
    ],
    outputs=[
        Field("relevant_numbers", "The numbers of the relevant passages; empty if none apply", "list"),
        Field("rationale", "One sentence explaining the choice", "text"),
    ],
)

ReasonAboutIntent = Judgment(
    instruction=(
        "Resolve an intent into a final, concrete decision given its context. "
        "Reason as the assembled capabilities: the persona instructions and "
        "the stacked skill prompts describe who is acting and how -- follow "
        "them when reaching the decision."
    ),
    inputs=[
        Field("intent", "The natural-language intent to resolve"),
        Field("context", "Structured context relevant to the intent"),
        Field("capabilities", "The stacked personas and skill prompts composed for this intent"),
    ],
    outputs=[Field("decision", "The concrete decision reached, with a brief justification", "text")],
)

ExplainDecision = Judgment(
    instruction=(
        "Write a short, human-readable explanation of why a decision was "
        "reached, given the evidentiary basis for it."
    ),
    inputs=[
        Field("basis", "The evidentiary basis the decision rests on"),
        Field("decision"),
    ],
    outputs=[Field("explanation", "One or two plain-English sentences", "text")],
)

SummarizeHistory = Judgment(
    instruction=(
        "Summarize execution history into a short paragraph, preserving any "
        "decisions, amounts and outcomes that later reasoning might need."
    ),
    inputs=[Field("history")],
    outputs=[Field("summary", "A short paragraph", "text")],
)

DistillInsight = Judgment(
    instruction=(
        "State one durable lesson, in one sentence, from aggregated execution "
        "experience that should bias future decisions."
    ),
    inputs=[Field("experience_summary")],
    outputs=[Field("insight", "One sentence", "text")],
)

RefineInstruction = Judgment(
    instruction=(
        "Improve a reasoning instruction so the model follows it more "
        "reliably. Read the current instruction and worked examples of what "
        "it produced -- especially any a reviewer judged wrong -- and rewrite "
        "the instruction to fix those failures while preserving everything it "
        "already gets right. Return only the improved instruction, in the "
        "same imperative voice; never narrow it to the examples."
    ),
    inputs=[
        Field("current_instruction", "The instruction being improved"),
        Field("examples", "Worked examples: intent, the decision reached, and any reviewer verdict"),
    ],
    outputs=[Field("improved_instruction", "The rewritten instruction", "text")],
)

ChooseToolAction = Judgment(
    instruction=(
        "You are reasoning toward a decision and may use tools to get facts "
        "first. Read the tools and what you have gathered so far. If a tool "
        "would help, call exactly one -- name it and give its arguments. "
        "Otherwise, give the final decision. Never invent a tool that is not "
        "listed."
    ),
    inputs=[
        Field("intent", "The intent to resolve"),
        Field("context", "Structured context relevant to the intent"),
        Field("capabilities", "The stacked personas and skill prompts for this intent"),
        Field("tools", "One 'name(parameters): description' line per available tool"),
        Field("gathered", "Results of tool calls made so far, or 'none yet'"),
    ],
    outputs=[
        Field("tool", "The name of the one tool to call now, or empty to decide instead", "str"),
        Field("arguments", "The tool's arguments as '- name: value' lines; empty when deciding", "list"),
        Field("decision", "The final decision, given only when no tool is called", "text"),
    ],
)

# The registry of every declared Judgment in this module, by name -- what
# the Optimizer refines and what persisted instructions apply to.
REGISTRY = {name: value for name, value in list(globals().items()) if isinstance(value, Judgment)}
