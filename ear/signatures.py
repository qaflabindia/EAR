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
knowledge relevance, passage gisting for the retrieval index, contract
conformance, decision-quality grading, pairwise preference between stacks,
and panel deliberation. Mechanical stages (composing, validating) stay
plain Python.
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

RankRelevantSkills = Judgment(
    instruction=(
        "Identify which of a persona's skills are relevant to handling the "
        "given intent, most relevant first, so only those need be stacked "
        "into reasoning."
    ),
    inputs=[
        Field("intent_text"),
        Field("available_skills", "One 'name: instruction' pair per line"),
    ],
    outputs=[Field("relevant_skill_names", "Names of the relevant skills, most relevant first", "list")],
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

AuthorCode = Judgment(
    instruction=(
        "Write a small, self-contained Python script that provides the "
        "requested capability. The script MUST follow this contract exactly: "
        "read one JSON object of arguments from sys.argv[1] (default to an "
        "empty object when absent), do its work using only the Python "
        "standard library, and print exactly one JSON object of results to "
        "stdout. No network access, no reading outside the working directory, "
        "no shelling out. Keep it short and correct. Also provide a sample "
        "input and a substring the sample run's output must contain, so the "
        "script can be trialled before it is trusted."
    ),
    inputs=[
        Field("spec", "What the capability must do, in plain English"),
    ],
    outputs=[
        Field("name", "A short snake_case identifier for the capability", "str"),
        Field("description", "One line describing what the capability does", "text"),
        Field("code", "The complete Python script following the contract", "text"),
        Field("sample_input", "A JSON object to trial the script with", "str"),
        Field("expected", "A substring the trial output must contain", "str"),
    ],
)

JudgeCodeReasonable = Judgment(
    instruction=(
        "Review a script the system wrote for itself, as a security-minded "
        "reviewer who must be convinced BEYOND ANY REASONABLE SUSPICION before "
        "approving. Read the code against its stated purpose and decide "
        "whether it does only what it claims and nothing else -- nothing "
        "hidden, deceptive, or harmful. Withhold approval on any sign of an "
        "attempt to reach the network, read or exfiltrate data outside its "
        "task, persist beyond its sandbox, obfuscate what it does, mislead the "
        "reviewer, or widen its own privileges. Approve ONLY when you have no "
        "reasonable suspicion; if you have any doubt at all, do not approve, "
        "and name the suspicion plainly."
    ),
    inputs=[
        Field("purpose", "What the capability is supposed to do"),
        Field("code", "The complete script under review"),
    ],
    outputs=[
        Field("reasonable", "True only if the code is safe beyond any reasonable suspicion", "bool"),
        Field("suspicion", "Any concern that gives pause, or 'none'", "text"),
        Field("rationale", "One sentence explaining the verdict", "text"),
    ],
)

JudgeKnowledgeAdmission = Judgment(
    instruction=(
        "Decide whether a claim should be admitted into the governed "
        "knowledge base, the way a careful epistemics reviewer would. Judge "
        "whether it is well-formed, sourced, and plausible on its face; score "
        "its epistemic quality from 0 (unsupported rumour) to 1 (well-sourced, "
        "verifiable fact). Admit only what the organization should reason from "
        "as if it were true; when the claim is unsourced, vague, or "
        "speculative, do not admit it."
    ),
    inputs=[
        Field("claim", "The claim or document text under consideration"),
        Field("source", "Where the claim came from, if stated"),
        Field("existing", "A short summary of what the knowledge base already holds"),
    ],
    outputs=[
        Field("admit", "True to admit the claim into the knowledge base", "bool"),
        Field("score", "Epistemic quality from 0 to 1", "str"),
        Field("rationale", "One sentence explaining the judgment", "text"),
    ],
)

JudgeContradiction = Judgment(
    instruction=(
        "Decide whether a new claim contradicts anything the knowledge base "
        "already holds. A contradiction is a direct factual conflict, not "
        "mere difference of topic or emphasis. Name the conflicting passage "
        "when there is one."
    ),
    inputs=[
        Field("new_claim", "The claim being admitted"),
        Field("existing_passages", "The passages already held, one per line"),
    ],
    outputs=[
        Field("contradicts", "True if the new claim conflicts with an existing passage", "bool"),
        Field("passage", "The source of the conflicting passage, if any", "str"),
        Field("rationale", "One sentence explaining the judgment", "text"),
    ],
)

JudgeReasoningQuality = Judgment(
    instruction=(
        "Audit a piece of the runtime's own reasoning for epistemic quality, "
        "the way a rigorous critic would. Flag whether it rests on a biased "
        "premise (a protected attribute, an unwarranted stereotype, a "
        "one-sided framing) or on an unsupported assumption stated as fact. Be "
        "specific and fair -- do not manufacture a flaw where the reasoning is "
        "sound."
    ),
    inputs=[Field("reasoning", "The reasoning excerpt to audit")],
    outputs=[
        Field("biased", "True if the reasoning rests on a biased premise", "bool"),
        Field("unsupported", "True if it rests on an unsupported assumption stated as fact", "bool"),
        Field("rationale", "One sentence naming the specific concern, or affirming soundness", "text"),
    ],
)

JudgeWorkflowLegitimacy = Judgment(
    instruction=(
        "Decide whether a machine-created change to the runtime is legitimate "
        "-- fit to exist before it is applied. A legitimate change carries a "
        "clear purpose and explanation, stays within the organization's "
        "constitution, and proposes a sound role topology (it delegates to "
        "personas that make sense for the work). Reject a change that is "
        "unexplained, that a constitutional rule forbids, or whose structure "
        "is incoherent."
    ),
    inputs=[
        Field("kind", "The kind of change (skill prompt, workflow, ...)"),
        Field("name", "What the change is called"),
        Field("description", "What the change does"),
        Field("explanation", "Why the change was proposed"),
        Field("constitution", "A short summary of the governing constitutional rules"),
    ],
    outputs=[
        Field("legitimate", "True if the change is fit to exist and be applied", "bool"),
        Field("rationale", "One sentence explaining the judgment", "text"),
    ],
)

JudgeTaskComplexity = Judgment(
    instruction=(
        "Decide whether this task genuinely needs a large, expensive model, "
        "or whether a small fast one is adequate -- the way a pragmatic lead "
        "assigns work. Simple lookups, formatting, extraction, routine "
        "classification and short summaries are light work; multi-step "
        "reasoning, nuanced judgment, high-stakes decisions and long-form "
        "synthesis are heavy. When honestly uncertain, say heavy -- a wasted "
        "large call costs money; a botched hard task costs more."
    ),
    inputs=[
        Field("intent_text", "The task being routed"),
        Field("context", "The intent's context values"),
    ],
    outputs=[
        Field("heavy", "True if the task needs the large model", "bool"),
        Field("rationale", "One sentence naming why", "text"),
    ],
)

SynthesizeParallel = Judgment(
    instruction=(
        "Fold several independent partial results -- each produced in "
        "parallel over a part of one task -- into a single coherent answer. "
        "Reconcile disagreements and remove redundancy the way a careful "
        "editor would; do not merely concatenate the parts, and do not drop a "
        "part's substance. State the one combined result."
    ),
    inputs=[
        Field("task", "What the parallel work was solving, as one line"),
        Field("parts", "The partial results, one per line, each 'part N: ...'"),
    ],
    outputs=[Field("synthesis", "The single combined result", "text")],
)

FlagForAdversarialReview = Judgment(
    instruction=(
        "Decide whether an intent warrants an adversarial safety review "
        "before it is carried out. Flag it if the action is high-impact, "
        "hard to reverse, security- money- or safety-sensitive, or otherwise "
        "the kind of thing a careful red team would want to challenge first. "
        "Do not flag routine, low-stakes, easily reversible work -- an "
        "adversarial pass is deliberation when triggered, not a tax on every "
        "cycle."
    ),
    inputs=[
        Field("intent_text", "The action being considered"),
        Field("context", "The intent's context values"),
    ],
    outputs=[
        Field("flag", "True if the intent warrants an adversarial review", "bool"),
        Field("reason", "One sentence naming why it does or does not warrant review", "text"),
    ],
)

AdversarialChallenge = Judgment(
    instruction=(
        "Stress-test a flagged intent the way a rigorous red team would. "
        "First argue, as a determined adversary, the strongest concrete case "
        "that carrying out this action would cause harm, breach a control, or "
        "be exploited -- the challenge. Then argue, as the defender, why the "
        "action is nonetheless sound and safe to proceed -- the defense. Then "
        "return a single verdict: 'uphold' if the defense clearly answers the "
        "challenge, 'escalate' if a human must decide, 'overturn' if the "
        "challenge stands and the action should not proceed."
    ),
    inputs=[
        Field("intent_text", "The flagged action under adversarial review"),
        Field("context", "The intent's context values"),
        Field("concern", "Why the intent was flagged for adversarial review"),
        Field("decision", "The decision reached so far, if any, to stress-test"),
    ],
    outputs=[
        Field("challenge", "The strongest adversarial case against proceeding", "text"),
        Field("defense", "The case that the action is sound and safe", "text"),
        Field("verdict", "One of: uphold, escalate, overturn", "str"),
        Field("rationale", "One sentence explaining the verdict", "text"),
    ],
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

RouteAfterLeg = Judgment(
    instruction=(
        "A journey through authored workflow steps has just completed one "
        "step. Read the authored routing prose and the step's outcome, and "
        "choose what to walk next: answer with the number of an authored "
        "step to jump to, 'next' to continue in order, or 'conclude' to "
        "end the journey complete. Choose only among the authored steps -- "
        "never invent one -- and follow the routing prose exactly; when no "
        "route applies, answer 'next'."
    ),
    inputs=[
        Field("routes", "The authored routing prose, verbatim"),
        Field("completed_step", "The step just completed, as 'number: instruction'"),
        Field("outcome", "The decision that step reached"),
        Field("steps", "Every authored step, one 'number: instruction' per line"),
    ],
    outputs=[
        Field("next_step", "An authored step number, 'next', or 'conclude'", "str"),
        Field("rationale", "One sentence explaining the routing choice", "text"),
    ],
)

ChooseNextSpeaker = Judgment(
    instruction=(
        "Moderate a panel deliberation: read the authored pattern, the "
        "personas at the table and the transcript so far, and choose who "
        "should speak next -- or whether the panel has genuinely converged "
        "and should conclude. Answer with exactly one listed persona's "
        "name, or 'conclude' when another turn would add nothing; when the "
        "pattern prescribes an order, follow it."
    ),
    inputs=[
        Field("intent_text", "What the panel is deliberating"),
        Field("pattern", "The authored deliberation pattern, in plain English"),
        Field("personas", "One 'name: instructions' line per persona at the table"),
        Field("transcript", "The turns so far, speakers bracketed"),
    ],
    outputs=[
        Field("speaker", "Exactly one listed persona's name, or 'conclude'", "str"),
        Field("rationale", "One sentence explaining the choice", "text"),
    ],
)

SpeakOrUseTool = Judgment(
    instruction=(
        "Speak one turn in a panel deliberation, entirely as the given "
        "persona -- and you may first use tools to get facts. If a tool "
        "would ground the turn, call exactly one: name it and give its "
        "arguments, leaving the statement empty. Otherwise give the "
        "statement: engage concretely with earlier speakers, honor the "
        "authored pattern, and keep it to a few sentences. Never invent a "
        "tool that is not listed."
    ),
    inputs=[
        Field("intent_text", "What the panel is deliberating"),
        Field("persona", "Who is speaking: instructions and stacked skills"),
        Field("pattern", "The authored deliberation pattern, in plain English"),
        Field("transcript", "The turns so far, speakers bracketed"),
        Field("tools", "One 'name(parameters): description' line per available tool"),
        Field("gathered", "Results of tool calls made so far, or 'none yet'"),
    ],
    outputs=[
        Field("tool", "The name of the one tool to call now, or empty to speak", "str"),
        Field("arguments", "The tool's arguments as '- name: value' lines; empty when speaking", "list"),
        Field("statement", "This persona's turn, a few sentences; given only when no tool is called", "text"),
    ],
)

JudgeGoalProgress = Judgment(
    instruction=(
        "You are checking whether a session's goal has been met. Read the "
        "goal -- a completion condition in plain English -- and what has "
        "happened so far, and decide if it is genuinely satisfied. If it is "
        "not, name exactly one blocker: 'goal_not_met_yet' (more work would "
        "help), 'needs_user_input' (a human must supply something), "
        "'external_wait' (waiting on an outside event or system), "
        "'missing_evidence' (the work cannot be verified from what is here), "
        "or 'run_failed' (it went wrong and cannot recover). Ground the "
        "verdict in the visible evidence, and when the blocker is "
        "'goal_not_met_yet', give the single next step that would move it "
        "forward. Never claim satisfaction the evidence does not show."
    ),
    inputs=[
        Field("goal", "The completion condition, in plain English"),
        Field("progress", "What has happened so far -- the latest outcome and recent history"),
    ],
    outputs=[
        Field("satisfied", "True only if the goal is genuinely met", "bool"),
        Field("blocker", "Exactly one blocker word, empty when satisfied", "str"),
        Field("evidence", "The visible evidence behind the verdict", "text"),
        Field("next_step", "The single next step when goal_not_met_yet; empty otherwise", "text"),
    ],
)

GistPassage = Judgment(
    instruction=(
        "Write a one-line gist of a reference passage for a retrieval "
        "index. Say what the passage covers in plain, everyday words -- "
        "including the synonyms a searcher might use for its key terms -- "
        "so a differently-phrased question still finds it. One line, no "
        "heading, no quotation."
    ),
    inputs=[Field("passage", "The passage, headed by its [source]")],
    outputs=[Field("gist", "One plain-English line covering the passage and its synonyms", "text")],
)

JudgePreference = Judgment(
    instruction=(
        "Two configurations of a runtime answered the same intent. Judge "
        "which outcome better satisfies the evaluator's expectation, the "
        "way a careful reviewer would: ground the preference in the "
        "expectation alone, never in length or style, and call it a tie "
        "when neither is genuinely better."
    ),
    inputs=[
        Field("expected", "What the evaluator said should happen, in plain English"),
        Field("outcome_a", "The outcome runtime A reached"),
        Field("outcome_b", "The outcome runtime B reached"),
    ],
    outputs=[
        Field("preference", "Exactly one of: A, B, tie", "str"),
        Field("rationale", "One sentence explaining the preference", "text"),
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
        "listed. No blind calls: every call must be aimed at a specific fact "
        "or output. When an attempt fails, the unchanged action may fail "
        "only once: do not simply fire it again. Diagnose the actual failure "
        "from the result you already have, change the input (the arguments, "
        "the script, the approach), and only then retry; an unchanged failed "
        "call will be refused before it runs again. "
        "When an argument's value needs more than one line -- the "
        "source of a script, a whole file's content -- write it as a "
        "blockquote rather than cramming it onto one bullet line; see the "
        "arguments field below for exactly how."
    ),
    inputs=[
        Field("intent", "The intent to resolve"),
        Field("context", "Structured context relevant to the intent"),
        Field("capabilities", "The stacked personas and skill prompts for this intent"),
        Field("tools", "One 'name(parameters): description' line per available tool"),
        Field("gathered", "Results of tool calls made so far, or 'none yet'"),
    ],
    # `gathered` is the one input that grows across the tool loop's otherwise-
    # identical calls; rendering it last makes intent+context+capabilities+tools
    # a stable prefix the provider re-reads from cache each iteration.
    cache_boundary="gathered",
    outputs=[
        Field("tool", "The name of the one tool to call now, or empty to decide instead", "str"),
        Field(
            "arguments",
            (
                "The tool's arguments; empty when deciding instead. A short "
                "value is a '- name: value' bullet, same as always. A value "
                "spanning more than one line -- a script's source, a whole "
                "file -- cannot go on a bullet line: write 'name:' alone, "
                "then the value as a blockquote, every line (blank ones "
                "too) starting with '> '. Example with one short argument "
                "and one multi-line one:\n"
                "- path: workspace/script.py\n"
                "content:\n"
                "> import openpyxl\n"
                ">\n"
                "> wb = openpyxl.load_workbook('uploads/data.xlsx')\n"
                "> print(wb.sheetnames)"
            ),
            "map",
        ),
        Field("decision", "The final decision, given only when no tool is called", "text"),
    ],
)

SummarizeToolResult = Judgment(
    instruction=(
        "Compress a tool call's result into short, caveman-style text for "
        "another model to act on next turn. This text has already been run "
        "through a deterministic compressor that only ever deletes matched "
        "filler words -- you may compress further, but you may generate new "
        "wording, so the following constraints are absolute:\n"
        "\n"
        "No fabrication. Never state a fact, number, path, name, or outcome "
        "that is not literally present in the result. If you are not certain "
        "a detail survived from the input, quote it verbatim rather than "
        "paraphrase it -- a quoted fact cannot drift.\n"
        "No shallowness. Preserve every fact that changes what the next turn "
        "should do -- exact error text, exact file paths, exact row/column "
        "counts, exit codes, exact names -- not just the headline outcome.\n"
        "No fluff. Cut filler, hedging, pleasantries, and restating the "
        "question; every remaining word must carry information.\n"
        "No sloppiness. A shorter sentence that becomes ambiguous, or drops "
        "a qualifier that changes the meaning, is wrong, not concise.\n"
        "No context loss or distortion. Whether the call succeeded or "
        "failed is never optional to state. When in doubt between "
        "compressing a number and keeping it exact, keep it exact."
    ),
    inputs=[
        Field("tool", "The tool that was called"),
        Field("arguments", "The arguments it was called with"),
        Field("result", "The tool's full raw result"),
    ],
    outputs=[Field("summary", "The compressed, caveman-style summary", "text")],
)

ConsolidateGatheredContext = Judgment(
    instruction=(
        "You are checkpointing an agent's tool-use history partway through a "
        "task. Read every gathered tool result below -- accumulated over "
        "several turns -- and produce ONE consolidated statement carrying "
        "every fact that still matters for what happens next: exact file "
        "paths, exact numbers (row counts, exit codes, byte sizes), exact "
        "error text, what has been verified versus merely attempted, and any "
        "decision already reached. This checkpoint REPLACES the individual "
        "entries below for every turn from here on, so anything it drops is "
        "gone for good -- no fabrication (state only what the gathered text "
        "actually shows), no shallowness (drop only genuine repetition and "
        "resolved dead ends, never a fact because it seems minor), no fluff, "
        "no context loss or distortion of any number, path, or outcome."
    ),
    inputs=[Field("gathered_so_far", "Every tool result gathered in this cycle so far, in order")],
    outputs=[Field("checkpoint", "The single consolidated statement of what still matters", "text")],
)

# The registry of every declared Judgment in this module, by name -- what
# the Optimizer refines and what persisted instructions apply to.
REGISTRY = {name: value for name, value in list(globals().items()) if isinstance(value, Judgment)}
