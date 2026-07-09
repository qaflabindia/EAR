#!/usr/bin/env python3
"""Dump the exact prompt text EAR's native tool loop sends to the model, and
the exact reply it gets back, turn by turn -- for investigating whether a
tool's result (success or failure) actually reaches the model on its next
call. Uses a scripted stand-in LM, so this costs zero API calls; it exists
to let a human read the literal wire text, not to test correctness (see
tests/test_stack.py::test_a_failed_tool_call_reaches_the_model_on_the_next_prompt
for the assertion version of the same scenario).
"""

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ear import Intent, Runtime  # noqa: E402
from ear.reasoner import Reasoner  # noqa: E402
from ear.tool_binder import BoundTool  # noqa: E402


class RecordingLM:
    """Same shape as the real LM's .complete(prompt, system) -- records every
    call verbatim instead of hitting the network, and answers from a fixed
    script, exactly like tests/test_stack.py's ScriptedLM."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.history = []
        self.calls = []  # (system, prompt, reply) triples, in order

    def complete(self, prompt, system=""):
        reply = self.replies.pop(0) if self.replies else "## decision\n\nok\n"
        self.calls.append((system, prompt, reply))
        self.history.append({"usage": {"prompt_tokens": 10, "completion_tokens": 3}, "latency_ms": 7, "retries": 0})
        return reply


def _tool_action(tool="", args="", decision=""):
    return f"## tool\n\n{tool}\n\n## arguments\n\n{args}\n\n## decision\n\n{decision}\n"


def main():
    attempts = {"count": 0}

    def flaky(path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom: no such file")
        return f"read {path} fine"

    tool = BoundTool(name="read_file", description="read a file", handler=flaky)
    runtime = Runtime(name="FailureFeedback")
    lm = RecordingLM([
        _tool_action(tool="read_file", args="- path: missing.xlsx"),  # fails
        _tool_action(tool="read_file", args="- path: missing.xlsx"),  # retried, succeeds
        _tool_action(decision="Read the file after one failed attempt."),
    ])

    decision = Reasoner._reason_with_tools(
        Intent(text="read a file"), runtime, lm, context={}, capabilities="none", tools=[tool], max_iterations=6
    )

    out_path = TOOLS_DIR.parent / "logs" / "06-tool-loop-prompt-reply-dump.log"
    lines = [
        "Scenario: read_file(missing.xlsx) raises on the first call, succeeds on the",
        "second. This dump shows the exact prompt text sent to the model on every",
        "turn, and the exact reply -- so 'does a tool failure reach the model on the",
        "next call' is answered by reading the wire text directly, not by trusting",
        "a description of the code.",
        "",
        f"Final decision returned: {decision!r}",
        f"Tool handler invocation count: {attempts['count']}",
        "",
    ]
    for turn, (system, prompt, reply) in enumerate(lm.calls, start=1):
        lines += [
            "=" * 78,
            f"TURN {turn} -- system instruction",
            "=" * 78,
            system,
            "",
            "-" * 78,
            f"TURN {turn} -- full prompt sent to the model",
            "-" * 78,
            prompt,
            "",
            "-" * 78,
            f"TURN {turn} -- model's raw reply",
            "-" * 78,
            reply,
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
