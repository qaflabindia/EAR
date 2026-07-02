"""llm -- EAR's own LLM client, built on the Python standard library alone.

EAR is an independent package: it has no third-party dependencies, so it
does not reach reasoning out to DSPy, LiteLLM or any provider SDK. This
module is the whole of it -- a small HTTPS client (`urllib`, `json`,
`ssl`) that speaks two wire protocols directly:

- Anthropic's Messages API (`provider="anthropic"`), the default;
- the OpenAI chat-completions shape (`provider="openai"` or any
  OpenAI-compatible endpoint via `api_base` -- Azure, Ollama, together,
  vLLM, and so on).

Credentials are read from the environment, never hardcoded. Outbound
requests honour the standard proxy and CA-bundle environment variables, so
the same code runs behind a corporate proxy unchanged. Each call is
appended to `history` with its token usage, so the Runtime's per-cycle
accounting reads real numbers with nothing extra wired up.

Adding a provider is one function and one branch here -- the judgment
stages never change, because they speak to `LM.complete`, not to a wire
format.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_MAX_TOKENS = 2048
ANTHROPIC_VERSION = "2023-06-01"


class LMError(RuntimeError):
    """A provider call failed -- network, auth or a malformed response."""


@dataclass
class LM:
    """A minimal, dependency-free chat LLM. `complete(prompt, system=...)`
    returns the model's text; every call is recorded in `history` with its
    token usage."""

    model: str
    provider: str = "anthropic"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, prompt: str, system: str = "") -> str:
        if self.provider == "anthropic":
            url, headers, body, parse = self._anthropic(prompt, system)
        else:
            url, headers, body, parse = self._openai(prompt, system)
        raw = self._post(url, headers, body)
        text, usage = parse(raw)
        self.history.append({"usage": usage, "cost": 0.0})
        return text

    # -- provider wire formats ------------------------------------------------

    def _anthropic(self, prompt: str, system: str):
        base = self.api_base or "https://api.anthropic.com"
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self.api_key or "",
        }
        body: dict[str, Any] = {
            "model": self._bare_model,
            "max_tokens": self.params.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        if "temperature" in self.params:
            body["temperature"] = self.params["temperature"]

        def parse(data: dict[str, Any]):
            blocks = data.get("content") or []
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            usage = data.get("usage") or {}
            return text, {
                "prompt_tokens": int(usage.get("input_tokens") or 0),
                "completion_tokens": int(usage.get("output_tokens") or 0),
            }

        return f"{base}/v1/messages", headers, body, parse

    def _openai(self, prompt: str, system: str):
        base = self.api_base or "https://api.openai.com/v1"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key or ''}",
        }
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict[str, Any] = {"model": self._bare_model, "messages": messages}
        if "temperature" in self.params:
            body["temperature"] = self.params["temperature"]
        if "max_tokens" in self.params:
            body["max_tokens"] = self.params["max_tokens"]

        def parse(data: dict[str, Any]):
            choices = data.get("choices") or [{}]
            text = (choices[0].get("message") or {}).get("content", "")
            usage = data.get("usage") or {}
            return text, {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }

        return f"{base}/chat/completions", headers, body, parse

    @property
    def _bare_model(self) -> str:
        return self.model.split("/", 1)[1] if "/" in self.model else self.model

    # -- transport ------------------------------------------------------------

    @staticmethod
    def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        context = _ssl_context()
        try:
            with urllib.request.urlopen(request, context=context, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:  # noqa: PERF203 -- surface the provider's message
            detail = error.read().decode("utf-8", "replace")[:500]
            raise LMError(f"LLM call to {url} failed ({error.code}): {detail}") from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise LMError(f"LLM call to {url} failed: {error}") from error


def _ssl_context() -> ssl.SSLContext:
    """A default-verifying TLS context that honours the standard CA-bundle
    environment variables, so requests behind a proxy that presents its own
    certificate verify correctly without ever disabling verification."""
    ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca_file and os.path.exists(ca_file):
        return ssl.create_default_context(cafile=ca_file)
    return ssl.create_default_context()
