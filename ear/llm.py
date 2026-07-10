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

import http.client
import json
import logging
import os
import socket
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("ear.llm")

DEFAULT_MAX_TOKENS = 2048
ANTHROPIC_VERSION = "2023-06-01"

# Transient-failure handling: retried statuses (rate limits, overload,
# gateway trouble), total attempts, and the pause before each retry.
# Mechanics constants -- auth and malformed-request errors fail fast.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0)


class LMError(RuntimeError):
    """A provider call failed -- network, auth or a malformed response.
    `retryable` marks failures worth another attempt (rate limits,
    overload, network) as opposed to ones that never will be (auth,
    malformed requests)."""

    retryable: bool = False


@dataclass
class LM:
    """A minimal, dependency-free chat LLM. `complete(prompt, system=...)`
    returns the model's text; every call is recorded in `history` with its
    token usage."""

    model: str
    provider: str = "anthropic"
    api_key: Optional[str] = field(default=None, repr=False)  # a credential -- never shown by repr/str
    api_base: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, prompt: str, system: str = "", cache_prefix: str = "") -> str:
        """`cache_prefix` is the stable leading span of `prompt` that will
        repeat across calls -- a provider-neutral hint. An adapter that caches
        prefixes for free (OpenAI-compatible) ignores it; one that needs an
        explicit marker (Anthropic) caches exactly that span. Empty (the
        default) keeps every request byte-identical to the uncached form, so
        this is inert until a caller declares a boundary."""
        if self.provider == "anthropic":
            url, headers, body, parse = self._anthropic(prompt, system, cache_prefix)
        else:
            url, headers, body, parse = self._openai(prompt, system)
        logger.info("LM call starting: %s (%d prompt chars, %d system chars)", self._bare_model, len(prompt), len(system))
        started = time.monotonic()
        raw, retries = self._post_with_retries(url, headers, body)
        text, usage = parse(raw)
        latency_ms = int((time.monotonic() - started) * 1000)
        self.history.append({"usage": usage, "latency_ms": latency_ms, "retries": retries})
        logger.info(
            "LM call finished: %s in %dms (retries=%d, %s+%s tok)",
            self._bare_model,
            latency_ms,
            retries,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return text

    # -- provider wire formats ------------------------------------------------

    def _anthropic(self, prompt: str, system: str, cache_prefix: str = ""):
        base = self.api_base or "https://api.anthropic.com"
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self.api_key or "",
        }
        # Anthropic caches only spans marked with cache_control. When the
        # caller declares a stable prefix, split the user content there and
        # mark the stable half; the provider re-reads it at ~0.1x price on the
        # next call. A missing or non-matching boundary leaves the content a
        # plain string -- byte-identical to the uncached request.
        if cache_prefix and prompt.startswith(cache_prefix) and 0 < len(cache_prefix) < len(prompt):
            content: Any = [
                {"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt[len(cache_prefix):]},
            ]
        else:
            content = prompt
        body: dict[str, Any] = {
            "model": self._bare_model,
            "max_tokens": self.params.get("max_tokens", DEFAULT_MAX_TOKENS),
            "messages": [{"role": "user", "content": content}],
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
                "cache_read_tokens": int(usage.get("cache_read_input_tokens") or 0),
                "cache_write_tokens": int(usage.get("cache_creation_input_tokens") or 0),
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
            # OpenAI-compatible providers cache prefixes automatically and
            # report the served-from-cache count under prompt_tokens_details;
            # there is no separate cache-write charge to report.
            details = usage.get("prompt_tokens_details") or {}
            return text, {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "cache_read_tokens": int(details.get("cached_tokens") or 0),
                "cache_write_tokens": 0,
            }

        return f"{base}/chat/completions", headers, body, parse

    @property
    def _bare_model(self) -> str:
        return self.model.split("/", 1)[1] if "/" in self.model else self.model

    # -- transport ------------------------------------------------------------

    @classmethod
    def _post_with_retries(cls, url: str, headers: dict[str, str], body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """POST with retry on transient failures only -- rate limits,
        overload, gateway errors, network drops -- backing off between
        attempts. Auth and malformed-request errors fail fast. Returns the
        parsed response and how many retries it took, so the retry count is
        on the record, never silent."""
        last_error: Optional[LMError] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return cls._post(url, headers, body), attempt
            except LMError as error:
                last_error = error
                if not error.retryable or attempt == MAX_ATTEMPTS - 1:
                    logger.warning("LM call to %s failed, not retrying: %s", url, error)
                    raise
                wait = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                logger.warning(
                    "LM call to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    url,
                    attempt + 1,
                    MAX_ATTEMPTS,
                    wait,
                    error,
                )
                time.sleep(wait)
        raise last_error if last_error is not None else LMError(f"LLM call to {url} failed")

    @staticmethod
    def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with ipv4_opener(_ssl_context()).open(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:  # noqa: PERF203 -- surface the provider's message
            detail = error.read().decode("utf-8", "replace")[:500]
            failure = LMError(f"LLM call to {url} failed ({error.code}): {detail}")
            failure.retryable = error.code in RETRYABLE_STATUS
            raise failure from error
        except (urllib.error.URLError, TimeoutError) as error:
            failure = LMError(f"LLM call to {url} failed: {error}")
            failure.retryable = True
            raise failure from error
        except ValueError as error:
            failure = LMError(f"LLM call to {url} returned malformed JSON: {error}")
            failure.retryable = False
            raise failure from error


def _ssl_context() -> ssl.SSLContext:
    """A default-verifying TLS context that honours the standard CA-bundle
    environment variables, so requests behind a proxy that presents its own
    certificate verify correctly without ever disabling verification."""
    ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca_file and os.path.exists(ca_file):
        return ssl.create_default_context(cafile=ca_file)
    return ssl.create_default_context()


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    """An HTTPSConnection that resolves and connects over IPv4 only.

    Some networks silently blackhole IPv6 to a dual-stack host -- no RST,
    no ICMP unreachable, just silence -- while IPv4 to the same host
    answers instantly. Stdlib's `socket.create_connection` (what
    `http.client` uses by default) tries `getaddrinfo`'s addresses in
    order with no "happy eyeballs" racing the way curl or a browser does,
    so a host whose DNS answer puts an AAAA record first stalls for the
    full OS-level TCP connect timeout -- observed here as ~15s per call,
    every call, silently, against a host `curl` reached in half a second
    -- before ever falling back to IPv4. Forcing IPv4 here removes that
    stall deterministically rather than hoping the OS falls back fast."""

    def connect(self) -> None:
        info = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)
        sock = socket.socket(*info[0][:3])
        if self.timeout is not None:
            sock.settimeout(self.timeout)
        sock.connect(info[0][4])
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, context: ssl.SSLContext) -> None:
        super().__init__()
        self._context = context

    def https_open(self, req):  # noqa: ANN001 -- matches urllib.request.HTTPSHandler's own signature
        return self.do_open(
            lambda host, **kwargs: _IPv4HTTPSConnection(host, context=self._context, **kwargs), req
        )


def ipv4_opener(context: ssl.SSLContext) -> urllib.request.OpenerDirector:
    """An opener that speaks HTTPS over IPv4 only -- see
    `_IPv4HTTPSConnection` for why. Shared by every native HTTPS call in
    this package (`ear/llm.py`, `ear/web.py`) so none of them are exposed
    to a silently blackholed IPv6 route."""
    return urllib.request.build_opener(_IPv4HTTPSHandler(context))


def fetch_text(url: str, timeout: int = 60) -> str:
    """GET a text document over the same native transport the LM client
    uses -- standard library, proxy and CA-bundle aware, verification
    always on. Used by the loader for URL knowledge sources."""
    request = urllib.request.Request(url, headers={"user-agent": "ear-knowledge"})
    try:
        with ipv4_opener(_ssl_context()).open(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as error:
        raise LMError(f"Fetch of {url} failed: {error}") from error
