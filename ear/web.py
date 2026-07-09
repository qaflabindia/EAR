"""Web -- native internet access and search, spoken from the standard
library alone (`urllib`), the same way `ear/llm.py` speaks to model
providers with no SDK underneath.

Two of the ten basic toolsets a runtime may declare (`Toolsets` in
memory.md): `internet_access` (fetch a public URL) and `internet_search`
(query a search API). Both are mechanics -- there is nothing to reason
about in *how* an HTTP GET works -- so they ship as ready BoundTools
rather than something the model derives itself each time it needs the
web, the same split every native tool in this package follows: the model
judges *when* to reach for the web, code handles *how*.

`internet_search` needs a provider and an API key -- declared the same
way Model Selection declares its own credential: an environment-variable
*name*, never a key written in memory.md. Only Tavily is wired today;
an unsupported provider fails loudly rather than pretending."""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

from .llm import ipv4_opener

DEFAULT_TIMEOUT = 20.0
MAX_RESPONSE_BYTES = 500_000


class WebError(RuntimeError):
    """A fetch or search call failed -- unreachable host, a non-2xx
    status, a missing credential, or a malformed response. Loud by
    design: wrapped by the binder, it returns to the model as text like
    any other tool failure, never swallowed."""


@dataclass
class WebAccess:
    """`fetch_url` and `web_search`, confined to a response-size cap and a
    timeout -- the network equivalent of Sandbox's resource limits."""

    timeout: float = DEFAULT_TIMEOUT
    search_provider: str = ""
    search_api_key_env_var: str = ""

    def fetch_url(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "ear/0.1.0"})
        try:
            # IPv4-forced (see ear.llm.ipv4_opener): a dual-stack host on a
            # network that blackholes IPv6 otherwise stalls every call for
            # the OS's full TCP connect timeout before ever falling back.
            with ipv4_opener(ssl.create_default_context()).open(request, timeout=self.timeout) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, ValueError, OSError) as error:
            raise WebError(f"could not fetch {url!r}: {error}") from error
        text = body.decode("utf-8", errors="replace")
        if len(body) > MAX_RESPONSE_BYTES:
            text = text[:MAX_RESPONSE_BYTES] + "\n…[truncated]"
        return text

    def web_search(self, query: str, max_results: int = 5) -> str:
        if not self.search_provider or not self.search_api_key_env_var:
            raise WebError(
                "web_search needs a declared provider and API key env var -- "
                "see Toolsets: internet_search in memory.md"
            )
        api_key = os.environ.get(self.search_api_key_env_var)
        if not api_key:
            raise WebError(f"environment variable '{self.search_api_key_env_var}' is not set")
        provider = self.search_provider.lower()
        if provider == "tavily":
            return self._tavily(query, max_results, api_key)
        raise WebError(f"unsupported search provider '{self.search_provider}' -- supported: tavily")

    def _tavily(self, query: str, max_results: int, api_key: str) -> str:
        payload = json.dumps({"api_key": api_key, "query": query, "max_results": max_results}).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with ipv4_opener(ssl.create_default_context()).open(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ValueError, OSError) as error:
            raise WebError(f"web search failed: {error}") from error
        results = result.get("results") or []
        lines = [
            f"{item.get('title', '(no title)')} -- {item.get('url', '')}\n{item.get('content', '')}"
            for item in results[:max_results]
        ]
        return "\n\n".join(lines) if lines else "no results"

    def as_tools(self, enabled: set) -> list:
        """`fetch_url` when `internet_access` is enabled, `web_search`
        when `internet_search` is -- BoundTools ready to bind, no handler
        the caller has to supply."""
        from .tool_binder import BoundTool

        tools = []
        if "internet_access" in enabled:
            tools.append(
                BoundTool(name="fetch_url", description="Fetch the text content of a public URL.", handler=self.fetch_url)
            )
        if "internet_search" in enabled:
            tools.append(
                BoundTool(
                    name="web_search",
                    description="Search the web for a query and return the top results.",
                    handler=self.web_search,
                )
            )
        return tools
