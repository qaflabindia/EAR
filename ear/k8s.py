"""k8s -- run EAR instances on Kubernetes, spoken natively from the
standard library.

The user's platform runs every process on Kubernetes containers, live for
the recurring occurrence of tasks. This is the execution provider for that
-- and it keeps EAR's invariant: Kubernetes' API is JSON over HTTPS, so
EAR speaks it with `urllib` and `ssl`, no `kubernetes` SDK, no dependency,
the same way it speaks to LLM providers and MCP servers. The whole client
is a few dozen lines because the protocol *is* the spec.

The mapping is direct:

- a **runtime instance** runs in a **Job** -- one pod, one governed cycle,
  via the in-pod entrypoint `python -m ear.run <stack>` (see `ear/run.py`),
  the intent handed in through the environment;
- a **recurring task** is a **CronJob** -- Kubernetes owns the recurrence,
  so an instance stays live for the recurring occurrence of a task without
  anything of EAR's staying resident;
- and the `KubeProvider` can be the **Kernel's dispatcher**, so the Kernel
  stays the single scheduler while each firing runs in its own pod.

Configuration is the standard in-cluster service-account (token, CA and
namespace files, the API server from the environment) or an explicit
`KubeConfig` for out-of-cluster use. The manifest builders are pure
functions -- testable without a cluster -- and the client's transport is
injectable, so the whole provider is unit-tested against a faithful fake.
It has *not* been exercised against a live cluster from this repository; it
speaks the real Kubernetes REST API, and the tests hold it to that shape.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_SA = "/var/run/secrets/kubernetes.io/serviceaccount"


class KubeError(RuntimeError):
    """A Kubernetes API call failed, or the environment is not a cluster."""


# -- configuration -----------------------------------------------------------


@dataclass
class KubeConfig:
    """How to reach the API server, and as whom."""

    api_server: str
    token: str = field(default="", repr=False)  # a bearer credential -- never shown by repr/str
    ca_cert: Optional[str] = None
    namespace: str = "default"
    verify: bool = True

    @classmethod
    def in_cluster(cls, base: str = _SA) -> "KubeConfig":
        """The standard in-cluster configuration: the service-account token,
        CA and namespace mounted into the pod, and the API server from the
        environment."""
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        if not host:
            raise KubeError("not running in a cluster (KUBERNETES_SERVICE_HOST is unset)")
        token_path, ns_path, ca_path = Path(base) / "token", Path(base) / "namespace", Path(base) / "ca.crt"
        if not token_path.exists():
            raise KubeError(f"no service-account token at {token_path}")
        return cls(
            api_server=f"https://{host}:{port}",
            token=token_path.read_text(encoding="utf-8").strip(),
            ca_cert=str(ca_path) if ca_path.exists() else None,
            namespace=ns_path.read_text(encoding="utf-8").strip() if ns_path.exists() else "default",
        )


# -- the client (stdlib REST over the Kubernetes API) ------------------------


@dataclass
class KubeClient:
    """A minimal Kubernetes API client. `transport` is injectable --
    `(method, url, headers, body) -> (status, data)` -- so the client is
    testable without a cluster; unset, it speaks real HTTPS."""

    config: KubeConfig
    transport: Optional[Callable] = None
    timeout: int = 30

    def request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = self.config.api_server.rstrip("/") + path
        headers = {"Authorization": f"Bearer {self.config.token}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        status, data = (self.transport or self._http)(method, url, headers, body)
        if status >= 400:
            message = data.get("message", data) if isinstance(data, dict) else data
            raise KubeError(f"{method} {path} -> {status}: {message}")
        return data if isinstance(data, dict) else {}

    def _http(self, method: str, url: str, headers: dict, body: Optional[dict]):  # pragma: no cover - needs a cluster
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, context=self._ssl(), timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return response.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as error:
            try:
                detail = json.loads(error.read().decode("utf-8"))
            except (ValueError, OSError):
                detail = {"message": str(error)}
            return error.code, detail
        except (urllib.error.URLError, TimeoutError) as error:
            raise KubeError(f"{method} {url} failed: {error}") from error

    def _ssl(self) -> ssl.SSLContext:  # pragma: no cover - needs a cluster
        if not self.config.verify:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        if self.config.ca_cert and os.path.exists(self.config.ca_cert):
            return ssl.create_default_context(cafile=self.config.ca_cert)
        return ssl.create_default_context()

    # -- batch objects --------------------------------------------------------

    def _batch(self, kind: str, name: str = "") -> str:
        base = f"/apis/batch/v1/namespaces/{self.config.namespace}/{kind}"
        return f"{base}/{name}" if name else base

    def create_job(self, manifest: dict) -> dict:
        return self.request("POST", self._batch("jobs"), manifest)

    def get_job(self, name: str) -> dict:
        return self.request("GET", self._batch("jobs", name))

    def delete_job(self, name: str) -> dict:
        return self.request("DELETE", self._batch("jobs", name))

    def list_jobs(self, label_selector: Optional[str] = None) -> dict:
        path = self._batch("jobs")
        if label_selector:
            path += "?labelSelector=" + urllib.parse.quote(label_selector)
        return self.request("GET", path)

    def create_cronjob(self, manifest: dict) -> dict:
        return self.request("POST", self._batch("cronjobs"), manifest)

    def delete_cronjob(self, name: str) -> dict:
        return self.request("DELETE", self._batch("cronjobs", name))


# -- manifest builders (pure) ------------------------------------------------


def _dns_name(name: str) -> str:
    """An RFC-1123 object-name-safe rendering of an instance name."""
    safe = "".join(ch if (ch.isalnum() or ch == "-") else "-" for ch in name.strip().lower())
    safe = safe.strip("-") or "ear"
    return safe[:52]


def _labels(instance: str, extra: Optional[dict]) -> dict:
    return {"app": "ear", "ear/instance": _dns_name(instance), **(extra or {})}


def container_spec(
    image: str,
    command: Optional[list] = None,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    cpu: Optional[str] = None,
    memory: Optional[str] = None,
    name: str = "ear",
) -> dict:
    spec: dict = {"name": name, "image": image}
    if command:
        spec["command"] = list(command)
    if args:
        spec["args"] = list(args)
    if env:
        spec["env"] = [{"name": str(key), "value": str(value)} for key, value in env.items()]
    limits = {}
    if cpu:
        limits["cpu"] = str(cpu)
    if memory:
        limits["memory"] = str(memory)
    if limits:
        spec["resources"] = {"limits": limits, "requests": limits}
    return spec


def job_manifest(
    name: str,
    image: str,
    *,
    command: Optional[list] = None,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    namespace: str = "default",
    labels: Optional[dict] = None,
    backoff_limit: int = 2,
    ttl_seconds: int = 3600,
    cpu: Optional[str] = None,
    memory: Optional[str] = None,
) -> dict:
    tags = _labels(name, labels)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": _dns_name(name), "namespace": namespace, "labels": tags},
        "spec": {
            "backoffLimit": backoff_limit,
            "ttlSecondsAfterFinished": ttl_seconds,
            "template": {
                "metadata": {"labels": tags},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [container_spec(image, command, args, env, cpu, memory)],
                },
            },
        },
    }


def cronjob_manifest(
    name: str,
    image: str,
    schedule: str,
    *,
    command: Optional[list] = None,
    args: Optional[list] = None,
    env: Optional[dict] = None,
    namespace: str = "default",
    labels: Optional[dict] = None,
    cpu: Optional[str] = None,
    memory: Optional[str] = None,
) -> dict:
    tags = _labels(name, labels)
    return {
        "apiVersion": "batch/v1",
        "kind": "CronJob",
        "metadata": {"name": _dns_name(name), "namespace": namespace, "labels": tags},
        "spec": {
            "schedule": schedule,
            "concurrencyPolicy": "Forbid",
            "successfulJobsHistoryLimit": 3,
            "failedJobsHistoryLimit": 3,
            "jobTemplate": {
                "spec": {
                    "template": {
                        "metadata": {"labels": tags},
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [container_spec(image, command, args, env, cpu, memory)],
                        },
                    }
                }
            },
        },
    }


def every_to_cron(seconds: float) -> str:
    """A cron schedule for a period in seconds -- minutes, whole hours, or
    whole days. Cron granularity is one minute; a sub-minute period is
    refused with a clear steer to the in-process Kernel scheduler, which
    has no such floor. Periods that don't fall on a clean minute/hour/day
    are refused too, with a steer to a literal schedule string."""
    minutes = seconds / 60
    if minutes < 1:
        raise KubeError(
            "CronJob granularity is one minute; use the in-process Kernel scheduler for sub-minute periods"
        )
    whole = int(round(minutes))
    if whole < 60:
        return f"*/{whole} * * * *"
    if whole % 60 == 0:
        hours = whole // 60
        if hours < 24:
            return f"0 */{hours} * * *"
        if hours == 24:
            return "0 0 * * *"  # daily at midnight
        if hours % 24 == 0 and hours // 24 <= 28:
            return f"0 0 */{hours // 24} * *"  # every N days
    raise KubeError(
        f"a period of {seconds}s does not map to a simple cron schedule; pass a literal schedule string instead"
    )


def _intent_env(intent: Any, context: Optional[dict]) -> dict:
    text = getattr(intent, "text", None) or str(intent)
    ctx = context if context is not None else dict(getattr(intent, "context", {}) or {})
    return {"EAR_INTENT": text, "EAR_CONTEXT": json.dumps(ctx)}


# -- the provider ------------------------------------------------------------


@dataclass
class KubeProvider:
    """Runs EAR instances on the cluster: one-off Jobs for a single cycle,
    CronJobs for recurring tasks, and an adapter so it can be the Kernel's
    dispatcher. The container is the EAR image, running the in-pod entry
    `python -m ear.run <stack_mount>` with the intent in the environment."""

    client: KubeClient
    image: str
    stack_mount: str = "/stack"
    cpu: Optional[str] = None
    memory: Optional[str] = None

    def run(self, instance: str, intent: Any, context: Optional[dict] = None, labels: Optional[dict] = None) -> dict:
        """Create a Job that runs one governed cycle for `instance`."""
        job = job_manifest(
            f"{_dns_name(instance)}-{uuid.uuid4().hex[:8]}",
            self.image,
            command=["python", "-m", "ear.run", self.stack_mount],
            env=_intent_env(intent, context),
            namespace=self.client.config.namespace,
            labels={"ear/instance": _dns_name(instance), **(labels or {})},
            cpu=self.cpu,
            memory=self.memory,
        )
        return self.client.create_job(job)

    def schedule(self, instance: str, intent: Any, every: float, context: Optional[dict] = None) -> dict:
        """Create a CronJob so the cluster runs `instance` on a period --
        Kubernetes owns the recurrence."""
        cron = cronjob_manifest(
            f"{_dns_name(instance)}-cron",
            self.image,
            every_to_cron(every),
            command=["python", "-m", "ear.run", self.stack_mount],
            env=_intent_env(intent, context),
            namespace=self.client.config.namespace,
            labels={"ear/instance": _dns_name(instance)},
            cpu=self.cpu,
            memory=self.memory,
        )
        return self.client.create_cronjob(cron)

    def status(self, job_name: str) -> dict:
        return self.client.get_job(job_name)

    def stop(self, job_name: str) -> dict:
        return self.client.delete_job(job_name)

    def as_dispatcher(self) -> Callable:
        """Adapt this provider to the Kernel's dispatcher seam: each task
        firing becomes a Job on the cluster, so the Kernel stays the single
        scheduler while the work runs in its own pod. Returns
        `(status, summary)` for the Kernel's Dispatch record."""

        def dispatch(task: Any, runtime: Any) -> tuple:
            created = self.run(task.instance, task.intent)
            return "dispatched", f"job {created.get('metadata', {}).get('name', '?')} created"

        return dispatch
