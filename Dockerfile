# Reference image for an EAR runtime instance (ear/k8s.py's KubeProvider.image) --
# one pod, one governed cycle, via `python -m ear.run <stack>`.
#
# EAR's own package has zero third-party dependencies (see pyproject.toml:
# `dependencies = []`) -- `pip install .` here pulls in nothing. Node/npm
# below are *not* a Python dependency of EAR: they are OS-level toolchains
# the sandboxed agent may shell out to (Sandbox.run, the code_executor /
# environment_admin toolset names) when a task calls for them. Keeping that
# distinction is the point -- EAR's own code stays dependency-free; what
# the *agent* can reach inside its confined workspace is a deployment
# choice, made here, not something ear/ imports.
FROM python:3.12-slim

# nodejs/npm from Debian's own apt repo -- no external script piped into
# a shell, no nvm. Pin nothing more precisely than the base image already
# does; an operator who needs a specific Node version should override this
# layer, not patch EAR.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ear ./ear
RUN pip install --no-cache-dir .

# Verified, not assumed -- fails the build loudly if either toolchain
# didn't actually land, the same "checked, never assumed" rule
# Sandbox.capabilities() applies at runtime.
RUN python3 --version && node --version && npm --version && pip --version

ENTRYPOINT ["python", "-m", "ear.run"]
