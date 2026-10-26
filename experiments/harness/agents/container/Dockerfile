# Agent-only image for Jittor Harness trials.
#
# Contains node, DSH and a plain Python without torchport. Nothing from the controller
# repository is copied in: the build context is this directory only.
#
#   docker build -t jittor-harness-agent:0.1 experiments/harness/agents/container
FROM node:24-bookworm-slim

ARG DSH_VERSION=0.2.0-rc.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git ripgrep g++ make \
        python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g "@deepseek-ai/dsh@${DSH_VERSION}" && npm cache clean --force

# The trial mounts an anonymous volume at /work/.scratch; Docker initializes it from this
# image path, so it must already exist and belong to the Agent user (uid 1000).
RUN mkdir -p /work/.scratch && chown node:node /work/.scratch

# uid 1000 matches the default owner of WSL drvfs mounts, so /work stays writable.
USER node
ENV HOME=/home/node \
    DSH_HOME=/home/node/.dsh \
    DSH_TELEMETRY_DISABLED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PATH=/home/node/.local/bin:$PATH

# Materialize the headless profile at build time so each trial starts from the same tree.
RUN dsh --profile headless --dump-config > /dev/null

WORKDIR /work
