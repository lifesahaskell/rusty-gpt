#!/usr/bin/env bash
# Verifies the host's docker daemon can pass through an NVIDIA GPU to
# containers — i.e. that `--gpus all` and the compose-style
# `deploy.resources.reservations.devices: [{driver: nvidia, ...}]`
# request will both succeed.
#
# Catches the failure mode we hit when nvidia-container-toolkit isn't
# installed:
#     docker: Error response from daemon: could not select device driver ""
#     with capabilities: [[gpu]]
#
# Usage:  bash scripts/test_cuda_passthrough.sh
# Exits non-zero with diagnostic guidance on failure.
#
# Skipping: set RUSTY_GPT_SKIP_CUDA_TEST=1 to no-op (e.g. on CPU-only CI).

set -euo pipefail

if [[ "${RUSTY_GPT_SKIP_CUDA_TEST:-}" == "1" ]]; then
    echo ":: SKIPPED (RUSTY_GPT_SKIP_CUDA_TEST=1)"
    exit 0
fi

CUDA_BASE="${RUSTY_GPT_CUDA_TEST_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"
COMPOSE_FILE="$(git rev-parse --show-toplevel 2>/dev/null)/compose.yaml"

fail() {
    echo
    echo ":: FAIL — $1" >&2
    if [[ -n "${2:-}" ]]; then
        echo
        echo ":: To fix:" >&2
        echo "$2" >&2
    fi
    exit 1
}

# --- 1. Host driver visibility -------------------------------------------
echo ":: [1/4] host driver visibility"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    # WSL2 stashes nvidia-smi under /usr/lib/wsl/lib if it isn't on PATH
    if [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        export PATH="/usr/lib/wsl/lib:$PATH"
    else
        fail "nvidia-smi not found on host" \
"On WSL2: nvidia-smi ships with the Windows driver under /usr/lib/wsl/lib/.
Install the latest NVIDIA driver on Windows (not inside WSL).
On bare Linux: install the NVIDIA proprietary driver for your distro."
    fi
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || fail "nvidia-smi failed on the host" \
"Driver/GPU is misconfigured. Fix host driver before trying container passthrough."

# --- 2. nvidia-container-toolkit installed ------------------------------
echo ":: [2/4] nvidia-container-toolkit installed"
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    fail "nvidia-ctk not found — nvidia-container-toolkit is not installed" \
"sudo bash scripts/install_nvidia_container_toolkit.sh"
fi
echo ":: nvidia-ctk $(nvidia-ctk --version | head -1)"

# --- 3. Direct --gpus all works ------------------------------------------
echo ":: [3/4] docker --gpus all passthrough"
gpus_output="$(docker run --rm --gpus all "$CUDA_BASE" nvidia-smi -L 2>&1 || true)"
if ! grep -q '^GPU [0-9]\+:' <<<"$gpus_output"; then
    echo "$gpus_output" >&2
    if grep -q 'could not select device driver' <<<"$gpus_output"; then
        fail "docker cannot resolve the 'gpu' capability" \
"This is the canonical missing-toolkit signature. Run:
  sudo bash scripts/install_nvidia_container_toolkit.sh
If already run, ensure the docker daemon restart took effect:
  sudo nvidia-ctk runtime configure --runtime=docker && sudo service docker restart"
    fi
    fail "GPU passthrough failed (see error above)" \
"Inspect 'docker info | grep -i runtime' and /etc/docker/daemon.json"
fi
echo ":: container saw: $(echo "$gpus_output" | head -1)"

# --- 4. Compose-style device reservation works --------------------------
# This is the actual mechanism compose.yaml uses for the trainer service,
# so we exercise it end-to-end rather than just --gpus all.
echo ":: [4/4] docker compose deploy.resources.reservations.devices passthrough"
TMPDIR_COMPOSE="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_COMPOSE"' EXIT
cat >"$TMPDIR_COMPOSE/compose.yaml" <<EOF
services:
  gpucheck:
    image: $CUDA_BASE
    command: ["nvidia-smi", "-L"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]
EOF

compose_output="$(docker compose -f "$TMPDIR_COMPOSE/compose.yaml" run --rm gpucheck 2>&1 || true)"
if ! grep -q '^GPU [0-9]\+:' <<<"$compose_output"; then
    echo "$compose_output" >&2
    fail "compose-style device reservation failed" \
"Likely an older docker compose (<2.3) or missing CDI spec.
Run: nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
Check: docker compose version (need >= 2.3)"
fi

echo
echo ":: PASS — GPU passthrough works for both 'docker run --gpus' and compose 'deploy.resources.reservations.devices'"
echo ":: trainer service in [$COMPOSE_FILE] is good to go"
