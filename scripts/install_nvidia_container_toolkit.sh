#!/usr/bin/env bash
# Installs nvidia-container-toolkit on this WSL2 / Ubuntu host so that
# `docker run --gpus all ...` and compose's `deploy.resources.reservations.devices`
# can pass the host GPU into containers. Required for the `trainer` service.
#
# Run from anywhere; must be run as root:
#   sudo bash scripts/install_nvidia_container_toolkit.sh
#
# Safe to re-run: apt will report "already the newest version" on repeat runs
# and nvidia-ctk runtime configure is idempotent.
#
# Notes for WSL2 users:
#   - Install ONLY inside the WSL distro. Do NOT install the Linux NVIDIA
#     driver — WSL ships libcuda via /usr/lib/wsl/lib from the Windows driver.
#   - The Docker daemon must be restarted to pick up the new runtime config.
#     We use `service docker restart` since WSL2 typically runs without systemd.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (sudo bash $0)" >&2
    exit 1
fi

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
APT_LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "[1/4] Registering NVIDIA apt source"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o "$KEYRING"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" \
    > "$APT_LIST"

echo "[2/4] apt update + install nvidia-container-toolkit"
apt-get update
apt-get install -y nvidia-container-toolkit

echo "[3/4] Configuring docker runtime"
nvidia-ctk runtime configure --runtime=docker

echo "[4/4] Restarting docker daemon"
if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running --quiet 2>/dev/null; then
    systemctl restart docker
else
    service docker restart
fi

echo
echo "Done. Verify with:"
echo "  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
echo
echo "Then run the regression test:"
echo "  bash scripts/test_cuda_passthrough.sh"
