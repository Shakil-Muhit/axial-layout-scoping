#!/bin/bash
# ml-beast environment bootstrap — reproduces the EFFECTS of
# sm120-nulltest/setup.sh (handoff section 2: "same image/setup") on a
# no-sudo workstation instead of a root pod:
#   [B1] venv with the pinned stack (torch 2.8.0+cu128 -> triton 3.4.0) +
#        the msst minimal dep list from sm120-nulltest@a997e27 setup.sh [5/6]
#   [B2] user-space Nsight Systems (dpkg -x, no root) with cuda-graph-trace
#   [B3] msst clone @ the pin from sm120-nulltest/configs/msst_commit.txt
#   [B4] checkpoint via sm120-nulltest/scripts/fetch_checkpoint.sh (rule 2)
# Everything on LOCAL disk (~/axial): the repo mount is sshfs/noexec and slow.
# Idempotent: every step checks-before-does.
set -u
AX="$HOME/axial"
REPO_MNT="/mnt/Muhit/Home/ms/axial-layout-scoping"
SM120="/mnt/Muhit/Home/ms/sm120-nulltest"
mkdir -p "$AX"

echo "=== [B1] venv + pinned stack ==="
if [ ! -x "$AX/venv/bin/python" ]; then python3 -m venv "$AX/venv" || exit 1; fi
V="$AX/venv/bin"
"$V/pip" install -q --upgrade pip || exit 1
"$V/python" - <<'EOF' 2>/dev/null || NEED=1
import torch; assert torch.__version__ == '2.8.0+cu128'
EOF
if [ "${NEED:-0}" = "1" ]; then
    "$V/pip" install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || exit 1
fi
"$V/pip" install -q librosa einops pyyaml omegaconf ml_collections soundfile wandb beartype rotary_embedding_torch tqdm packaging loralib pandas || exit 1
"$V/python" -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__)" || exit 1
CUDA_VISIBLE_DEVICES=4 "$V/python" -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0), '| cap', torch.cuda.get_device_capability(0))" || exit 1

echo "=== [B2] user-space nsys ==="
NSYS_BIN="$(ls -d "$AX"/nsight/opt/nvidia/nsight-systems*/*/bin/nsys 2>/dev/null | sort -V | tail -1)"
if [ -z "$NSYS_BIN" ]; then
    mkdir -p "$AX/nsight" && cd "$AX/nsight"
    REPO_URL="https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64"
    # newest versioned CLI package from the repo index ('^nsight-systems-[0-9]'
    # discipline — the sm120 shakedown's nsight-systems-target lesson)
    DEB=$(wget -qO- "$REPO_URL/Packages" | grep -A8 "^Package: nsight-systems-20" \
          | grep "^Filename:" | awk '{print $2}' | sort -V | tail -1)
    [ -n "$DEB" ] || { echo "FATAL: no nsight-systems deb found in repo index"; exit 1; }
    echo "fetching $DEB"
    # Filename: is relative to the flat repo root
    wget -q "$REPO_URL/$(echo "$DEB" | sed 's|^\./||')" -O ns.deb \
        || wget -q "$REPO_URL/$(basename "$DEB")" -O ns.deb || exit 1
    dpkg -x ns.deb "$AX/nsight" || exit 1
    rm -f ns.deb
    NSYS_BIN="$(ls -d "$AX"/nsight/opt/nvidia/nsight-systems*/*/bin/nsys 2>/dev/null | sort -V | tail -1)"
fi
[ -n "$NSYS_BIN" ] || { echo "FATAL: nsys binary not found after extract"; exit 1; }
"$NSYS_BIN" --version | head -1
"$NSYS_BIN" profile --help 2>/dev/null | grep -q cuda-graph-trace \
    && echo "graph-trace: SUPPORTED" || { echo "FATAL: no cuda-graph-trace"; exit 1; }
echo "$NSYS_BIN" > "$AX/nsys_path.txt"

echo "=== [B3] msst clone @ pin ==="
PIN=$(grep -v '^[[:space:]]*#' "$SM120/configs/msst_commit.txt" | grep -m1 '[^[:space:]]' | tr -d '[:space:]')
if [ ! -d "$AX/msst/.git" ]; then git clone -q https://github.com/ZFTurbo/Music-Source-Separation-Training.git "$AX/msst" || exit 1; fi
if [ "$(git -C "$AX/msst" rev-parse HEAD)" != "$PIN" ]; then
    git -C "$AX/msst" fetch -q origin || true
    git -C "$AX/msst" checkout -q --detach "$PIN" || { echo "FATAL: pin checkout failed"; exit 1; }
fi
echo "msst @ $(git -C "$AX/msst" rev-parse HEAD)"

echo "=== [B4] checkpoint via sm120 fetch script (handoff rule 2) ==="
CKPT_DIR="$AX/ckpt" MSST_DIR="$AX/msst" bash "$SM120/scripts/fetch_checkpoint.sh" || exit 1

echo "BOOTSTRAP-DONE"
