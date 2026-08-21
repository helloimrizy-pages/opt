#!/usr/bin/env bash
# Provision a CUDA host (RunPod, Lambda, a local box) for the Stage 1 grid.
#
#   bash optimizer_state_tta/scripts/setup_gpu_host.sh
#
# Idempotent: safe to re-run.  If optimizer_state_tta/data already holds
# CIFAR-10-C and the checkpoints (e.g. extracted from stage_assets.sh) nothing
# is downloaded.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
REPO="$(dirname "$PROJ")"
cd "$REPO"

VENV="${VENV:-$REPO/.venv-optstate}"
DATA_DIR="${DATA_DIR:-$PROJ/data}"
CKPT_DIR="${CKPT_DIR:-$PROJ/data/ckpt}"
PY="$VENV/bin/python"

# cuBLAS needs this set before the first CUDA call for deterministic GEMMs.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

banner () { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

banner "host"
uname -a
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  N_GPUS_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
else
  echo "nvidia-smi not found — this script targets CUDA hosts"
  N_GPUS_DETECTED=0
fi

banner "python environment"
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if [ ! -x "$PY" ]; then
  uv venv --python 3.11 "$VENV"
  # On Linux the default PyPI torch wheel is the CUDA build.  Pin the index
  # explicitly if a specific CUDA runtime is required.
  VIRTUAL_ENV="$VENV" uv pip install ${TORCH_INDEX:+--index-url "$TORCH_INDEX"} \
      torch torchvision
  VIRTUAL_ENV="$VENV" uv pip install numpy scipy pandas matplotlib pytest \
      requests tqdm "setuptools<81" pypdf
  VIRTUAL_ENV="$VENV" uv pip install \
      "robustbench @ git+https://github.com/RobustBench/robustbench.git"
fi
"$PY" -W ignore -c "
import torch, torchvision, robustbench, sys
print('python', sys.version.split()[0], '| torch', torch.__version__,
      '| torchvision', torchvision.__version__,
      '| robustbench', getattr(robustbench, '__version__', 'installed'))
print('cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', i, torch.cuda.get_device_name(i))
"

banner "assets"
mkdir -p "$DATA_DIR" "$CKPT_DIR"
if [ ! -f "$DATA_DIR/CIFAR-10-C/labels.npy" ]; then
  echo "CIFAR-10-C not staged; downloading ~2.9 GB from Zenodo record 2535967"
  "$PY" -W ignore -c "
from robustbench.data import load_cifar10c
load_cifar10c(200, 5, '$DATA_DIR', False, ['gaussian_noise'])
print('CIFAR-10-C ready')
"
else
  echo "CIFAR-10-C already present ($(ls "$DATA_DIR/CIFAR-10-C" | wc -l | tr -d ' ') files)"
fi
if [ ! -f "$CKPT_DIR/cifar10/corruptions/Standard.pt" ]; then
  echo "checkpoint not staged; fetching from Google Drive (gdown may be rate-limited)"
  "$PY" -W ignore -c "
from robustbench.utils import load_model
load_model('Standard', '$CKPT_DIR', 'cifar10', 'corruptions')
load_model('Hendrycks2020AugMix_WRN', '$CKPT_DIR', 'cifar10', 'corruptions')
print('checkpoints ready')
" || { echo "checkpoint download failed — stage it with stage_assets.sh instead" >&2; exit 1; }
else
  echo "checkpoints already present"
fi

banner "verification"
"$PY" -m pytest "$PROJ/tests" -q
"$PY" -W ignore "$HERE/check_determinism.py" --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR"

banner "throughput probe"
"$PY" -W ignore - <<'PYPROBE'
import sys, time, torch
from pathlib import Path
ROOT = Path("optimizer_state_tta")
sys.path.insert(0, str(ROOT / "src"))
from optstate.model import (load_source_model, configure_tent_model,
                            collect_bn_params, make_adam)
from optstate.tent_core import tent_step
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = configure_tent_model(load_source_model("Standard", str(ROOT / "data/ckpt"), dev))
ps, _ = collect_bn_params(m)
opt = make_adam(ps, 1e-3, 0.9, 0.999, 0.0)
x = torch.randn(200, 3, 32, 32, device=dev)
y = torch.randint(0, 10, (200,), device=dev)
for i in range(5):
    tent_step(m, opt, x, y, i)
if dev.type == "cuda":
    torch.cuda.synchronize()
t = time.time()
for i in range(30):
    tent_step(m, opt, x, y, i)
if dev.type == "cuda":
    torch.cuda.synchronize()
s = (time.time() - t) / 30
print(f"{s*1000:.0f} ms per Tent step (batch 200, WRN-28-10, fwd+bwd+Adam)")
print(f"full grid is 119,175 steps -> {119175*s/3600:.2f} h on one worker")
PYPROBE

banner "ready"
cat <<EOF
Run the whole study (baseline, grid, analysis, figures, verdict):

  N_GPUS=${N_GPUS_DETECTED:-2} WORKERS_PER_GPU=2 \\
    bash optimizer_state_tta/scripts/run_optimizer_state_stage1.sh

Or drive only the experiment grid:

  $PY optimizer_state_tta/scripts/run_parallel.py \\
      --gpus ${N_GPUS_DETECTED:-2} --workers-per-gpu 2 --device cuda

Add --dry-run first to see the plan and the step budget.
Tune WORKERS_PER_GPU with the ms/step figure above: this workload
under-occupies a datacentre GPU, so 2-3 usually helps.
EOF
