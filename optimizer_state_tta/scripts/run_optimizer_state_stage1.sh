#!/usr/bin/env bash
# Stage 1 phenomenon validation for optimizer-state memory under continual TTA.
# Reproduces the whole study from a clean checkout on one machine.
#
#   bash optimizer_state_tta/scripts/run_optimizer_state_stage1.sh
#
# Stages can be skipped with environment variables, e.g.
#   SKIP_PRIMARY=1 SKIP_BETA1=1 bash .../run_optimizer_state_stage1.sh
#
# On a multi-GPU host, set N_GPUS (and optionally WORKERS_PER_GPU) to run the
# experiment grid through the worker pool in scripts/run_parallel.py instead of
# the sequential loops.  The runs are independent, so this changes only
# scheduling, never results:
#   N_GPUS=2 WORKERS_PER_GPU=2 bash .../run_optimizer_state_stage1.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
REPO="$(dirname "$PROJ")"
cd "$REPO"

VENV="${VENV:-$REPO/.venv-optstate}"
PY="$VENV/bin/python"
DATA_DIR="${DATA_DIR:-$PROJ/data}"
CKPT_DIR="${CKPT_DIR:-$PROJ/data/ckpt}"
RAW="$PROJ/results/optimizer_state_stage1/raw"
LOGS="$PROJ/logs"
mkdir -p "$RAW/boundary" "$RAW/baseline" "$RAW/toy" "$LOGS" \
         "$PROJ/figures/optimizer_state_stage1" "$PROJ/reports"

N_GPUS="${N_GPUS:-0}"                   # 0 = sequential, single device
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-0 1 2}"
ORDERS="${ORDERS:-conventional perm1 perm2 perm3}"
SWEEP_DOMAINS="${SWEEP_DOMAINS:-8}"     # 7 boundaries for the beta1 / lr subsets

banner () { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- #
banner "0. environment"
if [ ! -x "$PY" ]; then
  echo "creating $VENV with uv (python 3.11)"
  uv venv --python 3.11 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install torch torchvision numpy scipy pandas matplotlib \
      pytest requests tqdm "setuptools<81" pypdf
  VIRTUAL_ENV="$VENV" uv pip install \
      "robustbench @ git+https://github.com/RobustBench/robustbench.git"
fi
"$PY" -W ignore -c "
import torch, torchvision, robustbench, sys
print('python', sys.version.split()[0], '| torch', torch.__version__,
      '| torchvision', torchvision.__version__,
      '| robustbench', getattr(robustbench, '__version__', 'installed'))
print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())
"

banner "1. data + checkpoint"
if [ ! -d "$DATA_DIR/CIFAR-10-C" ]; then
  echo "downloading CIFAR-10-C (~2.9 GB) from Zenodo record 2535967"
  "$PY" -W ignore -c "
from robustbench.data import load_cifar10c
load_cifar10c(200, 5, '$DATA_DIR', False, ['gaussian_noise'])
print('CIFAR-10-C ready')
"
fi
if [ ! -f "$CKPT_DIR/cifar10/corruptions/Standard.pt" ]; then
  "$PY" -W ignore -c "
from robustbench.utils import load_model
load_model('Standard', '$CKPT_DIR', 'cifar10', 'corruptions')
load_model('Hendrycks2020AugMix_WRN', '$CKPT_DIR', 'cifar10', 'corruptions')
print('checkpoints ready')
"
fi

banner "2. unit tests (optimizer-state manipulation)"
"$PY" -m pytest "$PROJ/tests" -q

banner "2b. backend reproducibility check"
"$PY" -W ignore "$HERE/check_determinism.py" --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR"

banner "3. toy mechanistic sanity check (not evidence for the CV phenomenon)"
"$PY" -W ignore "$HERE/run_toy.py"

if [ -z "${SKIP_BASELINE:-}" ]; then
  banner "4. baseline reproduction (source / norm / tent, severity 5)"
  "$PY" -W ignore "$HERE/run_baseline.py" \
      --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR" 2>&1 | tee "$LOGS/baseline.log"
fi

run_boundary () {  # mode order seed beta1 lr max_domains
  local mode="$1" order="$2" seed="$3" b1="$4" lr="$5" maxd="$6"
  local tag="${mode}_${order}_seed${seed}_b1${b1}_lr${lr}"
  if [ -f "$RAW/boundary/${tag}.meta.json" ]; then
    echo "  [skip] $tag already complete"; return 0
  fi
  "$PY" -W ignore "$HERE/run_boundary_experiment.py" \
      --mode "$mode" --order "$order" --seed "$seed" --beta1 "$b1" --lr "$lr" \
      --max-domains "$maxd" --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR" \
      --out "$RAW/boundary" --tag "$tag" --device "$DEVICE" \
      2>&1 | tee -a "$LOGS/boundary.log"
}

# Stage order front-loads the decision-critical arms (H1 -> H2 -> H4) so partial
# results are already interpretable; runs are independent, so order does not
# affect any result.

if [ "$N_GPUS" -gt 0 ]; then
  banner "5-9. experiment grid on $N_GPUS GPU(s) x $WORKERS_PER_GPU worker(s)"
  STAGES=""
  [ -z "${SKIP_PRIMARY:-}"  ] && STAGES="$STAGES primary"
  [ -z "${SKIP_CONTROL:-}"  ] && STAGES="$STAGES control"
  [ -z "${SKIP_BETA1:-}"    ] && STAGES="$STAGES beta1"
  [ -z "${SKIP_LR:-}"       ] && STAGES="$STAGES lr"
  [ -z "${SKIP_GRADUAL:-}"  ] && STAGES="$STAGES gradual"
  [ -z "${SKIP_SEQUENCE:-}" ] && STAGES="$STAGES sequence"
  if [ -n "$STAGES" ]; then
    GRID_DEVICE="$DEVICE"
    [ "$GRID_DEVICE" = "auto" ] && GRID_DEVICE="cuda"
    "$PY" -W ignore "$HERE/run_parallel.py" \
        --gpus "$N_GPUS" --workers-per-gpu "$WORKERS_PER_GPU" \
        --device "$GRID_DEVICE" --stages $STAGES \
        --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR" \
        --out "$RAW/boundary" --log-dir "$LOGS/jobs" --python "$PY" \
        2>&1 | tee "$LOGS/parallel.log"
  fi
  SKIP_PRIMARY=1; SKIP_CONTROL=1; SKIP_BETA1=1
  SKIP_LR=1; SKIP_GRADUAL=1; SKIP_SEQUENCE=1
fi

if [ -z "${SKIP_PRIMARY:-}" ]; then
  banner "5a. primary matched-branch boundary experiment, conventional order (ORACLE_BOUNDARY_DIAGNOSTIC)"
  for seed in $SEEDS; do
    run_boundary primary conventional "$seed" 0.9 0.001 15
  done
fi

if [ -z "${SKIP_CONTROL:-}" ]; then
  banner "6. stationary pseudo-boundary control (matched checkpoint)"
  for seed in $SEEDS; do
    run_boundary control conventional "$seed" 0.9 0.001 15
  done
fi

if [ -z "${SKIP_BETA1:-}" ]; then
  banner "7. beta1 mechanistic validation"
  for b1 in 0 0.5 0.99; do
    for seed in $SEEDS; do
      run_boundary primary conventional "$seed" "$b1" 0.001 "$SWEEP_DOMAINS"
    done
  done
  # matched beta1=0.9 arm on the same truncated subset
  for seed in $SEEDS; do
    run_boundary primary conventional "$seed" 0.9 0.001 "$SWEEP_DOMAINS"
  done
fi

if [ -z "${SKIP_PRIMARY:-}" ]; then
  banner "5b. primary matched-branch boundary experiment, permuted corruption orders"
  for order in $ORDERS; do
    [ "$order" = "conventional" ] && continue
    for seed in $SEEDS; do
      run_boundary primary "$order" "$seed" 0.9 0.001 15
    done
  done
fi

if [ -z "${SKIP_LR:-}" ]; then
  banner "8. learning-rate robustness"
  for lr in 0.0003 0.003; do
    for seed in $SEEDS; do
      run_boundary primary conventional "$seed" 0.9 "$lr" "$SWEEP_DOMAINS"
    done
  done
fi

if [ -z "${SKIP_GRADUAL:-}" ]; then
  banner "9. optional gradual-shift control (severity 1->2->3->4->5)"
  for seed in $SEEDS; do
    run_boundary gradual gradual "$seed" 0.9 0.001 5
  done
fi

if [ -z "${SKIP_SEQUENCE:-}" ]; then
  banner "9b. SECONDARY whole-sequence strategy comparison (descriptive only)"
  for pol in CARRY_ALL RESET_M_KEEP_V_STEP RESET_V_KEEP_M_STEP RESET_MV_KEEP_STEP \
             RESET_STEP_ONLY FRESH_ADAM; do
    for seed in $SEEDS; do
      tag="sequence_conventional_seed${seed}_${pol}"
      if [ -f "$RAW/boundary/${tag}.meta.json" ]; then echo "  [skip] $tag"; continue; fi
      "$PY" -W ignore "$HERE/run_boundary_experiment.py" --mode sequence \
          --sequence-policy "$pol" --order conventional --seed "$seed" \
          --data-dir "$DATA_DIR" --ckpt-dir "$CKPT_DIR" --out "$RAW/boundary" \
          --tag "$tag" 2>&1 | tee -a "$LOGS/sequence.log"
    done
  done
fi

banner "10. analysis"
"$PY" -W ignore "$HERE/analyze_stage1.py" 2>&1 | tee "$LOGS/analysis.log"

banner "10b. baseline report"
"$PY" -W ignore "$HERE/make_baseline_report.py"

banner "11. figures"
"$PY" -W ignore "$HERE/make_figures.py"

banner "11b. report tables"
"$PY" -W ignore "$HERE/make_stage1_report.py" || true

banner "12. verdict"
"$PY" -W ignore "$HERE/verdict.py" 2>&1 | tee "$LOGS/verdict.log"
"$PY" -W ignore "$HERE/make_stage1_report.py" || true

echo
echo "raw data : $RAW"
echo "summary  : $PROJ/results/optimizer_state_stage1/summary.csv"
echo "figures  : $PROJ/figures/optimizer_state_stage1"
echo "report   : $PROJ/reports/optimizer_state_stage1.md"
