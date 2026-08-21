#!/usr/bin/env bash
# Bundle CIFAR-10-C and the RobustBench checkpoints for transfer to a GPU host.
#
# Why: the checkpoints come from Google Drive via gdown, which is frequently
# rate-limited from datacentre IPs, and CIFAR-10-C is a 2.9 GB Zenodo download.
# Staging them once locally makes provisioning deterministic and fast.
#
#   bash optimizer_state_tta/scripts/stage_assets.sh [out.tar]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
DATA="${DATA_DIR:-$PROJ/data}"
OUT="${1:-$PROJ/optstate_assets.tar}"

for required in "$DATA/CIFAR-10-C/labels.npy" \
                "$DATA/ckpt/cifar10/corruptions/Standard.pt"; do
  [ -f "$required" ] || { echo "missing $required — run the download step first" >&2; exit 1; }
done

CORRUPTIONS=(gaussian_noise shot_noise impulse_noise defocus_blur glass_blur \
             motion_blur zoom_blur snow frost fog brightness contrast \
             elastic_transform pixelate jpeg_compression)

# Only the 15 standard corruptions are needed; CIFAR-10-C also ships 4 extras
# (gaussian_blur, saturate, spatter, speckle_noise) that this study never reads.
FILES=("CIFAR-10-C/labels.npy")
for c in "${CORRUPTIONS[@]}"; do FILES+=("CIFAR-10-C/${c}.npy"); done
FILES+=("ckpt/cifar10/corruptions/Standard.pt")
[ -f "$DATA/ckpt/cifar10/corruptions/Hendrycks2020AugMix_WRN.pt" ] && \
  FILES+=("ckpt/cifar10/corruptions/Hendrycks2020AugMix_WRN.pt")

echo "bundling ${#FILES[@]} files from $DATA"
tar -C "$DATA" -cf "$OUT" "${FILES[@]}"
shasum -a 256 "$OUT" > "$OUT.sha256" 2>/dev/null || sha256sum "$OUT" > "$OUT.sha256"
ls -lh "$OUT"; cat "$OUT.sha256"
cat <<EOF

Transfer, then on the GPU host:
  mkdir -p optimizer_state_tta/data
  tar -C optimizer_state_tta/data -xf $(basename "$OUT")
  bash optimizer_state_tta/scripts/setup_gpu_host.sh
EOF
