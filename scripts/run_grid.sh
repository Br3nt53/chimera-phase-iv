#!/usr/bin/env bash
set -e
echo "[run_grid] Starting dry-run grid for E1..."
for SEED in 0 1 2; do
  echo "[run_grid] Baseline UNet seed ${SEED}"
  python trainers/train_cfd2d.py     --experiment-config configs/experiments/E1_cfd2d.yaml     --model-config configs/models/unet_small.yaml     --data-config configs/data/cfd2d_ind_v1.yaml     --seed ${SEED} --dry-run
  echo "[run_grid] Fractal UNet seed ${SEED}"
  python trainers/train_cfd2d.py     --experiment-config configs/experiments/E1_cfd2d.yaml     --model-config configs/models/fractal_unet_small.yaml     --data-config configs/data/cfd2d_ind_v1.yaml     --seed ${SEED} --dry-run
done
echo "[run_grid] Done."
