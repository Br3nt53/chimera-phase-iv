#!/usr/bin/env python3
"""Simple command-line interface to run training tasks.

This script provides a minimal front-end for researchers to start
experiments without manually invoking training modules.  It wraps the
``train_once`` function from ``trainers.train_cfd2d`` and exposes a
single ``train`` command with common hyperparameters.
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainers.train_cfd2d import train_once
from sample_data import ensure_sample_data

def main():
    parser = argparse.ArgumentParser(description="Research CLI")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Run a training session")
    train_p.add_argument("--model-config", default="configs/models/unet_small.yaml", help="Model config YAML")
    train_p.add_argument("--ind-cfg", default="configs/data/cfd2d_ind_v1.yaml", help="IND data config")
    train_p.add_argument("--ood-cfg", default="configs/data/cfd2d_ood_v1.yaml", help="OOD data config")
    train_p.add_argument("--out-root", default="artifacts/sample_run", help="Output directory root")
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--epochs", type=int, default=1)
    train_p.add_argument("--batch-size", type=int, default=2)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "train":
        # Ensure tiny sample dataset is present for quick experiments
        ensure_sample_data()
        train_once(
            model_cfg_path=args.model_config,
            ind_cfg_path=args.ind_cfg,
            ood_cfg_path=args.ood_cfg,
            out_root=args.out_root,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
