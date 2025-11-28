#!/usr/bin/env python3
"""Example driver to run a hyperparameter sweep for GRU behavior training."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Dict, List


def generate_sweep_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    dataset_template = {
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "trials": args.trials,
        "window_size": args.window_size,
        "stride": args.stride,
        "normalization_mode": args.normalization_mode,
        "percentile_range": args.percentile_range,
        "percentile_clip": args.percentile_clip,
        "activity_sigma_mult": args.activity_sigma_mult,
        "outlier_sigma_mult": args.outlier_sigma_mult,
        "resting_labels": args.resting_labels,
        "include_labels": args.include_labels,
    }

    training_grid = list(
        itertools.product(
            args.lr_list,
            args.batch_list,
            args.hidden_list,
            args.layer_list,
            args.dropout_list,
        )
    )

    sweep = []
    for idx, (lr, batch, hidden, layers, dropout) in enumerate(training_grid):
        cfg = {
            "name": f"{args.run_prefix}_lr{lr}_bs{batch}_h{hidden}_L{layers}_d{dropout}",
            "dataset": dataset_template,
            "training": {
                "epochs": args.epochs,
                "batch_size": batch,
                "lr": lr,
                "weight_decay": args.weight_decay,
                "train_ratio": args.train_ratio,
                "hidden_size": hidden,
                "num_layers": layers,
                "dropout": dropout,
                "log_interval": args.log_interval,
                "eval_interval": args.eval_interval,
                "checkpoint_interval": args.checkpoint_interval,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
        }
        sweep.append(cfg)
    return sweep


def write_sweep_file(sweep: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)


def run_training_script(train_script: Path, sweep_file: Path, log_dir: Path) -> None:
    cmd = [
        "python",
        str(train_script),
        "--data-dir",
        sweep_file,  # placeholder, overridden by sweep config
    ]
    # Using Sweep file flag
    cmd = [
        "python",
        str(train_script),
        "--data-dir",
        str("/"),  # dummy (overridden by sweep config)
        "--sweep-config",
        str(sweep_file),
        "--log-dir",
        str(log_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and run GRU behavior sweep.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/neural_behavior"))
    parser.add_argument("--trials", nargs="*", default=None)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--normalization-mode", choices=["none", "zscore", "percentile"], default="percentile")
    parser.add_argument("--percentile-range", type=float, nargs=2, default=(1.0, 99.0))
    parser.add_argument("--percentile-clip", action="store_true")
    parser.add_argument("--activity-sigma-mult", type=float, default=0.0)
    parser.add_argument("--outlier-sigma-mult", type=float, default=10.0)
    parser.add_argument("--resting-labels", nargs="*", default=["resting"])
    parser.add_argument(
        "--include-labels",
        nargs="*",
        default=["resting", "walking", "climbing", "standing", "grooming"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-list", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--batch-list", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--hidden-list", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--layer-list", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--dropout-list", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--log-dir", type=Path, default=Path("runs/gru_behavior"))
    parser.add_argument("--train-script", type=Path, default=Path("train_gru_behavior.py"))
    parser.add_argument("--run-prefix", type=str, default="sweep")
    parser.add_argument("--sweep-file", type=Path, default=Path("configs/gru_sweep.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = generate_sweep_configs(args)
    write_sweep_file(sweep, args.sweep_file)
    print(f"Generated {len(sweep)} configs at {args.sweep_file}")
    run_training_script(args.train_script, args.sweep_file, args.log_dir)


if __name__ == "__main__":
    main()

