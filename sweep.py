#!/usr/bin/env python3
"""
Hyperparameter sweep launcher for train_gru_behavior.py.

- Define a baseline config.
- Define per-parameter sweep values (1D sweeps, one param at a time).
- Run experiments in parallel across multiple GPUs.
- Collect metrics into a pandas DataFrame and save to disk.

Usage:
    python sweep_hyperparams.py
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Make sure this file is in the same folder as train_gru_behavior.py
from train_gru_behavior import (
    DatasetConfig,
    TrainingConfig,
    run_single_training,
)


# --------------------------------------------------------------------------------------
# USER CONFIG: GPUs & PARALLELISM
# --------------------------------------------------------------------------------------

# Which GPUs to use for the sweep
GPU_IDS: List[int] = [0, 1, 2, 3]  # e.g. [0] for single GPU, [0,1,2] for multi-GPU

# Max number of concurrent training processes per GPU
MAX_JOBS_PER_GPU: int = 1

# Root directory for logs of ALL sweep runs
LOG_ROOT = Path("/media/NAS_179_2_josh_2/snu-pni-decoding/results_sweep")


# --------------------------------------------------------------------------------------
# USER CONFIG: BASELINE HYPERPARAMETERS
# (match your Bash script defaults)
# --------------------------------------------------------------------------------------

BASELINE_DATASET_CFG = {
    "data_dir": Path("/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116"),
    "cache_dir": Path("/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116/cache"),
    "trials": None,  # or list of trial filenames if you want to restrict
    "window_size": 2441,
    "stride": 2441,  # will always be kept equal to window_size
    "normalization_mode": "percentile",  # 'none', 'zscore', 'percentile'
    "percentile_range": (1.0, 99.0),      # (1.0, 99.0) if using percentile
    "percentile_clip": False,
    "activity_sigma_mult": 0.0,
    "outlier_sigma_mult": float("inf"),
    "resting_labels": [],          # [] == use default in Dataset if you want
    "include_labels": ["walking", "climbing", "standing", "grooming"],
}

# enforce baseline stride == window_size (in case you edit one later)
BASELINE_DATASET_CFG["stride"] = BASELINE_DATASET_CFG["window_size"]

BASELINE_TRAINING_CFG = {
    "epochs": 201,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "train_ratio": 0.8,
    "hidden_size": 64,
    "num_layers": 4,
    "dropout": 0.0,
    "model_type": "gru",
    "transformer_d_model": 256,
    "transformer_nhead": 4,
    "transformer_dim_feedforward": 512,
    "transformer_num_layers": 4,
    "transformer_activation": "gelu",
    "transformer_dropout": 0.1,
    "transformer_pooling": "cls",
    "log_interval": 50,
    "eval_interval": 10,
    "checkpoint_interval": 10,
    "num_workers": 4,
    "seed": 42,
    # imbalance handling
    "use_class_weights": True,
    "use_weighted_sampler": False,
}


# --------------------------------------------------------------------------------------
# USER CONFIG: HYPERPARAMETER SWEEP
# Each entry is a list of values. The script will:
# - run the baseline once
# - for each param, run baseline+modified for each value in the list
#
# Keys are dotted paths:
#   - "training.lr"  -> TrainingConfig.lr
#   - "training.batch_size"
#   - "dataset.window_size"
# etc.
#
# IMPORTANT: window_size == stride is enforced in code below.
#            You only need to sweep "dataset.window_size"; "stride" will follow it.
# --------------------------------------------------------------------------------------

HYPERPARAM_SWEEP: Dict[str, List[Any]] = {
    # --------------------
    # Training hyperparams
    # --------------------
    "training.lr": [1e-4],
    "training.batch_size": [64],
    "training.hidden_size": [8, 64, 512],
    "training.num_layers": [1, 2, 4],
    "training.dropout": [0.0, 0.5],
    "training.weight_decay": [0.0, 1e-4],
    "training.model_type": ["gru", "transformer"],
    "training.transformer_d_model": [128, 256, 512],
    "training.transformer_nhead": [2, 4, 8],
    "training.transformer_dim_feedforward": [256, 512, 1024],
    "training.transformer_num_layers": [2, 4, 6],
    "training.transformer_activation": ["relu", "gelu"],
    "training.transformer_dropout": [0.1, 0.3],
    "training.transformer_pooling": ["cls", "mean"],

    # --------------------
    # Dataset hyperparams
    # --------------------

    # window_size sweep; stride will be forced equal in make_experiment_configs()
    "dataset.window_size": [16, 512, 2441],

    # normalization mode
    "dataset.normalization_mode": ["none", "zscore", "percentile"],

    # each value here is a tuple -> matches DatasetConfig.percentile_range: Optional[Tuple[float,float]]
    # These only make sense when normalization_mode='percentile'; code below enforces that.
    "dataset.percentile_range": [
        (1.0, 99.0),
        (0.1, 99.9),
    ],

    # whether to clip before scaling in percentile mode
    "dataset.percentile_clip": [False, True],

    # activity-based filtering threshold
    "dataset.activity_sigma_mult": [0.0, 2.0, 5.0],

    # outlier rejection threshold (in stds)
    "dataset.outlier_sigma_mult": [float("inf"), 10.0],

    # imbalance variants
    "training.use_weighted_sampler": [False, True],
    "training.use_class_weights": [False, True],
}


# --------------------------------------------------------------------------------------
# EXPERIMENT GENERATION
# --------------------------------------------------------------------------------------


def make_experiment_configs() -> List[Dict[str, Any]]:
    """
    Generate a list of experiments from the baseline + 1D sweeps.
    Each experiment dict contains:
      - name
      - dataset (dict)
      - training (dict)
      - log_dir (str)
      - gpu_id (assigned later)
      - sweep_param (str | 'baseline')
      - sweep_value
    """
    experiments: List[Dict[str, Any]] = []

    # 1) Add baseline run
    baseline_exp = {
        "name": "baseline",
        "dataset": copy.deepcopy(BASELINE_DATASET_CFG),
        "training": copy.deepcopy(BASELINE_TRAINING_CFG),
        "sweep_param": "baseline",
        "sweep_value": None,
        "log_dir": str(LOG_ROOT),
    }
    # enforce window_size == stride on baseline, just in case
    baseline_exp["dataset"]["stride"] = baseline_exp["dataset"]["window_size"]
    experiments.append(baseline_exp)

    # 2) For each hyperparam, sweep each value while keeping others at baseline
    for param_name, values in HYPERPARAM_SWEEP.items():
        for v in values:
            exp = {
                "name": f"{param_name.replace('.', '_')}={v}",
                "dataset": copy.deepcopy(BASELINE_DATASET_CFG),
                "training": copy.deepcopy(BASELINE_TRAINING_CFG),
                "sweep_param": param_name,
                "sweep_value": v,
                "log_dir": str(LOG_ROOT),
            }

            prefix, key = param_name.split(".", 1)

            if prefix == "dataset":
                # Main assignment
                if key in ("window_size", "stride"):
                    # enforce window_size == stride
                    exp["dataset"]["window_size"] = v
                    exp["dataset"]["stride"] = v
                else:
                    exp["dataset"][key] = v

                # Dependency 1: percentile-specific params require normalization_mode='percentile'
                if key in ("percentile_clip", "percentile_range"):
                    exp["dataset"]["normalization_mode"] = "percentile"

                # Dependency 2: if we explicitly sweep normalization_mode to 'percentile',
                # ensure percentile_range is not None, or dataset will raise.
                if key == "normalization_mode" and v == "percentile":
                    if exp["dataset"].get("percentile_range") is None:
                        exp["dataset"]["percentile_range"] = (1.0, 99.0)

                # Always enforce stride == window_size after any dataset change
                exp["dataset"]["stride"] = exp["dataset"]["window_size"]

            elif prefix == "training":
                exp["training"][key] = v
            else:
                raise ValueError(f"Unknown param prefix {prefix} in {param_name}")

            experiments.append(exp)

    # Assign GPU in round-robin; concurrency will be limited by ProcessPoolExecutor
    if not GPU_IDS:
        raise ValueError("GPU_IDS is empty. Set GPU_IDS to e.g. [0] or [0,1].")

    for idx, exp in enumerate(experiments):
        exp["gpu_id"] = GPU_IDS[idx % len(GPU_IDS)]

    return experiments


# --------------------------------------------------------------------------------------
# WORKER FUNCTION (RUNS IN SEPARATE PROCESS)
# --------------------------------------------------------------------------------------


def run_experiment(exp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs a single training experiment in a separate process on a specific GPU.

    Returns a flat dict with:
      - sweep metadata
      - best metrics
      - flattened dataset/training configs
    """
    # Pin this process to one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(exp["gpu_id"])

    # Import inside worker to respect CUDA_VISIBLE_DEVICES in some environments
    from train_gru_behavior import DatasetConfig, TrainingConfig, run_single_training  # type: ignore

    dataset_cfg = DatasetConfig(**exp["dataset"])
    training_cfg = TrainingConfig(**exp["training"])
    log_root = Path(exp["log_dir"])

    metrics = run_single_training(
        run_name=exp["name"],
        dataset_cfg=dataset_cfg,
        train_cfg=training_cfg,
        log_root=log_root,
    )

    # Flatten result row
    row: Dict[str, Any] = {
        "run_name": metrics.get("run_name", exp["name"]),
        "sweep_param": exp["sweep_param"],
        "sweep_value": exp["sweep_value"],
        "gpu_id": exp["gpu_id"],
        "best_epoch": metrics.get("best_epoch"),
        "best_test_acc": metrics.get("best_test_acc"),
        "best_test_loss": metrics.get("best_test_loss"),
        "train_loss_at_best": metrics.get("train_loss_at_best"),
        "train_acc_at_best": metrics.get("train_acc_at_best"),
    }

    # Also record all config values in the row
    for k, v in exp["dataset"].items():
        row[f"dataset_{k}"] = v
    for k, v in exp["training"].items():
        row[f"training_{k}"] = v

    return row


# --------------------------------------------------------------------------------------
# MAIN SWEEP DRIVER
# --------------------------------------------------------------------------------------


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    experiments = make_experiment_configs()
    n_exps = len(experiments)
    print(f"[SWEEP] Total experiments: {n_exps}")
    for i, exp in enumerate(experiments):
        print(f"  [{i:03d}] {exp['name']}  (GPU {exp['gpu_id']})")

    max_workers = len(GPU_IDS) * MAX_JOBS_PER_GPU
    print(
        f"[SWEEP] Using GPUs: {GPU_IDS}, max {MAX_JOBS_PER_GPU} jobs per GPU "
        f"-> max_workers = {max_workers}"
    )

    rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_experiment, exp) for exp in experiments]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Experiment failed: {e}")
                continue
            rows.append(result)
            print(
                f"[SWEEP] Completed {i}/{n_exps}: {result['run_name']} "
                f"(param={result['sweep_param']} value={result['sweep_value']})"
            )

    if not rows:
        print("[SWEEP] No successful runs. Exiting.")
        return

    df = pd.DataFrame(rows)
    results_csv = LOG_ROOT / "hyperparam_sweep_results.csv"
    results_pkl = LOG_ROOT / "hyperparam_sweep_results.pkl"

    df.to_csv(results_csv, index=False)
    df.to_pickle(results_pkl)

    print(f"[SWEEP] Saved results to:\n  {results_csv}\n  {results_pkl}")
    print("[SWEEP] Done.")


if __name__ == "__main__":
    main()
