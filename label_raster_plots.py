#!/usr/bin/env python3
"""Generate raster-style plots for selected CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116"),
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/yedam/workspace/snu-pni-decoding/label_raster_plots"),
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Exact CSV filename to plot. If omitted, the available files will be listed.",
    )
    parser.add_argument("--sample-num", type=int, default=20, help="Signals per label to plot.")
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=1.5,
        help="Spacing multiplier controlling distance between stacked traces.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--resting-labels",
        type=str,
        default="resting",
        help="Comma-separated list of labels that should bypass firing-rate filtering.",
    )
    parser.add_argument(
        "--outlier-sigma-mult",
        type=float,
        default=10.0,
        help="Drop rows when any sample exceeds mean +/- (mult * global std).",
    )
    parser.add_argument(
        "--activity-sigma-mult",
        type=float,
        default=3.0,
        help="Number of global standard deviations used to decide if a sample counts toward activity.",
    )
    parser.add_argument(
        "--min-spike-count",
        type=int,
        default=20,
        help="Minimum number of above-threshold samples required inside a row (non-rest labels).",
    )
    return parser.parse_args()


def sanitize_label(label_value: str) -> str:
    label_str = str(label_value)
    label_str = re.sub(r"[^A-Za-z0-9_-]+", "_", label_str)
    return label_str.strip("_") or "label"


def list_csvs(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.csv"))


def main() -> int:
    args = parse_args()
    resting_labels = {label.strip().lower() for label in args.resting_labels.split(",") if label.strip()}
    if not resting_labels:
        resting_labels = {"resting"}

    if not args.data_dir.exists():
        print(f"Data directory does not exist: {args.data_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list_csvs(args.data_dir)

    if not csv_files:
        print(f"No CSV files found in {args.data_dir}", file=sys.stderr)
        return 1

    if args.target is None:
        print("Available CSV files:")
        for path in csv_files:
            print(f"  - {path.name}")
        print("\nPass --target <filename> to generate plots for a specific file.")
        return 0

    target_path = args.data_dir / args.target
    if not target_path.exists():
        print(f"Target CSV not found: {target_path}", file=sys.stderr)
        return 1

    print(f"Loading {target_path} ...")
    df = pd.read_csv(target_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least two columns (features + label)")

    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = df[label_col]
    unique_labels = labels.unique()

    # Establish a consistent vertical scale across all plots
    global_low = np.nanpercentile(features.values, 1)
    global_high = np.nanpercentile(features.values, 99)
    if not np.isfinite(global_low) or not np.isfinite(global_high) or global_low == global_high:
        raise ValueError("Unable to determine global percentile range for features.")
    global_range = global_high - global_low
    global_mean = np.nanmean(features.values)
    global_std = np.nanstd(features.values)
    if not np.isfinite(global_std) or global_std == 0:
        raise ValueError("Unable to compute global standard deviation for activity threshold.")
    activity_delta = args.activity_sigma_mult * global_std

    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    total_rows_all = len(features)
    after_outlier_all = 0
    after_spike_all = 0

    for label_value in unique_labels:
        label_mask = labels == label_value
        label_rows = features[label_mask]
        total_rows = len(label_rows)
        if label_rows.empty:
            continue

        label_key = str(label_value).strip().lower()
        numeric_rows = label_rows.to_numpy()
        outlier_mask = np.any(np.abs(numeric_rows - global_mean) >= args.outlier_sigma_mult * global_std, axis=1)
        if outlier_mask.any():
            label_rows = label_rows.iloc[~outlier_mask]
            numeric_rows = numeric_rows[~outlier_mask]
            if label_rows.empty:
                print(f"  Skipping label '{label_value}' (all {total_rows} rows flagged as extreme outliers)")
                continue
        filtered_rows = len(label_rows)
        after_outlier_all += filtered_rows
        if label_key not in resting_labels:
            spike_counts = (np.abs(numeric_rows - global_mean) >= activity_delta).sum(axis=1)
            spike_mask = spike_counts >= args.min_spike_count
            if not spike_mask.any():
                print(
                    f"  Skipping label '{label_value}' "
                    f"(no rows meet firing-rate threshold; {filtered_rows}/{total_rows} rows after outlier filter)"
                )
                continue
            label_rows = label_rows.iloc[spike_mask]
            numeric_rows = numeric_rows[spike_mask]
        kept_rows = len(label_rows)
        after_spike_all += kept_rows
        print(f"Label '{label_value}': {total_rows} rows -> {kept_rows} after filtering")

        n_rows = min(args.sample_num, len(label_rows))
        sampled = (
            label_rows.sample(n=n_rows, random_state=args.random_state)
            if len(label_rows) > n_rows
            else label_rows
        )
        signals = sampled.to_numpy()
        signals = np.clip(signals, global_low, global_high)

        offsets = np.arange(n_rows) * global_range * args.vertical_scale

        fig_height = max(3, n_rows * 0.4)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        for idx, signal in enumerate(signals):
            ax.plot(signal + offsets[idx], linewidth=0.8)

        ax.set_title(f"{target_path.name} | Label: {label_value} | {n_rows} signals", fontsize=12)
        ax.set_xlabel("Channel Index (0-based)")
        ax.set_ylabel("Sample Index (stacked)")
        ax.set_xlim(0, signals.shape[1] - 1)
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"sample_{i+1}" for i in range(n_rows)])
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()

        outfile = args.output_dir / f"{target_path.stem}_label-{sanitize_label(label_value)}_raster.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"  Saved plot for label '{label_value}' -> {outfile}")

    print(
        f"\nTotals: {total_rows_all} rows -> {after_outlier_all} after outlier filter -> {after_spike_all} final"
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
