#!/usr/bin/env python3
"""Visualize neural signals using NeuralBehaviorDataset parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neural_behavior_dataset import NeuralBehaviorDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='Directory with CSV trials.')
    p.add_argument('--cache-dir', type=Path, default=Path('/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116/cache'))
    p.add_argument('--trials', nargs='*', help='List of CSV filenames to include (default all).')
    p.add_argument('--window-size', type=int, default=64)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--normalization-mode', choices=['none', 'zscore', 'percentile'], default='none')
    p.add_argument('--percentile-range', type=float, nargs=2, metavar=('LOW','HIGH'),
                   help='Required when normalization-mode=percentile.')
    p.add_argument('--no-percentile-clip', dest='percentile_clip', action='store_false', help='If set, keeps values beyond percentiles.')
    p.set_defaults(percentile_clip=True)
    p.add_argument('--activity-sigma-mult', type=float, default=2.5)
    p.add_argument('--min-spike-count', type=int, default=40)
    p.add_argument('--outlier-sigma-mult', type=float, default=8.0)
    p.add_argument('--resting-labels', nargs='*', default=['resting'])
    p.add_argument('--labels', nargs='*', help='Only include these labels (case-insensitive).')
    p.add_argument('--num-windows', type=int, default=5, help='Number of windows per label to plot.')
    p.add_argument('--output-dir', type=Path, default=Path('visualizations'))
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--plot-percentiles', type=float, nargs=2, metavar=('LOW','HIGH'),
                   help='Percentiles used to set y-limits across plots.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.normalization_mode == 'percentile' and not args.percentile_range:
        raise SystemExit("Error: --percentile-range must be set when --normalization-mode=percentile")

    dataset = NeuralBehaviorDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        trials=args.trials,
        cache_dir=args.cache_dir,
        normalization_mode=args.normalization_mode,
        percentile_range=tuple(args.percentile_range) if args.percentile_range else None,
        percentile_clip=args.percentile_clip,
        activity_sigma_mult=args.activity_sigma_mult,
        min_spike_count=args.min_spike_count,
        outlier_sigma_mult=args.outlier_sigma_mult,
        resting_labels=args.resting_labels,
        include_labels=args.labels,
        use_cache=True,
    )

    label_names = dataset.label_encoder.inverse_transform(np.arange(len(dataset.label_encoder.classes_)))
    per_label_count = {name: 0 for name in label_names}

    collected = []
    values_for_range = []

    for idx in range(len(dataset)):
        signal, label_idx = dataset[idx]
        label_name = label_names[label_idx]
        if per_label_count[label_name] >= args.num_windows:
            continue

        per_label_count[label_name] += 1
        signal_np = signal.numpy().squeeze()
        collected.append((signal_np, label_name, per_label_count[label_name], idx))
        values_for_range.append(signal_np)

        if all(count >= args.num_windows for count in per_label_count.values()):
            break

    if not collected:
        print("No windows collected with the specified parameters.")
        return

    if args.plot_percentiles:
        stacked = np.concatenate(values_for_range)
        y_low = np.nanpercentile(stacked, args.plot_percentiles[0])
        y_high = np.nanpercentile(stacked, args.plot_percentiles[1])
    else:
        stacked = np.concatenate(values_for_range)
        y_low = float(stacked.min())
        y_high = float(stacked.max())

    by_label = {label: [] for label in per_label_count.keys()}
    for signal_np, label_name, window_idx, dataset_idx in collected:
        by_label[label_name].append((signal_np, window_idx, dataset_idx))

    for label_name, windows in by_label.items():
        if not windows:
            continue
        fig, axes = plt.subplots(len(windows), 1, figsize=(12, 3 * len(windows)), sharex=True)
        if len(windows) == 1:
            axes = [axes]
        for ax, (signal_np, window_idx, dataset_idx) in zip(axes, windows):
            ax.plot(signal_np, linewidth=1.2)
            ax.set_ylim(y_low, y_high)
            ax.set_ylabel("Norm amp")
            ax.set_title(f"{label_name} | window #{window_idx} | idx={dataset_idx}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time step")
        fig.tight_layout()
        outfile = args.output_dir / f"label-{label_name}_stack.png"
        fig.savefig(outfile, dpi=args.dpi)
        plt.close(fig)

    print('Saved figures:')
    for label, count in per_label_count.items():
        print(f"  {label}: {count} windows")


if __name__ == '__main__':
    main()
