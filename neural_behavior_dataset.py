#!/usr/bin/env python3
"""PyTorch dataset for behavior-label decoding from CSV-based neural recordings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: Path
    trials: Optional[Tuple[str, ...]]
    window_size: int
    stride: int
    activity_sigma_mult: float
    outlier_sigma_mult: float
    normalization_mode: str
    percentile_range: Optional[Tuple[float, float]]
    percentile_clip: bool
    include_labels: Optional[Tuple[str, ...]]
    resting_labels: Tuple[str, ...]

    def cache_suffix(self) -> str:
        cfg_dict = {
            "window_size": self.window_size,
            "stride": self.stride,
            "trials": self.trials,
            "activity_sigma_mult": self.activity_sigma_mult,
            "outlier_sigma_mult": self.outlier_sigma_mult,
            "normalization_mode": self.normalization_mode,
            "percentile_range": self.percentile_range,
            "percentile_clip": self.percentile_clip,
            "include_labels": self.include_labels,
            "resting_labels": self.resting_labels,
        }
        cfg_json = json.dumps(cfg_dict, sort_keys=True)
        return hashlib.md5(cfg_json.encode()).hexdigest()


class NeuralBehaviorDataset(Dataset):
    """Sliding-window neural dataset for behavior classification."""

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 32,
        stride: int = 1,
        trials: Optional[Sequence[str]] = None,
        cache_dir: str | Path = "cache",
        normalization_mode: str = "none",  # "none", "zscore", or "percentile"
        percentile_range: Optional[Tuple[float, float]] = None,
        percentile_clip: bool = True,
        activity_sigma_mult: float = 3.0,
        outlier_sigma_mult: float = 10.0,
        resting_labels: Optional[Iterable[str]] = None,
        include_labels: Optional[Iterable[str]] = None,
        dtype: torch.dtype = torch.float32,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.stride = stride
        normalization_mode = normalization_mode.lower()
        if normalization_mode not in {"none", "zscore", "percentile"}:
            raise ValueError("normalization_mode must be 'none', 'zscore', or 'percentile'")
        self.normalization_mode = normalization_mode
        self.percentile_range = percentile_range
        self.percentile_clip = percentile_clip
        self.activity_sigma_mult = activity_sigma_mult
        self.outlier_sigma_mult = outlier_sigma_mult
        self.resting_labels = tuple(sorted(l.lower() for l in (resting_labels or {"resting"})))
        self.include_labels = tuple(sorted(l.lower() for l in include_labels)) if include_labels else None
        self.dtype = dtype
        self.use_cache = use_cache

        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if percentile_range is not None:
            if len(percentile_range) != 2:
                raise ValueError("percentile_range must be a tuple of (low, high)")
            p_low, p_high = percentile_range
            if not (0 <= p_low < p_high <= 100):
                raise ValueError("percentile_range values must satisfy 0 <= low < high <= 100")

        self.trial_names = self._select_trials(trials)
        if not self.trial_names:
            raise ValueError("No CSV files selected. Check trials list and data directory.")

        self.label_encoder = LabelEncoder()
        self.config = DatasetConfig(
            data_dir=self.data_dir,
            trials=tuple(self.trial_names) if self.trial_names is not None else None,
            window_size=window_size,
            stride=stride,
            activity_sigma_mult=activity_sigma_mult,
            outlier_sigma_mult=outlier_sigma_mult,
            normalization_mode=self.normalization_mode,
            percentile_range=percentile_range,
            percentile_clip=percentile_clip,
            include_labels=self.include_labels,
            resting_labels=self.resting_labels,
        )

        all_labels = self._collect_labels()
        self.label_encoder.fit(all_labels)

        self.windows: List[Tuple[int, int, int, int]] = []  # (file_idx, signal_idx, start_step, end_step)
        self.file_arrays: List[np.ndarray] = []
        self.file_labels: List[np.ndarray] = []
        self.filter_stats = {
            "original": 0,
            "after_outlier": 0,
            "after_activity": 0,
            "after_label_filter": 0,
            "removed_outlier": defaultdict(int),
            "removed_activity": defaultdict(int),
            "removed_label_filter": defaultdict(int),
        }
        self._load_files()

    # ------------------------------------------------------------------
    def _select_trials(self, trials: Optional[Sequence[str]]) -> List[str]:
        csv_files = sorted(self.data_dir.glob("*.csv"))
        names = [f.name for f in csv_files]
        if trials is None:
            return names
        wanted = set(trials)
        missing = wanted - set(names)
        if missing:
            raise FileNotFoundError(f"Trials not found: {sorted(missing)}")
        return [name for name in names if name in wanted]

    def _collect_labels(self) -> np.ndarray:
        labels = []
        for name in self.trial_names:
            csv_path = self.data_dir / name
            sample = pd.read_csv(csv_path, nrows=1)
            if sample.empty:
                continue
            last_col = sample.columns[-1]
            df = pd.read_csv(csv_path, usecols=[last_col])
            cur_labels = df.iloc[:, 0].astype(str).tolist()
            if self.include_labels:
                cur_labels = [lbl for lbl in cur_labels if lbl.lower() in self.include_labels]
            labels.extend(cur_labels)
        return np.array(labels, dtype=object)

    def _load_files(self) -> None:
        cache_suffix = self.config.cache_suffix()
        for name in tqdm(self.trial_names, desc="Loading CSVs"):
            cache_file = self.cache_dir / f"{Path(name).stem}_{cache_suffix}.npz"
            if self.use_cache and cache_file.exists():
                cached = np.load(cache_file, allow_pickle=True)
                if "stats" in cached:
                    features = cached["features"]
                    labels = cached["labels"]
                    stats = cached["stats"].item()
                    label_max = labels.max() if labels.size > 0 else -1
                    if label_max >= len(self.label_encoder.classes_):
                        features, labels, stats = self._process_csv(self.data_dir / name)
                        np.savez_compressed(
                            cache_file,
                            features=features,
                            labels=labels,
                            stats=np.array([stats], dtype=object),
                        )
                else:
                    features, labels, stats = self._process_csv(self.data_dir / name)
                    np.savez_compressed(
                        cache_file,
                        features=features,
                        labels=labels,
                        stats=np.array([stats], dtype=object),
                    )
            else:
                features, labels, stats = self._process_csv(self.data_dir / name)
                np.savez_compressed(
                    cache_file,
                    features=features,
                    labels=labels,
                    stats=np.array([stats], dtype=object),
                )

            self._accumulate_stats(stats)
            if features.shape[1] < self.window_size:
                continue
            self.file_arrays.append(features)
            self.file_labels.append(labels)
            self._add_windows(file_idx=len(self.file_arrays) - 1)

        if not self.windows:
            raise RuntimeError("No data windows available after processing. Relax filtering parameters?")
        self._print_filter_summary()

    def _process_csv(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(f"CSV needs >=2 columns (features + label): {csv_path}")
        features = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        labels = df.iloc[:, -1].astype(str).to_numpy()

        raw_features = features.copy()

        # Outlier rejection must be based on raw data
        raw_mean = np.nanmean(raw_features)
        raw_std = np.nanstd(raw_features)
        if not np.isfinite(raw_std) or raw_std == 0:
            raw_std = 1.0

        if np.isfinite(self.outlier_sigma_mult):
            outlier_thresh = self.outlier_sigma_mult * raw_std
            deviations = np.max(np.abs(raw_features - raw_mean), axis=1)
            keep_mask = deviations < outlier_thresh
            removed_outlier_labels = labels[np.logical_not(keep_mask)]
        else:
            keep_mask = np.ones(len(raw_features), dtype=bool)
            removed_outlier_labels = np.array([], dtype=labels.dtype)

        original_count = len(raw_features)
        features = raw_features[keep_mask]
        labels = labels[keep_mask]
        removed_by_outlier = original_count - len(features)
        after_outlier = len(features)

        if self.normalization_mode == "percentile":
            if self.percentile_range is None:
                raise ValueError("percentile_range must be provided when normalization_mode='percentile'")
            p_low, p_high = self.percentile_range
            low_val = np.nanpercentile(features, p_low)
            high_val = np.nanpercentile(features, p_high)
            if not np.isfinite(low_val) or not np.isfinite(high_val) or low_val == high_val:
                raise ValueError(f"Invalid percentile bounds for {csv_path}")
            if self.percentile_clip:
                features = np.clip(features, low_val, high_val)
            denom = high_val - low_val
            if denom == 0:
                denom = 1.0
            features = (features - low_val) / denom
        elif self.normalization_mode == "zscore":
            mean_val = np.nanmean(features)
            std_val = np.nanstd(features)
            if not np.isfinite(std_val) or std_val == 0:
                std_val = 1.0
            features = (features - mean_val) / std_val

        global_mean = np.nanmean(features)
        global_std = np.nanstd(features)
        if not np.isfinite(global_std) or global_std == 0:
            raise ValueError(f"Unable to compute std for {csv_path}")

        # Outlier rejection
        if np.isfinite(self.outlier_sigma_mult):
            outlier_thresh = self.outlier_sigma_mult * global_std
            deviations = np.max(np.abs(features - global_mean), axis=1)
            keep_mask = deviations < outlier_thresh
            removed_outlier_labels = labels[np.logical_not(keep_mask)]
        else:
            keep_mask = np.ones(len(features), dtype=bool)
            removed_outlier_labels = np.array([], dtype=labels.dtype)

        original_count = len(features)
        features = features[keep_mask]
        labels = labels[keep_mask]
        removed_by_outlier = original_count - len(features)
        after_outlier = len(features)

        # Activity filter (skip resting labels)
        removed_activity_labels = []

        if self.activity_sigma_mult > 0:
            activity_thresh = self.activity_sigma_mult * global_std
            filtered_features = []
            filtered_labels = []
            for row, label in zip(features, labels):
                label_key = label.lower()
                if label_key in self.resting_labels:
                    filtered_features.append(row)
                    filtered_labels.append(label)
                    continue
                if np.max(np.abs(row - global_mean)) >= activity_thresh:
                    filtered_features.append(row)
                    filtered_labels.append(label)
                else:
                    removed_activity_labels.append(label_key)
            features = np.array(filtered_features, dtype=np.float32)
            labels = np.array(filtered_labels)
        after_activity = len(features)

        if len(features) == 0:
            stats = self._compose_stats(
                original=original_count,
                after_outlier=after_outlier,
                after_activity=after_activity,
                after_label_filter=0,
                removed_outlier_labels=removed_outlier_labels,
                removed_activity_labels=removed_activity_labels,
                removed_label_filter_labels=[],
            )
            return (
                np.empty((0, features.shape[1] if features.ndim == 2 else 0), dtype=np.float32),
                np.empty((0,)),
                stats,
            )

        if self.include_labels:
            include_mask = np.array([lbl.lower() in self.include_labels for lbl in labels])
            removed_label_filter_labels = labels[np.logical_not(include_mask)]
            features = features[include_mask]
            labels = labels[include_mask]
            if len(features) == 0:
                stats = self._compose_stats(
                    original=original_count,
                    after_outlier=after_outlier,
                    after_activity=after_activity,
                    after_label_filter=0,
                    removed_outlier_labels=removed_outlier_labels,
                    removed_activity_labels=removed_activity_labels,
                    removed_label_filter_labels=removed_label_filter_labels,
                )
                return (
                    np.empty((0, features.shape[1]), dtype=np.float32),
                    np.empty((0,)),
                    stats,
                )
        else:
            removed_label_filter_labels = []

        stats = self._compose_stats(
            original=original_count,
            after_outlier=after_outlier,
            after_activity=after_activity,
            after_label_filter=len(features),
            removed_outlier_labels=removed_outlier_labels,
            removed_activity_labels=removed_activity_labels,
            removed_label_filter_labels=removed_label_filter_labels,
        )

        labels = labels.astype(str)
        encoded = self.label_encoder.transform(labels)
        return features.astype(np.float32), encoded.astype(np.int64), stats

    def _add_windows(self, file_idx: int) -> None:
        signals = self.file_arrays[file_idx]  # shape: [num_signals, num_timepoints]
        num_signals, num_steps = signals.shape
        if num_steps < self.window_size:
            return
        for signal_idx in range(num_signals):
            for start in range(0, num_steps - self.window_size + 1, self.stride):
                end = start + self.window_size
                self.windows.append((file_idx, signal_idx, start, end))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, signal_idx, start, end = self.windows[idx]
        window = self.file_arrays[file_idx][signal_idx, start:end]
        label_idx = self.file_labels[file_idx][signal_idx]
        signal = torch.from_numpy(window).to(dtype=self.dtype)
        label = torch.tensor(label_idx, dtype=torch.long)
        return signal, label

    # ------------------------------------------------------------------
    def _compose_stats(
        self,
        original: int,
        after_outlier: int,
        after_activity: int,
        after_label_filter: int,
        removed_outlier_labels,
        removed_activity_labels,
        removed_label_filter_labels,
    ) -> dict:
        def count_labels(iterable):
            counter = defaultdict(int)
            for lbl in iterable:
                counter[str(lbl).lower()] += 1
            return dict(counter)

        return {
            "original": original,
            "after_outlier": after_outlier,
            "after_activity": after_activity,
            "after_label_filter": after_label_filter,
            "removed_outlier": count_labels(removed_outlier_labels),
            "removed_activity": count_labels(removed_activity_labels),
            "removed_label_filter": count_labels(removed_label_filter_labels),
        }

    def _accumulate_stats(self, stats: dict) -> None:
        self.filter_stats["original"] += stats["original"]
        self.filter_stats["after_outlier"] += stats["after_outlier"]
        self.filter_stats["after_activity"] += stats["after_activity"]
        self.filter_stats["after_label_filter"] += stats["after_label_filter"]
        for key in ("removed_outlier", "removed_activity", "removed_label_filter"):
            for label, count in stats[key].items():
                self.filter_stats[key][label] += count

    def _print_filter_summary(self) -> None:
        total_removed_outlier = sum(self.filter_stats["removed_outlier"].values())
        total_removed_activity = sum(self.filter_stats["removed_activity"].values())
        total_removed_label = sum(self.filter_stats["removed_label_filter"].values())

        print("Filter summary:")
        print(
            f"  Signals: {self.filter_stats['original']} -> {self.filter_stats['after_label_filter']} "
            "after filters"
        )
        print(f"  Removed by outlier threshold: {total_removed_outlier}")
        for label, count in sorted(self.filter_stats["removed_outlier"].items()):
            print(f"    {label}: {count}")
        print(f"  Removed by activity threshold: {total_removed_activity}")
        for label, count in sorted(self.filter_stats["removed_activity"].items()):
            print(f"    {label}: {count}")
        print(f"  Removed by include_labels filter: {total_removed_label}")
        for label, count in sorted(self.filter_stats["removed_label_filter"].items()):
            print(f"    {label}: {count}")


__all__ = ["NeuralBehaviorDataset"]
