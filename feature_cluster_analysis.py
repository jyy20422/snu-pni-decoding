#!/usr/bin/env python3
"""Feature-based clustering and classification diagnostics for E003 dataset.

This script loads all labelled CSV recordings, extracts a rich set of temporal,
spectral, wavelet, and autocorrelation features, and then:
  1. Clusters the feature vectors (KMeans) to see whether label structure
     emerges from the handcrafted descriptors.
  2. Runs a lightweight classifier (logistic regression) to quantify
     separability when using:
         * all trials together,
         * only trials from the same recording week,
         * each trial independently.

In addition to the feature CSV, the script saves:
  - clustering metrics (JSON + table),
  - classification metrics per grouping,
  - optional PCA scatter plots comparing true labels vs cluster IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from scipy import signal as sp_signal
except ImportError:
    sp_signal = None

try:
    import pywt
except ImportError:
    pywt = None


METRIC_GUIDE = """\
Metric quick-reference (higher is better unless noted):
- Adjusted Rand Index (ARI): 1.0 = identical clustering, ~0 = random, <0 = worse than random.
- Normalized Mutual Information (NMI): 0–1 overlap between clusters and labels.
- Homogeneity/Completeness/V-measure: 0–1; homogeneity asks “clusters contain one label,” completeness asks
  “label members end up in one cluster,” and their harmonic mean is V.
- Silhouette: -1 to 1. Close to 1 means samples align with their cluster better than neighboring clusters.
- Accuracy: fraction of correctly classified samples (0–1).
- Macro F1: class-balanced harmonic mean of precision/recall (0–1); punishes ignoring rare behaviors.
"""


# --------------------------------------------------------------------------- #
# Feature extraction helpers
# --------------------------------------------------------------------------- #

def _ensure_array(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.ravel()
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, copy=False)
    return arr


def _zero_crossing_rate(arr: np.ndarray) -> float:
    signs = np.sign(arr)
    signs[signs == 0] = -1
    crossings = np.where(np.diff(signs))[0]
    return float(len(crossings) / max(len(arr) - 1, 1))


def _hjorth_params(arr: np.ndarray) -> Tuple[float, float, float]:
    first_deriv = np.diff(arr)
    second_deriv = np.diff(first_deriv)
    var0 = np.var(arr)
    var1 = np.var(first_deriv)
    var2 = np.var(second_deriv)
    activity = var0
    mobility = math.sqrt(var1 / var0) if var0 > 0 else 0.0
    complexity = math.sqrt(var2 / var1) / mobility if var1 > 0 and mobility > 0 else 0.0
    return float(activity), float(mobility), float(complexity)


def _rolling_stat(arr: np.ndarray, window: int, stat: str) -> float:
    if window <= 1 or len(arr) < window:
        return float("nan")
    strides = len(arr) - window + 1
    stacked = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
    if stat == "var":
        return float(np.mean(np.var(stacked, axis=1)))
    if stat == "energy":
        return float(np.mean(np.mean(stacked**2, axis=1)))
    return float("nan")


def _signal_envelope(arr: np.ndarray) -> np.ndarray:
    if sp_signal is not None:
        analytic = sp_signal.hilbert(arr)
        return np.abs(analytic)
    warnings.warn(
        "scipy is unavailable; using absolute-value envelope approximation.",
        RuntimeWarning,
    )
    return np.abs(arr)


def _frequency_features(arr: np.ndarray) -> Dict[str, float]:
    fft_vals = np.fft.rfft(arr)
    power = np.square(np.abs(fft_vals))
    power_sum = np.sum(power)
    if power_sum == 0:
        power_sum = 1.0
    power_norm = power / power_sum
    freqs = np.fft.rfftfreq(len(arr), d=1.0)

    # Relative band powers (normalized frequency bands)
    band_edges = [
        (0.0, 0.02, "ultra_low"),
        (0.02, 0.05, "very_low"),
        (0.05, 0.1, "low"),
        (0.1, 0.2, "mid"),
        (0.2, 0.35, "high"),
        (0.35, 0.5, "ultra_high"),
        (0.5, 1.0, "nyquist_plus"),
    ]
    feats = {}
    for low, high, name in band_edges:
        mask = (freqs >= low * freqs[-1]) & (freqs < high * freqs[-1])
        feats[f"bandpower_{name}"] = float(np.sum(power_norm[mask]))

    centroid = float(np.sum(freqs * power_norm))
    spectral_spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power_norm)))
    rolloff_threshold = 0.85
    cumulative = np.cumsum(power_norm)
    rolloff_idx = np.searchsorted(cumulative, rolloff_threshold)
    rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    sorted_idx = np.argsort(power)[::-1]
    dominant_freqs = freqs[sorted_idx[:3]]
    bandwidth = float(freqs[np.argmax(power)] - freqs[np.argmin(power)])
    flux = float(np.sqrt(np.sum(np.diff(power_norm, prepend=power_norm[0]) ** 2)))
    entropy = float(-np.sum(power_norm * np.log(power_norm + 1e-12)))

    feats.update(
        {
            "spectral_centroid": centroid,
            "spectral_spread": spectral_spread,
            "spectral_rolloff": rolloff_freq,
            "spectral_flux": flux,
            "spectral_entropy": entropy,
            "dominant_freq_1": float(dominant_freqs[0]) if len(dominant_freqs) > 0 else 0.0,
            "dominant_freq_2": float(dominant_freqs[1]) if len(dominant_freqs) > 1 else 0.0,
            "dominant_freq_3": float(dominant_freqs[2]) if len(dominant_freqs) > 2 else 0.0,
            "spectrum_bandwidth": bandwidth,
        }
    )
    return feats


def _wavelet_features(arr: np.ndarray) -> Dict[str, float]:
    if pywt is None:
        warnings.warn("pywt not available; skipping wavelet packet features.", RuntimeWarning)
        return {}
    max_level = 3
    wp = pywt.WaveletPacket(arr, wavelet="db4", mode="symmetric", maxlevel=max_level)
    feats = {}
    for level in range(1, max_level + 1):
        nodes = wp.get_level(level, order="freq")
        energies = np.array([np.sum(node.data**2) for node in nodes], dtype=np.float32)
        total = np.sum(energies)
        if total == 0:
            total = 1.0
        normalized = energies / total
        feats[f"wavelet_L{level}_energy_mean"] = float(np.mean(normalized))
        feats[f"wavelet_L{level}_energy_std"] = float(np.std(normalized))
        feats[f"wavelet_L{level}_entropy"] = float(-np.sum(normalized * np.log(normalized + 1e-12)))
    return feats


def _autocorr_features(arr: np.ndarray) -> Dict[str, float]:
    arr_centered = arr - np.mean(arr)
    acf_full = np.correlate(arr_centered, arr_centered, mode="full")
    acf = acf_full[len(arr) - 1 :]
    if acf[0] == 0:
        return {"acf_lag1": 0.0, "acf_lag5": 0.0, "acf_energy": 0.0, "acf_first_peak_lag": 0.0}
    acf /= acf[0]
    lags = np.arange(len(acf))
    acf_energy = float(np.sum(acf**2) / len(acf))
    lag1 = float(acf[1]) if len(acf) > 1 else 0.0
    lag5 = float(acf[5]) if len(acf) > 5 else lag1
    # Find first significant peak beyond lag=1
    peak_lag = 0.0
    for idx in range(2, min(len(acf), 500)):
        if acf[idx] > acf[idx - 1] and acf[idx] > acf[idx + 1]:
            peak_lag = float(idx)
            break
    return {
        "acf_lag1": lag1,
        "acf_lag5": lag5,
        "acf_energy": acf_energy,
        "acf_first_peak_lag": peak_lag,
    }


def extract_features(signal: np.ndarray) -> Dict[str, float]:
    arr = _ensure_array(signal)
    feats: Dict[str, float] = {}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    rms = float(np.sqrt(np.mean(arr**2)))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    peak_to_peak = float(np.ptp(arr))
    skewness = float(np.mean(((arr - mean) / (std + 1e-12)) ** 3))
    kurtosis = float(np.mean(((arr - mean) / (std + 1e-12)) ** 4) - 3.0)
    zero_cross = _zero_crossing_rate(arr)
    envelope = _signal_envelope(arr)
    hjorth_activity, hjorth_mobility, hjorth_complexity = _hjorth_params(arr)
    teager = arr[1:-1] ** 2 - arr[:-2] * arr[2:] if len(arr) > 2 else np.array([0.0])
    line_length = float(np.sum(np.abs(np.diff(arr))))
    rolling_var_64 = _rolling_stat(arr, 64, "var")
    rolling_energy_128 = _rolling_stat(arr, 128, "energy")

    feats.update(
        {
            "mean": mean,
            "std": std,
            "var": float(std**2),
            "rms": rms,
            "median": float(np.median(arr)),
            "iqr": iqr,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "peak_to_peak": peak_to_peak,
            "zero_cross_rate": zero_cross,
            "envelope_mean": float(np.mean(envelope)),
            "envelope_std": float(np.std(envelope)),
            "envelope_max": float(np.max(envelope)),
            "rolling_var_64": rolling_var_64,
            "rolling_energy_128": rolling_energy_128,
            "hjorth_activity": hjorth_activity,
            "hjorth_mobility": hjorth_mobility,
            "hjorth_complexity": hjorth_complexity,
            "teager_mean": float(np.mean(np.abs(teager))),
            "line_length": line_length,
        }
    )

    # Multi-scale window energies
    for window in (32, 64, 128, 256):
        if len(arr) >= window * 2:
            trimmed = arr[: (len(arr) // window) * window]
            segments = trimmed.reshape(-1, window)
            energy = np.mean(segments**2, axis=1)
            feats[f"ms_window{window}_energy_mean"] = float(np.mean(energy))
            feats[f"ms_window{window}_energy_std"] = float(np.std(energy))

    feats.update(_frequency_features(arr))
    feats.update(_wavelet_features(arr))
    feats.update(_autocorr_features(arr))
    return feats


# --------------------------------------------------------------------------- #
# Data loading and metadata
# --------------------------------------------------------------------------- #

WEEK_TRIAL_PATTERN = re.compile(r"E\d+_(\d+W)_?[_-]?Trial(\d+)", re.IGNORECASE)


@dataclass
class SampleRecord:
    file_path: Path
    row_index: int
    label: str
    week: str
    trial: str
    signal: np.ndarray


def parse_week_trial(file_name: str) -> Tuple[str, str]:
    match = WEEK_TRIAL_PATTERN.search(file_name)
    if match:
        week, trial = match.groups()
        return week.upper(), f"Trial{trial}"
    return "unknown_week", Path(file_name).stem


def load_dataset(data_dir: Path, max_samples: Optional[int] = None) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    for csv_path in tqdm(csv_files, desc="Reading CSV files"):
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            warnings.warn(f"Skipping {csv_path}: expected feature columns + label.")
            continue
        features = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        labels = df.iloc[:, -1].astype(str).to_numpy()
        week, trial = parse_week_trial(csv_path.name)
        for idx, (row, label) in enumerate(zip(features, labels)):
            records.append(
                SampleRecord(
                    file_path=csv_path,
                    row_index=idx,
                    label=label,
                    week=week,
                    trial=trial,
                    signal=row,
                )
            )
    if max_samples and len(records) > max_samples:
        np.random.seed(0)
        indices = np.random.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in indices]
    return records


def records_to_feature_frame(records: Iterable[SampleRecord]) -> pd.DataFrame:
    rows = []
    meta = []
    for rec in tqdm(records, desc="Extracting handcrafted features"):
        feats = extract_features(rec.signal)
        rows.append(feats)
        meta.append(
            {
                "label": rec.label,
                "week": rec.week,
                "trial": rec.trial,
                "file": rec.file_path.stem,
                "row_index": rec.row_index,
            }
        )
    feature_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)
    combined = pd.concat([meta_df, feature_df], axis=1)
    return combined


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #

def _plot_pca_panels(
    feats_pca: np.ndarray,
    labels: pd.Series,
    cluster_ids: np.ndarray,
    group_name: str,
    output_dir: Path,
) -> None:
    label_values = labels.astype(str).to_numpy()
    unique_labels = pd.unique(labels.astype(str))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    label_cmap = plt.colormaps.get_cmap("tab20").resampled(max(2, len(unique_labels)))
    for idx, label_name in enumerate(unique_labels):
        mask = label_values == label_name
        axes[0].scatter(
            feats_pca[mask, 0],
            feats_pca[mask, 1],
            s=12,
            alpha=0.75,
            color=label_cmap(idx),
            label=f"{label_name} (n={mask.sum()})",
        )
    axes[0].set_title(f"{group_name}: Behaviors in PCA space")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, title="Behavior label")

    unique_clusters = np.unique(cluster_ids)
    cluster_cmap = plt.colormaps.get_cmap("viridis").resampled(max(2, len(unique_clusters)))
    for idx, cluster in enumerate(unique_clusters):
        mask = cluster_ids == cluster
        axes[1].scatter(
            feats_pca[mask, 0],
            feats_pca[mask, 1],
            s=12,
            alpha=0.75,
            color=cluster_cmap(idx),
            label=f"Cluster {cluster} (n={mask.sum()})",
        )
    axes[1].set_title(f"{group_name}: KMeans clusters")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, title="Cluster ID")

    plot_path = output_dir / f"{group_name}_pca.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _format_table_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _plot_metric_table(df: pd.DataFrame, columns: List[str], title: str, path: Path) -> None:
    display_df = df[columns]
    cell_text = [[_format_table_value(val) for val in row] for row in display_df.to_numpy()]
    fig_height = max(4, 0.35 * len(display_df))
    fig, ax = plt.subplots(figsize=(min(18, max(10, len(columns) * 2)), fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bar(df: pd.DataFrame, metric: str, title: str, xlabel: str, path: Path) -> None:
    if df.empty:
        return
    display_df = df.sort_values(metric, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(display_df))))
    bars = ax.barh(display_df["group"], display_df[metric], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for bar, value in zip(bars, display_df[metric]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_metric_tables_and_charts(cluster_df: pd.DataFrame, clf_df: pd.DataFrame, output_dir: Path) -> None:
    if cluster_df.empty or clf_df.empty:
        return
    top_cluster = cluster_df.sort_values("ARI", ascending=False).head(20)
    top_clf = clf_df.sort_values("accuracy_mean", ascending=False).head(20)
    _plot_metric_table(
        top_cluster,
        ["group", "n_samples", "n_labels", "ARI", "NMI", "silhouette"],
        "Top clustering metrics (sorted by ARI)",
        output_dir / "clustering_metrics_table.png",
    )
    _plot_metric_table(
        top_clf,
        ["group", "n_samples", "n_labels", "accuracy_mean", "macro_f1_mean"],
        "Top classification metrics (sorted by accuracy)",
        output_dir / "classification_metrics_table.png",
    )
    _plot_metric_bar(
        cluster_df,
        "ARI",
        "Adjusted Rand Index by group",
        "ARI (higher = closer to labels)",
        output_dir / "clustering_ari_bar.png",
    )
    _plot_metric_bar(
        clf_df,
        "accuracy_mean",
        "Cross-validated accuracy by group",
        "Accuracy",
        output_dir / "classification_accuracy_bar.png",
    )


def _write_metric_guide(output_dir: Path) -> None:
    guide_path = output_dir / "metric_guide.txt"
    guide_path.write_text(METRIC_GUIDE)


# --------------------------------------------------------------------------- #
# Analysis utilities
# --------------------------------------------------------------------------- #

def run_clustering(
    features: pd.DataFrame,
    labels: pd.Series,
    group_name: str,
    output_dir: Path,
    random_state: int,
) -> Dict[str, float]:
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(features)
    n_components = min(10, feats_scaled.shape[1], feats_scaled.shape[0])
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=random_state)
        feats_pca = pca.fit_transform(feats_scaled)
    else:
        feats_pca = feats_scaled
    n_clusters = len(np.unique(labels))
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_ids = model.fit_predict(feats_pca)

    ari = adjusted_rand_score(labels, cluster_ids)
    nmi = normalized_mutual_info_score(labels, cluster_ids)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels, cluster_ids)
    silhouette = (
        silhouette_score(feats_pca, cluster_ids) if len(features) > n_clusters and n_clusters > 1 else float("nan")
    )

    result = {
        "group": group_name,
        "n_samples": int(len(labels)),
        "n_labels": int(n_clusters),
        "ARI": float(ari),
        "NMI": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "silhouette": float(silhouette),
    }

    if feats_pca.shape[1] >= 2:
        _plot_pca_panels(feats_pca, labels, cluster_ids, group_name, output_dir)

    return result


def run_classification(
    features: pd.DataFrame,
    labels: pd.Series,
    group_name: str,
    random_state: int,
) -> Dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {
            "group": group_name,
            "n_samples": int(len(labels)),
            "n_labels": int(len(np.unique(labels))),
            "accuracy_mean": float("nan"),
            "accuracy_std": float("nan"),
            "macro_f1_mean": float("nan"),
            "macro_f1_std": float("nan"),
        }
    features_array = features.to_numpy(dtype=np.float32)
    label_array = labels.to_numpy()
    k = min(5, np.min(np.bincount(pd.factorize(labels)[0])))
    k = max(k, 2)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    accuracy_scores: List[float] = []
    macro_f1_scores: List[float] = []
    for train_idx, test_idx in cv.split(features_array, label_array):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features_array[train_idx])
        X_test = scaler.transform(features_array[test_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, label_array[train_idx])
        preds = clf.predict(X_test)
        accuracy_scores.append(accuracy_score(label_array[test_idx], preds))
        macro_f1_scores.append(f1_score(label_array[test_idx], preds, average="macro"))

    return {
        "group": group_name,
        "n_samples": int(len(labels)),
        "n_labels": int(len(np.unique(labels))),
        "accuracy_mean": float(np.mean(accuracy_scores)),
        "accuracy_std": float(np.std(accuracy_scores)),
        "macro_f1_mean": float(np.mean(macro_f1_scores)),
        "macro_f1_std": float(np.std(macro_f1_scores)),
    }


# --------------------------------------------------------------------------- #
# Main CLI
# --------------------------------------------------------------------------- #

def _save_dataframe_all_formats(df: pd.DataFrame, base_path: Path) -> None:
    """Persist dataframe as CSV and (best-effort) Parquet."""
    csv_path = base_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except (ImportError, ValueError, OSError) as exc:
        warnings.warn(
            f"Skipping Parquet export for {base_path.name} ({exc}). "
            "Install pyarrow or fastparquet for Parquet support.",
            RuntimeWarning,
        )


def analyze(
    data_dir: Path,
    output_dir: Path,
    max_samples: Optional[int],
    random_state: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_dataset(data_dir, max_samples=max_samples)
    combined_df = records_to_feature_frame(records)
    
    # Filter to only include specific behavior labels
    target_labels = {"grooming", "climbing", "resting", "walking"}
    # Normalize labels (case-insensitive, strip whitespace) for matching
    combined_df["label_normalized"] = combined_df["label"].astype(str).str.lower().str.strip()
    target_labels_normalized = {label.lower().strip() for label in target_labels}
    before_count = len(combined_df)
    combined_df = combined_df[combined_df["label_normalized"].isin(target_labels_normalized)].copy()
    after_count = len(combined_df)
    print(f"\nFiltered dataset: {before_count} -> {after_count} samples (keeping only: {', '.join(sorted(target_labels))})")
    if after_count == 0:
        raise ValueError(f"No samples found with target labels: {target_labels}")
    # Restore original label values (drop normalized column)
    combined_df = combined_df.drop(columns=["label_normalized"])
    
    feature_cols = [col for col in combined_df.columns if col not in {"label", "week", "trial", "file", "row_index"}]
    feature_df = combined_df[feature_cols]
    _save_dataframe_all_formats(combined_df, output_dir / "handcrafted_features")

    groupings = {"all_data": combined_df.index}

    # Group by week
    for week, group in combined_df.groupby("week"):
        if len(group["label"].unique()) < 2:
            continue
        groupings[f"week_{week}"] = group.index

    # Group by trial/file
    for trial, group in combined_df.groupby("file"):
        if len(group["label"].unique()) < 2:
            continue
        groupings[f"trial_{trial}"] = group.index

    cluster_results = []
    clf_results = []
    for group_name, idx in groupings.items():
        subset = combined_df.loc[idx]
        labels = subset["label"]
        features_subset = subset[feature_cols]
        if len(labels.unique()) < 2:
            continue
        print(f"\n=== Analyzing {group_name} ({len(labels)} samples, {labels.nunique()} labels) ===")
        cluster_result = run_clustering(features_subset, labels, group_name, output_dir, random_state)
        cluster_results.append(cluster_result)
        clf_result = run_classification(features_subset, labels, group_name, random_state)
        clf_results.append(clf_result)

    cluster_df = pd.DataFrame(cluster_results)
    cluster_df.to_csv(output_dir / "clustering_metrics.csv", index=False)
    cluster_df.to_json(output_dir / "clustering_metrics.json", orient="records", indent=2)
    clf_df = pd.DataFrame(clf_results)
    clf_df.to_csv(output_dir / "classification_metrics.csv", index=False)
    clf_df.to_json(output_dir / "classification_metrics.json", orient="records", indent=2)
    _plot_metric_tables_and_charts(cluster_df, clf_df, output_dir)
    _write_metric_guide(output_dir)

    print("\nClustering metrics summary:")
    print(cluster_df.sort_values("ARI", ascending=False).to_string(index=False))
    print("\nClassification metrics summary:")
    print(clf_df.sort_values("accuracy_mean", ascending=False).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features and cluster neural behavior signals.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116"),
        help="Directory containing labelled CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/feature_clusters"),
        help="Directory for saving features, plots, and metrics.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on total samples (useful for quick iterations).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible clustering/classification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.data_dir, args.output_dir, args.max_samples, args.random_state)


if __name__ == "__main__":
    main()

