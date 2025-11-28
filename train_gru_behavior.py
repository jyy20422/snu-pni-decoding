#!/usr/bin/env python3
"""Train a GRU classifier on NeuralBehaviorDataset with hyperparameter sweeps."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from neural_behavior_dataset import NeuralBehaviorDataset


# --------------------------------------------------------------------------------------
# Utility structures
# --------------------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    data_dir: Path
    cache_dir: Path
    trials: Optional[List[str]] = None
    window_size: int = 1000
    stride: int = 500
    normalization_mode: str = "percentile"  # 'none', 'zscore', 'percentile'
    percentile_range: Optional[Tuple[float, float]] = (1.0, 99.0)
    percentile_clip: bool = False
    activity_sigma_mult: float = 0.0
    outlier_sigma_mult: float = 10.0
    resting_labels: Sequence[str] = field(
        default_factory=lambda: ["resting", "standing"]
    )
    include_labels: Sequence[str] = field(
        default_factory=lambda: ["resting", "walking", "climbing", "standing", "grooming"]
    )


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_ratio: float = 0.8
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    model_type: str = "gru"  # 'gru' or 'transformer'
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_dim_feedforward: int = 256
    transformer_num_layers: int = 4
    transformer_activation: str = "gelu"
    transformer_dropout: float = 0.1
    transformer_pooling: str = "cls"  # 'cls' or 'mean'
    log_interval: int = 50
    eval_interval: int = 1
    checkpoint_interval: int = 1
    num_workers: int = 4
    seed: int = 42

    # Imbalance handling
    use_class_weights: bool = False
    use_weighted_sampler: bool = False


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class GRUBehaviorModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerBehaviorModel(nn.Module):
    """Transformer encoder classifier for behavioral decoding."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        pooling: str,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, d_model))
            if pooling == "cls"
            else None
        )
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(x)
        if self.pooling == "cls":
            cls_token = self.cls_token.expand(features.size(0), -1, -1)
            features = torch.cat([cls_token, features], dim=1)
        encoded = self.pos_encoder(features)
        encoded = self.encoder(encoded)
        if self.pooling == "cls":
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled)


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def build_model(train_cfg: TrainingConfig, num_classes: int) -> nn.Module:
    if train_cfg.model_type == "gru":
        return GRUBehaviorModel(
            input_size=1,
            hidden_size=train_cfg.hidden_size,
            num_layers=train_cfg.num_layers,
            dropout=train_cfg.dropout,
            num_classes=num_classes,
        )
    if train_cfg.model_type == "transformer":
        return TransformerBehaviorModel(
            input_size=1,
            d_model=train_cfg.transformer_d_model,
            nhead=train_cfg.transformer_nhead,
            num_layers=train_cfg.transformer_num_layers,
            dim_feedforward=train_cfg.transformer_dim_feedforward,
            dropout=train_cfg.transformer_dropout,
            activation=train_cfg.transformer_activation,
            pooling=train_cfg.transformer_pooling,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown model_type: {train_cfg.model_type}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(
    fig_cm: np.ndarray,
    labels: Sequence[str],
    title: str = "Confusion Matrix",
    value_format: str = "d",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(fig_cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(fig_cm.shape[1]),
        yticks=np.arange(fig_cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = fig_cm.max() / 2.0 if fig_cm.size > 0 else 0.0
    for i in range(fig_cm.shape[0]):
        for j in range(fig_cm.shape[1]):
            ax.text(
                j,
                i,
                format(fig_cm[i, j], value_format),
                ha="center",
                va="center",
                color="white" if fig_cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return fig


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    preds: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device).unsqueeze(-1)  # (B, T, 1)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            preds.append(logits.argmax(dim=1).cpu())
            targets_list.append(targets.cpu())
    preds_cat = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    targets_cat = torch.cat(targets_list) if targets_list else torch.empty(0, dtype=torch.long)
    preds_np = preds_cat.numpy()
    targets_np = targets_cat.numpy()
    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else float("nan")
    acc = accuracy_score(targets_np, preds_np) if targets_np.size > 0 else float("nan")
    cm = confusion_matrix(
        targets_np,
        preds_np,
        labels=np.arange(len(np.unique(targets_np))) if targets_np.size > 0 else None,
    )
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_true": targets_np,
        "y_pred": preds_np,
    }


def prepare_dataloaders(
    dataset: NeuralBehaviorDataset,
    train_ratio: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, Sequence[int]]:
    total_len = len(dataset)
    train_len = max(1, int(total_len * train_ratio))
    test_len = max(1, total_len - train_len)
    if train_len + test_len > total_len:
        train_len = total_len - test_len
    generator = torch.Generator().manual_seed(seed)

    train_ds, test_ds = random_split(
        dataset,
        [train_len, total_len - train_len],
        generator=generator,
    )

    # Indices into the original dataset for the training subset
    train_indices: Sequence[int] = train_ds.indices

    sampler = None
    if use_weighted_sampler:
        # Compute inverse-frequency weights per sample for the training subset
        labels: List[int] = []
        for idx in train_indices:
            _, target = dataset[idx]  # target is a tensor
            labels.append(int(target))

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        class_counts = torch.bincount(labels_tensor)
        class_weights = 1.0 / class_counts.float().clamp(min=1.0)
        sample_weights = class_weights[labels_tensor]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not use_weighted_sampler,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_indices


# --------------------------------------------------------------------------------------
# Training routine
# --------------------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def _serialize_config(obj: dict) -> dict:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (set, tuple)):
            return list(value)
        return value

    return {k: convert(v) for k, v in obj.items()}


def run_single_training(
    run_name: str,
    dataset_cfg: DatasetConfig,
    train_cfg: TrainingConfig,
    log_root: Path,
) -> Dict[str, Any]:
    set_seed(train_cfg.seed)
    start_ts = int(time.time())
    unique_run_name = f"{run_name}_{start_ts}"
    run_dir = log_root / unique_run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Per-run log dir (no HParams plugin nonsense)
    writer = SummaryWriter(log_dir=str(run_dir))

    dataset = NeuralBehaviorDataset(
        data_dir=dataset_cfg.data_dir,
        window_size=dataset_cfg.window_size,
        stride=dataset_cfg.stride,
        trials=dataset_cfg.trials,
        cache_dir=dataset_cfg.cache_dir,
        normalization_mode=dataset_cfg.normalization_mode,
        percentile_range=dataset_cfg.percentile_range,
        percentile_clip=dataset_cfg.percentile_clip,
        activity_sigma_mult=dataset_cfg.activity_sigma_mult,
        outlier_sigma_mult=dataset_cfg.outlier_sigma_mult,
        resting_labels=dataset_cfg.resting_labels,
        include_labels=dataset_cfg.include_labels,
        use_cache=True,
    )

    num_classes = len(dataset.label_encoder.classes_)
    label_names = list(dataset.label_encoder.classes_)

    print("Preparing dataloaders ...")
    train_loader, test_loader, train_indices = prepare_dataloaders(
        dataset,
        train_cfg.train_ratio,
        train_cfg.batch_size,
        train_cfg.num_workers,
        train_cfg.seed,
        use_weighted_sampler=train_cfg.use_weighted_sampler,
    )

    serialized_dataset = _serialize_config(asdict(dataset_cfg))
    serialized_training = _serialize_config(asdict(train_cfg))

    # Log configs as text
    writer.add_text("config/dataset", json.dumps(serialized_dataset, indent=2), global_step=0)
    writer.add_text("config/training", json.dumps(serialized_training, indent=2), global_step=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(train_cfg, num_classes).to(device)

    # Optionally compute class weights on the training subset
    class_weights_tensor: Optional[torch.Tensor] = None
    if train_cfg.use_class_weights:
        labels: List[int] = []
        for idx in train_indices:
            _, target = dataset[idx]
            labels.append(int(target))

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        class_counts = torch.bincount(labels_tensor, minlength=num_classes)
        class_weights = 1.0 / class_counts.float().clamp(min=1.0)

        print("Class counts (train):", class_counts.tolist())
        print("Class weights (inverse freq):", class_weights.tolist())
        class_weights_tensor = class_weights

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor.to(device) if class_weights_tensor is not None else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    global_step = 0
    best_acc = -1.0
    best_metrics: Dict[str, Any] = {}

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_examples = 0

        # For train confusion matrix (use predictions during training)
        train_preds_epoch: List[torch.Tensor] = []
        train_targets_epoch: List[torch.Tensor] = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress, start=1):
            inputs = inputs.to(device).unsqueeze(-1)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds_batch = logits.argmax(dim=1)

            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds_batch == targets).sum().item()
            total_examples += inputs.size(0)
            global_step += 1

            # Store predictions and labels for confusion matrix
            train_preds_epoch.append(preds_batch.detach().cpu())
            train_targets_epoch.append(targets.detach().cpu())

            if batch_idx % train_cfg.log_interval == 0:
                train_loss = running_loss / total_examples
                train_acc = running_correct / total_examples
                writer.add_scalar("Loss/train", train_loss, global_step)
                writer.add_scalar("Accuracy/train", train_acc, global_step)

        if epoch % train_cfg.eval_interval == 0 or epoch == train_cfg.epochs:
            # Train metrics for this epoch
            train_metrics = {
                "loss": running_loss / total_examples if total_examples > 0 else float("nan"),
                "accuracy": running_correct / total_examples if total_examples > 0 else float("nan"),
            }

            # Train confusion matrix using epoch predictions
            if train_preds_epoch and train_targets_epoch:
                train_preds_all = torch.cat(train_preds_epoch).numpy()
                train_targets_all = torch.cat(train_targets_epoch).numpy()
                cm_train_raw = confusion_matrix(
                    train_targets_all,
                    train_preds_all,
                    labels=np.arange(num_classes),
                )
                row_sums_train = cm_train_raw.sum(axis=1, keepdims=True)
                cm_train_norm = np.divide(
                    cm_train_raw.astype(float),
                    row_sums_train,
                    where=row_sums_train != 0,
                )
                cm_fig_train_raw = plot_confusion_matrix(
                    cm_train_raw,
                    label_names,
                    title="Train Confusion Matrix (Counts)",
                    value_format="d",
                )
                cm_fig_train_norm = plot_confusion_matrix(
                    cm_train_norm,
                    label_names,
                    title="Train Confusion Matrix (Normalized)",
                    value_format=".2f",
                )
                writer.add_figure("Raw Confusion Matrix/Raw TRAIN Confusion Matrix", cm_fig_train_raw, epoch)
                writer.add_figure("Normalize Confusion Matrix/Normalized TRAIN Confusion Matrix", cm_fig_train_norm, epoch)
                cm_fig_train_raw.savefig(run_dir / f"confusion_matrix_train_counts_epoch{epoch}.png", dpi=160)
                cm_fig_train_norm.savefig(run_dir / f"confusion_matrix_train_norm_epoch{epoch}.png", dpi=160)
                plt.close(cm_fig_train_raw)
                plt.close(cm_fig_train_norm)

            # Test metrics & confusion matrix
            test_metrics = evaluate(model, test_loader, criterion, device)
            writer.add_scalar("Loss/train_epoch", train_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train_epoch", train_metrics["accuracy"], epoch)
            writer.add_scalar("Loss/test", test_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/test", test_metrics["accuracy"], epoch)

            cm_raw = test_metrics["confusion_matrix"]
            if cm_raw.size > 0:
                row_sums = cm_raw.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm_raw.astype(float), row_sums, where=row_sums != 0)
                cm_fig_raw = plot_confusion_matrix(
                    cm_raw, label_names, title="Test Confusion Matrix (Counts)", value_format="d"
                )
                cm_fig_norm = plot_confusion_matrix(
                    cm_norm, label_names, title="Test Confusion Matrix (Normalized)", value_format=".2f"
                )
                writer.add_figure("Raw Confusion Matrix/Raw TEST Confusion Matrix", cm_fig_raw, epoch)
                writer.add_figure("Normalize Confusion Matrix/Normalized TEST Confusion Matrix", cm_fig_norm, epoch)
                cm_fig_raw.savefig(run_dir / f"confusion_matrix_test_counts_epoch{epoch}.png", dpi=160)
                cm_fig_norm.savefig(run_dir / f"confusion_matrix_test_norm_epoch{epoch}.png", dpi=160)
                plt.close(cm_fig_raw)
                plt.close(cm_fig_norm)

            if test_metrics["accuracy"] is not None and test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_metrics = {
                    "epoch": epoch,
                    "test_acc": test_metrics["accuracy"],
                    "test_loss": test_metrics["loss"],
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy"],
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "train_cfg": asdict(train_cfg),
                        "dataset_cfg": asdict(dataset_cfg),
                    },
                    checkpoint_dir / f"best_epoch_{epoch}.pt",
                )

        if epoch % train_cfg.checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_dir / f"epoch_{epoch}.pt",
            )

    # Ensure defaults if something went weird
    best_metrics.setdefault("epoch", train_cfg.epochs)
    best_metrics.setdefault("test_acc", best_acc)
    best_metrics.setdefault("test_loss", float("nan"))
    best_metrics.setdefault("train_loss", float("nan"))
    best_metrics.setdefault("train_acc", float("nan"))

    writer.flush()
    writer.close()

    return {
        "run_name": unique_run_name,
        "best_epoch": best_metrics["epoch"],
        "best_test_acc": best_metrics["test_acc"],
        "best_test_loss": best_metrics["test_loss"],
        "train_loss_at_best": best_metrics["train_loss"],
        "train_acc_at_best": best_metrics["train_acc"],
        "config": {
            "dataset": serialized_dataset,
            "training": serialized_training,
        },
    }


# --------------------------------------------------------------------------------------
# CLI and sweep handling
# --------------------------------------------------------------------------------------


def load_sweep_config(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Sweep configuration must be a list of configs.")
    return data


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    percentile_range = tuple(args.percentile_range) if args.percentile_range else None
    dataset_cfg = DatasetConfig(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        trials=args.trials,
        window_size=args.window_size,
        stride=args.stride,
        normalization_mode=args.normalization_mode,
        percentile_range=percentile_range,
        percentile_clip=args.percentile_clip,
        activity_sigma_mult=args.activity_sigma_mult,
        outlier_sigma_mult=args.outlier_sigma_mult,
        resting_labels=args.resting_labels,
        include_labels=args.include_labels,
    )
    training_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_ratio=args.train_ratio,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_num_layers=args.transformer_num_layers,
        transformer_activation=args.transformer_activation,
        transformer_dropout=args.transformer_dropout,
        transformer_pooling=args.transformer_pooling,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        num_workers=args.num_workers,
        seed=args.seed,
        use_class_weights=args.use_class_weights,
        use_weighted_sampler=args.use_weighted_sampler,
    )
    return {
        "name": args.run_name or "single_run",
        "dataset": asdict(dataset_cfg),
        "training": asdict(training_cfg),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/neural_behavior"))
    parser.add_argument("--trials", nargs="*", default=None)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument(
        "--normalization-mode",
        choices=["none", "zscore", "percentile"],
        default="percentile",
    )
    parser.add_argument("--percentile-range", type=float, nargs=2)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--model-type",
        choices=["gru", "transformer"],
        default="gru",
        help="Model architecture to train.",
    )
    parser.add_argument("--transformer-d-model", type=int, default=128)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=256)
    parser.add_argument("--transformer-num-layers", type=int, default=4)
    parser.add_argument(
        "--transformer-activation",
        choices=["relu", "gelu"],
        default="gelu",
    )
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--transformer-pooling",
        choices=["cls", "mean"],
        default="cls",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/gru_behavior"))
    parser.add_argument("--sweep-config", type=Path)
    parser.add_argument("--run-name", type=str)

    # Imbalance-handling CLI flags
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropyLoss to handle class imbalance.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Use WeightedRandomSampler to draw class-balanced mini-batches.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_config:
        sweep = load_sweep_config(args.sweep_config)
        configs = sweep
    else:
        configs = [build_config_from_args(args)]

    results = []
    timestamp = int(time.time())
    for run_idx, cfg in enumerate(configs, start=1):
        dataset_cfg = DatasetConfig(**cfg["dataset"])
        training_cfg = TrainingConfig(**cfg["training"])
        run_name = cfg.get("name", f"run_{timestamp}_{run_idx}")
        print(f"\n=== Starting {run_name} ===")
        metrics = run_single_training(run_name, dataset_cfg, training_cfg, args.log_dir)
        results.append(metrics)

    df = pd.DataFrame(results)
    results_path = args.log_dir / f"sweep_results_{timestamp}.csv"
    df.to_csv(results_path, index=False)
    df.to_excel(results_path.with_suffix(".xlsx"), index=False)
    print(f"\nSweep complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
