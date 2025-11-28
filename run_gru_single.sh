#!/usr/bin/env bash
set -euo pipefail

###############################################
# IMBALANCE MODE — SET IT HERE
###############################################
# Options: "none", "class_weights", "sampler", "both"
# IMBALANCE_MODE="none"
# IMBALANCE_MODE="class_weights"
IMBALANCE_MODE="sampler"
# IMBALANCE_MODE="both"
###############################################

# Default parameters for a single training run (edit as needed)
DATA_DIR="/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116"
CACHE_DIR="/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116/cache"
TRIALS=("E003_5W_Trial12_labelled_raw_data_.csv" "E003_5W_Trial11_labelled_raw_data_.csv" "E003_5W_Trial10_labelled_raw_data_.csv")
WINDOW_SIZE=2441
STRIDE=2441
NORMALIZATION_MODE="none"   # 'none', 'zscore', or 'percentile'
PERCENTILE_RANGE_LOW=1
PERCENTILE_RANGE_HIGH=99
PERCENTILE_CLIP=""
ACTIVITY_SIGMA_MULT=0.0
OUTLIER_SIGMA_MULT="inf"
RESTING_LABELS=()
INCLUDE_LABELS=("walking" "climbing" "standing" "grooming")

EPOCHS=999
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=1e-4
TRAIN_RATIO=0.8
HIDDEN_SIZE=32
NUM_LAYERS=4
DROPOUT=0.0
MODEL_TYPE="gru" # 'gru' or 'transformer'
TRANSFORMER_D_MODEL=16
TRANSFORMER_NHEAD=2
TRANSFORMER_DIM_FEEDFORWARD=256
TRANSFORMER_NUM_LAYERS=1
TRANSFORMER_ACTIVATION="gelu"
TRANSFORMER_DROPOUT=0.1
TRANSFORMER_POOLING="cls" # 'cls' or 'mean'
LOG_INTERVAL=50
EVAL_INTERVAL=1
CHECKPOINT_INTERVAL=1
NUM_WORKERS=4
SEED=42

LOG_DIR="/media/NAS_179_2_josh_2/snu-pni-decoding/results_same_week"
RUN_NAME="251126_gru_same_week_run"

TRIAL_ARGS=()
if [ ${#TRIALS[@]} -gt 0 ]; then
  TRIAL_ARGS=(--trials "${TRIALS[@]}")
fi

RESTING_ARGS=()
if [ ${#RESTING_LABELS[@]} -gt 0 ]; then
  RESTING_ARGS=(--resting-labels "${RESTING_LABELS[@]}")
fi

PERCENTILE_ARGS=()
if [ "${NORMALIZATION_MODE}" = "percentile" ]; then
  PERCENTILE_ARGS=(--percentile-range "${PERCENTILE_RANGE_LOW}" "${PERCENTILE_RANGE_HIGH}")
  if [ -n "${PERCENTILE_CLIP}" ]; then
    PERCENTILE_ARGS+=("${PERCENTILE_CLIP}")
  fi
fi

###############################################
# MAP IMBALANCE_MODE TO FLAGS
###############################################
IMBALANCE_ARGS=()

case "${IMBALANCE_MODE}" in
  none)
    echo "[INFO] Using NO imbalance handling"
    ;;
  class_weights)
    echo "[INFO] Using CLASS WEIGHTS"
    IMBALANCE_ARGS=(--use-class-weights)
    ;;
  sampler)
    echo "[INFO] Using WEIGHTED SAMPLER"
    IMBALANCE_ARGS=(--use-weighted-sampler)
    ;;
  both)
    echo "[INFO] Using CLASS WEIGHTS + WEIGHTED SAMPLER"
    IMBALANCE_ARGS=(--use-class-weights --use-weighted-sampler)
    ;;
  *)
    echo "ERROR: Unknown IMBALANCE_MODE: ${IMBALANCE_MODE}"
    exit 1
    ;;
esac

###############################################
# RUN TRAINING
###############################################
python train_gru_behavior.py \
  --data-dir "${DATA_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  "${TRIAL_ARGS[@]}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --normalization-mode "${NORMALIZATION_MODE}" \
  "${PERCENTILE_ARGS[@]}" \
  --activity-sigma-mult "${ACTIVITY_SIGMA_MULT}" \
  --outlier-sigma-mult "${OUTLIER_SIGMA_MULT}" \
  "${RESTING_ARGS[@]}" \
  --include-labels "${INCLUDE_LABELS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --train-ratio "${TRAIN_RATIO}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --num-layers "${NUM_LAYERS}" \
  --dropout "${DROPOUT}" \
  --model-type "${MODEL_TYPE}" \
  --transformer-d-model "${TRANSFORMER_D_MODEL}" \
  --transformer-nhead "${TRANSFORMER_NHEAD}" \
  --transformer-dim-feedforward "${TRANSFORMER_DIM_FEEDFORWARD}" \
  --transformer-num-layers "${TRANSFORMER_NUM_LAYERS}" \
  --transformer-activation "${TRANSFORMER_ACTIVATION}" \
  --transformer-dropout "${TRANSFORMER_DROPOUT}" \
  --transformer-pooling "${TRANSFORMER_POOLING}" \
  --log-interval "${LOG_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --log-dir "${LOG_DIR}" \
  --run-name "${RUN_NAME}" \
  "${IMBALANCE_ARGS[@]}"
