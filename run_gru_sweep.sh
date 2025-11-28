#!/usr/bin/env bash
set -euo pipefail

# Default arguments (edit as needed)
DATA_DIR="/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116"
CACHE_DIR="/media/NAS_179_2_josh_2/snu-pni-decoding/E003_12weeks_labelled_raw_data_251116/cache"
TRIALS=("E003_10W_Trial1_labelled_raw_data_.csv")
WINDOW_SIZE=1000
STRIDE=1000
NORMALIZATION_MODE="percentile"
PERCENTILE_RANGE_LOW=1
PERCENTILE_RANGE_HIGH=99
ACTIVITY_SIGMA_MULT=0.0
OUTLIER_SIGMA_MULT=10.0
RESTING_LABELS=("resting")
INCLUDE_LABELS=("resting" "walking" "climbing" "standing" "grooming")
EPOCHS=20
WEIGHT_DECAY=1e-4
TRAIN_RATIO=0.8
LOG_INTERVAL=50
EVAL_INTERVAL=1
CHECKPOINT_INTERVAL=1
NUM_WORKERS=4
SEED=42
LR_LIST=("1e-3" "5e-4")
BATCH_LIST=(32 64)
HIDDEN_LIST=(128 256)
LAYER_LIST=(1 2)
DROPOUT_LIST=(0.1 0.3)
LOG_DIR="runs/gru_behavior"
RUN_PREFIX="sweep"
SWEEP_FILE="configs/gru_sweep.json"
TRAIN_SCRIPT="train_gru_behavior.py"

# Build arguments
python run_gru_sweep.py \
  --data-dir "${DATA_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --trials "${TRIALS[@]}" \
  --window-size "${WINDOW_SIZE}" \
  --stride "${STRIDE}" \
  --normalization-mode "${NORMALIZATION_MODE}" \
  --percentile-range "${PERCENTILE_RANGE_LOW}" "${PERCENTILE_RANGE_HIGH}" \
  --activity-sigma-mult "${ACTIVITY_SIGMA_MULT}" \
  --outlier-sigma-mult "${OUTLIER_SIGMA_MULT}" \
  --resting-labels "${RESTING_LABELS[@]}" \
  --include-labels "${INCLUDE_LABELS[@]}" \
  --epochs "${EPOCHS}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --train-ratio "${TRAIN_RATIO}" \
  --log-interval "${LOG_INTERVAL}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --lr-list "${LR_LIST[@]}" \
  --batch-list "${BATCH_LIST[@]}" \
  --hidden-list "${HIDDEN_LIST[@]}" \
  --layer-list "${LAYER_LIST[@]}" \
  --dropout-list "${DROPOUT_LIST[@]}" \
  --log-dir "${LOG_DIR}" \
  --run-prefix "${RUN_PREFIX}" \
  --sweep-file "${SWEEP_FILE}" \
  --train-script "${TRAIN_SCRIPT}"

