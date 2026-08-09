#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/root/autodl-tmp/EdgeCloudRuntime/worktrees/adjust
PYTHON=/root/miniconda3/bin/python
DATASET="$WORKTREE/data/BoxCars116k_kaggle/BoxCars116k"
MODEL="$WORKTREE/models/shared/InternViT-6B-224px"
BASELINE="$WORKTREE/checkpoints/shared/boxcars_make_baseline/best.pth"
SOURCE_HEAD="$WORKTREE/outputs/internvit6b_boxcars_head_20260809"
HEAD_RUN="$WORKTREE/outputs/internvit6b_boxcars_noise_head_20260809"
TEACHER_CACHE="$WORKTREE/outputs/internvit6b_boxcars_teacher_cache_20260809.pth"
ADAPTER_DIR="$WORKTREE/checkpoints/internvit6b_cloud_teacher_adapter_20260809"
SUMMARY_DIR="$WORKTREE/outputs/internvit6b_cloud_teacher_pipeline_20260809"

cd "$WORKTREE"

# Phase 1: seconds/minutes only. Reuse the completed 6B feature cache and stop
# the whole pipeline unless one head preserves clean accuracy and fixes noise.
if [[ ! -f "$HEAD_RUN/selected_head.pt" || ! -f "$HEAD_RUN/metrics.json" ]]; then
  "$PYTHON" module_edge_perception/retrain_boxcars_cloud_teacher_head_from_cache.py \
    --train-cache "$SOURCE_HEAD/train_clean_drift_features.pt" \
    --val-cache-dir "$SOURCE_HEAD" \
    --output-dir "$HEAD_RUN" \
    --extraction-batch-size 4 \
    --noise-repeat 3 \
    --epochs 120 \
    --min-clean 0.928 \
    --min-noise 0.795
fi

# Phase 2: formal task logits/features from the already extracted random-
# severity train cache. No second 6B forward and no validation-severity leakage.
"$PYTHON" module_edge_perception/build_boxcars_teacher_cache_from_feature_cache.py \
  --feature-cache "$SOURCE_HEAD/train_clean_drift_features.pt" \
  --head-checkpoint "$HEAD_RUN/selected_head.pt" \
  --output "$TEACHER_CACHE" \
  --extraction-batch-size 4 \
  --severity-min 0.3 \
  --severity-max 1.0

# Phase 3a: exact-data label-only control.
"$PYTHON" module_edge_perception/train_boxcars_cloud_teacher_adapter.py \
  --dataset-path "$DATASET" \
  --baseline-checkpoint "$BASELINE" \
  --teacher-cache "$TEACHER_CACHE" \
  --expert-name label_only_control \
  --save-dir "$ADAPTER_DIR" \
  --random-train-severity \
  --epochs 4 \
  --max-train-batches 256 \
  --seed 42 \
  --ce-weight 1.0 \
  --kd-weight 0 \
  --teacher-feature-weight 0 \
  --anchor-weight 0.2

# Phase 3b: supervised cloud-teacher-guided experiment.
"$PYTHON" module_edge_perception/train_boxcars_cloud_teacher_adapter.py \
  --dataset-path "$DATASET" \
  --baseline-checkpoint "$BASELINE" \
  --teacher-cache "$TEACHER_CACHE" \
  --expert-name cloud_hybrid \
  --save-dir "$ADAPTER_DIR" \
  --random-train-severity \
  --epochs 4 \
  --max-train-batches 256 \
  --seed 42 \
  --ce-weight 1.0 \
  --kd-weight 0.35 \
  --teacher-feature-weight 0.25 \
  --anchor-weight 0.2

# Phase 3c: deployment-realistic refresh without uploaded ground-truth labels.
"$PYTHON" module_edge_perception/train_boxcars_cloud_teacher_adapter.py \
  --dataset-path "$DATASET" \
  --baseline-checkpoint "$BASELINE" \
  --teacher-cache "$TEACHER_CACHE" \
  --expert-name cloud_unlabeled \
  --save-dir "$ADAPTER_DIR" \
  --random-train-severity \
  --epochs 4 \
  --max-train-batches 256 \
  --seed 42 \
  --ce-weight 0 \
  --kd-weight 0.7 \
  --teacher-feature-weight 0.3 \
  --anchor-weight 0.2

mkdir -p "$SUMMARY_DIR"
"$PYTHON" module_edge_perception/summarize_boxcars_cloud_teacher_pipeline.py \
  --adapter-dir "$ADAPTER_DIR" \
  --head-metrics "$HEAD_RUN/metrics.json" \
  --output-json "$SUMMARY_DIR/summary.json" \
  --output-md "$SUMMARY_DIR/summary.md"
