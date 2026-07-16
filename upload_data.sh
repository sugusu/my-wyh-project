#!/bin/bash
# Upload large data files to HuggingFace
# Requirements: hf (HuggingFace CLI) - pip install huggingface-hub
# Usage: ./upload_data.sh
# First: hf auth login (or set HF_TOKEN env var)

set -e

HF_REPO="sugusu/my-pro"
echo "Uploading data to HuggingFace: $HF_REPO"
echo ""

# Upload DeformTransGS experiments
echo "==> DeformTransGS/experiments/..."
hf upload "$HF_REPO" DeformTransGS/experiments/stage4_0_R2A_GT1_gt_optical_semantics_closure \
  DeformTransGS/experiments/stage4_0_R2A_GT1_gt_optical_semantics_closure --repo-type model
hf upload "$HF_REPO" DeformTransGS/experiments/stage5_0_R3_C2_perspective_v2_validity \
  DeformTransGS/experiments/stage5_0_R3_C2_perspective_v2_validity --repo-type model
hf upload "$HF_REPO" DeformTransGS/experiments/stage4_0_attribute_sufficiency_gate \
  DeformTransGS/experiments/stage4_0_attribute_sufficiency_gate --repo-type model

# Upload RecycleGS data
echo "==> RecycleGS/data/translab_full/..."
hf upload "$HF_REPO" RecycleGS/data/translab_full RecycleGS/data/translab_full --repo-type model

# Upload RecycleGS baselines
echo "==> RecycleGS/baselines/..."
hf upload "$HF_REPO" RecycleGS/baselines/tsgs_official_scene01_30k_v4 \
  RecycleGS/baselines/tsgs_official_scene01_30k_v4 --repo-type model

# Upload RecycleGS outputs
echo "==> RecycleGS/outputs/..."
hf upload "$HF_REPO" RecycleGS/outputs/recovery RecycleGS/outputs/recovery --repo-type model
hf upload "$HF_REPO" RecycleGS/outputs/prune_only RecycleGS/outputs/prune_only --repo-type model

echo ""
echo "Done! All data uploaded."
