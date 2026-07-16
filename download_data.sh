#!/bin/bash
# Download large data files from HuggingFace
# Requirements: hf (HuggingFace CLI) - pip install huggingface-hub
# Usage: ./download_data.sh

set -e

HF_REPO="sugusu/my-pro"
echo "Downloading data from HuggingFace: $HF_REPO"
echo ""

# Download DeformTransGS experiments (GT evaluation data)
echo "==> DeformTransGS/experiments/..."
hf download "$HF_REPO" --repo-type model --local-dir /tmp/hf_data 2>/dev/null || \
  mkdir -p DeformTransGS/experiments && \
  for dir in stage4_0_R2A_GT1_gt_optical_semantics_closure \
             stage5_0_R3_C2_perspective_v2_validity \
             stage4_0_attribute_sufficiency_gate; do
    echo "  Downloading $dir..."
    hf download "$HF_REPO" "DeformTransGS/experiments/$dir" --repo-type model \
      --local-dir . 2>/dev/null || true
  done

# Download RecycleGS data
echo "==> RecycleGS/data/translab_full/..."
hf download "$HF_REPO" "RecycleGS/data/translab_full" --repo-type model \
  --local-dir . 2>/dev/null || true

# Download RecycleGS baselines
echo "==> RecycleGS/baselines/..."
hf download "$HF_REPO" "RecycleGS/baselines/tsgs_official_scene01_30k_v4" \
  --repo-type model --local-dir . 2>/dev/null || true

# Download RecycleGS outputs
echo "==> RecycleGS/outputs/..."
hf download "$HF_REPO" "RecycleGS/outputs/recovery" --repo-type model \
  --local-dir . 2>/dev/null || true
hf download "$HF_REPO" "RecycleGS/outputs/prune_only" --repo-type model \
  --local-dir . 2>/dev/null || true

echo ""
echo "Done! All data downloaded."
