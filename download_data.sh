#!/bin/bash
set -e

echo "Downloading large data files from HuggingFace..."
echo ""

# From model repo (sugusu/my-pro)
echo "==> RecycleGS/data/translab_full/..."
hf download sugusu/my-pro RecycleGS/data/translab_full --repo-type model --local-dir . 2>/dev/null || true

echo "==> RecycleGS/baselines/tsgs_official_scene01_30k_v4/..."
hf download sugusu/my-pro RecycleGS/baselines/tsgs_official_scene01_30k_v4 --repo-type model --local-dir . 2>/dev/null || true

echo "==> RecycleGS/outputs/..."
hf download sugusu/my-pro RecycleGS/outputs/recovery --repo-type model --local-dir . 2>/dev/null || true
hf download sugusu/my-pro RecycleGS/outputs/prune_only --repo-type model --local-dir . 2>/dev/null || true

echo "==> DeformTransGS/experiments/GT data from model repo..."
hf download sugusu/my-pro DeformTransGS/experiments/stage4_0_R2A_GT1_gt_optical_semantics_closure --repo-type model --local-dir . 2>/dev/null || true
hf download sugusu/my-pro DeformTransGS/experiments/stage5_0_R3_C2_perspective_v2_validity --repo-type model --local-dir . 2>/dev/null || true

# From dataset repo (sugusu/my-pro-data)
echo "==> DeformTransGS/experiments/GT data from dataset repo..."
hf download sugusu/my-pro-data DeformTransGS/experiments/stage4_0_attribute_sufficiency_gate --repo-type dataset --local-dir . 2>/dev/null || true

echo "==> ReliablePeakGS data from dataset repo..."
hf download sugusu/my-pro-data ReliablePeakGS/data --repo-type dataset --local-dir . 2>/dev/null || true
hf download sugusu/my-pro-data ReliablePeakGS/external --repo-type dataset --local-dir . 2>/dev/null || true
hf download sugusu/my-pro-data ReliablePeakGS/environment --repo-type dataset --local-dir . 2>/dev/null || true

echo ""
echo "Done! All data downloaded."
