#!/bin/bash
set -e

echo "Uploading large data files to HuggingFace..."
echo ""

# Upload to model repo
echo "==> Uploading RecycleGS..."
hf upload sugusu/my-pro RecycleGS/data/translab_full RecycleGS/data/translab_full --repo-type model
hf upload sugusu/my-pro RecycleGS/baselines/tsgs_official_scene01_30k_v4 RecycleGS/baselines/tsgs_official_scene01_30k_v4 --repo-type model
hf upload sugusu/my-pro RecycleGS/outputs/recovery RecycleGS/outputs/recovery --repo-type model
hf upload sugusu/my-pro RecycleGS/outputs/prune_only RecycleGS/outputs/prune_only --repo-type model

echo "==> Uploading DeformTransGS to model repo..."
hf upload sugusu/my-pro DeformTransGS/experiments/stage4_0_R2A_GT1_gt_optical_semantics_closure \
  DeformTransGS/experiments/stage4_0_R2A_GT1_gt_optical_semantics_closure --repo-type model
hf upload sugusu/my-pro DeformTransGS/experiments/stage5_0_R3_C2_perspective_v2_validity \
  DeformTransGS/experiments/stage5_0_R3_C2_perspective_v2_validity --repo-type model

echo "==> Uploading DeformTransGS to dataset repo..."
hf upload sugusu/my-pro-data DeformTransGS/experiments/stage4_0_attribute_sufficiency_gate \
  DeformTransGS/experiments/stage4_0_attribute_sufficiency_gate --repo-type dataset

echo "==> Uploading ReliablePeakGS to dataset repo..."
hf upload sugusu/my-pro-data ReliablePeakGS/data ReliablePeakGS/data --repo-type dataset
hf upload sugusu/my-pro-data ReliablePeakGS/external ReliablePeakGS/external --repo-type dataset
hf upload sugusu/my-pro-data ReliablePeakGS/environment ReliablePeakGS/environment --repo-type dataset

echo ""
echo "Done! All data uploaded."
