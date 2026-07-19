#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS
TSGS="$ROOT/third_party/TSGS"
SCENE="$ROOT/data/translab/scene_01"
MODEL="$ROOT/checkpoints/baseline/scene_01"
PYTHON=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
GPU="${TRGRGS_GPU:-2}"
case "$GPU" in 2|3) ;; *) echo "ERROR: only physical GPUs 2 and 3 are allowed" >&2; exit 64;; esac
NORMAL_FOLDER="$($PYTHON "$ROOT/tools/detect_normal_folder.py" "$SCENE")"
test "$NORMAL_FOLDER" = "normals" # scene01 transnormals is currently incomplete (303/400)
mkdir -p "$MODEL" "$ROOT/logs" "$ROOT/reports"
cd "$TSGS"
export PYTHONPATH="$ROOT/compat:$TSGS:${PYTHONPATH:-}"

TRAIN=("$PYTHON" train.py -s "$SCENE" -m "$MODEL" -d -n --mask_background --use_asg
  --asg_degree 24 --sd_normal_until_iter 30000 --iterations 30000
  --normal_cos_threshold_iter 3000 --eval --delight_iterations 15000 --resolution 2
  --ncc_loss_from_iter 7000 --nofix_position --nofix_scaling --nofix_rotation
  --seed 42 --normal_folder "$NORMAL_FOLDER" --use_transparencies_map)
RENDER=("$PYTHON" render.py -m "$MODEL" -d -n --mask_background --use_asg --asg_degree 24
  --iteration 30000 --quiet --num_cluster 5 --voxel_size 0.002 --max_depth 10.0
  --mesh_expname mesh --window_size 0.03 --start_threshold 0.0 --end_threshold 0.2
  --transparency_threshold 0.15 --use_transparent_depth)
EVAL=("$PYTHON" scripts/eval_translab/eval.py
  --data "$MODEL/mesh/tsdf_fusion_post_30000.ply" --scan scene_01
  --vis_out_dir "$MODEL/mesh" --dataset_dir "$ROOT/data/translab"
  --mode mesh --downsample_density 0.002)
{
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${TRAIN[@]}"; printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${RENDER[@]}"; printf '\n'
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$GPU"; printf '%q ' "${EVAL[@]}"; printf '\n'
} > "$MODEL/command.txt"
cp "$MODEL/command.txt" "$ROOT/outputs/scene_01/command.txt"
CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN[@]}" 2>&1 | tee "$ROOT/logs/scene01_baseline_train.log"
test -s "$MODEL/point_cloud/iteration_30000/point_cloud.ply"
CUDA_VISIBLE_DEVICES="$GPU" "${RENDER[@]}" 2>&1 | tee "$ROOT/logs/scene01_baseline_render.log"
test -s "$MODEL/mesh/tsdf_fusion_post_30000.ply"
CUDA_VISIBLE_DEVICES="$GPU" "${EVAL[@]}" 2>&1 | tee "$ROOT/logs/scene01_baseline_geometry.log"
"$PYTHON" "$ROOT/tools/stage0_report.py"
