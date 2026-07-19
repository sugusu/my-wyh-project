#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wyh/TRGRGS
TSGS="$ROOT/third_party/TSGS"
PY=/data/wyh/RecycleGS_Mainline/envs/tsgs_mainline/bin/python
train_one() {
  local tag="$1" gpu="$2"
  local scene="$ROOT/data/stage2s/split_${tag}/scene_01" model="$ROOT/checkpoints/split_${tag}/scene_01"
  if [[ -e "$model/point_cloud/iteration_30000/point_cloud.ply" ]]; then echo "Refusing to overwrite completed split_${tag}" >&2; return 64; fi
  mkdir -p "$model"
  local train=("$PY" train.py -s "$scene" -m "$model" -d -n --mask_background --use_asg --asg_degree 24 --sd_normal_until_iter 30000 --iterations 30000 --normal_cos_threshold_iter 3000 --eval --delight_iterations 15000 --resolution 2 --ncc_loss_from_iter 7000 --nofix_position --nofix_scaling --nofix_rotation --seed 42 --normal_folder normals --use_transparencies_map)
  local render=("$PY" render.py -m "$model" -d -n --mask_background --use_asg --asg_degree 24 --iteration 30000 --quiet --num_cluster 5 --voxel_size 0.002 --max_depth 10.0 --mesh_expname mesh --window_size 0.03 --start_threshold 0.0 --end_threshold 0.2 --transparency_threshold 0.15 --use_transparent_depth)
  { printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu";printf '%q ' "${train[@]}";printf '\n';printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu";printf '%q ' "${render[@]}";printf '\n'; } > "$model/command.txt"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu" >> "$ROOT/command.txt";printf '%q ' "${train[@]}" >> "$ROOT/command.txt";printf '\n' >> "$ROOT/command.txt"
  cd "$TSGS";export PYTHONPATH="$ROOT/compat:$TSGS:${PYTHONPATH:-}"
  CUDA_VISIBLE_DEVICES="$gpu" "${train[@]}" 2>&1 | tee "$ROOT/logs/stage2s_split_${tag}_train.log"
  test -s "$model/point_cloud/iteration_30000/point_cloud.ply"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu" >> "$ROOT/command.txt";printf '%q ' "${render[@]}" >> "$ROOT/command.txt";printf '\n' >> "$ROOT/command.txt"
  CUDA_VISIBLE_DEVICES="$gpu" "${render[@]}" 2>&1 | tee "$ROOT/logs/stage2s_split_${tag}_render.log"
  test -s "$model/mesh/tsdf_fusion_post_30000.ply"
}
train_one a 2 & pa=$!
train_one b 3 & pb=$!
wait "$pa";wait "$pb"
