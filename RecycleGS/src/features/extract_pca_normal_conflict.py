import argparse, sys, os, numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from recyclegs.config import load_config, save_np, save_json
from recyclegs.tsgs_loader import load_scene, get_train_cameras

def load_ply_xyz(ply_path):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    vertex = ply['vertex']
    xyz = np.stack([np.asarray(vertex['x']), np.asarray(vertex['y']), np.asarray(vertex['z'])], axis=1)
    return xyz

def local_pca_normal(points):
    cov = np.cov(points.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argmin(evals)
    normal = evecs[:, idx]
    if normal[2] < 0:
        normal = -normal
    planarity = (evals[1] - evals[0]) / (evals[2] + 1e-10)
    return normal, float(evals[0]), float(evals[1]), float(evals[2]), float(planarity)

def weighted_median(arr):
    if len(arr) == 0:
        return np.nan
    return float(np.median(arr))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--iteration', type=int, required=True, choices=[15000])
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg['reliability_output_dir'])
    device = cfg['device']
    scene_dir = cfg['scene_dir']
    iter_str = f"iter_{args.iteration}"
    iter_dir = out_dir / iter_str
    os.makedirs(iter_dir, exist_ok=True)

    knn = cfg.get('pca_normal', {}).get('knn', 24)
    min_neighbors = cfg.get('pca_normal', {}).get('min_neighbors', 8)
    min_planarity = cfg.get('pca_normal', {}).get('min_planarity', 0.10)

    print("[1/6] Loading scene and cameras...")
    scene, gaussians, pipe = load_scene(cfg, device)
    cameras = get_train_cameras(scene)
    total = len(cameras)
    n_views = min(cfg['analysis_views']['count'], total)
    step = max(1, total // n_views)
    selected = list(range(0, total, step))[:n_views]
    print(f"  Using {len(selected)} cameras from {total} total")

    print("[2/6] Loading candidate Gaussians and computing PCA normals...")
    candidate_indices = np.load(iter_dir / 'candidate_indices.npy')
    ply_path = Path(cfg['checkpoint_path'])
    xyz_all = load_ply_xyz(str(ply_path))
    xyz_cand = xyz_all[candidate_indices]
    N = len(xyz_cand)
    print(f"  Candidate Gaussians: {N}")

    print(f"  Building kNN tree (k={knn})...")
    tree = cKDTree(xyz_cand)

    N_total_full = len(xyz_all)
    global_candidate = np.zeros(N_total_full, dtype=bool)
    global_candidate[candidate_indices] = True

    pca_normals = np.full((N, 3), np.nan, dtype=np.float32)
    pca_lambda0 = np.full(N, np.nan, dtype=np.float32)
    pca_lambda1 = np.full(N, np.nan, dtype=np.float32)
    pca_lambda2 = np.full(N, np.nan, dtype=np.float32)
    pca_planarity = np.full(N, np.nan, dtype=np.float32)
    pca_normal_valid = np.zeros(N, dtype=bool)
    pca_num_neighbors = np.zeros(N, dtype=np.int32)

    chunk = min(5000, N)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        batch_xyz = xyz_cand[start:end]
        dists, idxs = tree.query(batch_xyz, k=min(knn + 1, N))
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]

        for i in range(start, end):
            bi = i - start
            if len(idxs[bi]) < min_neighbors:
                continue
            neigh_xyz = xyz_cand[idxs[bi]]
            n, l0, l1, l2, plan = local_pca_normal(neigh_xyz)
            pca_normals[i] = n
            pca_lambda0[i] = l0
            pca_lambda1[i] = l1
            pca_lambda2[i] = l2
            pca_planarity[i] = plan
            pca_num_neighbors[i] = len(idxs[bi])
            pca_normal_valid[i] = (len(idxs[bi]) >= min_neighbors) and (plan >= min_planarity)

        if start % 20000 == 0:
            print(f"    kNN progress: [{start}/{N}]")

    num_valid_pca = int(pca_normal_valid.sum())
    print(f"  Valid PCA normals: {num_valid_pca}/{N}")

    print(f"[3/6] Projecting PCA normals to {len(selected)} views...")
    e_normal_per_view = np.full((N, len(selected)), np.nan, dtype=np.float32)

    mask_dir = os.path.join(scene_dir, 'transparent_masks')
    normal_dir = os.path.join(scene_dir, 'normals')

    for vi, cam_idx in enumerate(selected):
        cam = cameras[cam_idx]
        stem = cam.image_name.split('.')[0]

        import torch
        w2c = cam.world_view_transform[:3, :3].to(device)
        tvec = cam.world_view_transform[3, :3].to(device)
        K_mat = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]],
                             dtype=torch.float32, device=device)
        xyz_t = torch.from_numpy(xyz_cand).float().to(device)
        pts_cam = xyz_t @ w2c.T + tvec.unsqueeze(0)
        depths = pts_cam[:, 2]
        valid_depth = depths > cfg['projection']['min_positive_depth']
        pts_2d = pts_cam[:, :2] / depths.clamp(min=1e-8).unsqueeze(1)
        pts_px = pts_2d @ K_mat[:2, :2].T + K_mat[:2, 2].unsqueeze(0)
        u, v = pts_px[:, 0], pts_px[:, 1]
        in_frame = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height) & valid_depth

        mask_path = os.path.join(mask_dir, f'{stem}.png')
        if not os.path.exists(mask_path):
            continue
        mask_img = Image.open(mask_path).convert('L')
        mask_img_resized = mask_img.resize((cam.image_width, cam.image_height))
        mask_np = np.array(mask_img_resized, dtype=np.float32) / 255.0

        normal_path = os.path.join(normal_dir, f'{stem}_normal.png')
        if not os.path.exists(normal_path):
            continue
        nimg = Image.open(normal_path).convert('RGB')
        nimg_resized = nimg.resize((cam.image_width, cam.image_height))
        n_arr = np.array(nimg_resized, dtype=np.float32) / 255.0 * 2.0 - 1.0
        n_norm = np.linalg.norm(n_arr, axis=2, keepdims=True)
        n_arr = n_arr / (n_norm + 1e-8)

        in_frame_np = in_frame.cpu().numpy()
        valid_idx = np.where(in_frame_np & pca_normal_valid)[0]
        if len(valid_idx) == 0:
            continue

        u_np = pts_px[:, 0].detach().cpu().numpy()
        v_np = pts_px[:, 1].detach().cpu().numpy()

        for i in valid_idx:
            mval = mask_np[int(v_np[i]), int(u_np[i])] if (0 <= int(v_np[i]) < mask_np.shape[0] and 0 <= int(u_np[i]) < mask_np.shape[1]) else 0.0
            if mval < 0.5:
                continue

            n_prior = n_arr[int(v_np[i]), int(u_np[i])]
            n_prior_norm = np.linalg.norm(n_prior)
            if n_prior_norm < 0.5:
                continue

            n_pca_world = pca_normals[i]
            if np.any(np.isnan(n_pca_world)):
                continue
            n_pca_cam = n_pca_world @ w2c.cpu().numpy().T
            n_pca_cam = n_pca_cam / (np.linalg.norm(n_pca_cam) + 1e-8)

            dot = abs(np.dot(n_pca_cam, n_prior))
            e_normal = 1.0 - min(dot, 1.0)
            e_normal_per_view[i, vi] = e_normal

        if vi % 16 == 0:
            print(f"    [{vi+1}/{len(selected)}]")

    print("[4/6] Aggregating cross-view normal conflict...")
    E_normal_pca = np.full(N, np.nan, dtype=np.float32)
    normal_variance = np.full(N, np.nan, dtype=np.float32)

    for i in range(N):
        vals = e_normal_per_view[i, ~np.isnan(e_normal_per_view[i])]
        if len(vals) >= 3:
            E_normal_pca[i] = weighted_median(vals)
            normal_variance[i] = float(np.var(vals))

    print(f"  E_normal_pca valid: {np.isfinite(E_normal_pca).sum()}/{N}")

    print("[5/6] Saving outputs...")
    save_np(pca_normals, iter_dir / 'pca_normals.npy')
    save_np(pca_planarity, iter_dir / 'pca_planarity.npy')
    save_np(pca_normal_valid, iter_dir / 'pca_normal_valid.npy')
    save_np(E_normal_pca.astype(np.float32), iter_dir / 'pca_normal_conflict.npy')
    save_np(normal_variance.astype(np.float32), iter_dir / 'normal_variance_pca.npy')
    save_np(pca_num_neighbors.astype(np.int32), iter_dir / 'pca_num_neighbors.npy')
    save_np(pca_lambda0.astype(np.float32), iter_dir / 'pca_lambda0.npy')
    save_np(pca_lambda1.astype(np.float32), iter_dir / 'pca_lambda1.npy')
    save_np(pca_lambda2.astype(np.float32), iter_dir / 'pca_lambda2.npy')

    stats = {
        'iteration': args.iteration,
        'num_views': len(selected),
        'num_candidate_gaussians': N,
        'knn': knn,
        'min_neighbors': min_neighbors,
        'min_planarity': min_planarity,
        'num_valid_pca_normals': num_valid_pca,
        'num_valid_normal_conflict': int(np.isfinite(E_normal_pca).sum()),
        'pca_planarity_mean': float(np.nanmean(pca_planarity)),
        'E_normal_pca_mean': float(np.nanmean(E_normal_pca)),
        'E_normal_pca_median': float(np.nanmedian(E_normal_pca)),
    }
    save_json(stats, iter_dir / 'pca_normal_stats.json')

    print("[6/6] Done.")

if __name__ == '__main__':
    main()
