import sys, os, torch, numpy as np

def init_tsgs_env(tsgs_root):
    sys.path.insert(0, tsgs_root)
    sys.path.insert(0, os.path.join(tsgs_root, 'pytorch3d_stub'))
    os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

def load_scene(config, device='cuda:0'):
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from arguments import ModelParams, PipelineParams, GroupParams
    import argparse

    mp = ModelParams(argparse.ArgumentParser())
    pp = PipelineParams(argparse.ArgumentParser())

    dataset = GroupParams()
    dataset.source_path = config['scene_dir']
    dataset.model_path = config['model_dir']
    dataset.images = "images"
    dataset.resolution = 2
    dataset.sh_degree = 3
    dataset.asg_degree = 24
    dataset.eval = True
    dataset.preload_img = True
    dataset.white_background = False
    dataset.data_device = "cuda"
    dataset.delight = False
    dataset.normal = False
    dataset.normal_folder = "normals"
    dataset.mask_background = False
    dataset.use_delighted_normal = False
    dataset.use_transparencies_map = False
    dataset.not_delight_only_transparent = False
    dataset.load2gpu_on_the_fly = False
    dataset.is_real = False
    dataset.is_indoor = False
    dataset.add_val = False
    dataset.multi_view_num = 8
    dataset.multi_view_max_angle = 30
    dataset.multi_view_min_dis = 0.01
    dataset.multi_view_max_dis = 1.5
    dataset.ncc_scale = 1.0

    pipe = GroupParams()
    pipe.convert_SHs_python = False
    pipe.compute_cov3D_python = False
    pipe.debug = False

    gaussians = GaussianModel(dataset.sh_degree, dataset.asg_degree)
    gaussians.load_ply(config['checkpoint_path'])
    gaussians.active_sh_degree = 0

    scene = Scene(dataset, gaussians, load_iteration=config['checkpoint_iteration'],
                  shuffle=False)
    return scene, gaussians, pipe

def get_train_cameras(scene):
    return scene.getTrainCameras()

def render_view(gaussians, camera, pipe, bg_color, device='cuda:0', app_model=None):
    from gaussian_renderer import render
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
    rendered = render(camera, gaussians, pipe, bg, app_model=app_model)
    return rendered

def load_app_model(model_dir, iteration=15000, device='cuda:0'):
    from scene.app_model import AppModel
    weights_path = os.path.join(model_dir, "app_model", f"iteration_{iteration}", "app.pth")
    if not os.path.exists(weights_path):
        return None
    app_model = AppModel()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    app_model.load_state_dict(state_dict)
    app_model.cuda()
    app_model.eval()
    return app_model
