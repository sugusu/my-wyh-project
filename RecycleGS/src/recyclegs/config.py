import yaml, os, sys, json

def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('device', 'cuda:0')
    for k in ['debug_output_dir', 'reliability_output_dir', 'reference_output_dir', 'figure_output_dir']:
        os.makedirs(cfg.get(k, '/tmp/recyclegs'), exist_ok=True)
    return cfg

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_np(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import numpy as np
    np.save(path, arr)

def save_npz(path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import numpy as np
    np.savez_compressed(path, **kwargs)
