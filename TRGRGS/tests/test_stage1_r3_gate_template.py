from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_stage1_uses_exact_nonquantile_names_and_r3_gate():
 d=yaml.safe_load((ROOT/'configs/stage1_depth_sweep_r3.yaml').read_text());assert d['prerequisite']['required_status']=='PASS_PROTOCOL_EQUIVALENT';assert d['terminology']['renderer_depth']=='max_weight_threshold_window_depth';assert d['terminology']['uncertainty']=='threshold_conditioned_depth_dispersion';assert d['gate']['renderer_monotonicity_gate'] is False

