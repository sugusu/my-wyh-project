from pathlib import Path
def test_main_consensus_tool_does_not_open_gt():
 s=(Path(__file__).parents[1]/'tools/run_stage2a_consensus.py').read_text().lower();assert 'scene_mesh' not in s and 'raw_intersections' not in s
