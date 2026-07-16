# S0 Candidate Source Audit

Status: IMPLEMENTED-NOT-RUN.

Official source commit: `cac432ccf3fd9ecccb8da9afd35c323e6b5b5665`.

## Candidate Depth Rendering

- Source file: `/data/wyh/ReliablePeakGS/external/TSPE-GS/gaussian_renderer/__init__.py`
- Function: `render_threshold`
- Line range audited: 85-152
- Role: renders a stack of median depth maps for uniformly sampled transmittance thresholds.
- Key call: passes `threshold=<sample>` into `GaussianRasterizationSettings`, then calls the official CUDA rasterizer.

## CUDA Threshold Handling

- Source file: `/data/wyh/ReliablePeakGS/external/TSPE-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
- Relevant line range audited: 1148-1236
- Role: carries per-pixel depth/alpha state and projected depths inside the rasterizer.

## Peak Selection

- Source file: `/data/wyh/ReliablePeakGS/external/TSPE-GS/mesh_extract_opa_hotfix.py`
- Function: `remove_adjacent_duplicates`
- Line range audited: 273-303
- Algorithm:
  - compute mean depth for each threshold-rendered depth map;
  - apply Gaussian KDE to the sequence of mean depths;
  - run `scipy.signal.find_peaks`;
  - map each KDE peak back to the nearest sampled threshold index;
  - return full depth maps at those selected threshold indices.

## Local Export Wrapper

- Wrapper: `/data/wyh/ReliablePeakGS/reliable_peak_gs/candidate_export/export_tspe_candidates.py`
- Modification policy: does not edit official TSPE-GS source; reuses official renderer and reproduces the audited peak-selection logic.
- Status: script compiles; not yet run because AlphaSurf data download and baseline checkpoints/training are not complete.

## Known Gaps

The official Python API does not currently expose per-candidate contributing Gaussian IDs or contribution mass. These are recorded as schema placeholders until an auditable renderer instrumentation patch is added. That patch must not alter baseline predictions.
