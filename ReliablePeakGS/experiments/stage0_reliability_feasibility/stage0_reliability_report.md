# Stage 0 Reliability Report

## Research Question

Can cross-view evidence, extraction perturbation stability, cycle consistency, and layer-order stability estimate transparent depth-hypothesis reliability better than peak score, opacity, contribution mass, or other single-view confidence signals?

## Environment

The server was audited under the GPU policy `CUDA_VISIBLE_DEVICES=2,3`.

Summary:

- Driver: 550.54.15.
- CUDA reported by `nvidia-smi`: 12.4.
- Visible PyTorch devices under `CUDA_VISIBLE_DEVICES=2,3`: 2.
- Visible GPU names: Tesla V100-PCIE-32GB, Tesla V100-PCIE-32GB.
- Python 3.10.12 is available as `python3`.
- `python` is not available.
- `conda` is not available.
- `nvcc` is not available.
- PyTorch: 2.6.0+cu124.

Environment details are in `/data/wyh/ReliablePeakGS/environment/`.

## Official Source Provenance

Primary official baseline candidate:

- Method: TSPE-GS.
- Official project: https://nortonii.github.io/TSPE-GS/
- Official repository: https://github.com/nortonii/TSPE-GS
- Paper: https://arxiv.org/abs/2511.09944

The GitHub web page showed branch `main` and short commit `cac432c`. A full exact local commit lock was not possible because local source acquisition failed.

Related method:

- TSGS project: https://longxiang-ai.github.io/TSGS/
- TSGS repository: https://github.com/longxiang-ai/TSGS

TSGS was not selected as the primary baseline because the Stage 0 protocol requires a multi-depth or multi-peak transparent Gaussian depth candidate source.

## Source Acquisition Failure

The following local acquisition commands failed:

- `git clone https://github.com/nortonii/TSPE-GS.git TSPE-GS`
- `git clone --depth 1 https://github.com/nortonii/TSPE-GS.git TSPE-GS`
- `git ls-remote https://github.com/nortonii/TSPE-GS.git HEAD`
- `curl -L --retry 3 --retry-delay 2 -o TSPE-GS-main.zip https://github.com/nortonii/TSPE-GS/archive/refs/heads/main.zip`
- `git ls-remote https://github.com/longxiang-ai/TSGS.git HEAD`
- `curl -L --retry 3 --retry-delay 2 -o TSGS-main.zip https://github.com/longxiang-ai/TSGS/archive/refs/heads/main.zip`

Observed errors included:

- `gnutls_handshake() failed: The TLS connection was non-properly terminated`
- `TLS connect error: unexpected eof while reading`

No official source was modified.

## Dataset Audit

No dataset was downloaded. No scene was selected. AlphaSurf was recorded as a likely official TSPE-GS dataset dependency, but it was not audited locally because the official repository and its dataset instructions could not be acquired.

## Baseline Reproduction

Not run. Reproduction requires official source lock, official configuration, dataset acquisition, and checkpoints or training instructions.

## Candidate Definition And Export

Not run. Candidate source files, functions, line ranges, and exact algorithms could not be audited without local official source.

## GT Labeling Protocol

The label protocol was recorded, but no labels were generated. GT camera/scale validation, mesh rendering, nearest-triangle distances, and layer labels require selected scenes and verified GT geometry.

## Feature Definitions

No feature table was generated. No GT-derived feature leakage was introduced, but S0-G6 cannot pass because no reproducible feature pipeline exists.

## Scene-Level Splits

No scene split was created because no valid scenes were selected.

## Calibration And Metrics

No logistic regression, calibration, risk-coverage, layer-specific evaluation, feature ablation, or independent metric reproduction was run.

## Gates

- S0-G0: FAIL_BLOCKED_LOCAL_SOURCE_ACQUISITION.
- S0-G1: NOT-RUN.
- S0-G2: FAIL.
- S0-G3: NOT-RUN.
- S0-G4: NOT-RUN.
- S0-G5: NOT-RUN.
- S0-G6: NOT-RUN.
- S0-G7: NOT-RUN.
- G8-A through G8-G: NOT-RUN.

## Failure Cases

The run is blocked by local inability to clone or download official GitHub sources over HTTPS. The protocol requires exact official source lock before baseline reproduction and all downstream artifacts.

## Scientific Limitations

No scientific claim can be made about cross-view reliability, perturbation stability, calibration, AUPRC, Brier score, ECE, or top-50% geometry error. The only valid conclusion from this run is an infrastructure/source-acquisition failure.

## Final Gate Decision

Final CASE: BASELINE-REPRODUCTION-FAIL.

New primary line: STOP.

Stage 1 reliability-guided fusion: no.

Gaussian training modification: no.

Next exact research action: repair server GitHub HTTPS access or provide an official TSPE-GS source archive with verifiable commit SHA and license, then rerun Stage 0 from source lock onward using only `CUDA_VISIBLE_DEVICES=2,3`.
