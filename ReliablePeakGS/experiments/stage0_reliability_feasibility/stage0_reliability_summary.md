# Stage 0 Summary

Final CASE: IN-PROGRESS.

New primary line: IN-PROGRESS.

Stage 1 reliability-guided fusion is not allowed yet.

Gaussian training modification is not allowed.

The environment was audited with `CUDA_VISIBLE_DEVICES=2,3`. PyTorch sees exactly two visible V100 devices. The official TSPE-GS repository was identified and cloned after unsetting bad proxy variables.

TSPE-GS source is locked at `cac432ccf3fd9ecccb8da9afd35c323e6b5b5665`. AlphaSurf source is locked at `bd24e58fbc59f0624d59ab8afa87573d1b8249df`. The official AlphaSurf `data.zip` download from Google Drive is in progress. GT geometry validation, baseline reproduction, candidate export, labels, perturbations, features, calibration, evaluation, and independent metric reproduction have not completed.

Next exact research action: finish AlphaSurf data download, unpack and audit scenes, then run camera/GT geometry validation using only `CUDA_VISIBLE_DEVICES=2,3`.
