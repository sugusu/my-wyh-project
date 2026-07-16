# AttributeDeformGS Final Audit

## Final Decision

ACCEPT:

`CASE REAL-CANONICAL-CARRIER-INSUFFICIENT`

The current Stage4 canonical carrier line is stopped. Do not continue C4C, C4D, C4E, or further rescue of this fixed 4096-Gaussian TSGS-based canonical carrier.

## Scientific Status

`ATTRIBUTEDEFORMGS-HYPOTHESIS: UNTESTED`

The hypothesis, "which Gaussian optical attributes must dynamically evolve under semi-transparent deformation", is not scientifically falsified. The real R0-R7 dynamic attribute sufficiency comparison was never allowed to run because canonical carrier capacity was not established.

## Formal Blocker

The fixed 4096-Gaussian surfel-like carrier with the current TSGS Gaussian image-formation operator cannot sufficiently fit the canonical thin-transmission benchmark under the frozen corrected Gate.

Corrected real canonical metrics:

| Case | PSNR | median image-equivalent optical-depth Elog |
| --- | ---: | ---: |
| K0 | 21.231329 | 0.178887 |
| K1 | 21.231336 | 0.178926 |
| K2 | 17.881014 | 0.215660 |

Frozen C4R requires:

- PSNR >= 28 dB
- median tau_eq Elog <= 0.25

The tau_eq median criterion passes, but PSNR fails strongly. The capacity diagnosis is `TRAIN-FIT-INSUFFICIENT`.

## Final Labels

Stage3.5A: `ANALYTIC-KIOT-SUPPORTED`

Stage3.5A-R1: `KIOT-READY-FOR-RECONSTRUCTION-BRIDGE`

Stage3.5B: `RETIRED-SYNTHETIC-BRIDGE`

Stage3.5B-R4B: `NO-RECOVERABLE-TSGS-SURFACE-LAYER`

Stage3.6A-R4: `SOURCE-OBJ-RENDER-SCOPE-NOT-RECOVERABLE`

Stage4.0: `RETIRED-SYNTHETIC-ATTRIBUTE-STUDY`

Stage4.0-R2A-GT1: `CLEAN-GT-ESTABLISHED`

Stage4.0-R2A-E1: `RASTERIZER-ATTRIBUTE-DIFFERENTIABLE`

Stage4.0-R2A-F2: `REAL-O-C-V-GRAPH-VALIDATED`

Stage4.0-R2A-C4A: `CANONICAL-JOB-CASE-REUSE-FIXED`

Stage4.0-R2A-C4B: `ALPHA-OBSERVABLE-SEMANTIC-BUG-CONFIRMED`

FINAL: `REAL-CANONICAL-CARRIER-INSUFFICIENT`

ATTRIBUTEDEFORMGS-HYPOTHESIS: `UNTESTED`

PRIMARY-ATTRIBUTE-DEFORMATION-METHOD-LINE: `STOP`

## Hard Stop Rules

Do not repair this line by:

- increasing Gaussian count
- optimizing canonical xyz
- optimizing canonical covariance
- replacing the carrier with a regular-grid carrier
- lowering the PSNR Gate
- changing loss weights
- changing C4R thresholds
- adding extinction attributes
- adding RGB opacity
- adding an MLP
- returning to KIOT

