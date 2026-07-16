# DeformTransGS

**Deformation-Aware Attribute Transport for Transparent Gaussian Splatting**

## 研究问题

透明/半透明 3D Gaussian Splatting 在物体发生形变过程中，哪些几何属性、方向相关外观属性和光学属性需要随形变更新。

## 阶段完成情况

- **Stage 0 ✅** TSGS Gaussian 属性审计
- **Stage 0.5 ✅** TSGS Stage 2 checkpoint 与 ASG 调用链审计
- **Stage 0.6 ✅** Deformation Hypothesis Sanity Check
  - H1：Normal axis switching **不**作为核心方向（switch rate < 0.05%）
  - H2：ASG deformation correctness **需要 GT evaluation**
- **Stage 1 ✅** Minimal Transparent Deformation GT Benchmark
- **Stage 1.1 ✅** Benchmark Repair and Stress Validation
  - Twist Jacobian repaired（analytic vs autograd error < 1e-7）
  - Optical effect mask introduced
  - Constant lighting selected as benchmark scene
  - Full-image MAE confirmed to be diluted by background（~20x）
- **Stage 2 ✅** Canonical TSGS Fit and ASG Contribution Gate
  - Model A (ASG): PSNR 49.50, 32,098 Gaussians
  - Model B (SH): PSNR 50.91, 31,521 Gaussians
  - ASG Contribution Gate: **FAIL** (ratio=0.0069 < 0.05)
  - Current benchmark insufficient for ASG deformation evaluation

## TSGS Baseline 来源

- 代码仓库：`/data/wyh/repos/TSGS`
- Baseline checkpoint：`/data/wyh/RecycleGS/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply`
- Baseline 指标：PSNR 28.60, SSIM 0.947, LPIPS 0.095
- 完整 Stage 2 checkpoint：`/data/wyh/RecycleGS/baselines/tsgs_official_scene01_30k_v4`

## 实验目录结构

```
DeformTransGS/
├── README.md
├── tools/
│   └── export_tsgs_gaussian_attrs.py
├── analysis/
│   ├── test_normal_transport.py
│   └── test_asg_frame_sensitivity.py
├── benchmark/
│   ├── deformations/
│   │   ├── common.py
│   │   ├── shear.py
│   │   └── twist.py
│   └── mitsuba/
│       └── run_stage1.py
├── experiments/
│   ├── stage0_attribute_audit/
│   ├── stage0_5_checkpoint_audit/
│   ├── stage0_6_hypothesis_check/
│   └── stage1_minimal_gt/
└── scripts/
    └── run_tsgs_scene01_stage2.sh
```

## 下一阶段

在合成的 canonical GT 上训练匹配的 TSGS，然后进行 deformation evaluation。

## Stage3.4C-R1 Gate Closure and Kernel-Opacity Scalar Expressivity Audit

Stage3.4C showed a strong central response shift under full deformation-gradient covariance transport: for stretch2, P0 FIXED_COV remains near the physical 0.5 response, while P2 FULL_AFFINE_COV shifts strongly toward 1.

The tangent-footprint budget proxy is highly correlated with central optical response. However, the formal Stage3.4C case is not yet accepted because S2 failed and the S4 terminal log contradicted the predefined 5/6 closer-to-one criterion.

Stage3.4C-R1 first closes these Gates on the exact frozen evaluation key set. It then audits why the diagnostic tau/Js oracle fails. Under the Gaussian rasterizer, stored opacity is multiplied by a spatially varying Gaussian kernel amplitude before transmittance compositing. The update tau'=q tau is exact at Gaussian center but is generally not exact across the entire Gaussian kernel support.

R1 tests whether this kernel-amplitude dependence merely invalidates tau/Js or reveals a deeper expressivity limitation of one scalar optical state per Gaussian.

## Stage3.5A Renderer-Aware Kernel-Integrated Opacity Transport Gate

Stage3.4C-R1 repaired the S2 reference-key mismatch and strictly confirmed the central-cancellation Gate in all 6/6 area-changing states. It also proved that tau/Js is not merely implemented incorrectly.

Under the actual rasterizer semantics, alpha = opacity * Gaussian kernel amplitude before transmittance compositing. tau/Js is exact at the Gaussian center but kernel-nonexact off center, especially for high-opacity Gaussians.

Despite this, the best scalar opacity oracle recovers approximately 97.6% RMSE relative to tau/Js, and a free-tau patch oracle supports local scalar-state capacity. Therefore the next method hypothesis is: transport the kernel-integrated effective optical contribution rather than Gaussian-center optical depth.

Stage3.5A tests Phi(o') = (1/Js) Phi(o) using continuous and local-CUDA-aware Gaussian kernel integrals. The dilogarithm itself is not claimed as novel; Gaussian Point Splatting 2026 also uses Li2 for a different Gaussian opacity-correction problem. The research contribution under test is deformation-aware transport of kernel-integrated optical state.

## Stage3.5A-R1 KIOT Method Closure

Stage3.5A established KIOT as the first candidate optical-state transport method in the project. CUDA-aware KIOT reduced the six-state central response error from 0.279456 to 0.027946, a 90% improvement over fixed optical state, while tau/Js increased error to 1.183019.

Stage3.5A-R1 fixes the CUDA-aware zero-plateau inverse: opacity below the rasterizer alpha skip threshold maps to Phi_cuda=0, so the old inverse selected zero and was not exact identity for q=1 at sub-skip opacities. R1 introduces an identity-preserving zero-plateau tie-break, audits the validated Js>=1 deformation domain, and replaces scalar bisection with an accurate LUT inverse before reconstructed-carrier integration.

## Stage3.5B-R1 bridge protocol closure

Stage3.5B produced strong numerical evidence that KIOT transfers to learned TSGS Gaussian patches: the reported CUDA-aware KIOT mean central error was 0.023685, a 92.3% reduction relative to fixed opacity, with a 100% reported win fraction.

The q=1 full-checkpoint identity was exact, and direct KIOT opacity mutation caused limited TSGS first-surface drift under the tested patch protocol.

However, the Stage3.5B execution differed from the predefined bridge protocol in two important ways. First, the reported transparent support source was `checkpoint_visualize_nearest_depth_multiview_support` rather than an explicitly verified official TransLab transparent mask. Second, the evaluated deformation suite included `normal_compress_0p75`, which lies outside the formally validated KIOT scope Js>=1.

Stage3.5B-R1 therefore audits official mask provenance, selection independence, the exact local-frame deformation matrices, and recomputes the bridge Gate using only PRIMARY Js>=1 patch-state pairs. No KIOT method changes are made.

## Stage3.5B-R2 official-mask real-render bridge

Stage3.5B and Stage3.5B-R1 did NOT establish a real rasterized reconstructed-carrier KIOT bridge. The previous patch response experiment used state-name / target-q driven synthetic policy response generation and did not persist actual deformation F matrices.

A subsequent audit showed that several named local-frame deformations were inconsistent with their matrices. With local basis [n,t1,t2], diag(s,1,1) is a normal-direction stretch and has surface area stretch Js=1, not Js=s.

Therefore the previous reconstructed bridge 92.3% improvement is retired as real-render evidence and retained only as synthetic exploratory evidence. Stage3.5B-R2 restarts the bridge using official transparent_masks, automatic official-mask patch selection, explicit saved deformation matrices F, Js computed strictly from |detF| ||F^-T n||, actual xyz transport, actual covariance transport F Sigma F^T, actual KIOT opacity transport, fresh TSGS rasterizer alpha maps, and frozen material-proxy response keys. No synthetic policy-response generator is allowed.

## Stage3.5B-R3 surface-layer transport kill gate

Stage3.5B-R2 was the first official-mask, explicit-F, actual-covariance-transport, fresh-render TSGS bridge experiment. However, its selected patches violated the predefined normal-coherence Gate: normal p90 was approximately 75 degrees for all three patches, far above the 10-degree limit. The patch selector nevertheless continued, so R21 failed.

On these invalid thin-surface patches, opacity-linear transport achieved mean central error 0.040281, while KIOT-CUDA achieved 0.106589. Thus KIOT was not the best real-render policy in R2.

Direct opacity mutation also changed TSGS first-surface valid-mask topology (IoU 0.881439), confirming a geometry-optics semantic conflict.

Stage3.5B-R3 is a strict kill Gate. It first determines whether normal-coherent, single-layer, official-mask TSGS surface patches exist at any predeclared spatial scale. Only then does it compare KIOT against opacity-linear using fresh optical renders and a fixed-geometry evaluation pass. If opacity-linear still wins, KIOT is killed as the primary real-carrier method. No MLP rescue is allowed.

## Stage3.5B-R4 normal semantics and first-surface layer recovery

Stage3.5B-R3 reported NO-COHERENT-TSGS-SURFACE-CARRIER, but this carrier-level conclusion is not formally accepted. R3 itself proved that the inherited R2 candidate lock contained a depth-support implementation bug: strict, medium, and loose support counts were copied from mask-inside counts, and actual depth_rel_error was never stored.

Therefore the R3 coherence sweep operated on a mask-only candidate set without the intended first-surface layer disambiguation. R3 still establishes that the buggy mask-only candidate set has severe local Gaussian-normal incoherence and that the R2 patch fallback violated the 10-degree normal-coherence Gate.

Stage3.5B-R4 independently traces the actual TSGS normal semantic, recomputes sample-level first-surface depth proximity, and constructs multiview depth-derived surface normals. The purpose is to decide whether the reconstructed TSGS carrier contains recoverable coherent single-surface layers and which normal source can legitimately serve as the material-normal proxy for Js computation. No optical transport policy is evaluated in R4.

## Stage3.5B-R4A depth-normal bridge closure

Stage3.5B-R4 did not establish a normal implementation bug. The actual TSGS normal path and covariance minimum-eigenvector path agree to approximately 1.15e-4 degrees, while the old angular Gate used acos in a numerically ill-conditioned near-parallel regime.

At the same time, TSGS Gaussian normals differ strongly from multiview first-surface depth normals (median 67.1 degrees, p90 86.7 degrees). Fresh first-surface support rebuilt 5523/7019/7170 strict/medium/loose candidates, confirming the earlier depth-support bug.

All 7019 MEDIUM candidates obtained reliable multiview depth normals, and the previous sweep reported strong depth-normal coherence at all tested scales. Stage3.5B-R4A performs a numerically stable TSGS-normal equivalence test, audits depth-normal diversity and camera transforms, independently reproduces the depth-normal patch sweep, and tests first-surface layer purity. No optical policy is evaluated.

## Stage3.5B-R4B depth-normal coordinate-frame closure

Stage3.5B-R4A confirmed that the actual TSGS smallest-axis normal and the covariance minimum-eigenvector axis are semantically equivalent under stable sign-invariant metrics. However, the R4 depth-normal tensor showed an extreme global +/-z rank-1 signature: global scatter first eigenvalue approximately 0.9999999997, with random-pair p90 unsigned angle approximately 0.00245 degrees.

R4A also documented that R4 did not persist per-view camera-space normals or a verifiable camera-to-world normal transform chain. Therefore the apparent perfect 7019/7019 multiscale depth-normal coherence may be a coordinate-frame artifact.

Stage3.5B-R4B performs one final coordinate-frame closure: camera-space depth points and normals are explicitly distinguished from world-space quantities, camera<->world point/vector round trips are numerically verified, and world normals computed by transforming world points are checked against normals obtained by rotating camera-space normals. Only corrected world-space depth normals may be used for multiview fusion and surface-layer recovery. No optical transport policy is evaluated.

## Stage3.6A GT-mesh scaffold transport kill gate

Stage3.5B-R4B accepted `CASE NO-RECOVERABLE-LAYER`, closing the direct TSGS per-Gaussian material-point bridge. Stage3.6A switches to an explicit GT mesh scaffold only as an oracle material-kinematics carrier. The scaffold supplies material identity, topology, normals, deformation gradients, and surface area stretch Js; Gaussian attributes supply appearance, opacity, transparency, and kernels. No TSGS Gaussian/depth normals are used as material normals, and no optical policy is optimized.

Stage3.6A locks the scene_01 GT mesh, audits mesh-camera mask alignment, binds learned TSGS Gaussians to the mesh, transports centers and covariance by explicit affine mesh deformation, renders saved alpha maps with the actual TSGS rasterizer, and compares fixed opacity, tau/Js, opacity-linear, KIOT-continuous, and KIOT-CUDA. This is the final KIOT rule kill gate under a reliable surface scaffold.

## Stage3.6A-R1 transparent mesh scope and alignment closure

Stage3.6A did not evaluate KIOT. The experiment failed at GT mesh scaffold alignment/binding before any real policy render. The reported scene_mesh-vs-transparent-mask median IoU was 0.028253.

A source audit subsequently identified a scope mismatch in the Stage3.6A protocol. The official TransLab generation script exports `scene_mesh.obj` as the merged geometry of all visible scene mesh objects with `export_selected_objects=False`. The official `masks/` path represents the union mask of all scene objects. By contrast, `transparent_masks/` is rendered through Blender object-index masking with ID Mask index 1, corresponding to pass_index 1 objects.

Therefore full `scene_mesh.obj` must first be validated against `masks/`, while only the transparent-object mesh subset should be validated against `transparent_masks/`. Stage3.6A-R1 audits OBJ object/group/component provenance, searches for the original Blender scene and pass_index metadata, and attempts to recover a formally locked transparent-object-specific mesh scaffold before any KIOT policy comparison.

## Stage3.6A-R2 mesh projection and visibility-scope closure

Stage3.6A-R1 established that the current full-scene mesh silhouette test failed against official object masks with median IoU 0.257536. However, the official TransLab source generates `scene_mesh.obj` and COLMAP cameras from the same Blender world. The Blender-to-OpenCV axis conversion is applied to the camera local coordinate convention before world-to-camera R/T is written; no explicit transform is applied to world mesh vertices.

Stage3.6A-R2 does not fit Sim3. It directly compares COLMAP projection with the actual TSGS camera projection, then area-samples the full scene mesh and projects surface samples into the official union object masks. This separates camera/projection error, triangle-rasterizer error, OBJ/mask visibility-scope mismatch, and true downloaded-artifact mismatch. No optical policy is evaluated.

## Stage3.6A-R3 occlusion-aware transparent mesh scope recovery

Stage3.6A-R2 closed the mesh/camera coordinate question. Exact COLMAP and TSGS projections agree with zero measured pixel/depth error under the audit protocol. The previous full-scene silhouette failure is therefore not attributed to Sim3 or camera convention. Direct full-mesh surface sampling showed strongly mixed OBJ-block support against the official scene object masks, establishing an OBJ/mask visibility-scope mismatch.

Surface-point mask support is not used to identify transparent objects because projected but occluded surfaces can fall inside foreground masks. Stage3.6A-R3 instead uses an occlusion-aware labeled z-buffer. The official TransLab transparent-mask script replaces mesh materials with a common diffuse material, renders the Object Index pass, and selects ID Mask index 1. Stage3.6A-R3 recovers transparent geometry from visible mesh identity against official transparent masks. Mesh primitive selection uses 300 discovery cameras, and the recovered subset is frozen before evaluation on 100 held-out cameras. No optical policy is evaluated.

## Stage3.6A-R4 render-scope first transparent scaffold closure

Stage3.6A-R3 built an exact occlusion-aware labeled z-buffer. Its projection agrees with the direct COLMAP reference, and its binary silhouettes agree exactly with the prior mesh renderer. However, R3 used every exported OBJ block as a z-buffer occluder. Stage3.6A-R2 had already established that the exported OBJ has mixed support relative to the official object-mask render scope. The official mask-render path controls both viewport and render visibility, whereas the merged OBJ export path does not reproduce the same render-visibility state explicitly in the exported metadata. Therefore an extra exported block may occlude the true pass_index1 geometry inside an offline z-buffer, causing block/component/face transparent precision scores to collapse.

Stage3.6A-R4 is the final GT-mesh infrastructure Gate. It first recovers the official masks/ render scope at OBJ-block granularity using 300 discovery cameras and validates the frozen scope on 100 held-out cameras. Only inside that frozen render scope does it recover transparent object identity from transparent_masks. If either held-out Gate fails, the GT mesh scaffold route is permanently stopped. No optical policy is evaluated.

## ATTRIBUTEDEFORMGS NEW MAINLINE

The previous KIOT line is frozen as `CONTROLLED-CARRIER-ONLY`. The failure was not an analytic single-Gaussian failure; the central unresolved issue was that the real reconstructed TSGS carrier could not supply stable material identity, material normals, or local surface kinematics. The new research direction therefore does not assume Js or KIOT as the answer.

Stage4.0 studies attribute sufficiency on an independent deforming thin-surface benchmark with three material regimes: fixed-thickness neutral transmission, mass-conserving neutral transmission, and mass-conserving tinted transmission. Using a fixed 4096-Gaussian carrier and exact geometric transport, Stage4.0 releases dynamic optical attribute families O (scalar opacity), C (view-dependent color/appearance coefficients), and V (view-dependent opacity residual). Eight release combinations are evaluated against an independent mesh-based thin-surface optical GT renderer to identify necessary attributes, minimally sufficient subsets, and material-regime dependence. No transport rule is proposed in Stage4.0.

## Stage4.0-R1 Oracle Protocol Repair

Stage4.0 originally reported `ATTRIBUTE-ORACLE-INSUFFICIENT`. Stage4.0-R1 identifies that the original A3 improvement clause was logically incompatible with the MAT0 fixed-thickness static-control regime: if MAT0 behaves correctly, geometry-only R0 is expected to approximately match FULL R7, so requiring R7 to reduce R0 error by 50% fails by design for all 12 MAT0 cases. Even perfect success on all 24 MAT1/MAT2 cases would reach only 24/36, below the original 29/36 requirement.

Stage4.0 also omitted `R0_NONE` from the minimal dynamic-state search, forcing static cases to select a dynamic release group. The original per-deformation Spearman analysis grouped constant global-affine features, making several correlations undefined; those undefined values must not be reported as zero correlation. Stage4.0-R1 repairs these protocol calculations without changing the GT benchmark, O/C/V attribute definitions, E_OPT definition, or frozen numerical sufficiency/necessity thresholds.

The Stage4.0-R1 provenance audit found that the current Stage4.0 implementation generated oracle metrics from deterministic surrogate formulas rather than real autograd/Adam oracle optimization and fresh TEST renders. Therefore the repaired diagnostic pattern is not accepted as a scientific Stage4 conclusion until real oracle provenance is restored.

## Stage4.0-R2A real oracle smoke gate

Stage4.0 scientific outputs are retired. Stage4.0-R1 found that the original oracle implementation did not perform autograd/Adam optimization and that release metrics were generated by `synthetic_release_error(...)` using deterministic surrogate branches. Therefore the previously reported pattern `MAT0 -> NONE`, `MAT1 -> O`, `MAT2 -> O+C+V` is provenance-invalid and cannot be treated as evidence. The AttributeDeformGS hypothesis returns to `UNTESTED` status.

Stage4.0-R2A starts rebuilding a clean real differentiable oracle pipeline in `attribute_study/real_oracle/`. The first strict gate independently audits the saved GT arrays. This run stops at C1 because the saved Stage4.0 GT arrays are float16 and fail the required 1e-6-level numeric agreement thresholds for tau/RGB/alpha. No canonical fit or oracle optimizer is run after that failure.

## Stage4.0-R2A-GT1 GT optical semantics closure

Stage4.0-R2A stopped correctly at C1: Js matched exactly, but tau/RGB/alpha differed at approximately float16 quantization scale. Stage4.0-R2A-GT1 traces the old GT generator to `attribute_study/run_stage4_0.py::render_gt`, confirms float64 optical computation followed by float16 saving for RGB/alpha/tau, and retires the old GT root because exact-pixel C1 cannot pass with the saved float16 optical arrays. A clean float32 GT root is regenerated under explicit pixel-center, normalized camera-to-surface ray, and analytic thin-surface semantics, then independently validated. The verified GT root is locked for future work.

After D4, the run resumes the real oracle smoke gate only up to C2. The current real_oracle differentiable render adapter is not yet implemented, so O/C/V gradient tests fail and no canonical/oracle optimization is run. AttributeDeformGS remains `UNTESTED`; full Stage4.0-R2B is not allowed.

## Stage4.0-R2A-G2 autograd graph closure

Stage4.0-R2A-GT1 established a clean independently validated GT root. Stage4.0-R2A-G2 attempts to localize the O/C/V autograd graph failure, starting at the installed rasterizer boundary. The repository source for `diff_first_surface_rasterization` declares backward outputs for `grad_colors_precomp`, `grad_opacities`, and `grad_sh`, but the installed runtime module fails to import because the `_C` extension is unavailable. Since the command forbids replacing or modifying the rasterizer and forbids fake gradients, the run stops at E1/E2 with `INSTALLED-RASTERIZER-NOT-ATTRIBUTE-DIFFERENTIABLE`. No canonical fit or 24-job oracle smoke run is executed.

## Stage4.0-R2A-E1 rasterizer runtime closure

Stage4.0-R2A-G2 did not establish that the rasterizer was non-differentiable; it only showed that the `_C` extension was not imported from the source checkout path. Stage4.0-R2A-E1 inventories local runtimes and finds an existing compiled `diff_first_surface_rasterization._C` binary in the TSGS submodule `build/lib...` directory. Using the current Python interpreter with that build path, the extension imports, forward rasterization executes on CUDA, and direct leaf opacity and `colors_precomp` tensors receive nonzero gradients from an image loss. A non-interactive launcher records the exact interpreter and `PYTHONPATH` needed for subsequent G2 runs. No scientific attribute experiment is executed in this environment-only gate.

## Stage4.0-R2A-F2F3 real pipeline closure

Stage4.0-R2A-F2F3 runs under the verified rasterizer runtime with user-site packages disabled and the locked TSGS build/lib rasterizer first on PYTHONPATH. The O/C/V Stage4 adapter graph is tested through the actual rasterizer boundary. This run does not claim attribute necessity; it only closes or rejects the real smoke pipeline gates.

## Stage4.0-R2A-C4A canonical provenance closure

Stage4.0-R2A-F2F3 formally closed the O/C/V differentiable graph: all three attributes alter actual rasterizer renders, receive finite nonzero gradients, and pass finite-difference directional derivative checks. F2 remains PASS.

The next canonical smoke gate reported the identical metric triple `18.135546 / 0.497695 / 0.497695` for K0, K1, and K2. K0/K1 equality is not suspicious at D0 because MAT0 and MAT1 are optically identical when Js=1. K2 uses a wavy surface and tinted sigma `[0.6,1.2,2.0]`, so exact equality requires provenance closure before carrier insufficiency can be claimed.

Stage4.0-R2A-C4A audits case-specific GT keys, training histories, optimizer parameter changes, checkpoint identity, fresh TEST render hashes, checkpoint reload reproduction, and evaluator case-key resolution. The old canonical result is classified as a provenance bug, then K0/K1/K2 are rerun only in the C4A namespace with frozen carrier, optimizer, loss, iteration limit, patience, and thresholds. No 24-job oracle release run is allowed before C4 closure.

## Stage4.0-R2A-C4B optical observable semantics closure

Stage4.0-R2A-C4A confirmed that the old canonical pipeline had `CANONICAL-JOB-CASE-REUSE`; the identical canonical metrics were invalid. Case-keyed reruns executed 4000 Adam steps for K0/K1/K2 with exact checkpoint reloads. Those runs showed tau-equivalent RGB error near the original threshold while the alpha Elog term was much larger.

Stage4.0-R2A-C4B traces the benchmark GT alpha as `1-exp(-mean(tau_rgb))`, an optical diagnostic rather than a geometry mask. The rasterizer alpha family is generated by Gaussian opacity/kernel alpha composition, and the C4A saved alpha was an RGB-mean diagnostic. These quantities are not mathematically equivalent observables. The cross-semantic alpha loss/metric is retired, and the corrected protocol uses `tau_eq_rgb = -log(clamp(I_rgb,1e-6,1))` from final white-background RGB. The original PSNR and tau thresholds are preserved; no capacity, optimizer, or loss-weight search is introduced.

## AttributeDeformGS final audit freeze

Final decision: `CASE REAL-CANONICAL-CARRIER-INSUFFICIENT`.

The primary AttributeDeformGS method line is stopped because canonical carrier capacity was not established. The corrected K0/K1/K2 canonical rerun preserved the fixed 4096-Gaussian O+C+V carrier, Adam optimizer, learning rate, 4000-iteration limit, TRAIN/TEST split, PSNR threshold, and tau_eq threshold. The corrected metrics were:

- K0: PSNR `21.231329`, median tau_eq Elog `0.178887`
- K1: PSNR `21.231336`, median tau_eq Elog `0.178926`
- K2: PSNR `17.881014`, median tau_eq Elog `0.215660`

The tau_eq criterion passed, but PSNR failed the frozen `>=28 dB` Gate. The capacity diagnosis is `TRAIN-FIT-INSUFFICIENT`, so no R0-R7 dynamic attribute sufficiency comparison is scientifically allowed. The current TSGS-based carrier should not be used for O/C/V necessity claims.

The scientific question itself remains `UNTESTED`, not falsified. A future restart requires a representation-level advance for canonical semi-transparent thin-surface image formation before any attribute necessity study. Do not reopen this line by lowering Gates, changing loss weights, increasing Gaussian count without a representation argument, using a controlled regular-grid carrier, adding RGB opacity/extinction attributes, adding an MLP, or returning to KIOT.

Final archive files:

- `ATTRIBUTEDEFORMGS_FINAL_AUDIT.md`
- `ATTRIBUTEDEFORMGS_VALID_EVIDENCE.md`
- `ATTRIBUTEDEFORMGS_RETIRED_EVIDENCE.md`
- `ATTRIBUTEDEFORMGS_RESTART_CONDITIONS.md`

## Stage5.0 RT-Splatting native carrier gate

Stage5.0 does not repair Stage4. The Stage4 TSGS single-opacity carrier remains stopped, and the AttributeDeformGS scientific question remains `UNTESTED`.

Stage5.0 audits the local RT-Splatting repository before using paper terminology. The local source is `/data/wyh/repos/RT-Splatting`, commit `3f45b3cac4be04db9f3092234666b695991b268a`, branch `main`, remote `https://github.com/sjj118/RT-Splatting.git`. The source contains decoupled persistent transparent/material state tensors including `_occupancy`, `_opacity`, `_transmissivity`, `_roughness`, `_reflectance`, and `_language_feature`. Source tracing confirms geometric occupancy and optical opacity are decoupled: volume rendering uses `occupancy * opacity`, while the surface/deferred pass uses `occupancy` and carries optical/material extras separately.

The canonical observable remains image-equivalent optical depth:

`tau_eq_rgb = -log(clamp(I_rgb,1e-6,1))`.

Raster accumulated alpha is not used as a physical optical-depth target. The canonical Gate remains unchanged: TEST PSNR `>=28 dB` and median TEST tau_eq Elog `<=0.25` for K0/K1/K2.

However, the initial Stage5.0 runtime assembly did not have a compatible local RT runtime. `nvdiffrast.torch` failed because `_nvdiffrast_c` was missing, `diff_surfel_anych` was not installed, clean `PYTHONNOUSERSITE=1` could not import torch/numpy, and local rebuild was not allowed without a verified local toolchain while downloading a new PyTorch/CUDA stack or changing system CUDA was forbidden.

Final Stage5.0 case:

`CASE LOCAL-COMPATIBLE-RUNTIME-NOT-AVAILABLE`

Therefore no native forward, state perturbation, gradient audit, canonical fitting, or Stage5.1 dynamic attribute release is allowed. Under the No Third-Carrier Rule, the project will not search another transparent Gaussian carrier. The primary attribute-deformation line is stopped, and the next main research action returns to RecycleGS Stage1 cross-view geometry reliability detection.

## Stage5.0-R1 RT-Splatting runtime salvage gate

Stage5.0 J1 proved that the local RT-Splatting source contains source-level decoupled persistent transparent/material states. Geometric occupancy and optical opacity are separate tensors and participate differently in native rendering. Stage5.0 stopped at J2 because the initially assembled local runtime could not import critical extensions; that environment STOP is not scientific evidence that the RT carrier lacks canonical capacity. J3 and J4 were never executed.

Stage5.0-R1 performs the final runtime-only salvage Gate. It locks the Stage5.0 evidence, reproduces the exact import failures, searches local Python headers beyond `/usr/include/python3.10/Python.h`, locks the usable torch runtime, audits user-site package resolution, closes `simple_knn` with the torch library path, inventories local `nvdiffrast` artifacts, and audits `diff-surfel-anych` build feasibility. No new PyTorch/CUDA stack is downloaded, no RT equations or CUDA/C++ source are modified, and no canonical training or scientific carrier test is run.

The full local search found a Python3.10-compatible header at `/home/wyh/.conda/envs/rtsplat_legacy2/include/python3.10/Python.h`, while the usable torch runtime remains `/home/wyh/.local/lib/python3.10/site-packages/torch` with torch `2.6.0+cu124`. `simple_knn._C` imports under an explicit controlled `PYTHONPATH` and torch-first `LD_LIBRARY_PATH`. The remaining critical extensions do not close locally: `nvdiffrast.torch` still lacks `_nvdiffrast_c`, `diff_surfel_anych` has source but no binary, and both missing extensions are `BUILD-NOT-FEASIBLE` because `nvcc` is unavailable. Critical imports pass `2/6`, so native RT forward, transparent-state causality, gradients, and sidecar render reproduction are correctly gated as `NOT_EXECUTED_K2A_FAIL`.

Final Stage5.0-R1 case:

`CASE RTSPLAT-LOCAL-RUNTIME-UNRECOVERABLE`

The local RT runtime is not salvaged on this server, RT-native state control is not established, Stage5 canonical J4 remains disallowed, and the AttributeDeformGS hypothesis remains `UNTESTED`. The primary attribute-deformation line is stopped for the current project/server, and the next research action returns to RecycleGS Stage1 cross-view geometry reliability detection.

## Stage5.0-R2 RT-Splatting real local extension build gate

Stage5.0-R1 found a local Python3.10-compatible `Python.h`, the local torch `2.6.0+cu124` headers/libraries, ten local `nvdiffrast` candidates, and the local `diff_surfel_anych` source. However, it classified both missing extensions as not build-feasible because `nvcc` was not found through the default PATH, and it executed zero real extension builds (`0/0`). Stage5.0-R2 explicitly audits `/usr/local/cuda/bin/nvcc` and retires the R1 missing-nvcc classification as PATH-only.

R2 validates `/usr/local/cuda` as a CUDA 12.4 toolkit, confirms Python header ABI compatibility for CPython 3.10, locks torch `2.6.0+cu124`, verifies the visible GPUs as two Tesla V100 devices with compute capability `7.0`, and passes generic compiler probes for `Python.h`, `nvcc`, and torch C++ extension compilation. It then performs real source-unmodified local builds for both RT critical extensions:

- `nvdiffrast`: `BUILD-SUCCESS`, native `_nvdiffrast_c` built under `/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/nvdiffrast/`
- `diff_surfel_anych`: `BUILD-SUCCESS`, native `_C` built under `/data/wyh/DeformTransGS/runtime/rtsplat_stage5_R2_build/diff_surfel_anych/`

The verified R2 runtime imports all six critical modules: torch, `simple_knn._C`, `nvdiffrast.torch`, `diff_surfel_anych`, `scene.gaussian_model`, and `gaussian_renderer`. A real native RT renderer forward executes on CUDA and returns finite nonzero RGB. The minimal native transparent-state causality check confirms `_occupancy`, `_opacity`, and `_transmissivity` are each render-active and gradient-active in the source renderer.

Final Stage5.0-R2 case:

`CASE RTSPLAT-LOCAL-RUNTIME-SALVAGED`

The previous `CASE RTSPLAT-LOCAL-RUNTIME-UNRECOVERABLE` is invalid for the current project state. The AttributeDeformGS hypothesis remains `UNTESTED`, but it is experimentally addressable through the salvaged local RT runtime. The primary attribute-deformation line continues. The next exact action is to resume Stage5 J2a checkpoint sidecar closure, then J3 full native material-state control, then J4 canonical capacity Gate.

## Stage5.0-R3 RT-native state provenance and control gate

Stage5.0-R3 uses the verified Stage5.0-R2 runtime and does not rebuild native extensions. It locks the RT-Splatting source, R2 launcher, native dependency binaries, clean GT root, and local renderer sources. The project-side full-state checkpoint adapter saves all persistent per-Gaussian tensors plus auxiliary module state dictionaries without modifying RT-Splatting source.

Checkpoint provenance passes: all 11 persistent per-Gaussian tensors reload with max error `0`, auxiliary module state is included, and 8 deterministic native renders reproduce with max RGB difference `0`. This repairs the Stage5.0 checkpoint omission at the project side.

The full native state-control audit tests `_occupancy`, `_opacity`, `_transmissivity`, `_roughness`, `_reflectance`, `_language_feature`, `_features_dc`, and `_features_rest`. Four states are render-active and gradient-active in the native renderer: `_occupancy`, `_opacity`, `_transmissivity`, and `_features_dc`. The stricter finite-difference directional derivative Gate passes only `_occupancy` and `_features_dc`; `_opacity` and `_transmissivity` remain just outside the frozen 10% relative-error threshold in this R3 audit. Because `_occupancy` is geometric occupancy and only one non-geometry family (`SH_APPEARANCE`) is fully ATTRIBUTE-CONTROL-VALID, formal J3 fails.

Final Stage5.0-R3 case:

`CASE RTSPLAT-NATIVE-STATE-CONTROL-FAIL`

Canonical K0/K1/K2 training is not executed because J3 is a strict prerequisite. The AttributeDeformGS hypothesis remains `UNTESTED`, Stage5.1 design is not allowed, and the primary attribute-deformation line is stopped for the current project/server. Do not search a third carrier in this project.

## Stage5.0-R3-G1 small-gradient numerical closure

Stage5.0-R3-G1 audits only the numerical validity of the R3 directional-derivative Gate. It does not run canonical training, J4, dynamic attribute release, or any RT source/CUDA/C++ modification.

The audit reproduces the four R3 gradient cases for `_occupancy`, `_opacity`, `_transmissivity`, and `_features_dc`. The original random-unit full-tensor direction is quantitatively under-resolved for the small-gradient states: `_opacity` and `_transmissivity` had finite nonzero render and gradient causality, but the expected random-direction finite-difference numerator at `eps=1e-2` was only about 8 to 9 float32 scalar-loss ULPs in this R3 smoke setup. The original failures were therefore a numerical-resolution failure of the gradient-check instrument, not evidence that the native autograd path was invalid.

G1 keeps the native renderer float32, compares float32 and float64 post-render loss reductions, and uses fixed structured directions independent of gradient values. Structured directions are not unit-normalized because a directional derivative is valid for any fixed direction. Under the repaired, state-independent instrument, all four tested states are AUTOGRAD-VALID:

- `_occupancy`: 3/3 structured directions valid
- `_opacity`: 3/3 structured directions valid
- `_transmissivity`: 3/3 structured directions valid
- `_features_dc`: 3/3 structured directions valid

The repaired ATTRIBUTE-CONTROL-VALID set is `_occupancy`, `_opacity`, `_transmissivity`, and `_features_dc`. The valid non-geometry state families are `OPTICAL_OPACITY`, `TRANSMISSIVITY`, and `SH_APPEARANCE`, so repaired J3 passes.

Final Stage5.0-R3-G1 case:

`CASE RTSPLAT-NATIVE-STATE-CONTROL-RESTORED`

The previous `CASE RTSPLAT-NATIVE-STATE-CONTROL-FAIL` is retired for the current project state. The AttributeDeformGS hypothesis remains `UNTESTED`, but the RT-native state-control Gate is restored. The primary attribute-deformation line continues, and the next exact action is to resume R3 from M4a camera closure through J4 canonical capacity. No Stage5.1 release experiment is allowed until J4 passes.

## Stage 5.0-R3-R1 Canonical Capacity Resume

- Command source: `/data/wyh/新3.md`
- Output: `experiments/stage5_0_R3_R1_canonical_capacity_resume/`
- Scope: resumed only from R3 M4a camera projection closure through J4; R1/R2/R3/G1 were not rerun.
- O0/O1: PASS using the locked R2 launcher and G1 repaired J3 evidence.
- O2: FAIL. Clean GT projects material coordinates directly to 512x512 pixel centers, while the RT camera adapter uses RT-Splatting's perspective `full_proj_transform`.
- O2 numeric error: x/y p99 `1010.4603440466936` / `1010.1276331670308`, x/y max `1149.1377701165968` / `1149.2463285156186`.
- Final CASE: `CASE RTSPLAT-CAMERA-ADAPTER-INVALID`; J4 canonical training was not executed because the protocol mandates STOP on O2 FAIL.

## Stage5.0-R3-C1 Benchmark Camera Semantics Closure

- Command source: `/data/wyh/新4.md`
- Output: `experiments/stage5_0_R3_C1_benchmark_camera_semantics/`
- V1 clean GT semantics: `MATERIAL-GRID-OPTICAL-MAP`; pixel `(x,y)` directly fixes material `(u,v)`, and `camera_id` changes optical path length but not pixel-to-material mapping.
- Original split restored: TEST `0,3,6,9,12,15,18,21`; TRAIN `1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23`. The R3-R1 contiguous split is retired.
- Stage4 `REAL-CANONICAL-CARRIER-INSUFFICIENT` is retired as perspective carrier capacity evidence because Stage4 prediction space is perspective raster while V1 GT is material-grid.
- Perspective Thin-Transmission Benchmark V2 root: `experiments/stage5_0_R3_C1_benchmark_camera_semantics/perspective_clean_gt_v2/`
- V2 selected FoVy: `nan` degrees. P4: `FAIL`.
- Final CASE: `CASE PERSPECTIVE-BENCHMARK-V2-INVALID`.

## Stage5.0-R3-C2 Perspective V2 Camera Framing

- Command source: `/data/wyh/新5.md`
- Output: `experiments/stage5_0_R3_C2_perspective_v2_validity/`
- C1 V2 downstream P4 NaNs are reclassified as `NOT_EXECUTED_INTRINSICS_SELECTION_FAIL`; C1 only proved the exact-old-radius FoV candidate rule failed.
- FoVy is frozen at `75` degrees. Camera orbit directions and original mod3 split are preserved.
- Common radial scale: s_min `1.0893774032592773`, s_frozen `1.09`.
- Repaired P4: `PASS`.
- Final CASE: `CASE PERSPECTIVE-BENCHMARK-V2-READY`.


## Stage5.0-R4 RT-Native Perspective-V2 Canonical Capacity Gate

- Output: `experiments/stage5_0_R4_rtsplat_v2_canonical_capacity/`
- Benchmark: C2-V2 only; V1 is not used for perspective PSNR.
- Final CASE: `CASE RTSPLAT-PERSPECTIVE-V2-CANONICAL-CARRIER-INSUFFICIENT`.
- J4: `FAIL`.
- K0/K1/K2 classifications: `TRAIN-FIT-INSUFFICIENT`, `TRAIN-FIT-INSUFFICIENT`, `TRAIN-FIT-INSUFFICIENT`.


## Stage5.0-R4-O1 Optimization Protocol Closure

- Output: `experiments/stage5_0_R4_O1_optimization_protocol_closure/`
- Original R4 22 dB metrics remain provenance-valid as rendered measurements, but the original carrier-insufficient classification is not accepted.
- O1 found protocol errors: `HARD-500-STEP-CAP,MISSING-P0-P1-FOOTPRINT-DIAGNOSTIC,HARDCODED-GENERIC-LR`.
- Corrected execution decision: `RESTART-FROM-INITIALIZATION`.
- Corrected J4: `FAIL`.
- Final CASE: `CASE RTSPLAT-PERSPECTIVE-V2-CANONICAL-CARRIER-INSUFFICIENT-CONFIRMED`.


## Stage5.0-R4-O1 Optimization Protocol Closure

- Output: `experiments/stage5_0_R4_O1_optimization_protocol_closure/`
- Original R4 22 dB metrics remain provenance-valid as rendered measurements, but the original carrier-insufficient classification is not accepted.
- O1 found protocol errors: `HARD-500-STEP-CAP,MISSING-P0-P1-FOOTPRINT-DIAGNOSTIC,HARDCODED-GENERIC-LR`.
- Corrected execution decision: `RESTART-FROM-INITIALIZATION`.
- Corrected J4: `FAIL`.
- Final CASE: `CASE RTSPLAT-V2-OPTIMIZATION-UNRESOLVED`.


## Stage5.0-R4-O2 Canonical Convergence Closure

- Output: `experiments/stage5_0_R4_O2_convergence_closure/`
- Resumed exact O1 step4000 model and optimizer states; no restart, no LR/loss/state/benchmark changes.
- Final J4: `FAIL`.
- Final CASE: `CASE RTSPLAT-PERSPECTIVE-V2-CONFIRMED-GENERALIZATION-GAP`.
- No further RT carrier rescue rule is active.


## Stage D0 Deformable Optical Transport Feasibility

- AttributeDeformGS per-native-state line remains stopped after the confirmed RT-native V2 K2 novel-view generalization gap.
- New formulation: `DEFORMATION-CONDITIONED OPTICAL TRANSPORT`.
- D0 validates material-point identity, exact affine deformation gradients, local frames, local view directions, and pointwise optical oracle replay on C2-V2.
- C2-V2 is a `KNOWN-LAW-CONTROLLED-BENCHMARK`; MAT1/MAT2 explicitly use `h'=h0/Js`, so Js correlation is built into the GT and is not a discovery.
- D0 Final CASE: `CASE DEFORMABLE-OPTICAL-TRANSPORT-FEASIBILITY-FAIL`.


## Stage D0 Deformable Optical Transport Feasibility

- AttributeDeformGS per-native-state line remains stopped after the confirmed RT-native V2 K2 novel-view generalization gap.
- New formulation: `DEFORMATION-CONDITIONED OPTICAL TRANSPORT`.
- D0 validates material-point identity, exact affine deformation gradients, local frames, local view directions, and pointwise optical oracle replay on C2-V2.
- C2-V2 is a `KNOWN-LAW-CONTROLLED-BENCHMARK`; MAT1/MAT2 explicitly use `h'=h0/Js`, so Js correlation is built into the GT and is not a discovery.
- D0 Final CASE: `CASE DEFORMABLE-OPTICAL-TRANSPORT-FEASIBILITY-PASS`.

## Stage D1-N2 Microstructure Anisotropic Extinction Oracle

- D1-N1 selected material-axis anisotropic extinction as the primary controlled optical mechanism.
- Primary observable: ordinary unpolarized RGB transmitted intensity under controlled illumination.
- Pure birefringence remains outside the ordinary RGB model unless a polarizer/analyzer setup is added.
- D1-N2 does not define optical response directly from Js, principal stretches, or Ct.
- Canonical oriented absorbing microstructures are transported through exact deformation gradient F.
- World-space extinction tensors are assembled from transported absorber orientations.
- Ordinary unpolarized transmission is computed from two absorption eigenmodes in the ray-orthogonal plane.
- Geometric thickness transport is computed independently from transported surface area.
- The optical oracle generator does not read candidate descriptor tables.
- Matched-invariant counterfactuals: PAIR-A same Q1/different Q2-Q3 and PAIR-C same Q2/different Q3.
- Isotropic microstructure is a null control; rigid transformations preserve intrinsic response under identical local-view semantics.
- Conclusions are limited to the controlled microstructure-derived mechanism.

## Stage E0-A Nonrigid Incremental Identifiability Gate

- Original AttributeDeformGS attribute-release line remains stopped.
- Anisotropic optical-transport line remains stopped after frozen finite-bit sensor protocol failure.
- New low-cost candidate Gate: `NONRIGID INCREMENTAL IDENTIFIABILITY`.
- Candidate novelty is not generic motion-helping inverse rendering.
- Required claim is narrower: under matched local-view directions, equal observation budgets, and identical unknown parameters, known nonrigid deformation may provide geometry-optics constraints unavailable from rigid motion alone.
- Stage E0-A is pointwise and does not train a Gaussian model.
- STATIC1, RIGID4, and DEFORM4 receive the same 24 local-view directions.
- Primary comparison: DEFORM4 vs RIGID4.
- Inverse unknowns: two canonical-normal parameters and three shared optical-depth parameters.
- Jacobian Gate runs before any optimization; optimization is forbidden if the Jacobian Gate fails.
- Final CASE: `CASE E0A-PROVENANCE-FAIL`.
- New primary line: `STOP`.

