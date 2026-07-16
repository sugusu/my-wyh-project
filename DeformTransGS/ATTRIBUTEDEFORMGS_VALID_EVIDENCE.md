# AttributeDeformGS Valid Evidence

Valid retained evidence count: 13

1. `ANALYTIC-KIOT-SUPPORTED` on a controlled carrier.
2. `KIOT-CONTROLLED-CARRIER-ONLY`.
3. Clean independently validated thin-surface optical GT benchmark.
4. Exact deformation `Js` implementation validated against triangle area ratios.
5. Verified TSGS rasterizer runtime with real opacity and `colors_precomp` gradients.
6. Real differentiable O/C/V adapter.
7. O/C/V forward perturbation causality.
8. O/C/V finite nonzero autograd gradients.
9. O/C/V directional derivative validation.
10. `CANONICAL-JOB-CASE-REUSE` provenance bug discovery and repair.
11. GT optical diagnostic alpha and raster accumulated alpha are mathematically non-equivalent.
12. Image-equivalent optical depth, `tau_eq_rgb = -log(clamp(I_rgb,1e-6,1))`, as a representation-independent image-domain optical observable.
13. `REAL-CANONICAL-CARRIER-INSUFFICIENT`.

