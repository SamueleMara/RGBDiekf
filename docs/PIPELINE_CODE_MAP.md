# Pipeline code map: four-pass projective RGB-D IEKF

This map follows the contract `CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md`. Every file the production entry point `train.py` runs is listed with `file:line` references.

The previous rendered-splat pipeline (`tools/legacy_rendered_splat_train.py`, `estimator/point_renderer.py`, `observation_terms.py`, `sparse_iekf.py`, `covariance.py`, `restricted_posterior.py`, `supernodes.py`, `phase1.py`, `rigid_part_iekf.py`, `joint_iekf.py`, `joint_reduction/support.py`, `debug/viewer.py`, `debug/rigeuler_pages.py`) is kept as the labelled LEGACY_RENDERED_SPLAT_BASELINE. No production module imports it (see the import check in `reports/projective_four_pass_cuda_implementation.md`).

```
scene folder | dataset root
 └─ train.py:147 run_scene
     F1/B1  estimator/free_projective_run.py:48 run_free (per direction)  ──► free_{forward,backward}/
     E      estimator/projective_coherence.py:26 run_coherence             ──► coherence_{forward,backward}/{pinned_raw,measurement_supported}/
     F2/B2  estimator/rigid_kalman_cuda.py:154 run_rigid                   ──► rigid_{forward,backward}/
     G      joint_reduction/joint_projective.py:169 run_joint, :206 compare_directions ──► joint/{forward,backward}/, bidirectional/
            joint_reduction/support_projective.py:90 run_support           ──► joint/<dir>/support/
     H      debug/projective_debug.py:274 write_debug                      ──► debug/index.html (+ pages, data/, img/)
            tools/evaluate_projective_gt.py (separate process)            ──► evaluation_gt/
            tools/qa_projective_debug.py (separate process)               ──► debug/qa/
```

## Sensor geometry and measurement (Stage A)

| File | Role |
|---|---|
| `estimator/projective_camera.py:30` `CameraRig` | World→camera `T_cw` (persisted together with `T_wc`). `:62` `project` (all cameras; `jacobian=False` for visibility), `project_rows` (per-row `u, v, Z, J_u = J_π R, J_Z`), `ray`, `backproject` (rank-one ray-depth identity). |
| `estimator/projective_sensor.py:51` `MeasuredFrame` | One acquisition of all cameras on the GPU (RGB, metric depth, binary mask, raw validity) with source paths and sha256. |
| `estimator/projective_sensor.py:86` `sample` | Bilinear sampling of the MEASURED fields with the exact interpolant gradient. Interpolation variance `Σ w² σ²`. Reason codes (outside, mask, invalid depth, depth break). `:129` `sample_cpu_oracle` is the test oracle. |
| `estimator/projective_visibility.py:20` `visibility` | Per-pixel front-most point table (reuse control, self-occlusion with prior σ_z). `:67` `apply_frozen_assignment` implements `--pixel-reuse-assignment per-frame`. |
| `estimator/projective_terms.py:43` `build_terms` | `r = [a − I; Z − D]`, `H_x = [∇I J_u; ∇D J_u − J_Z]`, diagonal R, and the external-occlusion rule (measured surface in front beyond `n σ` of `S_zz`). `:65` `h_value` is the finite-difference oracle. |

## Free 6D IEKF (Stages B, C)

| File | Role |
|---|---|
| `estimator/point_kalman_cuda.py:38` `transition`, `:43` `process_noise`, `:50` `predict` | `F(Δt)`; signed white-acceleration `Q`. |
| `estimator/point_kalman_cuda.py:139` `PointMeasurement` | One linearization of all points: visibility, terms, deterministic segment sums of `HᵀR⁻¹H` and `HᵀR⁻¹r` per point, and point-range chunks (`--point-chunk-size`). |
| `estimator/point_kalman_cuda.py:200` `update_points` | Fixed-prior IEKF in information form (6×6 per point); per-point stopping; NIS at the prior. The posterior is `(Λ⁻ + HᵀR⁻¹H)⁻¹` at the final linearization; a point with no valid row keeps the prediction. Traces cover the explicit gain, the Joseph cross-check and finite-difference H. |
| `estimator/point_kalman_cuda.py:72` `jacobi_eigvalsh`, `:101` `spd_extreme_eigs` | Device eigen diagnostics (cuSOLVER's batched syev is unusable at 10⁵ matrices). |
| `estimator/free_projective_run.py:48` `run_free` | Endpoint catalogue (`scene/endpoint.py`), then subsampling, prior, the frame loop and memory maps in chronological order, with `numeric/frame_XXXX.json`, `trace_frame_XXXX.json` and `timeline.csv`. |
| `scene/canonical_subsample.py:72` `subsample`, `:41` `boundary_priority` | Deterministic factor; voxel-stratified selection of existing ids; diagnostics. |

## Coherence (Stage E)

| File | Role |
|---|---|
| `estimator/projective_coherence.py:26` `run_coherence` | Runs the unchanged pinned extractor (`estimator/motion_coherence_support.py`, `estimator/supported_coherence.py`, `rigeuler_ref/coherence_extraction.py`) twice, with `pinned_raw` and `measurement_supported` weights. Keeps seed and recruited nodes separate. Verdicts: RESOLVED, NO_IDENTIFIABLE_MOVING_CORE or UNRESOLVED_FULL_GRAPH_SELECTED. |

## Rigid 12D IEKF (Stage F)

| File | Role |
|---|---|
| `estimator/se3.py` | `[v, ω]` spatial-left convention: `exp` (matrix exponential), `log`, `Ad`, `ad`, `left_jacobian` (series), `point_jacobian` `B = [I, −[x]×]`. |
| `estimator/rigid_kalman_cuda.py:37` `rigid_predict` | `T⁻ = Exp(Δt ξ) T`, `F = [[Ad_E, Δt J_l], [0, I]]`, signed `Q`. |
| `estimator/rigid_kalman_cuda.py:47` `chain_measurement` | The free filter's projective terms chained by `G = ∂x/∂state` (rigid: `B`; joint: `∂x/∂q`). Static-compatible points act as occluders only. |
| `estimator/rigid_kalman_cuda.py:97` `rigid_update` | Fixed-prior Lie IEKF with `e = [Log(T T⁻⁻¹); ξ − ξ⁻]` and `L = blkdiag(J_l(e)⁻¹, I)`. Final-chart covariance; trace with the local-chart Joseph cross-check. |
| `estimator/rigid_kalman_cuda.py:139` `rigidity_check`, `:154` `run_rigid` | Pairwise invariance; the pass lifecycle; SKIPPED_UNRESOLVED_COHERENCE. |

## Joint, support, bidirectional (Stage G)

| File | Role |
|---|---|
| `joint_reduction/joint_projective.py:52` `reduce_and_decide` | Pinned `reduce_trajectory` (fed in processing order, anchor first). χ² static / prismatic / revolute adequacy on the filtered pose marginals, excluding the anchor, plus the nested likelihood-ratio test. |
| `joint_reduction/joint_projective.py:106` `axis_constancy`, `:130` `q_filter`, `:206` `compare_directions` | Per-increment screw axis vs the fixed line; the 2-state projective IEKF; the forward/backward comparison (never fused). |
| `joint_reduction/support_projective.py:90` `run_support` | Static vs joint-transport measured costs per frame; vectorized HAC one-sided tests (`:34`); BH-FDR (`:62`). |

## Debug, evaluation, QA, profiling (Stage H, Gate D)

| File | Role |
|---|---|
| `debug/projective_debug.py:274` `write_debug` | Exports `data/*.js` and copies source RGB/mask PNGs byte for byte (sha256), with depth colour maps. |
| `debug/assets/projective_*.html`, `projective_common.js` | Pages `free_*_3d`, `free_*_rgbd`, `coherence_*`, `rigid_*_3d`, `rigid_rgbd`, `joint_*`, `bidirectional`, `numerical_diagnostics`, `index`. |
| `tools/qa_projective_debug.py` | Headless Chrome: screenshots, underlay pixel equality, marker = π(T_cw x) in NumPy, sliders and selectors, canonical labels. |
| `tools/evaluate_projective_gt.py` | Post-hoc ground-truth metrics under `evaluation_gt/`. |
| `tools/profile_rgbd_projective_cuda.py` | Stage timings and memory per factor and precision; kernel table. |

## Tests

`tests/test_projective_camera.py`, `test_projective_measurement.py`, `test_point_kalman_cuda.py`, `test_rigid_kalman_cuda.py`, `test_canonical_subsample.py`, `test_joint_projective.py`, `test_four_passes.py`, and `test_coherence_parity.py` (unchanged extractor). Run with `python -m unittest discover -s tests -t .`.

## Opt-in depth-sorted temporal RGB-D F1/B1 (2026-10-30)

- `train.py`: explicit observation model/mode/noise/lag flags; legacy default;
  per-mode manifests, source hashes and incompatible-resume rejection.
- `estimator/temporal_zbuffer.py`: shared torch CPU/CUDA stable first-hit renderer
  and independent NumPy oracle. No measured appearance/depth chooses an owner.
- `estimator/temporal_reference.py`: camera-local accepted references, source
  identity preservation, bootstrap and processing-order age.
- `estimator/temporal_rgbd_terms.py`: common hard-valid raw keys, bilinear weights,
  temporal prediction, both state Jacobians and clean-pixel derivatives.
- `estimator/temporal_noise.py`: separate warp covariance and provenance; no
  inference of temporal warp noise from unrelated cross-camera residuals.
- `estimator/temporal_simplified_iekf.py`: fixed-reference conditional 6D IEKF.
- `estimator/temporal_full_graph.py`: unique pixel registry, coupled source
  backprojection, sparse J/JT, connected components, exact feasible-component
  Gaussian solve/covariance, correlated innovation and complete FEJ Schur prior.
- `estimator/temporal_run.py`: signed-time F1/B1, preserved output schema, graph
  preflight, per-camera diagnostics, selected cross blocks, live debug output.
- `debug/temporal_debug.py`: file-compatible live image/3D/convergence dashboard;
  `debug/projective_debug.py` still supplies the full existing viewer.
- `tools/test_temporal_rgbd.py`: machine-readable tiny CPU/CUDA gate results;
  `tools/verify_temporal_legacy_parity.py` reconstructs the captured dirty baseline
  in /tmp and compares every legacy array/full covariance snapshot.
- `tools/audit_temporal_inputs.py`: clearly labeled geometry/resource preflight,
  not reconstruction; can inspect actual entire point clouds without a warp R.
- `tools/run_temporal_rgbd_critical.sh` / `.py`: operator-run all-scene driver.
- `tools/evaluate_temporal_rgbd.py` / `compare_temporal_modes.py`: separate
  diagnostics and opt-in post-hoc GT; never imported by the estimator.

Limitations and implementation equations are in
`TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md`. Full-scale temporal estimation is
not validated; the full-scene component sizes after actual state updates and their computational
costs remain unvalidated. No simplified fallback occurs.

Temporal next-actions validation tools:

- `tools/calibrate_temporal_warp_noise.py`: separate GT-free identifiability study;
  per-camera conditional sensor estimates from calibration-block frames only,
  robust 4x4 candidate differences and disjoint held-out temporal blocks. No
  usable warp covariance is emitted when discrepancy is not identifiable.
- `tools/audit_temporal_regressions.py`: isolated captured pre-edit/active runs of
  the seven failures, with complete assertion hashes and preserved tracebacks.
- `tools/audit_temporal_saved_linearizations.py`: independent SciPy dense QR on
  actual saved bounded components, cached real CPU/CUDA geometry/warp parity.
- `tools/evaluate_temporal_points_gt.py`: post-inference only, original-ID motion
  errors and conditional NEES; never imported by inference or calibration.
- `tools/qa_temporal_debug.py`: browser checks of actual estimated point data,
  plotted trajectory/speed/covariance pixels, 3D frame and camera controls.
- `tools/report_temporal_next_validation.py`: seven required reports and the
  actual-results HTML index; incomplete runs and focal approximations stay explicit.
