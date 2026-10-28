# RGBDekf

Four independent directional RGB-D Kalman filters for articulated-object discovery
(`CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md`):

| Pass | Canonical cloud | Order | State |
|---|---|---|---|
| F1 free forward | measured frame-0 points | 0 → T | one independent 6D `[x, v]` IEKF per original point |
| B1 free backward | measured frame-T points | T → 0 | one independent 6D `[x, v]` IEKF per original point |
| F2 rigid forward | frame-0 moving core (coherence of F1) | 0 → T | one shared SE(3) pose + twist (12D) |
| B2 rigid backward | frame-T moving core (coherence of B1) | T → 0 | one shared SE(3) pose + twist (12D) |

The only observations are the actual masked, calibrated RGB-D frames, sampled at the projections of the transported endpoint points (`r = [a − I(u,v); Z − D(u,v)]`, bilinear, exact interpolant gradient). Every correction is a Kalman correction.

The pipeline uses none of the following: splatting, rendering, image energy, trust region, group tangent, point correspondence, or ground truth during inference.

Coherence (the pinned RigEuler 192-node extractor), joint reduction and decision, the optional `(q, q̇)` IEKF and support labelling run downstream.

**Status:** see [reports/projective_four_pass_cuda_implementation.md](reports/projective_four_pass_cuda_implementation.md), which lists what is implemented, tested, passing and failing.

## Run

Everything, with debug pages, ground-truth evaluation (separate process) and page QA:

```bash
python train.py --scene <scene folder | dataset root> [--output <new folder>]
```

Results and debug pages go to a new folder on the external SSD, by default `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/projective_<scene>_<YYYYmmdd_HHMM>` (`RESULTS_ROOT` in `train.py`). An output folder inside the code tree is refused. No code is copied into outputs: `run_manifest.json` records the git commit and the sha256 of every source file.

Common options:
- `--pcd-subsample-factor 4`: deterministic endpoint subsampling; images, cameras and frames are never reduced.
- `--debug none|light|full`
- `--direction forward|backward|both`
- `--stop-after free|coherence|rigid|joint|support`
- `--stage S --resume-from RUN`: reuse earlier stages read-only, after hash checks.
- `--stage debug --resume-from RUN`: reuse every inference stage and rebuild only the debug pages, GT evaluation and page QA into a new folder.
- `--sensor-noise measured|config` (default `measured`): the measurement covariance R is measured from the scene's own frames (`scene/noise_calibration.py`, written to `noise_calibration.json`). `config` uses the `--noise-config` values.
- `--velocity-prior-speed99 V`: a declared physical speed (m/s) that encloses 99% of the expected point speeds. It sets the free-pass P_vv to (V/3.368)² I. Without it, the uncalibrated 1 (m/s)² per axis is used.
- `--policy-override key=value ...`: one-variable ablations of `configs/projective_rgbd.json`. For example, `update_acceptance=final_iterate row_innovation_gate=depth_front_only final_visibility=frozen` restores the filter as it was before the 2026-10-28 audit fixes.

Free-filter safeguards (audit `RGBDiekf_static_point_explosion_code_audit_2026-10-28.md`, see [reports/static_point_explosion_fixes.md](reports/static_point_explosion_fixes.md)):
- Each row is gated on its innovation, linearized back to the prior, with a χ² test on RGB and depth together.
- Each point's result is accepted only if its posterior cost is consistent with the actual samples. Otherwise the last consistent iterate is used; if there is none, the point is prediction-only for that frame.
- Self-occlusion is rechecked at the accepted state.
- NaN-safe information sums.
- The per-point arrays `update_acceptance`, `posterior_cost` and `posterior_cost_dof` record each decision. The numeric JSON of every frame lists the largest accepted corrections with their ids, ready for `--debug-point-ids`.

Debug pages follow the RigEuler3DGS style (dark scene, side panel). The free 3D page shows the fixed start cloud (grey), the estimated cloud at the slider frame (blue) and tracked points with trails. Tracked points are chosen from estimates only: the coherence moving core, a spread over the cloud, and the largest displacements (off by default).

`python train.py --help` lists everything. `tools/run_rgbd_projective_pipeline.py` is the contract's name for the same entry point.

Outputs of one run:
- `free_{forward,backward}/`, `coherence_*/`, `rigid_*/`, `joint/{forward,backward}/` (with `support/`), `bidirectional/`;
- `debug/index.html`, `performance/`, `evaluation_gt/`;
- `run_manifest.json` and `result.json`.

## Layout

| Folder | Contents |
|---|---|
| `train.py` | four-pass driver (stages, resume, dataset roots) |
| `scene/` | dataset loading, timing, calibration, noise configuration, `noise_calibration.py` (measured R), endpoint catalogue, `canonical_subsample.py` |
| `estimator/` | `projective_camera`, `projective_sensor`, `projective_visibility`, `projective_terms`, `point_kalman_cuda`, `free_projective_run`, `projective_coherence`, `se3`, `rigid_kalman_cuda`; earlier rendered-splat modules kept as a labelled legacy baseline |
| `joint_reduction/` | `joint_projective` (reduction, decision, `(q, q̇)` IEKF, F/B comparison), `support_projective` |
| `debug/` | `projective_debug.py` and `assets/projective_*.html` (point-only measured RGB-D pages) |
| `rigeuler_ref/` | vendored pinned RigEuler3DGS sources (coherence extractor, joint reduction) |
| `tools/` | `evaluate_projective_gt.py`, `qa_projective_debug.py`, `profile_rgbd_projective_cuda.py`, audits; `legacy_rendered_splat_train.py` (LEGACY_RENDERED_SPLAT_BASELINE) |
| `tests/` | `python -m unittest discover -s tests -t .` |
| `configs/` | `projective_rgbd.json` (declared policy), `example_uncalibrated.json` (UNCALIBRATED noise), `image_joint_policy.json` (alpha) |
| `docs/`, `reports/` | code map, specifications, audits and validation reports |
| `attic/` | earlier pipelines, ignored by git |
