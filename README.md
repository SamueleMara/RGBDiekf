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
python train.py --scene <scene folder | dataset root> --output <new folder>
```

Common options:
- `--pcd-subsample-factor 4`: deterministic endpoint subsampling; images, cameras and frames are never reduced.
- `--debug none|light|full`
- `--direction forward|backward|both`
- `--stop-after free|coherence|rigid|joint|support`
- `--stage S --resume-from RUN`: reuse earlier stages read-only, after hash checks.

`python train.py --help` lists everything. `tools/run_rgbd_projective_pipeline.py` is the contract's name for the same entry point.

Outputs of one run:
- `free_{forward,backward}/`, `coherence_*/`, `rigid_*/`, `joint/{forward,backward}/` (with `support/`), `bidirectional/`;
- `debug/index.html`, `performance/`, `evaluation_gt/`;
- `run_manifest.json` and `result.json`.

## Layout

| Folder | Contents |
|---|---|
| `train.py` | four-pass driver (stages, resume, dataset roots) |
| `scene/` | dataset loading, timing, calibration, noise configuration, endpoint catalogue, `canonical_subsample.py` |
| `estimator/` | `projective_camera`, `projective_sensor`, `projective_visibility`, `projective_terms`, `point_kalman_cuda`, `free_projective_run`, `projective_coherence`, `se3`, `rigid_kalman_cuda`; earlier rendered-splat modules kept as a labelled legacy baseline |
| `joint_reduction/` | `joint_projective` (reduction, decision, `(q, q̇)` IEKF, F/B comparison), `support_projective` |
| `debug/` | `projective_debug.py` and `assets/projective_*.html` (point-only measured RGB-D pages) |
| `rigeuler_ref/` | vendored pinned RigEuler3DGS sources (coherence extractor, joint reduction) |
| `tools/` | `evaluate_projective_gt.py`, `qa_projective_debug.py`, `profile_rgbd_projective_cuda.py`, audits; `legacy_rendered_splat_train.py` (LEGACY_RENDERED_SPLAT_BASELINE) |
| `tests/` | `python -m unittest discover -s tests -t .` |
| `configs/` | `projective_rgbd.json` (declared policy), `example_uncalibrated.json` (UNCALIBRATED noise), `image_joint_policy.json` (alpha) |
| `docs/`, `reports/` | code map, specifications, audits and validation reports |
| `attic/` | earlier pipelines, ignored by git |
