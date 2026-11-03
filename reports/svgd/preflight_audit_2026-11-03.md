# SVGD Stage-1 preflight audit — 2026-11-03 (spec §0)

| item | value |
|---|---|
| `git rev-parse HEAD` | `d0b470b75e9ec9181f7561e1a80d2909fb75c7e5` (= audited base of the spec; no diff to reconcile) |
| `git log -1` | `d0b470b… 2026-11-03T15:39:08+01:00 new dense implmentation 4` |
| `git status --short` at start | `?? CODEX_RGBDiekf_SVGD_…_2026-11-03.md`, `?? docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf` (user files, untouched) |
| `python train.py --help` | exit 0 (legacy CLI unchanged; no new flag added to train.py) |
| CUDA | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB (shared machine), torch 2.11.0+cu128, CUDA 12.8; triton 3.6.0 installed (no Triton kernels implemented) |
| theory PDF sha256 | `33ebd5ff…` (`.tex` NOT in the repository: labels mapped to PDF equation numbers by content) |

## Baseline test suite before any change

`python -B -m unittest discover -s tests -t .` (collected before the first new file was written): **441 tests, 7 failures, 2 skipped, 2035 s**.
Pre-existing failures (legacy paths not touched by this work):

- `test_f1_definitive_drift_forensics.DefinitiveArchivedDataTests.test_G0a_exact_sequential_required_points_and_covariance`
- `test_image_rigid_joint.RigidFromZeroInitTest.test_image_iekf_recovers_hinge_without_phase1_motion`
- `test_image_rigid_joint.PipelineTest.test_prismatic_backward_chronological`, `test_prismatic_forward`, `test_revolute_forward`
- `test_image_support.SupportLabelsTest.test_labels_on_textured_scene`
- `test_rendered_mask_four_pass.RenderedMaskFourPassTest.test_on_keeps_a_static_scene_static`

That run rewrote the tracked `reports/image_iekf_dense_covariance_comparison.json` (last-digit roundoff, written by `tests/test_image_iekf.py`); it was restored with `git checkout` so the working tree holds only intended changes.

## Code read (spec §0 item 2)

`scene/{dataset,dataloader,endpoint,config,timing,noise_calibration}.py`, `estimator/{point_kalman_cuda,temporal_noise}.py`, `debug/projective_debug.py`,
`debug/assets/projective_common.js`, `tools/{run_status,qa_projective_debug}.py`, `train.py` (RESULTS_ROOT, STAGE_SOURCES, temporal-rgbd options).
Reused unchanged: `scene.endpoint.samples/backproject` (masked back-projection + covariance), strict masks via `scene.dataloader`, `scene.timing`
(MISSING_ACQUISITION_TIMING), `scene.noise_calibration.temporal_sensor_noise` via `estimator.temporal_noise.temporal_sensor_only`,
`point_kalman_cuda.transition/process_noise/jacobi_eigvalsh`, `debug.projective_debug` writers and assets.

## Existing files changed

Only `tools/run_status.py` (+20 lines): a branch that recognises `run_kind = svgd_support_stage1` runs via their `progress.json`; every other path unchanged.
