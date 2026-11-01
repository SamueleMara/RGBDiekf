# Temporal RGB-D implementation and measured validation

Status: **NOT accepted as a complete full-scene reconstruction implementation.**
The opt-in model and exact feasible-component numerical path exist, with the
limitations below. No real-scene temporal reconstruction completed and no
full-scene production job was launched. Geometry preflights are not reconstruction.

Git commit: `e1e663b9618d3c833e63158e65dbbb1edd732003`; dirty working tree deliberately preserved.
Pre-edit status, dirty patch and CLI help are in `reports/temporal_rgbd_baseline/`.
Exact file hashes, line counts and top-level symbol ranges are in
`reports/temporal_rgbd_validation.json`. New files are separate modules; legacy
remains the CLI default. Earlier untracked temporal IEKS modules were preserved
and are not relabeled as this model.

## Actual executed tests

- Latest focused CPU/CUDA suite: **29 tests**, 0 failures,
  0 errors, 0 skips; 16.022 seconds.
- Both modes, both directions, four-frame CPU/CUDA runs include an actual lag
  slide. Separate train.py integration checks both directions and the full
  existing HTML export. These are named tiny fixtures, not real reconstruction.
- Captured pre-edit dirty baseline vs current legacy default: **38
  arrays and 4 NPZ archives bit-for-bit equal**, both directions,
  three frames, including full covariance snapshots. Six preexisting tracked
  non-CLI source files are byte-identical to the captured pre-edit tree.
- Required broad discovery command executed: 316 tests in
  2707.672 seconds; **7 failures remain**.
  See the complete log and failure list below. Later numerical changes were
  rechecked by the focused suite; do not interpret this as a green full suite.

| Numerical witness | Actual maximum absolute discrepancy |
|---|---:|
| CPU dense inverse vs sparse component covariance | 1.1102230246251565e-16 |
| JVP/VJP dot-product identity | 4.440892098500626e-16 |
| Exact lag Schur marginal covariance | 0.0 |
| Two-time analytical vs central finite difference | 2.1201918692925759e-10 |
| Tiny CPU/CUDA mean | 2.220446049250313e-16 |
| Tiny CPU/CUDA covariance | 2.7755575615628914e-16 |

The full physical 6D tiny-pipeline CPU/CUDA covariance discrepancies, known-motion
results and per-check tolerances/durations are in `tests/temporal_test_results.json`.
Tests include both-state and latent-pixel derivatives, source-depth coupling,
shared-reference cross-point covariance, corrected correlated NIS, log-determinant
and covariance identities under lag elimination, exact no-data publication,
conditioning the reference out, source identity, hard support validity, camera
handover and stale references, deterministic ownership, and positive/negative
motion across reference cells. Monte Carlo NIS/NEES checks use a declared known
Gaussian generative model, not the real-scene noise model.

## Implemented mathematics and preserved behavior

See `docs/TEMPORAL_RGBD_IEKF_MATHEMATICAL_CONTRACT.md` for equations and code map.
The state is 6D world XYZ/velocity; original signed-time F and white-acceleration Q
are reused. Both modes call the same first-hit owner, camera-local lifecycle,
strict four-neighbor foreground sampler and two-time temporal operator.

Full mode has one clean RGB-D latent per unique raw key and one sensor likelihood
per key, regardless of reuse. Source XYZ is generated from that same clean depth,
pixel coordinate error and an independent velocity prior; it is not an additional
independent Pxx observation. The source chart is numerically centered on the
recorded canonical mean. Sparse J/JT operate on CUDA/CPU; exact QR solves and
square-root Schur projection retain all induced cross blocks inside feasible
linearized components. Exactly zero blocks may be omitted; no small nonzero
coupling is thresholded away. Current marginals stream across independent
components; full covariance is cached for configured traces and can be refactored
on demand. FEJ priors preserve marginalized evidence; outside-lag references are
unusable, not deterministic old points. No new process or physical movement gates
were added. Simplified covariance is explicitly approximate and conditional.

Reference interpolation cells refresh after nonlinear proposals. Newly touched
sensor pixels are registered once and old sensor factors are kept, so changing
support cannot erase a bad observation to improve cost. Active current-pixel /
owner changes and invalid reference support remain conservatively rejected.
Hard sort and boundary cases are piecewise approximations, with convergence and
invalidity diagnostics; no claim of global multimodal Bayesian inference is made.
No background RGB-D enters a factor. Existing background feasibility violations
are reported rather than repaired by an invented mask force. In full mode, lack
of a direct row is distinct from an indirect coupled marginal update.

A separate import bug was corrected in three forensic drivers: importing them,
or calling their mocked CLI from unittest, must not disable the caller's CUDA.
Their CPU-only command-line behavior remains. No legacy estimator mathematics
was changed to force its tests to pass.

## Real input and resource evidence

All ten actual synthetic scene folders exist, each with 60 frames, eight cameras,
and 59 Hz from `metadata/timing.json:fps`; no manual timing override is necessary.
No scene contains the required separately justified temporal-warp covariance.
Existing sensor MAD / cross-camera mismatch estimates are different quantities.
No invented positive R_warp was substituted. A real three-frame train preflight
returns `MISSING_TEMPORAL_WARP_NOISE` before optimization, as permitted by spec 16.
An explicit user-assumption file is acceptable; laboratory calibration is not
required. Required format: `configs/temporal_warp_noise.schema.json`.

Storage geometry-only preflight processed all **346437** source points and
all eight cameras at frames 0/1/2. CPU/CUDA owner disagreements were
[0, 0, 0], peak allocated CUDA bytes
829992960, total wall time
25.635 seconds. Hardware: NVIDIA GeForce RTX 4070 Laptop GPU, reported device memory
8444575744 bytes.

**Correction of the initial resource estimate:** counting zero-weight neighbors
incorrectly merged the whole nonlinear footprint. That 346,416-point / 276 TB
estimate is withdrawn for the initial linearization. With only exactly nonzero
weights, the preflight finds **143901 components, largest
64 points**. This is unchanged-cloud geometry,
not a post-estimation operator. Components may merge once motion changes weights.
No full inference resource conclusion follows from this preflight. The backend
still uses dense QR within bounded components, generic Python factor orchestration,
and may be slow or hit FULL_RESOURCE_LIMIT after actual updates. Full production
performance and sparse-prior fill have not been validated. Do not call that a
successful result or silently switch to simplified mode.

The Storage source-camera collision is real: point 8444 originates from camera 0,
pixel 151962 = (282,237), z=3.427461862564087 m. Unrestricted rendering there picks
45588, from camera 1/pixel 148711, z=3.4273588334329204 m. Both source identities
remain separate from rendered ownership. The measured three-frame collision
records and images are in the geometry audit; no optimized motion is implied.

## Acceptance status and remaining work

| Contract requirement | Status |
|---|---|
| Legacy default/source preservation and same-seed array parity | PASS on captured dirty-baseline fixture |
| Entire existing suite green | FAIL; list below, no thresholds weakened |
| Shared renderer, foreground validity, camera reference lifecycle | Tested tiny CPU/CUDA; all-point real geometry preflight |
| Full source/noise coupling and true current marginals | PASS on numerical fixtures |
| Actual per-camera full/simplified filtering + lag slide | PASS on tiny fixtures only |
| Actual positive/negative motion and new raw support | PASS on declared toy model only |
| Complete mandatory adversarial/physical calibration coverage | NOT exhaustive / NOT accepted |
| Real 3–11-frame temporal optimization and critical-point drift | NOT RUN: required R_warp missing |
| Real full-scene runtime/covariance/accuracy | NOT RUN / not validated |
| HTML image/3D diagnostics | Implemented; actual geometry HTML rendered in Chrome |
| Source-code snapshots in SSD result folders | Not created |

Full-scene scientific acceptance is therefore NOT satisfied. The missing warp
model and existing-suite failures must be resolved, and real forward/backward
accuracy and component-size evolution measured before promotion. Depth sort is
not physical correspondence; local translation can fail under rotation, occlusion
or silhouettes. Covariances are conditional local Gaussian approximations, not
independently calibrated physical accuracy. The generic factor orchestration is
not a validated high-throughput production CUDA implementation.

## Broad-suite failures (unmodified legacy/archive paths)

- `test_G0a_exact_sequential_required_points_and_covariance (tests.test_f1_definitive_drift_forensics.DefinitiveArchivedDataTests.test_G0a_exact_sequential_required_points_and_covariance)`
- `test_prismatic_backward_chronological (tests.test_image_rigid_joint.PipelineTest.test_prismatic_backward_chronological)`
- `test_prismatic_forward (tests.test_image_rigid_joint.PipelineTest.test_prismatic_forward)`
- `test_revolute_forward (tests.test_image_rigid_joint.PipelineTest.test_revolute_forward)`
- `test_image_iekf_recovers_hinge_without_phase1_motion (tests.test_image_rigid_joint.RigidFromZeroInitTest.test_image_iekf_recovers_hinge_without_phase1_motion)`
- `test_labels_on_textured_scene (tests.test_image_support.SupportLabelsTest.test_labels_on_textured_scene)`
- `test_on_keeps_a_static_scene_static (tests.test_rendered_mask_four_pass.RenderedMaskFourPassTest.test_on_keeps_a_static_scene_static)`

The archived USB covariance-parity gate remains strict. Other failures concern
legacy rigid/joint/support behavior and the legacy rendered-mask static case.
Those paths were not numerically redesigned by this F1/B1 task.

## Reproduction

```bash
conda activate rigeuler3dgs
cd "$HOME/Joint3DGS/RGBDekf"
python -B tools/test_temporal_rgbd.py
python -m unittest discover -s tests -t . -v
bash tools/run_temporal_rgbd_critical.sh --mode full,simplified --dry-run
```

After supplying a justified warp file, the operator command for ALL ten scenes,
ALL frames/cameras and both directions (not pre-cleared as successful full inference):

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --mode full,simplified --direction both \
  --temporal-warp-noise config \
  --temporal-warp-noise-config "/absolute/path/to/approved_temporal_warp_noise.json"
```

The driver defaults to the actual synthetic data root and requested external SSD
results parent, creates a unique timestamped child, logs commands/exit codes, and
never copies a source snapshot. Add `--frame-limit 11 --direction forward` for the
pilot. Full runs have no frame limit; backward then anchors at original frame 59.
Live pages: `SCENE_MODE/free_DIRECTION/temporal/index.html`; completed full viewer:
`SCENE_MODE/debug/index.html`.

Reviewed geometry-only HTML: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_rgbd_validation_rbylqk6a/storage_geometry_preflight/temporal/index.html`.

Report regeneration:

```bash
python tools/report_temporal_rgbd_validation.py \
  --suite-log /absolute/completed_suite.log \
  --geometry-audit /absolute/geometry_preflight/audit.json
```
