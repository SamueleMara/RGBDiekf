# CODEX — NEXT ACTIONS: UNBLOCK TEMPORAL RGB-D EXPERIMENTS WITHOUT FABRICATING WARP NOISE

**Priority:** P0. **Repository:** `/home/samuelemara/Joint3DGS/RGBDekf` (use the actual current working tree; do not assume GitHub `main` includes the new uncommitted modules). **Reference contract:** `RGBDiekf_Codex_Depth_Sorted_Temporal_RGBD_Full_FixedLag_IEKF.md` (or the local copy provided to you). **Observed implementation report:** `reports/temporal_rgbd_implementation.md`.

## Current factual baseline, not to be overstated

- New CPU/CUDA checks: **29 pass**; broad suite: **316 total, 7 failures, 2 skips**. The 7 failures have *not* yet been classified as pre-existing or introduced by this change.
- GPU/runtime, estimation over actual synthetic sequences, and successful full vs. simplified trajectories are **not validated**.
- The existing all-scene driver finds ten synthetic scenes. Attempts to run estimation stop at `MISSING_TEMPORAL_WARP_NOISE`; the supplied `/absolute/path/to/approved_temporal_warp_noise.json` is only a placeholder, not a real file.
- The HTML screenshot for Storage is a **geometry preflight**, not a temporal IEKF estimate. Do not present it as evidence of successful tracking.
- User expects **working commands, real synthetic inference and causal debugging**, not another iteration of tests that only exercise isolated components.

## Hard prohibitions

1. **Do not fill the JSON with invented RGB/depth covariance values** or call assumed values calibrated. Do not silently interpret sensor MAD, cross-camera discrepancy, or GT motion error as temporal-warp process noise.
2. **No GT, moving-part labels, joint values, object poses, synthetic segmentation labels, or evaluation output in any inference or noise-calibration path.** GT may be used strictly after inference for accuracy reporting.
3. Do not relax foreground masks. No background RGB/depth samples in either current or reference measurements. Avoid any reference bilinear cell that mixes invalid, different-surface, or occluder pixels.
4. No velocity clipping, world-coordinate drift clamps, arbitrary depth thresholds or static-point labels to make metrics look better.
5. Do not replace the full fixed-lag solver by the simplified estimator; both must remain distinct and selectable. Keep the original projective pipeline unchanged as the control.
6. Preserve original canonical point indices, camera calibrations, source-pixel provenance, 60 frames, all 8 cameras and measured timestamps (the synthetic dataset metadata reports **59 Hz**; do not assume 25 Hz).
7. Do not turn a prediction-only/no-reference trajectory into a 'successful tracking' metric. Disappearing measurements is not a victory.
8. Do not start a blanket 10-scene × 60-frame × 2-mode × 2-direction run before passing the bounded pilots and profiling checks below. Never overwrite previous results.

## Phase 0 — Audit the actual working tree before editing

1. Read `reports/temporal_rgbd_implementation.md`, `configs/temporal_warp_noise.schema.json`, `tools/run_temporal_rgbd_critical.sh`, `tools/test_temporal_rgbd.py`, `train.py`, `estimator/free_projective_run.py`, `estimator/point_kalman_cuda.py`, `estimator/projective_visibility.py`, `estimator/projective_sensor.py`, `debug/temporal_debug.py`, and the newly introduced temporal estimator/reference/noise modules. Use `git diff --stat`, `git diff --check`, `git status --short`, and `rg` to find the real local module/function names. **Do not reset, checkout, overwrite or commit without authorization.**
2. Produce an explicit implementation map: CLI → scene loader → source-identity initialization → depth-sorted owner map → per-camera reference → RGB-D temporal factor → full lag QR + Schur marginalization / simplified 6×6 update → output arrays and viewer.
3. Audit the 7 broad-suite failures with exact names and tracebacks. Run a baseline comparison in an isolated clean worktree **only if it can be done safely without modifying the active tree**. Classify each failure: `PRE_EXISTING`, `INTRODUCED`, `ENVIRONMENT`, `UNKNOWN`. Fix introduced regressions surgically. Report the seven; do not dismiss them generically as legacy.
4. Verify the new modules are listed in `train.py` `STAGE_SOURCES` and resume/manifest hashes. Mark the actual mode, noise provenance and observation/reference semantics. Do not leave `renderer_used=false` or `correspondences_used=false` if the new mode renders an index map and uses temporal correspondence.

## Phase 1 — Separate statistical quantities. This is the immediate blocker.

The temporal observation, for each current first-depth owner `i = kappa(c,t,p)` with valid same-camera reference at frame `tau`, is

\[
q=p-\pi_c(x_{i,t})+\pi_c(x_{i,\tau}),\quad
r=\begin{bmatrix}
I_{c,t}(p)-I_{c,\tau}(q)\\
D_{c,t}(p)-D_{c,\tau}(q)-[Z_c(x_{i,t})-Z_c(x_{i,\tau})]
\end{bmatrix}.
\]

The full model **must distinguish**:

- `R_current`: uncertainty of the current measured RGB-D observation, from the existing sensor noise model.
- `R_reference`: uncertainty of the camera-specific reference RGB-D samples, including covariance from bilinear interpolation and any shared raw reference pixel. If the reference pixel is a latent variable in the full graph, its observation is introduced **once**, not again as an independent residual each subsequent frame.
- `P_reference`, `P_current`, `P_cross`: reference/current point-state uncertainty and propagated cross-covariance; included in the full lag graph and its marginalization. In simplified mode, document what is conditioned upon or discarded.
- `R_warp_model`: **additional conditional error** from the local translational RGB-D warp assumption (surface rotation, nonrigid patch, imperfect same-surface model), distinct from sensor/reference/state uncertainty.

A raw empirical residual covariance combines these quantities and association failures. **Do not add it as `R_warp_model` while also including `R_current`, `R_reference` and the lag covariance: that double counts uncertainty.**

Audit what `covariance_rgbd` currently means in the implementation and schema. If it currently represents the entire observation covariance, first refactor/namespacify the distinction, preserving any old config interpretation through an explicit versioned configuration and manifest; never silently change units or semantics.

### P1A — A mathematically explicit ideal-warp diagnostic mode (unblocks a genuine inference run)

Implement a separately named *experimental* option, e.g. `--temporal-warp-noise ideal` or `--temporal-warp-model-error zero`, which sets the **additional** conditional model-discrepancy covariance to

\[
R_{\mathrm{warp\_model}}=0_{4\times4}\quad\text{(positive semidefinite)}.
\]

This is **not zero total measurement noise**: current and reference sensor noise and the full joint state/measurement uncertainty remain active. Confirm the full innovation covariance remains SPD; otherwise fail loudly with a numerical diagnosis. For the simplified estimator, include its declared `R_current + J_ref R_reference J_ref^T` and clearly state the ignored cross-time/state covariance. Do not add a tiny arbitrary diagonal to make a matrix SPD. If the full model currently cannot distinguish sensor vs. model discrepancy, fix that first.

This mode is labelled `IDEAL_WARP_ASSUMPTION_UNCALIBRATED` everywhere (CLI, manifest, reports, plots). It is suitable for exercising code and diagnosing feasibility, **not** for claiming probabilistically calibrated performance or validating the noise model. Keep strict `config` schema validation; allow PSD zero only for the separate additive model term, not for the total covariance.

### P1B — Actual data-based model-discrepancy estimation (separately validated)

Implement a reproducible calibration analysis (e.g. `tools/calibrate_temporal_warp_noise.py`) using the **measured RGB-D sequence only** and explicitly stated assumptions. Do not estimate warp noise from cross-camera color discrepancies. Do not use the filter's own unconstrained large corrections as ground-truth correspondences.

Required method/report:

1. Determine RGB encoding [0,1], metric depth [m], per-camera sensor covariance and interpolation covariance from real source pixels. Source RGB-D neighborhoods and same-surface validity are checked with existing `projective_sensor.sample()` logic.
2. Propose *independent, geometrically consistent temporal correspondence support*, preferably acquired using source-anchored and first-hit visibility, a forward/backward consistency check and measured-depth agreement propagated with sensor uncertainty. State precisely why the correspondence selection itself is not circular. If such independently defensible matches do not exist, explicitly report `MODEL_DISCREPANCY_NOT_IDENTIFIABLE` instead of fabricating a calibration.
3. Collect candidate same-camera RGB-D warp residuals from that support with per-frame/camera/surface-depth/gradient breakdown. Estimate a **robust full 4×4 residual covariance**, report effective sample counts and RGB/depth correlations. Compare the empirical covariance with the analytically propagated contributions of current sensor, warped reference sensor and reference/current state error; estimate only the **positive-semidefinite unexplained component** when identifiable. Do not clip unexplained negative eigenvalues and call the result validated; report mismatch/insufficient evidence and the PSD-constrained fit explicitly.
4. Split calibration support from held-out validation support by temporal blocks and/or cameras. Never choose `R_warp` by minimizing final GT tracking error. Report held-out normalized residuals, NIS coverage, dependence on depth, image gradient, mask edge, camera and warp magnitude; a single 4×4 global covariance may be insufficient.
5. Write a schema-valid, versioned calibration JSON with complete `assumption_class`, `method`, `units`, source frame/camera counts, held-out checks, model version, data hash, source-code hash and any reasons calibration is **conditional** rather than approved.
6. Optionally add `--temporal-warp-noise estimate` that executes this routine in a **separate calibration stage** and transparently fails when the assumptions are not supported. Do not allow run-time observations of the evaluation frames to tune parameters in a way that leaks testing information without declaring the calibration protocol.

Do not block P1A on completion of P1B. Both are required workstreams; P1A provides useful numerical/systems validation now.

## Phase 2 — Ensure the 'full' mode really preserves two-time dependencies

Before trusting any full-mode trajectory, verify and test:

1. For RGB and depth, finite-difference all Jacobians of `r_t(x_t,x_tau,y_t,y_tau)` with respect to **both** states and to the reference/current raw RGB-D samples, including pixel-interpolation weights and sign conventions. Explicitly test with nonzero depth gradient and a nonzero difference in camera depth.
2. The lag graph contains all active states and shared reference observations that influence factors in the current window. Same raw sensor pixel entering several interpolated factors **must generate nonzero off-diagonal factor covariance** or be represented once as a shared latent variable. Merely using a 4×4 block-diagonal R per factor is not equivalent.
3. The reference state and original source RGB-D observation are correlated at initialization. Audit both source back-projection and subsequent reference refresh: no duplicate likelihood for the same raw pixel and no reset of `P_{t,tau}`.
4. Marginalize the oldest state with exact QR/Schur in independently small dense toy examples. Compare full posterior mean and covariance to a joint batch Gaussian oracle for 2–4 frames, multiple camera references, pixel overlap and camera handover. Test the limiting case `R_warp_model=0` with strictly positive sensor noise.
5. Explicitly report approximations for cross-point shared-pixel ownership and lag component splitting. Separate `EXACT_WITHIN_COMPONENT` from `GLOBAL_APPROXIMATION`. **A bounded QR solve is not globally exact if splitting/deleting components loses coupling.**
6. Verify repeated use of an unchanged reference does not produce unjustified posterior variance collapse. Test one static point with noisy repeated reference samples, and another point acquiring camera B after losing camera A.
7. A reference for a newly visible camera is initialized **after** final acceptance of the current step; no same-frame self-measurement from that newly initialized reference. Older references are used only if prior owner and 2×2 source sample support are valid; never interpolate occluder/background into a reference.
8. For no eligible rows, full and simplified modes return the exact motion prediction and propagated covariance; no pseudo-update from missing pixels. Uncovered foreground counts remain explicit.

## Phase 3 — End-to-end integrity and causal traces

1. Create a deterministic **CPU dense oracle and CUDA parity** for the first-hit ownership, sampled RGB-D warp, analytic JVP/VJP, numerical residual, full QR Schur result and simplified information update, including 1e-12 pixel perturbations and nearly tied depth sorting.
2. At frame 0, verify the original source `(camera, pixel) → point_id` mapping exactly *before* the unrestricted cross-camera z-buffer. Log where a different camera's point supersedes it, including depth differences. Do not force source ownership at later frames, but report any frame-0 ambiguity; incorrect source initialization is a legitimate source of failure.
3. No accidental leakage of background RGB/depth, mask-invalid bilinear neighbors, NaN, infinite depth, or image samples from a different owner. Track invalid row reason without poisoning covariance; confirm hard-mask consistency.
4. The first-hit owner map is frozen for an **individual linear solve**, reprojected and resorted after accepted nonlinear changes. Do not conflate this with existing `--pixel-reuse-assignment per-frame` semantics. Accept proposed steps against the same observed-pixel objective, accounting for uncovered pixels, not merely the sum over whichever measurement rows survive.
5. Log state-position/velocity changes, reference age, previous/current camera owner, image coordinates, warped reference pixel, measured 2×2 RGB-D values, gradient, temporal innovation, Jacobians, sensor/reference/warp covariance blocks, cross-time covariance, information eigenvalues, covariance condition number, QR rank, component size, convergence and objective before/after. **Keep default storage bounded:** summary CSV/JSON each frame, full row/iteration records only for designated point IDs and frames. No automatic full-frame per-iteration PNG dumps.
6. Mandatory exact original point IDs when the canonical catalogue is unsampled: Storage `8444`, `45588`, `49199`; USB `67827`, `87290`, `85696`; Blade `14104`, `27016`. Confirm IDs against actual `canonical.npz` and `subsampled_to_full_index`; never silently identify a different point after subsampling.
7. Viewer: clearly distinguish `GEOMETRY_PREFLIGHT`, `ACTUAL_TEMPORAL_IEKF_ESTIMATION`, `SIMPLIFIED`, `FULL`, `IDEAL_WARP_ASSUMPTION`, `CALIBRATED` and `PREDICTION_ONLY`. Show real measured current image, selected owner map, valid/invalid temporal reference, warped RGB-D, residual, confidence/uncertainty and actual estimated point trajectory. Verify an estimated frame is truly populated, not a preflight placeholder.

## Phase 4 — First run real estimation, bounded and reproducible

**Order is mandatory:** tests → 3-frame smoke → 11-frame pilot → 60-frame critical case → ten-scene batch. Do not start all ten immediately.

1. Run all **29 temporal tests** plus new covariance/oracle/noise/ownership checks. Run the broad 316-test suite, classify and correct introduced regressions.
2. Run an **actual 3-frame full + simplified** test (not a geometry preflight) for Storage and USB, using P1A ideal-model assumption. Check numerical finiteness, nonzero actual measurement counts, source and reference integrity, QR stability, and no unintended covariance shrinkage. Full mode must complete at least one real `update` and save usable states.
3. Run the **11-frame forward pilot** for `storage_45135`, `USB_100109`, `blade_103706`, `stapler_103111`, both modes with matched initialization, point IDs and sensor model. Preserve full original points for the causal pilot (`--pcd-subsample-factor 1`), unless runtime makes this impossible; if so, report exact factor and ID mapping, then run focal-point targeted controls without changing identity.
4. Compare identical raw RGB-D pixels and source history across full/simplified/legacy. Report static GT error and moving GT error **only in post-inference evaluation**, and emphasize usable accepted measurements, camera handover, reference age, prediction-only rate, NIS/NEES, convergence and time per component.
5. Only after pilots are stable, run a **60-frame critical scene** (Storage and USB first; Blade if pilot is satisfactory) both directions/modes. Full 60-frame backward has frame 59 as anchor. `--frame-limit 11 --direction backward` is a backward run on the 0–10 truncated dataset, **not** a test of frames 59→49. Label it correctly.
6. Then run the ten-scene suite under a justified, independently documented model; `ideal` can be included as explicitly uncalibrated ablation but must never be labelled a validated production filter.

### Required timing / memory report

For both modes, record per frame:
- input transfer; projection; z-buffer sort; temporal interpolation; reference validity; Jacobian construction; lag component assembly; QR solve; Schur marginalization; reference refresh; covariance output; viewer export (separately).
- component count, max/median/p95 component dimensions, fill-in, QR rank/condition, active measurements per point and camera, GPU high-water memory, wall time, and CUDA device model.
- compare full vs simplified against same point count and frames. Trigger a **named non-success** status if any component is skipped, decoupled or truncated for memory/runtime; do not quietly approximate an exact graph.

## Phase 5 — Deliver evidence and user commands

Write the following under a **new** SSD output folder and in `reports/temporal_rgbd_next_validation.md`:

- `IMPLEMENTATION_MAP.md` with exact local file/function locations and Git commit + dirty-state diff summary.
- `NOISE_SEMANTICS_AND_CALIBRATION.md` with each covariance component, provenance, scale, units, formula, SPD/PSD checks, calibration split, limits, ideal-model label.
- `REGRESSION_7_FAILURES.md` naming the 7 cases and proof of pre-existing/introduced classification.
- `TEMPORAL_ORACLE_TEST_REPORT.md` documenting full/approximate posterior parity, finite differences, repeated reference reuse and camera handover.
- `CRITICAL_PILOT_RESULTS.md` with a per-scene, per-mode, per-point/time table and actual estimator traces.
- `PERFORMANCE_AND_COMPONENT_GROWTH.md`.
- **Actual computed** HTML diagnostics for pilot scenes, with image/geometry overlays and a note separating estimation from preflight.
- `RUN_COMMANDS.md`: exact runnable commands using verified CLI arguments and the true local dataset paths. Provide commands to the user for the **full 60-frame** and **all-ten-scene** runs; the user has requested the commands, not an unattended all-scene production run.

### CLI commands Codex should ensure work after implementation (do not claim they already do)

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/test_temporal_rgbd.py
python -B -m unittest discover -s tests -t . -v
```

**3-frame smoke using the newly implemented `ideal` diagnostic mode** (if you choose a different explicit option name, update this block and `RUN_COMMANDS.md` using verified `--help` output):

```bash
DATA_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
OUT_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_rgbd_smoke_$(date +%Y%m%d_%H%M%S)"
bash tools/run_temporal_rgbd_critical.sh \
  --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
  --mode full,simplified --direction forward \
  --frame-limit 3 \
  --scenes storage_45135,USB_100109 \
  --temporal-warp-noise ideal
```

**11-frame pilot:**

```bash
DATA_ROOT="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear"
OUT_ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_rgbd_pilot_$(date +%Y%m%d_%H%M%S)"
bash tools/run_temporal_rgbd_critical.sh \
  --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
  --mode full,simplified --direction forward \
  --frame-limit 11 \
  --scenes storage_45135,USB_100109,blade_103706,stapler_103111 \
  --temporal-warp-noise ideal
```

**Justified explicit model after calibration (do not use placeholder):**

```bash
bash tools/run_temporal_rgbd_critical.sh \
  --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
  --mode full,simplified --direction both \
  --temporal-warp-noise config \
  --temporal-warp-noise-config "/REAL/VERIFIED/PATH/temporal_warp_noise.json"
```

Replace the explicit config example with the **actual file created and verified** by calibration before presenting it as a runnable command. The `ideal` route is a diagnostic assumption, not a calibrated result.

## Definition of done

Do not announce full validation based on 29 unit tests, 316 broader tests or preflight screenshots. A useful intermediate milestone requires both modes to **execute actual synthetic multi-frame motion estimation** and produce inspected trajectories, covariance and visible per-frame rows. Full scientific acceptance additionally requires a statistically justified observation model, reference-correlation tests, no unresolved introduced regressions, meaningful runtime performance and post-hoc GT evaluation.

When all these are complete, provide the user the numerical result summary, measured performance and copy-pastable commands. **Do not say 'waiting for user to approve a covariance' as a substitute for implementing the diagnostic ideal-model run and a defensible calibration study.**
