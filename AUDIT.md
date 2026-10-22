> **22 October 2026 pointwise replacement:** [current audit](AUDIT_POINTWISE_2026-10-22.md), [execution commands](POINTWISE_RGBD.md), and [measured validation](reports/pointwise_2026-10-22_validation.md). The audit below concerns the superseded rigid/group architecture. Its model-count-prior blocker does not apply to the new 6D pointwise algorithm.

> Historical request results: [LATEST_REQUEST_VALIDATION.md](LATEST_REQUEST_VALIDATION.md). Earlier rigid reconstruction was **not accepted**; its UKF progressive-motion test still failed.

> Current authority: `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md` (20 October).
> Earlier visibility-policy requirements and background-depth ordering below
> are historical, superseded behavior. See the v2 update at the end of this file.

# Parallel point RGB-D IEKF / UKF implementation audit

Authoritative sources: all six documents in `specs/parallel_rgbd/` (symlinks to
the user-supplied originals at the repository root), including all 19 PDF pages.
Read completely before this replacement. Previous implementation preserved in
`/tmp/rgbdekf_before_parallel_20261019`; original scene and result files preserved.

## Existing components

| Component | Decision | Evidence / replacement |
|---|---|---|
| `rgbd/dataloader.py`: Dataset, camera_from_metadata, load_dataset, RGBDFrame, scan | Keep verified loading logic in common package | Directory/ZIP, metric optical depth, real foreground masks, fixed camera conditioning; no GT annotations |
| `rgbd/timing.py`: SceneFiles, resolve_frame_times | Keep in common package | Actual selected scene resolves 328 timestamps / 3 cameras / 25 Hz; asynchronous acquisitions rejected rather than relabeled synchronous |
| `rgbd/lie.py`: Exp/Log/Ad, State, source_map_covariance | Keep verified common mathematics | Translation-first, left pose error, additive spatial twist; process Jacobians belong exclusively to IEKF |
| `rgbd/geometry.py`: SurfelMap, support fusion | Remove from new production paths | Explicitly prohibited normals, support edges and surfels |
| `rgbd/projection.py`, `raster.cpp`, `raw_factors.py`, `visibility.py` | Replace | New measurement is conditional 6D raw-pixel point output, no first-hit surface renderer |
| `rgbd/associations.py` | Replace | Hardcoded static category violates equal treatment of estimated rigid groups |
| `rgbd/ekf.py`, `smoother.py` | Reimplement filter-specific modules | Reuse tested chart formulas, not old renderer or posterior model |
| `rgbd/backward_refinement.py` | Replace | No explicit UNSOLVED ownership or chronological frontier; full real scan exceeded incidence memory budget |
| `rgbd/outer_loop.py`, `objective.py` | Replace | Shared solved/unsolved transactional controller and common normalized point score required |
| `rgbd/operators.py`, `uncertainty.py` | Keep mathematical reference only | New sparse factor/latent queries must match new point factors and each filter's covariance |
| `main.py`, old launchers / output code | Replace production routing | Independent `main_iekf.py` and `main_ukf.py`; no inter-filter imports |
| UKF / Torch CUDA pipeline | Implement | Absent from old repository |

## Observed prior failure

The superseded raw-first-hit experiment completed a forward pass and RTS, then
its 328-frame reverse scan took 6,151 seconds. Subsequent surface-incidence
construction exceeded the image-local allocation budget. Status was
`NUMERICAL_BREAKDOWN`, peak RSS 11,093,909,504 bytes. No second forward pass or
completed reconstruction resulted. Its logs remain on the external SSD.

## Runtime boundary

CUDA is hidden by the filesystem sandbox (`/dev/nvidia*` absent). Outside that
sandbox the installed driver exposes the RTX 4070 Laptop GPU, 8,188 MiB,
driver 580.178.04. Actual GPU tests/runs must use the existing rigeuler3dgs
environment outside the sandbox; CPU-only tests cannot establish CUDA parity.

## Statistical contract and acceptance

Approved P0, local Q rate, RGB/depth covariance and no-return/background parameters
remain initialization assumptions from `configs/example_uncalibrated.json`, not
independent calibration. A source-image initialization is never an independent
landmark prior. Inner state uncertainty is conditional on fixed points and matches.
Shared source-pose factors retain pose-point and point-point correlations without
a dense covariance. Neither local association nor sigma cubature is exact global
Bayesian inference.

Implementation and full-scene acceptance are **in progress**. No result from the
superseded pipeline counts toward either new pipeline's acceptance gates.

## Symbol inventory of the superseded production package

Top-level symbols below inherit the module decision. Class methods inherit the class decision; no superseded package is importable from either new production entry.

| Original module | Symbols | Decision |
|---|---|---|
| associations.py | `weighted_rigid_fit`, `_world`, `rigid_motion_mixture`, `_feature_covariance`, `initialize_motion`, `ownership_from_log_evidence`, `initialize_ownership` | Remove old production path; replacement described above |
| assumptions.py | `documented_noise_guesses`, `load_effective_noise` | Split verified mathematical/configuration reference from removed production model |
| backward_refinement.py | `trajectory_evidence`, `update_ownership`, `backward_discover` | Remove old production path; replacement described above |
| config.py | `MissingNoiseModel`, `NoiseModel`, `covariance`, `validate_noise_model` | Split verified mathematical/configuration reference from removed production model |
| dataloader.py | `Dataset`, `camera_from_metadata`, `load_dataset`, `RGBDFrame`, `scan` | Keep verified common logic |
| diagnostics.py | `write_json`, `runtime_summary`, `validate_full_dataset` | Keep verified common logic |
| ekf.py | `FilterStep`, `FilterTrace`, `fixed_prior_update`, `forward_iekf`, `step_to_dict`, `step_from_dict` | Remove old production path; replacement described above |
| evaluation.py | `trajectory_error` | Remove old production path; replacement described above |
| geometry.py | `SurfelMap`, `empty_map`, `backproject`, `canonicalize`, `solve_covariance`, `inverse_covariance`, `log_gaussian`, `associate_surface`, `_coalesce`, `union_supports`, `fuse_observations`, `build_initial_map` | Remove old production path; replacement described above |
| lie.py | `skew`, `ad`, `left_jacobian`, `exp`, `log`, `inverse`, `adjoint`, `State`, `predict`, `source_map_covariance`, `inverse_transport`, `process_residual_jacobians` | Split verified mathematical/configuration reference from removed production model |
| objective.py | `DirectFactors` | Remove old production path; replacement described above |
| operators.py | `LocalFactor`, `JointInformation`, `schur_solve`, `RayBatch`, `StreamedJointInformation`, `CoverageBatch` | Split verified mathematical/configuration reference from removed production model |
| outer_loop.py | `CompleteResult`, `_fingerprint`, `initialization_digest`, `checkpoint`, `outer_loop` | Remove old production path; replacement described above |
| outputs.py | `write_ply`, `export_result` | Remove old production path; replacement described above |
| progress.py | `progress`, `write` | Keep verified common logic |
| projection.py | `Camera`, `projection`, `point_jacobians`, `reprojection_jacobian`, `fixed_ray_jacobian`, `clip_polygon`, `pixel_overlap`, `support_projection`, `ProjectedMap`, `project_map`, `bilinear`, `batch_residual_jacobians`, `raw_support_projection` | Remove old production path; replacement described above |
| raster.py | `rasterize`, `pixel_support_hits`, `local_evidence`, `aggregate_raw` | Remove old production path; replacement described above |
| raw_factors.py | `camera_factors` | Remove old production path; replacement described above |
| smoother.py | `ConditionalSmootherTrace`, `backward_rts` | Remove old production path; replacement described above |
| timing.py | `MissingAcquisitionTiming`, `SceneFiles`, `_timestamps`, `_camera_keys`, `_timestamp_file`, `resolve_frame_times` | Keep verified common logic |
| uncertainty.py | `Unidentifiable`, `NumericalBreakdown`, `sym`, `spd_solve`, `covariance_from_information`, `spectrum`, `depth_to_world_covariance`, `shared_source_point_covariance`, `innovation_covariance`, `van_loan`, `process_covariance`, `se3_predict`, `joseph_update` | Split verified mathematical/configuration reference from removed production model |
| visibility.py | `depth_order`, `first_hit_weights`, `background_log_probability` | Remove old production path; replacement described above |
| visual_debug.py | `mark_visual_status`, `preview`, `dashboard`, `motion_html` | Remove old production path; replacement described above |

## Streaming and operator corrections, 2026-10-19

The raw frame cache now normalizes RGB and unpacks mask bits after sparse
indexing. It caches the immutable full-domain unassigned likelihood baseline,
keyed by its actual sensor/mixture parameters. Incremental candidate scores keep
exact global association normalization and changed-ray competition; rejected
candidates do not clone all sparse factor tables. Point IDs are monotonic, and
original source-only self factors are excluded from motion corrections.

The UKF batches its nonlinear sigma reprojections. Only competitors in selected
factors' fixed external bins affect visibility, so unused bins are eliminated
exactly. The preserved scalar implementation under tests/oracle is a numerical
reference, not a production fallback. On the real 194,724-point, seven-group
checkpoint at frame 164, group 0 / three cameras: scalar moments took 1.967 s;
batched moments 0.291 s; A, M and b agreed exactly in this measurement. There
were 41,394 active and 145 deferred factors. This concurrent timing is an
operator measurement, not a clean whole-pipeline speed benchmark.

Tests before this last operator change: 59 passed, zero failures/errors/skips,
actual CUDA enabled and all 328 times / three cameras validated. A subsequent
full suite and independent full real runs are required for the new source.
Reports and preserved source snapshots are under the requested SSD directory
parallel_validation_20261019. Live objective charts show rejected proposals too;
3D display sampling retains every discovered group and never reduces estimation.

Remaining limitations must not be hidden: structural priors are explicit model
choices where the papers leave them symbolic; conditional uncertainty is not a
full joint map-motion posterior; metadata/export consistency is not independent
physical calibration; asynchronous acquisition is rejected. An exploratory
rotating/translating four-frame drawer test showed spurious groups and poor
pose recovery (roughly 0.109 combined pose-log error for its best moving-group
proposal in the earlier implementation). Passing derivative tests does not
establish general physical reconstruction accuracy. Full real reverse/frontier
completion and convergence remain unverified until their actual reports exist.

## Real-run corrections found after the initial test report

The requested SSD rejects symbolic links (EPERM). Exports now copy semantic
aliases as regular files/directories, refreshing them on each export. Actual
rotating-scene exports succeeded on that SSD; the failed v1 diagnostic and logs
were preserved. CLI resume now compares JSON-normalized manifests, so YAML
lists do not falsely differ from observation tuples. Actual path differences
still reject resume.

The strict IEKF checkpoint check found a 36.815 score difference after a full
rebuild at approximately 656 million. Inspection found changed float32 depths
and discrete pixel memberships, not merely a Gaussian summation discrepancy:
BLAS contraction order changed with the point batch size. Thus a changed-point
projection could differ from a full-map projection. The corrected projection
uses explicit per-coordinate operations with fixed order, including backward
batches. A 150,001-point actual-GPU test checks bitwise world/camera/UV/depth,
pixel and validity invariance against a 23,719-point subset.

Unversioned checkpoint scores are refused by default. An explicit
--rebase-legacy-projection-score migration preserves the old checkpoint/history,
logs both scores and their difference, and starts projection score epoch 2.
No tolerance was widened. Earlier accepted score history is NOT validated by
the corrected implementation and is excluded from current-epoch monotonicity
claims. Recovered maps/trajectories are warm starts; original P0/Q/R remain
unchanged and complete forward/discovery/frontier/new-map passes are required.
The previous point counts and component tests are not full reconstruction success.

## Group selection counterexample (unresolved)

With 328 times, approved Q and a zero-residual identity trajectory, the motion
negative log density contributes -16,257.021790260042 per group. Holding all
194,724 point positions, raw assignments and RGB-D predictions unchanged, the
implemented structural prior produces for splitting B identical-trajectory groups
into B+1:

    delta J = -16257.021790260042 + N log((B+1)/B) + log((B+2)/B).

At B=12 this is -670.6314; at B=20 it is -6756.3106. Thus proper p(B) alone does
not prevent redundant groups from improving a continuous MAP density in changing
dimension. This is a model-selection limitation, not a CUDA or noise-calibration
result, and does not itself prove which proposals the finite search will accept.
The supplied symbolic group/assignment prior does not uniquely resolve the
reference-measure/model-evidence issue. No arbitrary group cap, prior strength,
or cancellation of likelihood constants has been added to conceal it. The exact
algebra and values are reproducible with scripts/audit_group_score.py and saved
in the SSD reports/group_score_counterexample.json. Physical grouping and full
convergence must be assessed separately from a monotone implemented score.

Broken rgbdekf geometry/Lie compatibility shims still referred to the removed
rgbd package; they were archived under /tmp rather than redirected to a different
API. The python -m rgbdekf entry continues to invoke only the analytical main.
Unused old catalogue oracle modules were also archived. The retained tiny scalar
UKF oracle is used solely by tests/benchmark parity. An AST test enforces no
inter-filter imports and no estimator imports from the common package.

A headless browser rendered the offline real-map HTML. A second browser check
selected frame 327 and group colors; Plotly's actual plotted X coordinates
matched the saved final-pose projection. Displayed sample maximum displacement
was 2.260942400637932 m; this verifies the control/trajectory rendering, not
physical accuracy. Screenshots and browser DOM/logs are preserved separately.

## Hard foreground amendment audit

The current authoritative bundle additionally references
`CODEX_PARALLEL_IEKF_UKF_HARD_FOREGROUND_MASKS.md`. Its complete text was read.
The pre-amendment implementation selected foreground observations but did not
enforce the predicted-visible mask-zero pose exclusion. Those runs cannot be
accepted under the amendment. Only the two exact workers recorded by our
launcher were interrupted; their checkpoint/source/log artifacts were preserved.

Replacements now cover exact binary mask parsing, aligned required masks,
finite positive foreground backprojection, foreground-only raw likelihoods and
initializer imagery, shared GPU visibility/feasibility, each independent filter's
candidate check, full and incremental common-score feasibility, and per-frame
mask/ownership diagnostics. Neither mask differentiation nor sigma clipping is
used. Missing/occluded solved geometry remains solved; UNSOLVED contains only
unclaimed valid foreground raw IDs. All source paths and backward candidates
remain subject to the same full candidate score.

An unresolved input to the amendment is the visible/occluded/unknown decision
under Gaussian depth uncertainty. The amendment specifies no confidence level,
loss function or bounded noise support. Gaussian noise alone cannot imply a
finite confidence interval. The implementation therefore accepts explicit
`visibility.mode: equal-loss-order` or `credible-order` with an explicit
`minimum_probability`. It does not insert a production default. Unit fixtures
explicitly use 0.99 as a test assumption, never as approved physical calibration.
The user was asked for this missing rule while independent implementation and
tests continued.

Actual frame-0 evidence is saved in `frame0_hard_mask_audit/`: the preserved
160,943-point map has 1,342 / 2,988 / 1,514 frontmost projections onto valid-depth
background in cameras 0 / 1 / 2. Their median probabilities that the model is
closer than the measured background are approximately 0.99937 / 8.39e-34 /
0.88013 under the approved depth noise. These are probability diagnostics,
not a hidden confidence decision or a claim that all such points are visible.
Some discrepancies are tens of declared depth standard deviations. Candidate
extrinsics and cross-view mask consistency must be assessed; no mask dilation,
camera modification or fabricated depth threshold was introduced.

Old checkpoints are explicitly rejected by the new hard-mask version guard;
their unconstrained scores are not silently reused as constrained scores.
No full-scene hard-mask reconstruction or physical convergence has passed.

The user explicitly reserved the visibility decision: “i will tell you, stop
assuming.” Production configuration remains unset; no unit-fixture confidence
is used on the real scene. Input timing and approved P0/Q/R do resolve correctly.

The remaining mask-path audit also moved foreground backprojection and its full
RGB-D covariance pushforward to bounded Torch device batches, using float64
arithmetic and CPU reference checks. Negative-mask backward projections are now
classified before hypothesis acceptance; only positive-mask targets populate
RGB-D vectors. Invalid/no-return visibility has an undefined probability (NaN),
not an invented 0.5 prior. Raw source indices, observation ownership and Kalman
factor IDs each have explicit domain checks. Invalid initializer features carry
no background-derived XYZ.

Raw cache identity now records original RGB/depth/mask content fingerprints.
Resume checks fingerprints after the full input scan, and cached-frame reads
reject mismatched or unverified source identity. Thus unchanged filenames do
not silently authorize reuse of older masks. The source dataset must remain
immutable during an experiment. No input file is modified by either pipeline.

## V2 masked point-reprojection implementation — 20 October 2026

Authority: the complete 885-line `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md`,
its 26-page PDF, and the complete 460-line companion implementation contract.
The source 3DGS PDF `reference_3dgs_masked.pdf` Section 13.2 Eq.41 was checked for
observation preparation only. No research Gaussian renderer was modified.
Earlier policy-based visibility behavior described above is superseded.

| Current component / functions | Decision / implemented change |
|---|---|
| `foreground.exact_binary_mask`, `compact_valid_fg` | Keep exact parsing; add native Torch CPU/CUDA compaction of M AND Vraw AND finite positive depth |
| `dataloader.Dataset.observation`, `RGBDFrame` | Add independently loaded raw validity, native prepared RGB, explicit acquisition-premise provenance; no background depth validation can alter estimation |
| `dataloader.resolve_mask_premise` | New: read explicit acquisition Boolean; unknown stays unknown, never inferred from scene name or a confidence parameter |
| `foreground.depth_order_status` and configuration visibility policy | Remove the old probability-threshold classifier and missing-policy guard |
| `foreground.foreground_depth_status`, `projected_mask_status` | Rewrite: normalized match/unassigned evidence for foreground ordering, model self-occlusion, conditional negative-mask support; no background RGB/depth gathering |
| `projection_cuda.build_pixel_bins_and_zbuffer` | Add explicit immutable-point-ID tie sorting and bounded HxW first-index/depth buffers |
| `matching_cuda.match_rgbd`, `observation_model.unassigned_log_probability` | Foreground-valid raw target domain only; invalid depth has no categorical/geometric observation factor |
| `score.frame_score`, `ScoreCache`, `CachedDataset.background_total` | Apply identical valid-foreground domain; version raw/noise caches and checkpoints; retain exact full/incremental objective parity |
| `ScoreCache.front_depth_at`, `backward_cuda.backward_evidence` | Reuse accepted factor-table depth ordering at valid foreground rays; compare against full projector in tests; no all-frame XYZ catalogue or point-square array |
| `MeasurementProblem.feasible` | Trial feasibility outside Gaussian derivatives; exact mask-only fast path, with no depth likelihood evaluation when only feasibility is needed |
| `ukf.measurement.measurement_moments` | Replace fixed external competitor bins with complete map z-buffer checks at every sigma pose; defer discontinuous factors; use raw fixed IDs in continuous M4 output |
| `outer_loop.run_outer` | Attempt actual measured-filter/smoother feasibility restoration for inconsistent initial hypotheses; fail explicitly only if those candidates cannot yield finite full J |
| IEKF `F,G,H`, fixed-prior update, own RTS; UKF process/sigma/Woodbury/own smoother | Retain verified independent mathematics; no hybrid or copied IEKF Jacobian in UKF |
| `entry.run` | Remove visibility-policy preflight abort, print compact timing summary, save complete provenance and measured-memory preflight |
| `outputs`, launchers and tests | Version2 products and gate; exact mask/validity, both branch invariance, empty/once-visible cases, CUDA measurements and full-scene attempts |

The no-external-occluder premise is absent in the selected real scene. M3 requires
negative support to remain undetermined in that case. The loader reports this
limitation and still runs; it does not fabricate that premise or consume masked
background depth. All observation/fusion/measurement sources still require M1.

Conditional raw factors are independent only in the chosen fixed-calibration,
fixed-map, nonoverlapping-pixel model. Shared source-pose keys/sensitivities remain
retained for selected queries. Full map-motion marginal covariances are not
computed or claimed. The earlier group-count/model-selection counterexample
remains unresolved; this amendment does not supply an alternative group prior.

The UKF's mandated factor deferral is operationally significant. In the preserved
four-frame drawer diagnostic every matched factor at times 1..3 is deferred under
the authorized propagated prior and full-map sigma visibility. The update then
has M=I, b=0 and retains its prediction. The unchanged nonzero-motion assertion
fails. This is a recorded acceptance failure, not justification for restoring the
old fixed-bin approximation, shrinking P0/Q, or inventing measurement information.
See SSD `parallel_masked_v2_20261020/ukf_fixture_failure/` and `VALIDATION.md`.

The final UKF operator reuses fixed-group front-depth/ID buffers and identical
pose arrays within each independent sigma ensemble. It still evaluates the full
map ordering: `min(fixed competitors, moving group)` is the same stable full-map
z-buffer. Coupled groups use the full joint sigma chart. Comparison against the
saved full-map implementation on real frame1 yields exactly zero A/M/b difference
and identical active counts; CPU full/scalar regressions also compare. This is
exact reuse, not the prohibited fixed external competing-bin approximation.

A further conditional-model correction prevents a factor's own fixed raw target
from being reclassified as an independent foreground occluder when its sigma
predicted depth changes. That depth discrepancy belongs in the M4 Gaussian
measurement. Distinct foreground rays, model depth/ID competition, masks and
frustum events still defer inconsistent factors. A new isolated-depth test
compares the actual UKF correction/covariance with dense sigma innovation and
requires nonzero measurement information. It passes without changing P0/Q/R.

The final UKF sigma path prepares exact valid-foreground raw samples on the GPU
once per camera ensemble, preserving full correlated/heteroscedastic noise and
the normalized unassigned likelihood. No background RGB/depth is uploaded as a
measurement; no persistent all-frame GPU image or XYZ tensor is created. This
removes per-sigma CPU raw-image transfers. Real frame1 operator comparison in
`kernel_benchmark_device_fg.json` gives zero A/M/b difference from the corrected
full reference and identical active counts. The final full real UKF attempt is
`full_scene_04/ukf`; earlier UKF workers were preserved/stopped for these actual
corrections, not labeled successful reconstructions.

The final tie audit also corrected factor-cache and conditional membership ties
for maps whose storage rows are reordered: ties now use immutable point IDs,
matching the common projector. A reordered-map full/incremental/rollback test
passes. The running real maps retain monotonic IDs, so the frozen worker
snapshots already produce the same ordering there; their snapshots remain the
source of truth for each experimental run. No prior result is relabeled as a
run of a different snapshot.
