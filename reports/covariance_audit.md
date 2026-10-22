# Covariance audit — baseline before correction (2026-10-16)

Authority: `CODEX_RGBD_IEKS_SPARSE_COVARIANCE_AND_FULL_PIPELINE_FIX.md`.
Inspected actual implementations and call sites, not filenames. The original
sparse reference is present as `original_sparse_iekf_reference.pdf` (the requested
`theory_sparse_iekf_3dgs(2).pdf` filename is absent); sections 11–12 specify local
JVP/VJP information actions and rigid reduction. The RGB-D PDF and state-space
addendum retain the first-hit likelihood and left-error/right-noise conventions.

| Area | Existing implementation and callers | Baseline finding |
|---|---|---|
| Initial mean/prior/gauge | `rgbdekf.solver.solve`, `initial_free_indices`, `Objective.__init__`; CLI calls `solve` | Initial poses fixed by removing gauge coordinates; twist covariance supplied, but the launcher silently supplies invented example priors and a 1-second interval. No rank diagnostics. |
| Dynamics, F/G | `rgbdekf.lie.predict`; filter and process residuals call it | Correct translation-first analytical F/G with full body blocks. `qrate*dt` is an explicit random-walk rate, **not** integrated white acceleration; no Van Loan implementation or per-interval model. |
| Sensor and calibration noise | `rgbdekf.config.read_config`, `model.Sensor`; CLI instantiates sensor | Joint 4D RGB-D covariance and categorical no-return exist; example values only, no per-field provenance or independent calibration evidence; no pixel-to-3D covariance. Fixed camera calibration is implicit. |
| Forward correction | `solver.forward_iekf`, `local_minimize`; `solve` calls it | Correct fixed prediction tangent and original prior. Empirical full-score metric is explicitly approximate and final covariance excludes LM damping. However `model.frame` allocates a joint dense map matrix even when filter only needs 12B coordinates. |
| Cross-body information | `lie.predict`, `model.frame`, `Objective.evaluate` | Full inter-body blocks are retained mathematically; dense global storage prevents real use. |
| Source/map correlations | `lie.source_map_covariance`; tests only. `Objective.covariance` | B2 helper correct including dependent source noise. Final dense information can express correlations, but no sparse production source factor representation or selected covariance queries exists. |
| New points and bookkeeping | `data.build_map`, `model.SceneModel`; CLI | Catalogue from all source pixels; source is initialization only and measurements are not intentionally counted twice. Entire dataset held in memory, global `3^N` enumeration/early resource rejection, no per-owner canonical proposals, no stable factor-ID ledger. |
| Backward smoothing | `solver.rts`; `solve` and exporter call it | Conditional RTS transports covariance between predicted/smoothed charts; charts are not exported explicitly. Conditional trace is labeled, but cannot replace joint uncertain-map smoothing. |
| Joint information/Schur | `solver.Objective.evaluate`, `motion_metric`, `covariance`; outer solver | **Erroneous production architecture**: O((12BK+6N)^2) arrays, dense map precision, dense Schur formation and full inverse. No matrix-free operator, PCG, or selected covariance. |
| Discrete posterior | `model.SceneModel.discrete_posterior`; outer solver | Exact tiny enumeration implemented as the only path; exponential full-scene rejection is unnecessary. No declared scalable variational family or ELBO updates. |
| Outer loop/convergence | `solver.solve`, `Objective.evaluate`; CLI | Common expected log-ray likelihood/prior/entropy, Armijo and objective+gradient test present. Dense re-evaluation, no checkpoint/resume, no separate stalled/indefinite/unidentifiable status. |
| SPD/uncertainty semantics | `solver.spd_solve`, `covariance`, exporter | Cholesky without fabricated jitter; final metric undamped. Empirical score covariance is an approximation, conditional filter and joint fixed-q covariance are labeled. No observability spectrum, solve residuals, or uncertainty queries. |
| Full run and outputs | `rgbdekf.__main__`, `run_blade.sh`, `export.py` | Real 60×8 data scanned, but production exits on artificial enumeration/dense limit. Launcher opts into example uncertainty and invented dt. Full reconstruction, calibrated inference and convergence were **not** validated. |

## Corrections to verify

Replace production enumeration/dense joint information with a declared discrete
variational family and exact local factor actions, preserve the normalized ray
model and full scores, keep gauges eliminated, discretize noise with explicit
units/charts, validate per-field provenance, and refuse missing calibration/time.
Retain dense enumeration and inverse comparisons only in tiny tests. Sparse
joint covariance must be queried by converged solves, separately from conditional
RTS uncertainty. Report every Section 14 criterion separately, without equating
dataset integrity validation with successful production reconstruction.

## Implemented corrections and current call graph

The user's subsequent 0A clarification authorizes documented initial guesses
and automatic timing discovery. It supersedes the earlier demand for independent
calibration before any production run. Guesses are accepted as a defined model,
with exact source keys, and are never relabeled calibrated. No assumption that
one uploaded ZIP's metadata describes another scene is made.

| Corrected area | Actual production implementation | Verification / remaining qualification |
|---|---|---|
| Entry and loading | `main.main` → `rgbd.dataloader.load_dataset`, `timing.resolve_frame_times` | All available frames/cameras; directory/ZIP streaming; actual timestamps/FPS discovery and explicit overrides; asynchronous timing rejected without a model. |
| Effective noise and P0 | `rgbd.assumptions.load_effective_noise`, `config.validate_noise_model`, `outer_loop.outer_loop` | Existing guesses accepted by user authorization; typed per-field units/provenance; gauge coordinates absent, full 6B initial twist prior retained. Missing values consolidated. |
| F/G/Q | `rgbd.lie.predict`, `uncertainty.process_covariance`, `van_loan`, `se3_predict` | Nonzero SE(3) finite differences; CV limit including cross terms and zero dt; correlated bodies. Equivalent continuous local Q is frozen at original-prior reference to keep one fixed discrete model/objective. |
| Localization/source coupling | `uncertainty.depth_to_world_covariance`, `shared_source_point_covariance`, raw factors in `objective.FullObjective` | Jacobian/units tests and dependent source-noise tests. No data-derived independent source-point prior is inserted. Shared correlation is retained by raw joint factors rather than per-point independence. |
| Forward IEKF | `ekf.forward_iekf`, `fixed_prior_update`; outer loop calls them | Fixed prediction prior over all inner iterations, full ray score, actual nonlinear objective, full 12B covariance. Numerical damping excluded from final covariance; explicitly empirical-score approximation. |
| Backward RTS | `smoother.backward_rts`; outer loop and output trace | Predicted/smoothed chart derivatives recorded; nonzero rotational correction and inter-body blocks compared against a fixed-chart dense batch oracle. Conditional only. |
| Ray model | `visibility.Sensor`, `project_contributor`, `ray_logprob_and_score`; `objective.RayFactors.frame` | Normalized first hit, no-return/empty cases, finite overlap/near-plane clipping, visibility and heteroscedastic covariance score derivatives. Empty rays are included analytically in the all-pixel objective. |
| Sparse information | `operators.InformationOperator`, `factor_store.FactorStore`, `objective.FullObjective.evaluate` | Local factor rows and bounded disk chunks; vectorized JVP/VJP actions; only 1D global work vectors. Independent prior blocks are vectorized; ray cross-point structure remains. No dense joint matrices or full-scene enumerator. |
| Schur/marginal queries | `joint_information.JointInformation`, `load_joint_information` | Matrix-free nested Schur solves, residual checks, selected undamped inverse blocks, reopenable persisted operator. Tiny dense oracle includes off-diagonal map blocks and augmented-state smoothing. |
| Discrete posterior/ELBO | `posterior.ScenePosterior`, `objective.FullObjective` | Explicit coupled finite-support categorical variational family; exact expectation on its support, entropy included; **not** exact unrestricted scene posterior. Support restriction can miss real assignments. |
| Backward geometry | `geometry.materialize_catalogue`, `backward_refinement.backward_geometry_discovery` | Fixed catalogue includes all sources and all owners; component-conditioned inverse proposals, frozen-motion geometry update under the same full objective. No residual threshold, point quota, or GT labels. |
| Outer/global updates | `outer_loop.outer_loop`, `joint_motion_map_refinement`, `accept_direction` | Actual A/RTS/B/joint/C sequence; all stages monotone-test the same original-data ELBO; objective + continuous gradient + within-family KKT required. Numerical stalls/budget exhaustion remain incomplete statuses. |
| Checkpoints/outputs | `outer_loop.checkpoint`, `outputs.export_result` | Original-prior hash, full discrete family, factor IDs, charts, numerical settings; deterministic accepted-boundary resume tested. Full predictions and per-cycle conditional/joint covariance diagnostics implemented. |

### Explicit mathematical and validation limits

* The reported continuous covariance is the inverse of an undamped empirical
  full-score information approximation, not an exact mixture posterior or true
  likelihood-Hessian Laplace covariance. Discrete uncertainty remains variational.
* The chosen finite-support global-scene variational family is deliberately
  restricted; it does not represent arbitrary millions-of-label posterior modes.
* Proper SPD priors/process factors are supported in production. General singular
  process/prior equality constraints and uncertain shared camera-extrinsic nuisance
  estimation are not implemented; unsupported inputs are rejected, never jittered.
  Conditioning on the supplied fixed camera calibration is explicit.
* Conditional data-rank diagnostics are small-block score-information ranks,
  distinct from full joint observability; no exact global rank claim is made.
* Full integrity validation has run on both supplied real (328×3) and synthetic
  (60×8) scenes. Real recorded-index timing resolves automatically to 25 Hz;
  synthetic blade timing is still missing. Full-scene motion estimation,
  convergence, practical sparse throughput and reconstruction accuracy are not
  established. Real candidate extrinsics and unverified hardware synchronization
  remain explicit conditioning qualifications. Section 14 is not declared
  satisfied on the basis of tests or input scans.
