# CODEX IMPLEMENTATION CONTRACT — Sequence-wide SVGD-assisted moving-support discovery on RGBDiekf

**Repository:** <https://github.com/SamueleMara/RGBDiekf>  
**Audited base:** `main` at **`d0b470b75e9ec9181f7561e1a80d2909fb75c7e5`**, commit message `new dense implmentation 4` (GitHub committer date 2026-11-03 14:39:08 UTC).  
**Research basis and normative mathematical reference:** `Stein_SVGD_Moving_Support_Pipeline.pdf` / `.tex`, produced in the preceding discussion; Liu & Wang, *Stein Variational Gradient Descent*, NeurIPS 2016, <https://arxiv.org/abs/1608.04471>; reference code <https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py>.  
**Deliverable:** an **opt-in, point-only, GPU-accelerated first stage** that uses the **whole masked colored 3D sequence** to estimate *provisional transport and latent rigid coherence* and **reports only per-initial-point probabilities of belonging to a physically moving support**, together with a conservative, evidence-backed moving-seed set. Preserve the existing baseline paths; do not install or substitute a new rigid/joint estimator.

> **Codex: the THEORY-TO-CODE FIDELITY DIRECTIVE below is the overriding acceptance contract. Implement the stages and tests below, in order. This is a research implementation contract, not permission to fill missing probability models, calibration, camera visibility rules, or hyperparameters with convenient guesses. Each choice marked `PROPOSED` is an explicit model to implement and falsify; it is not a theorem about SVGD. Report failures honestly and stop at the relevant acceptance gate. Do not claim full-sequence acceptance from toy tests.**

---

## THEORY-TO-CODE FIDELITY DIRECTIVE — mandatory, normative, and higher priority than optimization convenience

> **NON-NEGOTIABLE USER REQUIREMENT:** The implemented inference must correspond **100% to the mathematical model selected from the accompanying theory document** (`Stein_SVGD_Moving_Support_Pipeline.pdf` and the matching `.tex`). No mathematical substitution is permitted silently. **Every implementation deviation, approximation, omission, surrogate, altered objective, altered normalization, new prior, changed state space, or changed data-support rule must be explicitly reported and mathematically justified.** If a deviation cannot be justified and authorized within the contract, do not implement it as if it were compliant. Record a `BLOCKED_THEORY_FIDELITY` or `BLOCKED_MODEL_UNSPECIFIED` state instead. Do not report an approximate implementation as a 100%-equivalent implementation.

### TF.1 What “100%” means, and what it does **not** mean

The requirement is **semantic and mathematical fidelity**, not an impossible claim of real-number-exact floating-point arithmetic. For a fully specified mathematical model, the code must use the same:

1. **State and latent variables:** initial canonical material-point IDs, provisionally transported XYZ and optional velocity, shared RGB appearance/reference uncertainty, full frame sequence, unknown same-rigid-support relations, unknown moving-support labels, visibility and unexplained observed geometry.
2. **Observation support and units:** *all* valid foreground masked XYZ/RGB points from every eligible frame/camera; calibrated world XYZ in meters; declared RGB representation; actual timestamps; per-sample and shared-source uncertainty; no background evidence, GT groups or 2D image registration.
3. **Functions and probability laws:** the *exact* selected KDE mixture including Gaussian normalizers; appearance-conditioned **XYZ** score; full Stein field including attraction/repulsion, correct gradients and global `1/N`; selected sequence-wide XYZ/RGB feature observation and covariance; temporal dynamics; rigid-distance constraints; normalized priors and model evidences.
4. **Conditioning/marginalization:** keep trajectory and support uncertainty; do not substitute a pointwise displacement threshold or a maximum-likelihood fit difference for the marginal moving-support posterior. Do not turn a proposal field, raw MMD value, held-out p-value or unnormalized energy into `P(dynamic)`.
5. **Causality of evidence:** no counting a source measurement twice, independently reusing shared frame-0 uncertainty, ignoring visibility events to decrease cost, or adding independent pseudo-observations derived from the same RGB-D samples.
6. **Published outputs:** per-original-point `p_dynamic`, `p_static`, explicit validity/approximation status and evidence-backed moving-seed selection. Provisional tracks and graph scores are diagnostics only. No final rigid/joint estimation in Phase 1.

Theoretical paper **itself distinguishes exact equations from proposals and approximations**. “100% theoretical correspondence” therefore requires Codex to (a) implement the *selected* proposed model exactly as defined, (b) clearly identify where that model is itself an approximation to raw-data Bayesian inference or to standard SVGD, and (c) never infer more from that model than justified. An exact implementation of *restricted physical-point SVGD* is **not** evidence that standard posterior-SVGD's convergence theory applies to physical points.

**A change that yields similar pictures or comparable accuracy is NOT mathematical equivalence.** Matching output dimensions, training loss trends or CUDA speed is also NOT mathematical equivalence.

### TF.2 Source-of-truth precedence and mandatory mathematical freeze (Gate TF0)

Before writing any inference operator, Codex must create and commit to the working changeset a new document `docs/SVGD_THEORY_TO_CODE_CONTRACT.md`. This is a frozen, equation-by-equation, machine-auditable transcription of the *selected* model, not a new design proposal. It must:

- Reference the theoretical `.tex` equation **labels** (e.g. `eq:target`, `eq:targetscore`, `eq:physkernel`, `eq:physicalsvgd`, `eq:featureobs`, `eq:featurepred`, `eq:jointobjective`, `eq:edge-delta`, `eq:edge-cov`), corresponding sections/equations in this document (A1–D*, etc.), and implementation function/test names.
- State the complete conditional/joint factorization, density support, dimensions, normalization constants, precise frame/camera sample weighting and full treatment of shared uncertain observations. If a quantity is not mathematically specified, explicitly mark **`NOT_DEFINED_IN_THEORY`**, rather than inventing a code default.
- Separate **(A) exact algebraic identities**, **(B) selected proposed mathematical model**, **(C) approximation to that model**, **(D) numerical implementation and floating-point error**. This classification is mandatory for **every** operative equation and probability claim.
- Document, before implementation, the values or legitimate estimation procedures and provenance for KDE `Lambda_X/Lambda_C`, Stein `H_X/H_C`, feature map `psi`, feature covariance `Gamma`, sensor/appearance uncertainty, visibility/coverage/missingness law, temporal motion priors, same-support and dynamic-label priors, model-evidence integration, and seed risk rule. **A missing law or prior is a mathematical gap, not an engineering detail.**
- Record which equations act as **optimization proposals** versus which define the **single normalized statistical likelihood**. The KDE score is a proposal; the feature residual may be a statistical observation only after its `Gamma`/support model is defined. They must not be added as independent sensor factors.
- State the model's use of the **entire** sequence and how its posterior uses later observations to revise earlier frames. Specify the status of data support when observed/cardinality changes or visibility cannot be resolved.
- Declare any unresolved model question as `BLOCKED_MODEL_UNSPECIFIED`; **do not start the affected implementation** until the mathematics is complete. Unrelated independent CPU/CUDA kernels with complete equations may still be implemented and tested; do not claim end-to-end readiness.

If the theoretical document and this Codex document seem inconsistent, **stop and write an explicit `THEORY_CONFLICT` record with both formulas and their consequences**. Do not silently choose the simpler one. Likewise, a *new* proposal added by Codex is **not** part of the authorized theory until separately reviewed; keep it quarantined behind an explicit `EXPERIMENTAL_DEVIATION` mode, never used for baseline fidelity tests.

### TF.3 Mandatory equation-to-code traceability matrix

The `docs/SVGD_THEORY_TO_CODE_CONTRACT.md` file must have **one row per equation and material assumption**, including at least these rows (additional equations from the PDF/TeX may not be omitted):

| Theory source | Mathematical invariant | Required implementation proof | Prohibited silent substitution |
|---|---|---|---|
| `eq:target`; spec (A1) | Normalized 3D XYZ × 3D RGB mixture over **actual masked observations**, with correct per-component log determinants and explicitly selected weights | Independent NumPy normalized-density oracle; log-sum-exp parity; integral/normalization checks | Nearest point, hard correspondence, unnormalized kernels, implicit sample weights |
| `eq:targetscore`; (A2) | `g=∂_x log p` with fixed query RGB and full heteroscedastic responsibilities | Five-point finite differences and CPU/CUDA agreement, including anisotropic covariance | Different Gaussian kernel, missing covariance inverse or color factor |
| `eq:physkernel`; (B1) | Exact selected positive-definite XYZ–RGB interaction kernel, separate from target/sensor bandwidth | Value/gradient FD and derivative-sign tests | Dropping color, changing bandwidth with scene depth without an explicit model |
| `eq:physicalsvgd`, `eq:repulsion`; (B2–B3) | Entire `sum_{l=1}^{N}` with fixed `1/N`, source-point gradient, correctly signed repulsion | Dense exact oracle and tiled CUDA parity per point and per term | Degree normalization, local neighbor-only sum, repulsion removal, unreported clipping |
| `eq:temporalprior`; (C4) | Whole-sequence motion prior using actual timestamps and shared initial source uncertainty | Small exact Kalman/trajectory-oracle and global-objective derivative test | Independent frame-to-frame registration passed off as sequence inference |
| `eq:featureobs`, `eq:featurepred`; (C1–C3) | Chosen full observation feature and comparable visible prediction; exact selected normalization and Jacobian, including weight derivatives where applicable | Exact kernel/feature oracle, JVP/VJP dot-product & finite-difference checks | Arbitrary feature replacement, fixed denominator with changing support, untracked RFF approximation |
| `eq:featcost`; (C5) | Single normalized feature observation likelihood, `Gamma` and log determinant with shared correlations | Dense small joint probability/NLL and covariance oracle; finite-difference full cost | Unweighted MMD + weighted MMD as two measurements; diagonal covariance labeled exact |
| `eq:edgeresid`, `eq:edge-delta`, `eq:edge-cov`; (D1–D*) | Complete distance time history, correct XYZ Jacobians, shared frame-0/cross-point covariance | Independent analytic/FD Jacobians and Monte Carlo covariance oracle | Treating distances at different frames or edges as independent |
| `eq:fullpost`, `eq:jointobjective`; (C6/E*) | One explicitly normalized **joint model** for trajectories, rigidity, visibility, label priors, evidence | Tiny exact enumeration+Gaussian/integration oracle; decomposition and score parity | Independent edge probabilities treated as a coherent partition; arbitrary energies summed as posterior |
| `eq:evidence`; support probability | Marginalize nuisance motion/support/visibility and penalize flexibility through model evidence | Tiny exact posterior normalization and label permutation tests | MAP improvement, sigmoid of displacement/Stein norm, p-value relabeled probability |
| Phase-1 output; later stages | High-confidence dynamic seeds with ambiguous points retained; no joint-dependent completion in Phase 1 | Seed-risk computation from posterior, original-ID integrity, downstream isolation test | Guessing near-hinge dynamic labels from GT/joint axis; fake legacy stage outputs |

**Each row additionally requires:** source-file path + exact commit hash; Python/CUDA function; test ID; dtype; units; implementation classification `EXACT_MODEL` / `NUMERICAL_APPROX` / `MODEL_APPROX` / `DEVIATION`; maximum measured discrepancy and independently justified numerical tolerance; CPU/CUDA outcomes; failure status; reference to the corresponding deviation-ledger ID or the literal word `NONE`. A blank row is a failed gate.

### TF.4 Deviations: mandatory register, review and explicit authorization

Create two artifacts **at the start**, not at the end:

- `reports/svgd/theory_deviation_register.md` — human-readable, **append-only** in the implementation history; initial entry `NO_DEVIATIONS_RECORDED` (this entry must be superseded once deviations are found).
- `reports/svgd/theory_deviation_register.json` — machine-readable list with stable IDs, including entries for every `NUMERICAL_APPROX`, `MODEL_APPROX` and `DEVIATION`. Empty array is allowed **only after** the full automatic deviation audit confirms the selected model is exact within stated floating-point tolerances.

For **every** mismatch, including performance-related substitutions, model choices discovered to be incomplete, numerical workarounds, third-party library approximations, sampling/subsampling, and omitted derivatives, Codex must record:

```yaml
id: DEV-0001
status: PROPOSED | AUTHORIZED | REJECTED | BLOCKED | REVERTED
classification: NUMERICAL_APPROX | MODEL_APPROX | THEORY_CHANGE | IMPLEMENTATION_BUG
original_theory: {document: ..., equation: ..., exact_expression: ...}
actual_implementation: {file: ..., function: ..., exact_expression_or_algorithm: ...}
reason: ...                                  # including necessity; "faster" alone does not justify validity
mathematical_difference: ...                 # algebraic change, normalization, conditioning, distribution, derivative, etc.
affected_outputs: ...                       # score, transport, evidence, p_dynamic, seeds, etc.
error_or_bias: {units: ..., measured: ..., bound_or_limitation: ...}
independent_oracle: {test: ..., result: ..., tolerance_basis: ...}
correctness_risk: ...                       # static false positives, missed seeds, double counting, etc.
alternatives_considered: ...                 # include exact blocked alternative
ablation_command: ...
authorization: {source: ..., decision: ...} # no implicit authorization
conclusion: EXACT_EQUIVALENT | CONTROLLED_APPROXIMATION | NON_EQUIVALENT | UNDETERMINED
```

**Authorization rule:** Technical convenience, CUDA runtime limits and library availability do **not** authorize a mathematical change. When there is no user approval for a non-equivalent change, Codex must (1) leave the exact-reference mode intact, (2) isolate the alternate implementation behind an opt-in flag, (3) log it as `PROPOSED/NOT_AUTHORIZED`, and (4) **exclude it from theory-fidelity acceptance and any advertised posterior probabilities**. If exact execution is infeasible at requested scale, report `EXACT_MODEL_RUNTIME_OR_MEMORY_INFEASIBLE` with measured hardware/resource evidence; **never silently replace the equation with a different algorithm**. Numerical approximation to a selected equation is allowed for performance experiments only with an explicit error assessment and disclosure, not as a new unlabeled mathematical method.

An `IMPLEMENTATION_BUG` is not a scientifically permissible deviation: fix it and document the resolution, then rerun every affected oracle. A mathematically equivalent rearrangement (tiling, stable log-sum-exp, exact segmented reductions, analytical sparsity) is **not a model deviation**, but still needs CPU/CUDA parity and floating-point error characterization.

**No unlogged changes.** Every implementation PR/commit must regenerate the traceability matrix and deviation list, compare to the previous revision, and fail its theory gate if code changed without an updated mapping. Source hashes tie the review to the actual files. Commit summaries and Codex completion notes must provide counts: `exact_model_operators`, `verified_equivalent_optimizations`, `controlled_numerical_approximations`, `unapproved_deviations`, `blocked_equations`, `untested_equations`.

### TF.5 Exact GPU semantics versus approximations — explicit classifications

| Engineering choice | Theory-fidelity decision |
|---|---|
| Streaming/tiled evaluation of **all** target-mixture components and **all** Stein pairs with the original weights, correct operation, and only bounded FP rounding | `EXACT_MODEL` with numerical parity; allowed |
| CUDA reductions, fused kernels, automatic differentiation **proven equal** to specified derivatives, stable log-sum-exp, matrix-free JVP/VJP, disk streaming | Equivalent implementation *only if* audited parity and active-set semantics are proven |
| Sparse kNN, radius cutoffs, voxel truncation, early candidate stopping, neighbor-degree reweighting, omitted low-mass terms | `NUMERICAL_APPROX` or `MODEL_APPROX`, **not** the exact Stein/KDE; require a deviation record, error bound and explicit authorization before use in accepted results |
| RFF/Nyström to replace a specified exact XYZ–RGB kernel | `NUMERICAL_APPROX`/`MODEL_APPROX`; cannot report **exact-kernel** output. **Exception:** if a finite feature map is explicitly chosen and frozen as the theoretical observation model `psi`, implementing *that finite map* exactly is `EXACT_MODEL` for the declared feature model, not an exact reproduction of the infinite-dimensional kernel |
| Subsampling *inference* points, dropping frames/cameras, replacing entire sequence with fixed lag, fitting only one moving support, fixed or GT-known group membership | Not the selected full-sequence model; reject in strict mode; may be separately labeled diagnostic/ablation |
| Replacing full covariance by diagonal, dropping shared source/reference errors, approximate evidence or pointwise independent edge probabilities | Statistical-model deviation: calibrated probabilities forbidden unless the approximation is separately justified, evaluated and explicitly authorized |
| Replacing the Stein field with nearest-neighbor ICP, per-point image gradients, 2D projections/photometric loss, OT, pure MMD gradient or a rigid Kabsch fit | Different algorithm or prohibited measurement, irrespective of qualitative similarity; not theory faithful |
| Choosing a step size, backtracking or PCG tolerance | A numeric solver policy, **not** a change to probability model, *provided* the accepted update is evaluated against the prescribed full objective; must log convergence and truncation/error; if the method differs from the theoretical update in (B3), state that optimizer modification explicitly |
| Omitting visibility events, conditioning on selected observations without their selection likelihood, or ignoring unexplained-new-support mass | Changes the target posterior; record deviation; cannot claim full posterior |
| Replacing posterior marginalization by plug-in MAP, a logistic regression, Bayesian information criterion alone, pseudo-likelihood or a discriminative classifier | `MODEL_APPROX`/`THEORY_CHANGE`; never label its output an exact marginal probability |

**Two levels of success must never be conflated:** (i) `THEORY_EXACT_ON_DEFINED_MODEL` (correct operators and normalized model, within certified numeric precision), versus (ii) `PHYSICAL_MOTION_RECOVERED` (empirical effectiveness on independent masked RGB-D sequences). Passing one does not establish the other.

### TF.6 Strict execution mode, invariants, and falsification tests

Add an opt-in `--svgd-theory-fidelity strict` (the **default for the new SVGD runner**, not for legacy `train.py`), plus a separate clearly named `experimental` mode for approved ablations. Strict mode must validate before allocating the full sequence:

- Every required theoretical term and normalization exists in `model_contract.json`; no `NOT_DEFINED_IN_THEORY` or unresolved `THEORY_CONFLICT` for the requested computation.
- No active `DEVIATION`/`MODEL_APPROX` path without explicit authorization. An accepted result must include `theory_fidelity=EXACT_MODEL` (or a separately authorized, fully labeled `CONTROLLED_APPROXIMATION`), **never** a bare boolean `100%` unsupported by operator evidence.
- Source data masks and time are valid; no dropped cameras or time windows masquerading as complete inference; canonical IDs stable; source-frame uncertainty shared.
- The KDE and Stein fields are computed using the exact declared kernel and global sums; for strict exact-kernel acceptance, **no sparse cutoff** without proven mathematically equal zero contribution.
- Whole-sequence likelihood and joint support posterior are correctly defined; no probability files until posterior normalization/approximation semantics have passed the corresponding gate.
- Failed covariance conditioning, singularity, missingness law, optimizer exhaustion, GPU OOM or insufficient data produces an explicit failure/status; never swaps to a different model, adds unexplained covariance jitter, substitutes zeros or claims convergence.

Mandatory theory-fidelity tests **in addition to existing gates**:

1. **Full force identity:** in a small fixture, compare *every* target component, responsibility, Stein attraction, Stein repulsion and accepted update to independent dense NumPy. Include unequal N/M, multiple cameras, heterogeneous 3×3 covariances, equal colors, colored near-overlap, far scores and N=1.
2. **Whole-sequence gradient identity:** finite-difference the **entire specified** negative log posterior against implemented analytic/JVP/VJP gradients on a 3–5 frame tiny scene, including dependence of visibility weights, reference color and shared initial XYZ. If active support changes nonsmoothly, name the conditional derivative and test objective acceptance across the transition; do not claim an invalid globally smooth Jacobian.
3. **No duplicate evidence:** duplicate a raw measurement *identifier* in the inference graph without creating a new independent observation. The posterior and `gamma` must be invariant to the bookkeeping duplication; separately test duplication of a genuinely different camera exposure against the selected camera-sampling law.
4. **Exactly normalized posterior (tiny oracle):** enumerate tiny binary support/edge/visibility hypotheses and integrate or quadrature the continuous latent trajectories. Compare normalizers, `p_static+p_dynamic=1`, all `gamma_i`, log Bayes factors and seed-risk to the production posterior method; separately measure approximation error if the production method is not exact.
5. **Permutation/camera/sampling invariances:** permute point order, source-camera loading order and debug-display density without changing physical input; verify original-ID-aligned outputs. Where camera permutation affects an explicitly modeled camera bias, report rather than assume invariance.
6. **Pair covariance and group consistency:** compare correlated distance-history covariance with Monte Carlo and ensure shared frame-0/source uncertainty is used once; verify group/edge hypotheses are coherent rather than independent non-transitive labels.
7. **Genuine motion tests:** static jitter, large rigid translation/rotation, a near-axis point whose own displacement is below measured noise, different moving supports, two supports initially touching, repeated RGB, textureless surface, occlusion, disocclusion and unknown support count. Check moving seed **precision first**; leave non-identifiable cases unresolved.
8. **GPU equivalence:** CPU FP64 versus tiled CUDA FP64 from inputs through all operative outputs, with per-operator max/p95 relative+absolute discrepancies, numerical conditioning, principled tolerances and error diagnostics. CUDA must not silently change accepted state or labels through different truncation/normalization.
9. **Mutation and regression tests:** deliberately remove repulsion, replace `1/N` by degree-normalization, drop color covariance normalizers, duplicate source observations, truncate neighbors, substitute MAP for model evidence, omit the reference covariance cross term. Every mutation must be caught by a named test/theory-contract audit.
10. **Real-code runtime verification:** check `git diff`, AST/source instrumentation or equivalent audit to demonstrate the tested functions are the functions used in the actual runner. A perfect independent oracle does not validate an uncalled production path.

**Gate outcomes:** `PASS_THEORY_EXACT`, `PASS_AUTHORIZED_APPROXIMATION` (not “100% exact”), `FAILED_THEORY_MISMATCH`, `BLOCKED_MODEL_UNSPECIFIED`, `BLOCKED_THEORY_CONFLICT`, `BLOCKED_EXACT_RUNTIME`, `NOT_TESTED`. **No global `COMPLETE` from partial operator success.** Publish an automated `reports/svgd/theory_parity_summary.json` containing per-equation status, deviation IDs, tolerances, hashes and evidence paths; include it in `run_manifest.json`, `status.json` and the visual debug numerical page. Mark outputs as **not suitable for later joint recovery** until all required Stage-1 inference gates pass.

### TF.7 Exact implementation completion report template (mandatory)

At task completion Codex must report, even on failure:

```text
CHECKOUT_COMMIT: <hash>
THEORY_REFERENCE: <PDF/TEX hashes + equation-label count>
SELECTED_MODEL_CONTRACT: <path + SHA256>
EXACT_EQUATIONS_IMPLEMENTED: <number and identifiers>
EQUIVALENT_CUDA_OPTIMIZATIONS: <number and identifiers>
NUMERICAL_APPROXIMATIONS: <number and DEV IDs>
MODEL_APPROXIMATIONS: <number and DEV IDs>
NON_EQUIVALENT_DEVIATIONS: <number and DEV IDs>
UNAPPROVED_DEVIATIONS: <number and DEV IDs>
BLOCKED_OR_UNSPECIFIED_EQUATIONS: <number and identifiers>
UNTESTED_EQUATIONS: <number and identifiers>
ALL_OPERATOR_PARITY_TESTS: PASS/FAIL/NOT_RUN <reports and measured discrepancies>
ALL_POSTERIOR_PROBABILITY_TESTS: PASS/FAIL/NOT_RUN <normalization and evidence errors>
CUDA_TESTS: PASS/FAIL/NOT_RUN <actual device and memory>
WHOLE_SEQUENCE_VALIDATION: PASS/FAIL/BLOCKED/NOT_RUN <frames, cameras and point counts>
VISUAL_DEBUG_QA: PASS/FAIL/NOT_RUN <screenshots and array parity>
LEGACY_REGRESSIONS: PASS/FAIL/NOT_RUN
OVERALL_THEORY_FIDELITY: <one of the TF.6 gate outcomes>
NEXT_UNRESOLVED_MATHEMATICAL_ITEM: <precise equation, missing law, or NONE>
```

**This directive supersedes any lower section that could be read as permitting a silent approximation, numerical heuristic or unreported model change.** The intended final status is not “Codex found a good-looking algorithm”; it is “every actual inference operator is traceable to a selected mathematical equation, every difference is quantified, and no unapproved difference is presented as theoretical fidelity.”

---

## 0. Mandatory preflight: inspect and pin the actual checkout

1. At the beginning of work execute `git status --short`, `git rev-parse HEAD`, `git log -1 --format='%H %cI %s'`, `python train.py --help`, enumerate the exact CUDA runtime/device, and record them in a dated audit file. The audited base of *this* document is `d0b470b…`. If HEAD differs, **diff HEAD against this base before touching code**; update the file map and explain compatibility. Never reset, discard or overwrite user changes.
2. Read the existing code, not merely the prior Markdown specs. Required code: `train.py`, `scene/dataset.py`, `scene/dataloader.py`, `scene/endpoint.py`, `scene/canonical_subsample.py`, `scene/noise_calibration.py`, `scene/config.py`, `scene/timing.py`, `estimator/projective_coherence.py`, `estimator/motion_coherence_support.py`, `estimator/supported_coherence.py`, `estimator/point_kalman_cuda.py`, `estimator/temporal_material_run.py`, `estimator/temporal_local_run.py`, `estimator/temporal_noise.py`, `utils/temporal_output.py`, `debug/projective_debug.py`, `debug/temporal_debug.py`, `debug/assets/projective_common.js`, `debug/assets/projective_free_3d.html`, `debug/assets/projective_coherence.html`, `tools/qa_projective_debug.py`, `tools/run_status.py`.
3. Freeze a baseline test report and source hashes **before** modifying anything; save successful commands and actual device, timing and results. The `README.md` describes the old four-pass path, but recent HEAD also offers opt-in `--observation-model temporal-rgbd` and modes `full`, `simplified`, `local-correlated`, `material-following` in `train.py`. This new stage must not hijack those flags.
4. Inspect `reports/temporal_dynamic_point_recovery.md`, `reports/temporal_local_correlated_validation.md`, `reports/temporal_measurement_information_study.md`, and `reports/gradient_motion_code_map.md`: the previously tested independent local temporal solvers did **not** reliably recover moving points even though they suppressed static drift. This SVGD implementation is a distinct experiment; never claim those failures have been solved in advance.
5. Use the repository's established scene clock, camera calibration, binary-mask loader, noise provenance, immutable run directories and original point IDs. Ground truth is **evaluation-only**, in a separate process after inference artifacts are committed.

### Verified current source map (base commit)

| Existing path | Verified purpose | Required action |
|---|---|---|
| `train.py` | CLI; stages `free, coherence, rigid, joint, support`; run manifests, source hashes, resume; legacy default | **Preserve**; optional explicit dispatch only after new runner passes |
| `scene/dataset.py`, `scene/dataloader.py` | Calibrated RGB-D dataset/timing/masks; `Dataset.observation(k, c)` yields `RGBDFrame`; `exact_binary_mask` rejects nonbinary inputs | Reuse unchanged |
| `scene/endpoint.py` | `samples(frame, config)` produces masked XYZ/RGB, source pixel IDs, and first-order XYZ covariance; `endpoint_samples(dataset, anchor, config)` | Reuse for canonical, validate raw sample shape |
| `scene/noise_calibration.py` | Temporal sensor noise and cross-camera mismatch diagnostics | Reuse as measured *inputs*, not as arbitrary SVGD bandwidths |
| `estimator/point_kalman_cuda.py` | 6D `[x,v]`, signed-time prediction, acceleration-noise covariance | Reuse process conventions if temporally smoothing candidates |
| `estimator/projective_coherence.py`, `estimator/supported_coherence.py` | Existing **downstream** trajectory coherence and moving-core selection; `run_coherence(free_dir, root, …)` expects a specific old free-pass schema | Do **not** call on fabricated SVGD outputs; useful diagnostic/control |
| `joint_reduction/joint_projective.py`, `joint_reduction/support_projective.py` | Later rigid/joint and static-versus-moving completion; current implementation tests static vs one accepted moving joint | Leave untouched in this phase |
| `debug/projective_debug.py` | `write_debug`, `export_free`, `export_coherence`; writes offline `debug/data/*.js`; `CAPS` controls display subsampling | Add sibling writer or explicitly compatible extension, not a replacement |
| `debug/temporal_debug.py` | Streaming temporal 3D/numerical viewer, full conditional traces | Reuse plotting/interaction concepts; don't reuse pixel-residual figures as the new measurement |
| `debug/assets/projective_common.js` | Shared Three.js viewer, point rendering, frusta, chart helpers | Reuse intact where possible |
| `tools/qa_projective_debug.py` | Headless Chrome screenshot and numerical projection checks | Add new independent QA for support pages; leave legacy QA passing |
| `utils/temporal_output.py` | Completion/journal and valid-frame checks | Reuse its atomic-completion discipline where appropriate |
| `tests/test_temporal_*`, `tests/test_coherence_parity.py` | Legacy and temporal numerical oracles | Must still pass |

**Do not change** the semantics of legacy source color, same-camera provenance, residual masks, update acceptance, Jacobians, camera ownership, old stage outputs, existing default CLI, or Lie-group rigid/joint code in Phase 1. No GT part count, GT joint, GT relative motion, physical pivot, mesh, surfel, supplied normals or hand-selected moving IDs are allowed as inference inputs. No image-space photometric or depth residuals in the new algorithm. Camera centers and rays are allowed **only as 3D visibility/provenance metadata**, not as an excuse to reintroduce pixel-matching.

---

## 1. Scientific objective and acceptance semantics

### 1.1 Inputs

At actual acquisition time `tau[t]`, define the observed foreground **colored 3D point set**

\[
Q_t=\{z_{jt}=(y_{jt},d_{jt},c_{jt},o_{jt},\Sigma_{jt})\}_{j=1}^{M_t},
\]

where `y ∈ R^3` is world XYZ [m], `d ∈ R^3` is RGB in the stated representation, `c` is source camera ID, `o` contains the source ray/camera-origin provenance, and `Sigma` contains the propagated measurement uncertainty (with geometric and RGB blocks identified separately). `j` is an **observation sample**, never a persistent material ID. Only provided mask-positive and valid-depth samples enter `Q_t`. `M_t` may vary arbitrarily.

Let the **canonical persistent catalog** from masked frame 0 be

\[
P_0=\{(x_i^0,a_i,\Sigma_i^0,\mathrm{source}_i,\mathrm{id}_i)\}_{i=1}^{N}.
\]

`x_i^0` is a world XYZ point, `a_i` its measured reference RGB **appearance** (do not confuse `a_i` with any density amplitude), `Sigma_i^0` its XYZ source covariance, and `id_i` its unchanged canonical ID. Choose a single documented RGB transfer/normalization policy; do not compare sRGB-intensity noise with linear-RGB noise as if equivalent. Propagate uncertainty of `a_i`; it is not noise-free.

### 1.2 Internal variables versus published output

Internally infer a provisional trajectory `X_i = {x_i(t)}` and optionally `v_i(t)`, visibility/support explanation variables, and an uncertain graph `Z` of same-rigid-support relationships. **These are nuisance variables used to discover motion; they are not the final point tracks or Stage-2 rigid poses.**

For each initial point define

\[
m_i = 1 \iff i\text{ belongs to a support with nonidentity physical motion during some part of }0{:}T.
\]

Report `gamma_i = P(m_i=1 | Q_0,…,Q_T, declared_model)` and `1-gamma_i`. Also report `probability_semantics`, model/evidence approximation and confidence diagnostics. Do **not** confuse `m_i` with `||v_i(t)||>threshold` or `displacement_i>threshold` at a particular frame. Near-hinge points may move less than RGB-D jitter and should remain unresolved rather than being forced static.

**The only authoritative Stage-1 deliverables** are `(original_point_id, p_static, p_dynamic, reliability_status)` per initial particle and an optional set of sufficiently supported *moving seeds*. Candidate trajectories, edge probabilities and score maps must be saved as debug/reproducibility artifacts, clearly identified as **provisional**.

### 1.3 Permitted and prohibited conclusions

- We seek **high precision of moving seeds**, not artificially high moving recall. Later rigid-motion recovery, joint inference and model comparison complete near-hinge/occluded support.
- Distinguish `NO_OBSERVATION`, `SCORE_OUT_OF_SUPPORT`, `NON_IDENTIFIABLE`, `NUMERICAL_FAILURE`, `UNRESOLVED_MODEL`, and `SUPPORTED_STATIC_OR_DYNAMIC`; do **not** turn missing evidence into `p_static=1`.
- With the point-cloud-only model, a uniformly colored tangentially translating plane is not individually trackable. An optimizer that produces a plausible cloud while swapping persistent material identity must **fail** the identity evaluation, even if MMD/likelihood improves.
- Relative-distance preservation alone is shared by **static and moving** supports. Boundary distance changes are evidence of relative motion only if alternative explanations (visibility, wrong provisional transport, noise) are considered.
- An exact hinge-axis point can be geometrically indistinguishable from a static point. It is acceptable and expected for Stage 1 to leave it uncertain; Stage 4 revisits it after joint recovery.

---

## 2. Theory: exactly what comes from SVGD and what does not

### 2.1 Original SVGD (reference derivation)

Standard Bayesian SVGD transports **samples `theta^l` of the same target distribution** `p(theta)` by

\[
\Phi(\theta)=\frac1L\sum_{l=1}^{L}\big[K(\theta^l,\theta)\nabla_{\theta^l}\log p(\theta^l)+\nabla_{\theta^l}K(\theta^l,\theta)\big],
\quad \theta^{(k+1)}=\theta^{(k)}+\eta_k\Phi(\theta^{(k)}).
\]

It is an RKHS-constrained first-order KL-descent direction under the Stein identity. The first term is kernel-averaged **score**; the second is the kernel-divergence (often repulsive) term. `k` counts optimization iterations, **not** frame time. The normalization constant of `p` is unnecessary; its score is mandatory. This theorem does **not** imply that treating each physical 3D surface point as an SVGD inference particle recovers physically correct trajectories or posterior support probabilities.

### 2.2 Our restricted *physical-particle* proposal (`PROPOSED`)

The physical particles `x_i(t)` have different persistent identities and fixed/noisy reference appearances `a_i`. We restrict transport to XYZ; RGB enters the target score and the pair kernel but is not transported as a mutable coordinate. This produces an **SVGD-inspired collective registration operator** rather than exact Bayesian SVGD on the joint posterior of all trajectories. Code/tests/logs must use the label `restricted_physical_particle_stein_transport` or equivalent; never claim the original KL theorem guarantees support inference.

### 2.3 Separate three unrelated scales and two times

1. `R_X,R_C` = actual measured **sensor/reference observation error covariances** (meters squared, normalized-color squared), possibly correlated temporally and within RGB.
2. `Lambda_X,Lambda_C` = **continuous target representation** covariances needed to obtain a spatial log-density/score from a discrete point set; these include sampling/model smoothing, not just sensor variance.
3. `H_X,H_C` = **SVGD particle interaction kernel** bandwidth covariances; these determine how corrections spread among persistent points and how strongly repulsion acts. They are *not* `Lambda` or `R`.
4. `t` = physical frame time; `k` = numerical optimization iteration. `eta_k` is a solver step, **not a velocity or physical dt**.

Every nonzero covariance must be SPD, unit annotated and provenance recorded. Do not add `jitter`, a bandwidth, or a kernel strength simply to obtain movement or prevent a singular solve. Numerical regularization, if mathematically unavoidable, must be separated from physical noise and quantified against a dense reference.

---

## 3. Exact appearance-guided 3D target score (CUDA operator A)

### 3.1 Define an explicit target, not an undefined attraction weight

For each frame, **candidate reference model**:

\[
\widehat p_t(x,a)=\sum_{j=1}^{M_t}\omega_{jt}\,\mathcal N_3(x;y_{jt},\Lambda^X_{jt})\,\mathcal N_3(a;d_{jt},\Lambda^C_{jt}),
\quad\omega_{jt}\ge0,\quad\sum_j\omega_{jt}=1.
\tag{A1}
\]

`omega_jt` means *sampling/exposure contribution of observed sample j*; it is **defined**, not an unexplained learned amplitude. Uniform `1/M_t` is a required **diagnostic baseline**, but can be biased when cameras oversample the same surface. A source-aware alternative must be accompanied by an explicit sampling/exposure model (e.g. equal camera contribution after legitimate per-camera surface coverage normalization); **do not invent physical surface area from a 3D point alone**. Record the policy and its effect in ablations.

`N_3` denotes the normalized 3D Gaussian density; a spatial `Lambda_X` is **not automatically equal to** 3D depth noise. The product assumes conditionally independent geometry/color target kernels; if using a joint 6D covariance, derive the joint score explicitly and test the cross-block terms.

Define unnormalized component `A_j = omega_j N_3(x;y_j,Lambda_X_j) N_3(a;d_j,Lambda_C_j)` and score responsibility `r_j=A_j/sum_m A_m`; these are **continuous density contributions only**, not material correspondences. Then

\[
\boxed{g_t(x,a)=\nabla_x\log\widehat p_t(x,a)=\sum_j r_j(x,a)(\Lambda^X_{jt})^{-1}(y_{jt}-x).}\tag{A2}
\]

This formula is exact for model (A1). The full Gaussian **normalizers** `|Lambda|^{-1/2}`, colour covariance and `omega` belong inside `A_j`; otherwise heteroscedastic responsibilities are wrong. Evaluate through stable log-sum-exp. Never normalize RGB differences with an unexplained constant or omit covariance units.

`g_t` is **not** `P(m_i=dynamic)`; it is a gradient of spatial/appearance target density. When the support is absent or occluded, the score must be marked unsupported rather than forcibly assigned to an arbitrary target sample. A separate explicit broad/background likelihood may be tested, but its prior mass and support must be declared, not silently inserted.

### 3.2 RGB flicker, geometry vibration and reference uncertainty

Preserve **measured** per-sensor XYZ and RGB error statistics, including RGB cross-channel covariance and temporal correlation when supported. `scene/noise_calibration.py` provides useful existing observations but does **not** fully determine the target bandwidth. Account for reference appearance uncertainty `Cov(a_i)` in colour comparison; the same uncertain reference must not be counted anew at every frame. Separate `Lambda_C` (density bandwidth) from RGB flicker covariance, and `Lambda_X` from resampling jitter.

Required noise diagnostics: stationary repeated XYZ component variances; 3x3 RGB increment covariance; increment anti-correlation; cross-camera bias; color changes at stable XYZ; quantized depth peaks; image/viewpoint-induced material change. These diagnostics are **input-noise studies**, not a new image-based estimation residual.

### 3.3 Mathematical and implementation checks

- CPU independent NumPy oracle for `log p`, responsibilities and `g_t`, finite-difference derivative of `log p` in x and, for diagnostic use, in appearance a.
- Permuting observations leaves the output unchanged within numerical reduction tolerance; duplicating samples is **not** automatically invariant under exposure-biased weights. Tests must distinguish these cases.
- Constant RGB under large enough `Lambda_C` reproduces spatial-only score; color-distinct nearby surfaces yield properly different scores.
- Empty masked frame: record `NO_OBSERVATION`, retain prior/uncertainty, never create a `NaN` score or a fake confident stationary label.
- Check kernel normalizers, units and log-sum-exp against float64 oracle, including scores far outside the observed support.
- Verify that reducing `Lambda_X` sharpens the target **and** narrows the attraction basin; do not claim coarse-to-fine removes fundamental identity ambiguity.

## 4. Restricted Stein transport of persistent XYZ particles (CUDA operator B)

### 4.1 Interaction kernel and exact derivatives

For any persistent particles `i,l` at the **same frame** define, as a *proposed* first model,

\[
K_{il}=\exp[-\tfrac12(x_l-x_i)^T H_X^{-1}(x_l-x_i)-\tfrac12(a_l-a_i)^T H_C^{-1}(a_l-a_i)].\tag{B1}
\]

`H_X` [m²] and `H_C` [normalized-RGB²] are declared SPD bandwidth matrices; `a_i` is the stored color of particle i. **Not** physical rigidity, not a noise covariance, not a material connectivity label. For `H_X=h_X² I`, `∇_{x_l}K_{li}=K_{li}(x_i-x_l)/h_X²`. Implement the full matrix-covariance version with the correct sign, i.e.

\[
\boxed{\Phi_{i,t}=\frac1N\sum_{l=1}^{N}[K_{li}\,g_t(x_{l,t},a_l)+K_{li}H_X^{-1}(x_{i,t}-x_{l,t})],}\tag{B2}
\]

\[
\boxed{x_{i,t}^{(k+1)}=x_{i,t}^{(k)}+\eta_k\Phi_{i,t}^{(k)}.}\tag{B3}
\]

This is a position-only physical-particle adaptation. For a color-aware kernel, the derivative still acts only on the **source XYZ**; reference color is fixed during each proposed transport step. Use true global `1/N` when comparing with the dense reference; don't silently renormalize by truncated neighbor count or graph degree. If a local approximation requires a different normalization, call it a distinct algorithm and ablate it.

Compute and store for each point and iteration **separately** `attraction`, `repulsion`, `total_stein`, accepted `delta_xyz`, all in declared dimensions. Implement `repulsion=off` (a required ablation), `kernel_color=off`, and `stein=off` (temporally regularized MMD-only motion) as explicit switches. No implicit amplitude cutoff, speed gate or rigid-part prior in the Stein operator.

### 4.2 SVGD is an optimizer proposal, not an extra independent measurement

We will evaluate a single **sequence-level** observation objective (Section 5) for statistical evidence. `Phi` is a candidate optimizer displacement that exploits the local target score; its magnitude is **not** a measured velocity, an independent Kalman innovation, a variance estimate, or a motion probability. A frame may take many inner iterations but has one physical timestamp. Do not stack `Phi` as a second observation row beside the score-derived distribution residual: **same RGB-D samples must not be counted twice**.

At each optimization proposal, accept/reject using the **complete declared nonlinear sequence objective**, including the same active observation set or a correct model that includes changes of support. Backtracking reduces *numeric step size* only; it must not suppress measurements or introduce a hidden motion threshold. Log scores for rejected proposals. If the objective has no descent direction, report `STALLED` (no fabricated convergence).

### 4.3 Why whole-sequence transport is required

Store `X[t,i,3]` for all frames (or memory-mapped tiled chunks) and **couple provisional motion across the full sequence**. Do not independently run unrelated pairwise SVGD registrations and subsequently interpret the resulting unrelated particles as the same material identities. Allow forward initialization and a backward refinement sweep; every accepted trajectory update is checked against the **global sequence objective**. Retain the frame-0 source-position uncertainty and optional latent canonical color uncertainty instead of treating every frame's source reference as independent evidence.

An initially stationary trajectory `x_i(t)=x_i^0` is a legitimate initial hypothesis. An optional warm start from **existing free-pass predictions** is an ablation, not a prerequisite; this new method must function from the actual canonical cloud and full observed sequence alone. Do not use GT poses, simulated joint parameters or GT moving labels to initialize transport.

---

## 5. Whole-sequence distribution observation (CUDA operator C): one objective, not blind matching

The **score** in (A2) is an optimization proposal. To obtain motion evidence and a covariance-aware data objective, specify a likelihood for **a sequence of observable cloud statistics**. Do not reinterpret `log p_t(x_i,a_i)` as an independent likelihood for every physical point; observed sampling, occlusions and re-use of the same raw data make that invalid.

### 5.1 A finite-dimensional XYZ–RGB feature measurement (`PROPOSED`)

Choose a *declared, fixed* differentiable feature map `psi(z) ∈ R^D`, where `z=[x (meters), a (color units)]` and `D` is the feature dimension. The theory contract must separately specify and freeze which exact observation feature model is selected. Provide two **independently identified** validation or execution backends (do not falsely label their distinct mathematical outputs interchangeable):

1. **CPU exact small-cloud kernel oracle:** `k(z,z')` is a Gaussian product kernel over XYZ/RGB using feature bandwidth `B_X,B_C`, with exact pairwise MMD for small N,M.
2. **GPU finite feature approximation:** fixed-seed random Fourier features or a documented deterministic Nyström approximation to the *same declared kernel*, with validation of feature approximation error and derivative accuracy against exact dense-kernel computations. For RFF: `psi_r(z)=sqrt(2/D) cos(w_r^T z+b_r)` with `w_r~N(0, diag(B_X^{-1},B_C^{-1}))`, `b_r~Uniform(0,2π)`, **drawn once per run** and hashed in the manifest; `∂psi_r/∂x = -sqrt(2/D) sin(w_r^T z+b_r) w_{r,XYZ}`. This approximation is explicitly statistical/numerical, not equivalent to the exact kernel at finite D.

For frame `t`, observed summary and predicted summary:

\[
\mu_t^Q=\sum_{j=1}^{M_t}\omega_{jt}\psi(y_{jt},d_{jt}),\qquad
\mu_t^P(X_t)=\sum_{i=1}^{N} \alpha_{it}(X_t)\psi(x_{it},a_i),\tag{C1}
\]

where `omega` sums to one on the defined observation support and `alpha` sums to one on the **comparable predicted visible surface support**. `alpha` is a **defined sampling/visibility contribution**, not motion-probability `gamma_i` and not an unmotivated learned weight. At the simplest *fully observed equal-support synthetic oracle* use `omega=1/M`, `alpha=1/N`. Real masked observations require Section 6's explicit visibility and coverage model.

The feature innovation and basic measurement model are

\[
e_t=\mu_t^Q-\mu_t^P(X_t),\qquad e_t\mid X_t\sim\mathcal N(0,\Gamma_t)\quad\text{(declared approximate observation model)}.\tag{C2}
\]

`Gamma_t` is the covariance of the *feature-mean discrepancy* from sensor error, RGB flicker, sampling and model mismatch; **not** arbitrary pointwise RGB or depth noise. It generally has cross-feature, cross-camera and cross-time dependencies. A diagonal `Gamma` is only an explicit `ASSUMED_DENSITY` ablation, not exact Bayesian inference. Estimate its structure from recorded noisy measurements and the declared sampling model, propagate shared canonical source uncertainty, and retain uncertainty if the data cannot identify it. Do not invert a singular covariance by silently adding a huge diagonal.

For fixed valid support, the full-cloud feature Jacobian for point `i` is

\[
H_{t,i}= -\alpha_{it}\,[\partial_x\psi(x_{it},a_i),\ 0_{D\times3}] \quad\text{for the residual }e_t=\mu_t^Q-\mu_t^P.\tag{C3}
\]

If normalized visibility weights `alpha` depend differentiably on `X`, their derivatives **must also be included**. If they are frozen inside a conditional linearization, label `H` as conditional; recompute and accept candidates against the actual nonlinear visibility/coverage-aware objective. Do not call (C3) the full Jacobian when `alpha(X)` is omitted.

`||mu_Q-mu_P||²` approximates squared MMD; with covariance `Gamma`, `eᵀGamma^{-1}e` is a *statistically scaled feature residual*. **Do not add both raw MMD and covariance-weighted MMD as independent data terms**, or add target-score likelihood as a second observation of the same raw frame.

### 5.2 Temporal dynamics and global objective

Use the existing 6D free state `s_it=[x_it;v_it]`, measured timestamps and signed process propagation from `estimator/point_kalman_cuda.py`. Denote the exact chosen transition `A_t` and process covariance `Q_t` with documented provenance. A candidate global Gaussian temporal term is

\[
E_{\rm time}=\tfrac12\sum_i\|s_{i0}-\bar s_{i0}\|^2_{P_{i0}^{-1}}
+\tfrac12\sum_{t=1}^{T}\sum_i\|s_{it}-A_ts_{i,t-1}\|^2_{Q_{it}^{-1}}.\tag{C4}
\]

For improper/zero process covariance, handle the exact constraint using an appropriate null-space method; **do not** force inversion of zero variance. Source XYZ and RGB priors require proper shared-source uncertainty bookkeeping. Prior velocities are unknown and must carry a **declared prior**, not an assumption that all points are stationary.

Define data term for the complete observed sequence

\[
E_{\rm obs}=\tfrac12\,e_{0:T}^{\top}\Gamma^{-1}e_{0:T}+\tfrac12\log\det\Gamma
\quad\text{(with correlated frame blocks when modeled)}.\tag{C5}
\]

The log determinant is necessary when `Gamma` depends on positions, bandwidth or observation support. If `Gamma` is fixed, it is constant for candidate-state comparisons but must remain in model-evidence calculations. `E_obs+E_time` is a **conditional Gaussian feature model**, not proof that the original raw point-cloud samples are i.i.d. Keep raw-versus-feature uncertainty semantics explicit in every report.

The accepted whole-sequence objective, once rigidity is enabled, is

\[
\boxed{E(X,V,Z,m)=E_{\rm obs}+E_{\rm time}+E_{\rm rigid}(X,Z)+E_{\rm support\ prior}(Z,m)+E_{\rm visibility/unexplained}.}\tag{C6}
\]

Every term must have a declared likelihood/prior interpretation or be tagged `SURROGATE_ONLY`, never silently combined as if a calibrated log-posterior. Do not multiply independent likelihood terms from the same sensor observations without a joint noise model.

### 5.3 The IEKF/IEKS connection, without falsely assimilating SVGD twice

The feature observation (C2) is the **actual** observation function for an optional feature-space iterated Kalman smoother. For each frame, `H = [H_{t,1},…,H_{t,N}]` and the joint observation information includes off-diagonal point coupling `H_iᵀ Gamma^{-1} H_l`. This differs from the legacy per-point material-following RGB-D IEKF: it is a **distribution-level joint observation**. A matrix-free `H u` and `Hᵀ v` implementation is required; never instantiate dense `(6N)²` or `(6NT)²` matrices. The normal equations include the fixed temporal prior and may be solved using matrix-free PCG or a proven sparse factorization; track true objective decreases.

SVGD may initialize/propose `Delta X`; the feature measurement supplies **one** likelihood and the temporal smoother supplies the process model. A correction made by SVGD is **not** treated as an additional Gaussian measurement. The posterior covariance from a linearized joint feature model is at most a **conditional Laplace/Gaussian approximation**; use Joseph/information consistency or a square-root solver as appropriate and verify with dense tiny oracles. Do not report the old independent 6x6 covariance as the exact posterior after common feature measurements or graph coupling.

### 5.4 Feasibility versus moving-support evidence

Feature agreement can be excellent even for **wrong material-point permutations**, particularly in repeated or flat texture. Evaluate trajectories and support evidence separately; a decrease in `E_obs` is necessary for transport plausibility, but not sufficient for a physical motion claim. If the target is missing because of occlusion, `Phi` may still be nonzero due to another surface; visibility/support must prevent promoting that into a confident moving label.

---

## 6. Visibility, unequal cardinality, raw masking and new observations (point-only)

1. **Mask input:** both the initial canonical point catalog and every observed `Q_t` consist **only** of source foreground mask-positive valid-depth samples. Call the existing strict mask loader and `scene.endpoint.samples`. No background sample may contribute to a target density, feature observation, noise estimator or new-point explanation. Record raw sampled frame/camera counts.
2. **No point deletion:** a persistent initial point that is out of view, behind another object, or unsupported at frame t retains its identity/state/uncertainty. Exclude invalid *evidence*, not the physical point. Its unobservability cannot be interpreted as stationarity or motion.
3. **3D visibility only:** retain per-sample camera center and ray provenance. Use 3D angular/ray frontmost ordering, the measured 3D point locations and their uncertainties to evaluate whether a candidate is occluded/visible; a camera may contribute to geometric evidence only when its **observed** 3D surface could represent that candidate. Never use RGB/depth **pixel residuals**, image gradients, or mask-as-image-energy in this branch. Existing `estimator/temporal_zbuffer.py` may be studied for ordering and determinism, but its first-hit rasterizer cannot be installed as a disguised image-domain motion measurement. If a trustworthy point-only visibility model is not implemented, return `VISIBILITY_UNRESOLVED` and do not issue confident point labels.
4. **Camera overlap and sampling:** duplicate/overlapping camera surface measurements are not independent physical area. Camera-source IDs, angular sampling footprints, calibration and uncertainty must be retained. Normalized `1/N`/`1/M` removes cardinality scaling but **not** this sampling bias. Show the ablation `uniform_sample_weights` versus a declared exposure-aware policy; assess bias under synthetic camera-density doubling.
5. **Newly uncovered observed geometry:** it does **not** require transporting an unrelated old point onto that surface. Model unexplained observations with an explicit unrepresented/new-surface explanation (latent coverage term or normalized background component) and report its mass per frame/camera. This **Phase 1 does not initialize or commit new canonical IDs**. Actual point births and missing-support completion are downstream. A moving model must not win by ignoring unrepresented observations; the likelihood must include coverage/missingness consistently.
6. **Coherent but wrong occluder:** construct the `static handle hidden by moving blade` oracle. An occluded static point must not be scored using the blade's colour/geometry as if it were the point's material appearance. Visibility likelihood and `UNOBSERVED` status are mandatory, not optional cosmetic diagnostics.
7. **State-dependent visibility:** when moving candidates change visibility, compare full observation probabilities, including the likelihood of the *visibility event*. A proposal cannot appear to decrease cost merely by making inconvenient valid samples disappear. If this likelihood has not yet been specified and tested, freeze the comparison support during the conditional step and mark results `VISIBILITY_CONDITIONAL_APPROXIMATION`, not a validated complete posterior.
8. **No latent GT exposure:** never use segmentation into moving/static parts, joint state, joint angle, true material IDs or scene GT transforms in target construction, SVGD initialization, camera weighting, visibility or support inference.

### 6.1 Required observation accounting

Persist counts per frame/camera: raw pixels, valid-depth, foreground-valid, exported world points, target density mass, visible initial particles, occluded particles, unsupported particles, unexplained foreground observations, accepted/failed transport, contribution of each camera to `E_obs`, and number of unique source IDs. When a count is zero, make its status visible. Store enough source provenance to attribute a spurious motion force to the contributing 3D observations.

---

## 7. Whole-sequence rigidity and relative appearance (CUDA operator D)

### 7.1 A sparse **candidate** relation graph, not a known part segmentation

Create a graph `G=(V,E)` with one node for each **original persistent ID**, not for each raw observed sample. Graph construction is a *hypothesis generator* from frame-0 XYZ/RGB and their measured uncertainty. Support multiple components, multiple moving parts, and parts that are initially in contact. Use sparse 3D reference-space neighborhoods; do not declare that a kNN edge means the endpoints are on one rigid part. Preserve IDs and edge provenance; make the candidate graph and its coverage testable. Spatial search neighbor count/cell dimension are computational hyperparameters and must be distinguished from physical/statistical bandwidth.

Possible prior proposal: build a deterministic, sparse source-neighborhood graph, with an **uncertain** edge variable `z_il ∈ {0,1}` meaning that points i,l share one rigid motion throughout the sequence. Let `rho_il=P(z_il=1 | observations, declared_model)`. These are **not** moving-versus-stationary probabilities. Do not choose a fixed part count `K`, assume the hinge axis, or turn reference proximity into a hard same-part label.

### 7.2 Pairwise rigidity invariant, including the exact Jacobian

Given provisional trajectories `x_i(t)` and `x_l(t)`, define

\[
r_{il,t}=x_i(t)-x_l(t),\quad d_{il,t}=\|r_{il,t}\|,\quad
 e_{il,t}=d_{il,t}-d_{il,0}.\tag{D1}
\]

For genuine same-rigid-support points `e_il,t=0` for all t, irrespective of translation, rotation or unknown pivot. Do not use self-edges or coincident pairs with undefined distance Jacobians. When `d>0`, the derivative of `d_il,t` wrt `(x_i(t), x_l(t))` is `(u_il,tᵀ, -u_il,tᵀ)` where `u_il,t=r_il,t/d_il,t`. Since the **same noisy frame-0 distance** occurs in every `e_il,t`, the whole-sequence rigidity covariance is NOT diagonal:

\[
\Omega_{il}[t,s] \approx J_{il,t}\,\mathrm{Cov}(X)\,J_{il,s}^{\top},\tag{D2}
\]

with `J_il,t` formed from derivatives of `d_il,t - d_il,0` wrt *all participating frame-0, frame-t and frame-s point positions*. Preserve **shared reference** covariance and any cross-point correlations carried by the chosen approximation. Full joint covariance may be impractical: implement a named diagonal-across-points approximation plus a tiny exact-oracle comparison, and report lost cross-point information. If initial source point observations from different cameras are correlated, do not silently treat them as independent.

For same-rigid hypothesis `H^R_il`, a basic conditional likelihood is

\[
p(e_{il,1:T}\mid H_{il}^R,\mathrm{other\ latents})=\mathcal N(e_{il,1:T};0,\Omega_{il}+\Omega^{model}_{il}),\tag{D3}
\]

where `Omega^model` is **declared** rigidity-model discrepancy, not arbitrary numerical damping. Compare with `H^I_il` (independent motions) using its properly specified temporal prior and **integrated likelihood**. Never let the more flexible alternative win by comparing only raw optimized residuals. A pair with low strain does **not** prove moving membership: two static points also have low strain.

### 7.3 Differential consistency and changing cross-support edges

For diagnostic velocity trajectories, verify the rigid differential identity

\[
(x_i-x_l)^\top(v_i-v_l)=\tfrac12\frac{d}{d\tau}\|x_i-x_l\|^2\approx0\quad (z_{il}=1).\tag{D4}
\]

This is a check on the estimated motion, not an additional independent sensor measurement. Boundary relationships connecting independently moving bodies may change; however, a cross-support pair on a special rotation geometry can keep constant distance. Use many edges and many frames. A sparse rigidity graph can contain flexes; low edge strain alone does **not** guarantee a globally rigid set. Diagnose rigidity rank, and on candidate connected supports perform a **held-out** common-rigid-motion fit (unknown `R_t,b_t`) as an *internal model-evidence test*. This nuisance fit is **not** Stage-2 final motion reconstruction and must not be passed as a ground-truth joint or imposed physical center.

### 7.4 Appearance in support inference: do not use the same RGB observations twice

Keep material reference colors attached to original IDs through transport. The observed **spatiotemporal placement of those colors** is already evaluated in `E_obs` and `g_t`. Relative reference RGB differences `a_i-a_l` may condition a graph *prior* or Stein kernel, but this is **not** independent new temporal evidence. If tracking an additional per-particle latent appearance `A_i(t)` for relative invariance, define its observation model, illumination nuisance factors and cross-channel/temporal covariance; do not score the same raw RGB again as a statistically independent likelihood. Required ablations: spatial-only vs XYZ–RGB, independent RGB noise vs measured cross-channel noise, and shared temporal flicker/illumination nuisance on/off.

---

## 8. Infer static/dynamic probabilities from the **whole sequence** (operator E)

### 8.1 Formal latent-variable target

For each initial particle `i`, let `m_i∈{0,1}` mean membership in a rigid support undergoing nonidentity motion at *any* time during the sequence. Let `Z={z_il}` be the unknown same-rigid-support relationships; let `X,V` be provisional full trajectories; `U` denote visibility, unrepresented-surface and reference-appearance nuisance variables. The correct conceptual target is

\[
\boxed{p(X,V,Z,m,U\mid Q_{0:T})\propto p(Q_{0:T}\mid X,V,U)\,p(X,V\mid Z,m)\,p(Z,m)\,p(U).}\tag{E1}
\]

This is **one** posterior based on **one** observation model. The moving probability is a marginal

\[
\gamma_i = \sum_{m_{-i},Z}\int p(X,V,Z,m_i=1,m_{-i},U\mid Q_{0:T})\,dX\,dV\,dU.\tag{E2}
\]

Exact integration is infeasible at ~10^5 points; the implementation must name and validate an **approximation** (structured variational, message passing with candidate components, or marginal-likelihood approximation), not output a calibrated-looking sigmoid of displacement, strain or SVGD force. Saving internal optimized point trajectories does **not** supply `gamma_i` without this inference step.

### 8.2 Competing static versus moving explanations

For each *candidate coherent support* `S`, compare two generative explanations of the **same observed sequence**:

- `H_S`: its points remain at fixed world coordinates `x_i(t)=x_i^0` modulo canonical source uncertainty; measured samples vary because of sensor noise, RGB flicker, sampling, visibility and other supports.
- `H_M`: its points undergo a **common unknown** time-varying rigid transformation `x_i(t)=R_S(t)x_i^0+b_S(t)` up to declared geometric noise, where `R_S(t)∈SO(3)` and `b_S(t)∈R³` are *latent nuisance variables*, anchored to identity at frame 0. **No known physical pivot, no known joint type and no GT trajectory.** Optionally contrast with a completely free-transport alternative in diagnostic mode to expose model mismatch.

The motion model needs a **proper normalized temporal prior** on nuisance transformations/motion increments (e.g. smooth stochastic increments with declared covariance); the stationary model is the identity special case. Large motion, small rigid motion near a hinge and static high-frequency jitter are discriminated by **whole-sequence evidence of common motion**, not individual distance thresholds. A model with arbitrary per-frame motion has greater flexibility; its integrated evidence or explicit complexity/entropy term must reflect this.

For hypothesis `h`, write the normalized evidence

\[
L_S^h=\int p(Q_{0:T}\mid\Theta_h,U,h)\,p(\Theta_h,U\mid h)\,d\Theta_h\,dU,\quad
P(H_M\mid Q)={\pi_M L_S^M\over \pi_M L_S^M+(1-\pi_M)L_S^S}.\tag{E3}
\]

`Theta_h` includes motion nuisance and uncertain initial points under hypothesis h. **Both** likelihoods account for the same observation mechanisms, including hidden/unexplained data. An optional Gaussian/Laplace approximation near an actually validated local optimum is

\[
\log L_h\approx -E_h(\hat\Theta_h)+\frac{d_h}{2}\log(2\pi)-\frac12\log\det \nabla^2 E_h(\hat\Theta_h),\tag{E4}
\]

**only if** `E_h` contains all normalization constants (data likelihood, priors, log determinants), the Hessian is positive definite in an identified gauge, and the posterior is locally approximately Gaussian. Handle symmetries/multimodality with explicit multiple hypotheses or a different approximate integration method. Never call a raw optimized cost difference a Bayes factor. No arbitrary logistic or hard-coded large-displacement threshold.

### 8.3 Structured point marginalization

The graph induces soft memberships `rho_il` and candidate coherent supports. A point's `gamma_i` must reflect uncertainty over its possible support(s), including ambiguity at boundaries and near axes. A proposed structured variational approximation can maintain `q(Z,m,Theta)` and optimize its ELBO using the **same** full-sequence feature likelihood and priors. Discrete labels may use exact local enumeration in tiny tests, sum-product/message passing for small candidate graphs, and a documented structured variational relaxation at scale. The implementation must demonstrate that it does not silently assume independent labels or a known number of parts; enforce equivalence/transitivity of any *hard* same-support partition (uncertain edge marginals need not individually be transitive).

Do not overwrite unresolved particles with `static`. An unobserved hinge point may have `gamma_i` near its *declared prior*, while its reliable moving neighbors have high `gamma`. Later Stage 4 tests the point under the optimized joint. Similarly, two independent moving supports may exist: Stage 1 reports moving **membership** without requiring their final identities, but candidate graph/model evidence must not forcibly merge them.

### 8.4 A practical mandatory staging of inference

Implement **separately testable** passes within the joint iterative solver:

1. `INIT`: all candidate trajectories initialized (stationary copy or separately tagged baseline warm start), with canonical source/covariance.
2. `TARGET`: build XYZ–RGB observed target scores and feature observations at each selected frame; log target noise and coverage.
3. `STEIN_PROPOSAL`: compute attraction and repulsion with exact specified equations; propose a simultaneous XYZ update for the provisional particle paths.
4. `TEMPORAL_REFINEMENT`: jointly refine with the common feature likelihood and declared `[x,v]` process model (not a second score measurement). Check a true common objective.
5. `RIGID_GRAPH`: compute uncertain full-sequence rigidity residuals/covariances and candidate support graph, including held-out rigid-consistency tests.
6. `MODEL_EVIDENCE`: compare stationary and coherent-moving explanations with proper normalization/marginalization approximation; update `rho_il`, `gamma_i`, and unresolved status.
7. `JOINT_REFINEMENT`: use soft same-support constraints to refine provisional transport, **without** hard-coding moving membership or a part count; rerun stages 3–6 until a declared convergence criterion is met or report a named failure/cap.
8. `SEED_SELECTION`: export only evidence-backed moving seeds and the complete per-point posterior approximation/status.

Early test runs may stop after stages 2, 3, 4 or 5 with `DIAGNOSTIC_ONLY` outputs. **Do not** report final probabilities if model evidence (stage 6) is missing or uncalibrated. Any heuristic interim `motion_score` must use that name and not be represented as `p_dynamic`.

### 8.5 Moving-seed selection and uncertainty

Use a **declared** expected false-discovery/misclassification-risk policy. The existing joint policy `configs/image_joint_policy.json` has a declared tail probability; if reusing it, state that it is a *requested reliability level*, not automatically a calibrated posterior threshold. One simple risk rule for a set of seed IDs `S` is

\[
\mathrm{posterior\ expected\ false\ discovery\ fraction}(S)=\frac{\sum_{i\in S}(1-\gamma_i)}{\max(1,|S|)}.\tag{E5}
\]

Select a conservative set respecting a declared risk budget **only after** verifying probability calibration on held-out data. Report the achieved expected and empirical (post-hoc GT-only) false discovery rates separately. If the model has no evidence-calibrated probabilities, produce an **unresolved score map** and no falsely certified seeds.

### 8.6 Probability evaluation and proper calibration terminology

For synthetic validation with GT **outside inference**, test reliability curves, expected calibration error, Brier score, log loss, static false-moving rate, moving-seed precision and per-part coverage. If model fitting uses a learned calibration map from labeled data, record the trained model, train/test separation and GT usage explicitly; that is a **different, supervised calibration setting** and must not be silently introduced. Primary inference must remain GT-free.

In manifests label result `APPROXIMATE_MODEL_POSTERIOR` if a normalized approximate generative posterior was used, `UNCALIBRATED_SCORE` if only a heuristic was used, or `POSTERIOR_CALIBRATED_ON_HELDOUT_<source>` only after suitable independent empirical verification. Never claim universal frequentist calibration from a Bayes formula written on paper.

---

## 9. GPU/CUDA architecture: bounded memory, exact small oracle, no hidden CPU fallback

### 9.1 New code modules (proposed file split)

Implement under **new names**; inspect and adjust imports to the actual HEAD before writing:

```
scene/pointcloud_sequence.py             # strict masked, point-only frame materialization and source IDs
estimator/svgd_target_score.py           # Gaussian-mixture log p, XYZ score, CUDA sparse candidate scan
estimator/svgd_particle_operator.py      # Stein kernel, attraction, repulsion, accepted updates
estimator/svgd_feature_measurement.py    # XYZ–RGB MMD / feature map, H/H^T and covariance-aware residuals
estimator/svgd_visibility.py             # point-only 3D visibility/coverage; explicit unresolved reasons
estimator/svgd_temporal.py               # full sequence temporal prior/smoother, no duplicate measurement
estimator/svgd_rigidity.py               # reference graph, distance residuals, time covariance
estimator/svgd_support_posterior.py      # latent edge/support evidence, normalized model comparisons
estimator/svgd_support_run.py            # opt-in runner, journals, manifests and output contract
configs/svgd_support.json                # documented policy + all model provenance
utils/svgd_support_output.py             # atomic/checksummed write and frame validity

debug/svgd_support_debug.py              # writer reusing projective_common.js
  debug/assets/projective_svgd_support.html
  debug/assets/projective_svgd_numerical.html

tools/run_svgd_support.py                # explicit CLI, unique SSD run directory
tools/qa_svgd_support_debug.py           # headless Chrome + independent numeric checks
tools/profile_svgd_support_cuda.py       # CUDA events, memory, kernel timings

tests/test_svgd_target_score.py
tests/test_svgd_stein_operator.py
tests/test_svgd_feature_measurement.py
tests/test_svgd_mask_visibility.py
tests/test_svgd_temporal.py
tests/test_svgd_rigidity.py
tests/test_svgd_support_posterior.py
tests/test_svgd_support_runner.py
tests/test_svgd_cuda_parity.py
tests/test_svgd_visual_debug.py
```

Use explicit `__init__` exports only when needed. Do not import the old `estimator/phase1.py` or `estimator/sparse_iekf.py` as if they implement this experiment; the repository code map categorizes several such modules as **legacy rendered-splat baseline**. Reuse the existing **actual** dataset, noise, timestamp and GPU utilities.

### 9.2 CUDA primitives and layout

- Input SoA tensors, contiguous: `obs_xyz[M,3]`, `obs_rgb[M,3]`, `obs_cam[M]`, `obs_source_id[M]`, `obs_cov[M,3,3]`, canonical `xyz0[N,3]`, `rgb0[N,3]`, `original_id[N]`, trajectory tiled `(frames,point_chunk,3)`; keep provenance IDs as integer64. Never materialize `N×M`, `N×N` or `(6NT)×(6NT)` dense arrays for production.
- **Target-score search:** GPU spatial bins or sorted Morton/hashed 3D cells using Torch CUDA, `sort/searchsorted` or verified CUDA/Triton kernels. Enumerate only candidates with a *quantified* negligible Gaussian tail, or use an independently validated low-rank/fast-Gaussian-transform approximation. A fixed `top_k=32`, arbitrary radius, or nearest sample **is not an exact substitute** for (A1). Parameterize approximation by a declared numerical-error budget and prove/test its achieved log-density/score error against dense oracles on tractable clouds. For complex heterogeneous covariance, bound omitted tails conservatively; otherwise report `APPROX_SCORE_UNCERTIFIED`.
- **Log-sum-exp:** compute local Gaussian exponent, determinant normalizers and weighted score numerator with stable GPU segmented reductions; accumulate required denominators in FP64. Verify reduction order determinism to a documented tolerance. Never let underflow silently yield a zero score at useful support.
- **Stein pair kernel:** use the same neighborhood/indexing primitives to evaluate (B1) on persistent particles and gather `g_l`, positional deltas and repulsion. Keep the **global 1/N normalization** for faithful reference parity. Sparse truncation must be certified against dense full-kernel force on synthetic tests; include fallback or named error when active mass is insufficient.
- **MMD/feature map:** produce frame mean features and analytic JVP/VJP (`H u`, `H^T w`) on CUDA, without storing `D×N` dense Jacobians across all frames. RFF generation uses deterministic seed + fixed frequencies and is saved/hashed. The exact small-cloud kernel backend is CPU/NumPy-only oracle, optionally GPU for parity.
- **Temporal/rigidity:** use GPU batched 3D vector operations, sparse graph edge arrays `[E,2]`, segmented reductions for means and covariance accumulations, and matrix-free PCG with convergence/residual certificates for correlated feature measurements. Small test oracles may form dense covariances; production cannot.
- **Precision:** reference correctness in float64; explicit opt-in mixed precision (FP32 geometry and log kernels, FP64 scores, priors, Hessian/covariance/log evidence when needed) only after numerical acceptance. For tight spatial kernels and ~mm jitter, test cancellation and log-det conditioning. Disable TF32 in strict parity mode. Check NaN/Inf/PSD at operator boundaries.
- **GPU budget:** support 8GB and 32GB classes by streaming/tiled memory usage; `--svgd-workspace-mib` is an **operational memory budget**, never an undocumented statistical bandwidth or physical radius. Save intermediate sequence tensors as disk-backed `.npy`/compressed chunks when necessary, using existing SSD output constraints. Cache only live frames/chunks; never preload every full RGB-D frame on a limited-memory GPU.
- **Torch CUDA is a valid first accelerated implementation only when it implements the frozen mathematical operators.** If Triton is available, implement a second, opt-in kernel backend and prove parity. The existing `train.py --cuda-backend triton` explicitly rejects Triton: do not claim it is implemented by setting that existing flag. Use a new `--svgd-kernel-backend torch|triton` with strict availability checks.
- **No silent CPU fallback** after GPU failure. CPU is allowed only for independent reference or explicit `--device cpu` small tests. On CUDA OOM, save a journal with `CUDA_OOM` and current memory counters, then exit; retry only with explicitly changed chunk/workspace parameters. Do not quietly subsample inference points or frames.

### 9.3 Testable CUDA kernel interfaces

Document exact tensor shapes, dtype, device, units, validity reason and stride requirements. Suggested contracts:

```python
build_masked_cloud(dataset, frame:int, cameras:Sequence[int], config) -> PointCloudFrame
# xyz[M,3], rgb[M,3], xyz_cov[M,3,3], source_ids[M,3], ray_origin[M,3],
# frame/camera sampling provenance and validity counts

target_score(query_xyz[N,3], query_rgb[N,3], observed:PointCloudFrame,
             density_policy, workspace_bytes:int) -> ScoreResult
# log_density[N], grad_xyz[N,3], support_status[N], candidate_count[N],
# numerical-tail-bound[N], optionally responsible camera breakdown

stein_field(xyz[N,3], rgb[N,3], grad_xyz[N,3], kernel_policy,
            workspace_bytes:int) -> SteinResult
# attraction[N,3], repulsion[N,3], total[N,3], pair_count[N],
# truncation_error_bound[N]; global 1/N

feature_residual_and_actions(xyz[N,3], rgb[N,3], observed:PointCloudFrame,
                             visibility, feature_policy) -> FeatureOperator
# residual[D], calibrated/assumed covariance contract, forward JVP,
# VJP, (optionally) covariance-vector product, no dense N*D Jacobian

rigidity_evidence(x_seq[T+1,N,3], graph[E,2], covariance_contract,
                  time_sec[T+1]) -> RigidityResult
# pair residuals, full time covariance/declared approximation,
# support validity, node/edge diagnostics

support_posterior(sequence_evidence, graph_evidence,
                  normalized_priors, inference_policy) -> SupportPosterior
# gamma[N] or explicit uncalibrated failure; rho[E], model evidence,
# approximation status, uncertainty, reliable seed IDs
```

Allow NumPy and Torch entry points but do not silently move multi-million-point tensors from GPU to CPU each iteration. CPU synchronization must occur at planned checkpoints only.

## 10. CLI, run lifecycle, compatibility and reproducible output

### 10.1 Opt-in runner first; do **not** switch the existing default

Implement `python -B tools/run_svgd_support.py --scene SCENE --output NEW_EMPTY_DIR --device cuda --config configs/svgd_support.json --debug full --frame-limit 3` as the canonical **initial** entry point. Arguments should reuse the actual scene-folder/root, acquisition-timing, noise-source, canonical-subsample and original-ID conventions of `train.py`. Also support explicit `--dry-run`, `--debug-point-ids`, `--debug-frames`, `--svgd-workspace-mib`, `--svgd-kernel-backend torch|triton`, `--stop-after-step INIT|TARGET|STEIN_PROPOSAL|TEMPORAL_REFINEMENT|RIGID_GRAPH|MODEL_EVIDENCE|JOINT_REFINEMENT|SEED_SELECTION`, and an explicit output verbosity/memory policy. Use the repository's existing `RESULTS_ROOT` convention when output is omitted; results must be outside the repository tree and must not overwrite any nonempty directory. A `--frame-limit` run must be labeled `PREFIX_DIAGNOSTIC`, never a validated whole-sequence moving-support estimate.

After this runner passes its gates, an **optional** `train.py --pipeline svgd-support` dispatch can call the **same** tested runner functions. Preserve the default `train.py` CLI (`four-pass projective`) behavior and all old observation modes. Do not change the existing `INFERENCE=['free','coherence','rigid','joint','support']` stage ordering/resume logic to trick it into accepting a different schema. An SVGD run needs its **own run kind**, source hashes, paths, status semantics and resume validation. Reject incompatible combinations (e.g. `--pipeline svgd-support --temporal-mode material-following` or old `--stage rigid`). If new paths/data do not yet support resume, fail explicitly rather than copying fake complete stages.

### 10.2 Required configuration schema

Add `configs/svgd_support.json` with documented keys, types, units, provenance and value validation; do not disguise research choices as defaults calibrated from the data. Required named groups:

```text
schema_version / implementation_commit / research_mode
canonical: frame0 / source_ids / subsampling_policy / rgb_transfer
observations: strict_masks / provenance / camera_overlap_policy / visibility_model
sensor: depth_noise_source / rgb_noise_source / temporal_correlation_policy
reference: shared_source_covariance / color_reference_uncertainty
target_density: Lambda_X / Lambda_C / representation_scale_method / source_weights
stein: H_X / H_C / attraction_enabled / repulsion_enabled / kernel_color_enabled
stein_numerics: optimizer / step_policy / convergence / tail_error_budget
features: xyz_rgb_kernel_B / D / fixed_seed / feature_method / Gamma_model
motion_prior: process_model / Q_provenance / initial_P / timestamp_policy
rigid_graph: edge_proposal_method / covariance_approximation / support_prior
model_evidence: exact_tiny|laplace|structured_variational / priors / failure_policy
seed_selection: posterior_risk_alpha / required_probability_semantics
compute: device / workspaces / chunk_bytes / backend / precision
outputs: trace_ids / debug_frames / display_subsampling_only / save_internal_trajectories
```

**Every parameter must have one of**: `derived-from-measured-input`, `declared-research-assumption`, `numerical-error-budget`, `hardware-budget`. Do not invent specific millimeter/centimeter attraction radii, RGB thresholds, rigid strain tolerances, number of parts, or low/high motion speed gates. For unset statistical parameters, print `BLOCKED_MODEL_UNSPECIFIED` with a complete diagnostic suggesting a documented calibration/selection path rather than choosing a convenient hidden constant. Numeric iteration caps are permitted as operational safeguards but hitting a cap must report `ITERATION_CAP`, not convergence. Keep a config hash and all relevant source SHA-256 hashes in the manifest.

### 10.3 Immutable artifact contract

Use a unique new result directory, for example

```
RUN/
  run_manifest.json                  # git SHA, source hashes, scene, timing, policy, hardware
  result.json                        # COMPLETE / INCOMPLETE / FAILED / UNRESOLVED
  noise_calibration.json              # measured data and explicit qualifications
  svgd_support/
    canonical.npz                     # x0, rgb0, original ids, 3D source, covariance, selected_ids_sha256
    model_contract.json               # what each probability actually conditions on
    p_dynamic.npy                     # float64 [N], ONLY when a normalized posterior approximation exists
    p_static.npy                      # float64 [N] = 1-p_dynamic
    probability_status.npy            # [N] enum, distinct from probability value
    reliable_moving_seed_ids.npy       # original IDs, not local displayed indices
    support_edges.npz                 # endpoints, same-support rho, source provenance
    transport_provisional.npy          # optional float32/float64 [K,N,3], labeled INTERNAL_NOT_FINAL_TRACKS
    transport_status.npy              # [K,N] visibility/optimization status
    trajectory_covariance_summary.npz # precise approximation semantics
    observed_counts.csv               # frame × camera and reasons
    objective_timeline.csv             # complete common cost and decomposition per iteration
    posterior_evidence.json            # priors, normalization, approximations, uncertainties
    performance.json                  # CUDA/event timings, peak GPU memory, dense-oracle parity
    completed_frames.json             # checksums, status, actual processed frames
    status.json                        # failure modes, stage/gate, no fake completion
    trace_points/ORIGINAL_ID.json     # complete per-selected-point trace and contributing camera evidence
  debug/
    index.html
    svgd_support_3d.html
    svgd_support_numerical.html
    data/svgd_support_meta.js
    data/svgd_support_f0000.js ...   # display subset only; complete inference on full cloud
    qa/qa_report.json                # when QA executed
```

For noncomplete runs, do not create valid-looking `p_dynamic.npy` for uncomputed IDs/frames. Store `NaN` plus an explicit validity mask for incomplete numerical arrays, or omit final posterior files and write `UNCALIBRATED_SCORE` diagnostics. Never default to zero motion probability on an absent/failed update. Atomic rename or checksummed commit of outputs; partially written trajectories must not be displayed/evaluated as complete. Follow the current `utils/temporal_output.py` source-once/chunk integrity discipline. No code snapshots copied into SSD results; manifests reference SHA checksums and git commit.

### 10.4 Downstream boundary: preserve the existing rigid/joint machinery

Stage 1 **must not** directly write a fake `free_forward/posterior_xyz.npy` or `coherence_forward/moving_core_point_ids.npy` to trick `estimator/rigid_kalman_cuda.py` or `joint_reduction` into running. Those modules have different input contracts. Once moving seeds are validated, implement **a separately gated adapter** with explicit code review that passes verified original point IDs, canonical XYZ/RGB/covariance, and support reliability into the known rigid-motion estimator, while retaining the remaining points as potential occluders. Do not implement a new final rigid estimator, fixed-axis solver or joint model as part of this task. Later model comparison must allow `static`, `dynamic-A`, `dynamic-B`, and `unexplained`; the current `joint_reduction/support_projective.py` supports static versus **one** accepted moving joint and must not be inaccurately described as already multi-part.

---

## 11. Mandatory adaptation of existing visual debug — 3D first, probability-centered

**No new visualization framework.** Reuse the proven dark RigEuler-style HTML/Three.js viewer and `debug/assets/projective_common.js` (functions `PD.viewer`, `PD.points`, `PD.frustum`, `PD.chart`), the `write_js` + base64 array format in `debug/projective_debug.py`, its `CAPS`/display-ID convention, and trace-original-ID selection mechanics. The current `debug/assets/projective_free_3d.html` and `projective_coherence.html` are the visual references. Reuse assets by copying them into the **run** (including the local Three.js bundle and license), not by remotely fetching from the browser.

### 11.1 Required 3D synchronized views

Implement `debug/svgd_support_debug.py` writing `svgd_support_3d.html`, or a single page with synchronized panels and global frame selector, with the following **separately switchable** layers:

1. Original initial canonical colored PCD, **stationary gray or measured RGB** (never overwrite the initial positions).
2. Current frame **masked observed colored 3D cloud**, measured RGB; camera visibility toggles and calibrated camera frusta; true observed point count in side panel.
3. Internal **transported provisional cloud** at the selected frame; same original IDs and reference RGB; distinct glyph/colour from observed samples.
4. Pseudo-colour by `p_dynamic` in a continuous legend from `0` (static explanation) to `1` (moving explanation), with **unresolved/no evidence shown distinctly** (not mapped to 0). Display **probability semantics** in the legend; if outputs are `UNCALIBRATED_SCORE`, name the layer `motion evidence score`, not `probability`.
5. The **reliable moving seeds** in an overlaid high-contrast separate point set, with tracked individual IDs and their full provisional XYZ trajectories. Seeds must be a proper subset of the published canonical IDs, not of the decimated display set.
6. Rigidity candidate graph edges coloured by probability/strain and **boundary edge disagreement**, with per-edge trajectory on selection. Graph edges are hypotheses, not committed physical part labels.
7. Per-frame 3D vectors `target attraction`, `Stein repulsion`, `accepted step` for a **small explicitly selected display subset**, and an explanation of what each vector represents. **Do not render every vector at 100k points.**
8. Optional view of unsupported/occluded original particles and unrepresented observed surfaces, with reason codes. Mask-positive only; no displayed background points.

Every displayed frame is synchronized to the **true timestamp**, not playback rate. The UI must support orbit/pan/zoom, frame scrub/play, source-camera selection, point click-to-select original ID, and copying/linking selected point/frame in URL params (reuse the existing hash/query behavior if stable).

### 11.2 Required numerical diagnostic panes

Per frame/iteration: `E_obs`, `E_time`, `E_rigid`, visibility/unexplained cost, **actual common objective**, accepted/rejected step length, target-score norm distribution, attraction/repulsion norm distributions, observed and represented feature distances, coverage and unknown foreground fraction, number and reasons of unsupported particles, score/feature covariance condition and numerical tail bounds, pairwise-strain quantiles, mean/quantiles of `gamma`, number/risk of moving seeds, CPU/CUDA timings and peak allocation. Show `NO_ESTIMATION`, `FRAME_UNCOMMITTED` and `PROBABILITY_UNCALIBRATED` prominently when appropriate.

For clicked original ID, show over **all frames**: canonical XYZ and reference RGB, provisional XYZ trajectory, position prior uncertainty, target score, attraction, repulsion, accepted step, source-camera evidence contributions, visibility reason, rigid neighbor summaries (`rho_il`), temporal displacement and inferred `gamma`. Never imply the provisional trajectory is a final rigid-motion fit. Store raw numerical traces for selected IDs only; full per-frame arrays may be disk-backed.

### 11.3 Debug writer and headless-browser QA

- Add `debug/svgd_support_debug.py`; **do not modify** `debug/projective_debug.py` unless a genuinely shared, tested helper must be extracted. Existing `debug/index.html` should include a link only in a compatible explicit SVGD run; old projective page layouts must remain unchanged.
- `debug/data/*.js` must have the same documented base64-array discipline, stable IDs, known dtype, original timestamps and camera metadata; display downsampling strictly separate from inference. Save actual `N_full`, `N_display`, `M_t`, `M_display` on-screen. Never claim visualization flicker is sensor noise.
- Add `tools/qa_svgd_support_debug.py`, using the headless Chrome `Browser`/Three.js pattern in `tools/qa_projective_debug.py`: assert every page loads without JS errors and reaches an explicit `window.__state.ready=true`; frame and camera selectors change the data; selecting original ID fetches the **same** saved probability and source ID as the numeric arrays; displayed XYZ equals saved provisional XYZ within serialization tolerance; color legend endpoints and `NaN/unresolved` cases render correctly; viewer displays complete frames only; screenshots saved with hashes. Include a no-WebGL error/report path and an independent browser-unavailable status, not silent success.
- Require a numerical audit that `p_dynamic` values **never depend on display sampling factor**, camera-toggle state or HTML rendering. Test by changing display cap while preserving data and comparing every inference output hash.
- Preserve `--debug none|light|full`: `none` means no large debug payload, not no numerical log; `light` saves keyframes, `full` saves more diagnostics but **still** only a bounded display subset and requested point traces.

---

## 12. Implementation order and hard acceptance gates

**Implement one gate at a time. Run focused tests and save an actual report after each.** Do not launch full 60-frame/8-camera experiments before the indicated gates.

### Gate 0 — checkout, theory freeze (TF0), data and original baseline preservation

- Complete Gate TF0 (selected-model mathematical freeze, traceability matrix, deviation register), then verify HEAD, SHA/hashes, CLI and GPU; record unchanged baseline `python -B -m unittest discover -s tests -t .` results (a failing baseline must remain explicitly visible; do not assert it passed if it did not).
- Test strict binary masks, valid depth, original per-camera XYZ/RGB/source IDs and covariance using `scene/endpoint.py` against a NumPy backprojection oracle. Check known source point IDs remain invariant when display subsampling changes.
- Confirm actual acquisition times; fail on `MISSING_ACQUISITION_TIMING` when physical temporal prior requires unknown dt. Do not use HTML viewer FPS as acquisition time.
- Reject scene masks with missing/invalid pixels; no background.

### Gate 1 — target score: CPU reference and CUDA exact small-cloud parity

Build a toy colored point set (multiple distinct RGB modes, overlapping XYZ, anisotropic 3D noise). Compare `logp`, normalized responsibilities, 3D score and finite differences; test equal/unequal particle counts, huge number of observations, empty scene, far-from-cloud queries, reference RGB uncertainty and log-determinant normalizers. Dense versus sparse candidate search must have error certificates. Log maximum/median absolute and relative errors, not only PASS labels. CPU/CUDA exact small cloud should agree at the expected FP64 numeric precision; set justified tolerances from independent oracle roundoff tests.

### Gate 2 — Stein update sign, kernel, bounded GPU memory

Against Liu/Wang's formula implement RBF attraction and repulsion, test `N=1`, `N=2`, uneven XYZ, distinct colours, near coincident points, kernel gradients by finite difference and acceptance behavior. On analytically known Gaussian targets, show stable distributional improvement (e.g. KL estimate/MMD) without numerical collapse. Ensure correct sign of repulsion and global `1/N`; color-only changes affect the kernel when enabled but do not move RGB themselves. Compare attraction-only vs full Stein. Time and peak GPU memory; reject any dense `N²` allocation in realistic production configuration.

### Gate 3 — XYZ–RGB feature observation with correct Jacobian

Test exact Gaussian-kernel MMD vs finite RFF/Nyström feature approximations and JVP/VJP against independent finite differences; equality of observation features for identical **distributions**, not identical IDs. Verify noise covariance sampling including RGB cross-channel and source/color reuse, and no double assimilation of raw data. Show full-rank vs rank-deficient local information on synthetic textured/textureless surfaces, **without** fabricating point velocity. Test source camera density duplication and observation-count changes. Demonstrate that `stepping` the physical particles actually decreases the one common nonlinear data objective on a controlled scene.

### Gate 4 — whole-sequence temporal transport (toy first)

Use complete short synthetic sequences with **known evaluation-only** motion: (a) completely static XYZ+RGB with independent and anticorrelated depth noise, (b) RGB flicker with stationary geometry, (c) one translating rigid object, (d) one rotating rigid object, (e) two moving parts with different motion, (f) a moving object intermittently static, (g) moving foreground occluding a static background. Keep variable `M_t` and camera count. Verify whole-sequence objective is nonincreasing on accepted steps, true timestamps are used, source IDs are stable, and no independently re-fit frame permutation is mistaken for a material trajectory. The static-only case is especially important: the transport must not produce systematic drift merely to chase sample noise. Report failure when global optimization cannot find the motion from the stationary initialization.

### Gate 5 — rigidity, support and low-amplitude near-hinge evidence

Compute pairwise distance histories, full reference-correlation covariance and whitened residuals; compare exact tiny dense covariance to chosen production approximation. Test rigid translations and rotations preserve internal distances while many cross-support edges change; stationary/static edges also preserve distances; special axis pairs may have constant cross-support distance; sparse graphs can flex. Test a rotating body with points at different axis radii, including a **zero- or sub-jitter-amplitude** point; its support should be allowed to remain uncertain until group evidence resolves it, not forcibly classified static. Distinguish separate moving supports with different motions. If an unobservable hinge point remains genuinely ambiguous, correct behavior is `UNRESOLVED`, not invented certainty.

### Gate 6 — actual stationary-versus-moving **model evidence**, not arbitrary logits

On tiny discrete-support examples, enumerate candidate assignments/hypotheses and integrate or quadrature over toy nuisance trajectories to obtain an **independent** exact reference posterior. Check Bayes factor, prior odds, normalization, label marginal sums and response to measurement noise/visibility. Compare Laplace/variational approximation against oracle and report approximation error. Test `moving model fits random jitter better but has lower integrated evidence` where appropriate, because raw minimum-residual comparison alone overfits. Test no observations: posterior should remain at prior uncertainty. Check `p_static+p_dynamic=1` only for complete binary model, and `0≤p≤1` finite for every valid output. If a credible evidence approximation is not established: **do not emit false probability files; stop with diagnostic-only motion scores**.

### Gate 7 — 3D visibility and camera-overlap falsification

Synthetic blade-like occlusion with two nearby surfaces: ensure one support's color does not become the other's material evidence. Test partial occlusion, disocclusion, switching cameras, foreground-only masks, duplicate camera observations, observation count shifts, unseen points kept as latent, and no forced transport to newly revealed geometry. Compare results under source-weight policies. This gate must use the **point-only observation model** and original camera 3D rays; no use of a 2D image residual. A visibility discontinuity may be nonsmooth; characterize the active-set changes and make true objective comparison honest.

### Gate 8 — visual, CUDA and resource gates

- Build pages and run headless QA; verify pixel/data **provenance** is copied correctly if images are shown for context, but the new computation remains 3D point-based.
- Confirm display decimation never changes optimized states/probabilities.
- Demonstrate CPU dense-oracle versus Torch-CUDA parity; if implemented, Triton kernel parity under strict precision. Log accelerator utilization, per-kernel times and peak **allocated/reserved** GPU bytes; no silent CPU fallback. Quantify small (1k), medium (10k), near-full (≈100k) point workloads using both 8GB and 32GB configurations where available. On a smaller device a declared memory budget may force chunking, not statistical subsampling.
- Compile actual source change list and compare Git hashes to baseline. No performance claim without measured CPU/GPU wall-clock, workloads and accuracy results. A faster but statistically incorrect approximation fails.

### Gate 9 — bounded real-scene pilots, followed by complete sequence

First two actual scenes: `USB_100109` (revolute) and `blade_103706` (prismatic). Initial **3-frame** full-masked-cloud pilots: all available cameras, no GT inference. Only after Gates 1–8 and positive motion-evidence/visibility results, test **11 frames**. After correct acceptance, run their **complete 60-frame sequences**, with original source IDs, exact masks, actual timestamps, and full observation provenance. Add `storage_45135`, `stapler_103111`, `scissor_11100` and real `fridge_plastic_20261016_163314`, `white_drawer_20261021_161154_episode_000000` once the two pilots pass. Real data has stronger XYZ/RGB jitter and may expose independent calibration/occlusion problems; it is not optional to report them. Use post-hoc GT-only validation for synthetic scenes after inference files are frozen.

`points.zip`/HTML viewer `.js` samples from the discussion may be used as **diagnostic visual examples only**: they are subsampled viewer data, not a substitute for original camera RGB-D and foreground masks when claiming full-cloud physical reconstruction.

---

## 13. Specific scientific ablations and falsification tests

| Test | Why it is mandatory | What falsifies the proposed mechanism |
|---|---|---|
| `target=XYZ` vs `target=XYZ+RGB` | Attribute direction to appearance rather than just geometry | RGB worsens moving-seed precision under observed flicker |
| Stein attraction alone vs full attraction+repulsion | Test whether repulsion preserves support or physically scatters particles | Repulsion causes cross-boundary contamination/collapse/deformation |
| Spatial Stein kernel vs spatial+color kernel | Identify genuine benefit of appearance-conditioned evidence propagation | Color kernel merges or fragments wrong surfaces |
| Independent per-frame transport vs whole-sequence joint | Demonstrate temporal motion rather than jitter following | Full-sequence model still chases high-frequency noise |
| Mean-embedding feature likelihood vs raw score as optimizer | Avoid double counting and distinguish proposal from evidence | Scores do not correlate with integrated motion evidence |
| Exact dense score vs approximate sparse CUDA | Bound approximation bias introduced for acceleration | Large moving points lose informative gradient from truncation |
| Same-canonical RGB vs latent uncertain RGB | Measure effect of reference/flicker uncertainty | Confident false-moving classification from shared appearance error |
| Rigidity off vs uncertain rigidity on | Attribute near-hinge support recovery to rigidity | Static/dynamic boundary contamination increases |
| Uniform observed weights vs source-aware weighting | Test camera count and sampling-density robustness | Duplicated camera produces spurious motion evidence |
| Alternative reference graph construction | Test robustness to initialization sample density | Moving seeds depend primarily on arbitrary graph degree |
| Vary bandwidths and feature dimension at equal model evidence | Identify bandwidth sensitivity | Conclusion reverses under admissible calibrated scales |
| Independent vs correlated reference/time covariance | Test false certainty | Approximation systematically overconfident near hinge/static edges |
| Unknown/uncovered observation modeling on/off | Detect forced matching of missing geometry | Unseen surface causes known static point drift |
| Separate support evidence vs integrated score | Check whether reported gamma is really a posterior | Calibration curve badly disagrees with observed error rates |

**No thresholds chosen from synthetic GT during inference.** GT may set post-hoc evaluation axes or expose failure modes; do not tune hidden motion thresholds to the target scene. Graph proposal connectivity, kernel bandwidth, target bandwidth and probability prior each require separate provenance and sensitivity reports.

## 14. Complete proposed alternating algorithm — precise control-flow pseudocode

This is a proposed *implementation decomposition* for a difficult latent-variable model. The Stein update is exact only with an exact score and full kernel. The approximate posterior requires its separately specified evidence model. **Keep the boundaries between these components in code and logs**.

```python
# INPUT: actual Q_0:T from strict masked RGB-D frame observations,
#        all cameras calibrated into the same world frame,
#        no GT, known timestamps, declared sensor/representation covariances.

canonical = build_frame0_canonical_with_original_ids_and_uncertainty()
observed  = stream_masked_xyz_rgb_by_frame_and_camera()
model     = validate_policy_and_probability_semantics()
assert no_background_samples and no_unverified_timing

X, V, motion_prior = initialize_provisional_tracks_and_uncertainty(canonical)
# X[0] anchored probabilistically to noisy source XYZ; do not overwrite measured source.
graph = build_sparse_candidate_graph_from_canonical_samples(canonical)

for outer in numerical_iterations:
    # Stage A: direct score of observed *XYZ+RGB* points; not 2D samples.
    targets = build_target_distributions(observed, model.target_density,
                                         model.visibility_and_sampling)
    scores  = target_score(X, canonical.rgb, targets)  # includes support status

    # Stage B: RESTRICTED STEIN PROPOSALS across the physical particles.
    fields  = stein_field(X, canonical.rgb, scores.grad_xyz, model.stein)
    candidate_X = propose_transport(X, fields, model.step_policy)

    # Stage C: one actual data objective; do NOT assimilate the Stein force again.
    # Treat current source uncertainty, visibility, unexplained observations
    # and temporally correlated feature noise as explicitly modeled.
    features = build_joint_xyz_rgb_feature_observations(observed, candidate_X)
    candidate_X, candidate_V, candidate_cov = refine_temporal_sequence(
        candidate_X, V, features, motion_prior, soft_graph=graph,
        matrix_free=True, fixed_prior_for_iterated_correction=True)

    # Stage D: validate the same nonlinear sequence likelihood/regularizers.
    candidate_evidence = evaluate_global_sequence_objective(
        candidate_X, candidate_V, observed, canonical,
        model, include_visibility_and_normalizers=True)
    accepted = common_objective_acceptance(candidate_evidence, current_evidence)
    if not accepted:
        record_rejection_and_provenance()
        if no_valid_descent():
            stop_with_status("STALLED_NOT_CONVERGED")
        continue
    X, V, current_evidence = candidate_X, candidate_V, candidate_evidence

    # Stage E: uncertain same-support relationships from the WHOLE sequence.
    edge_stats = pairwise_distance_history_with_source_covariance(X, graph)
    graph = infer_soft_support_graph(edge_stats, all_observation_evidence=features,
                                     declared_priors=model.support_priors)

    # Stage F: moving-vs-static model evidence; nuisance group motions stay latent.
    posterior = approximate_normalized_support_marginals(
        X, graph, observed, motion_prior, model, initial_reference_uncertainty=True)
    # This step MUST return a named error/score-only status if evidence is not
    # normalized/calibratable; never fall back to sigmoid(displacement).

    write_atomic_iteration_journal(objective=current_evidence, posterior=posterior,
                                   scores=scores, fields=fields, graph=graph,
                                   convergence_and_error_budgets=True)
    if criteria_pass():
        break
else:
    stop_with_status("ITERATION_CAP_NOT_CONVERGED")

# No final rigid motion or joint estimates are written by this stage.
if not posterior.has_defensible_probability_semantics:
    save_diagnostics_and_stop("UNCALIBRATED_SCORE_ONLY")
else:
    seeds = select_seeds_by_declared_posterior_risk(posterior.gamma,
                                                     posterior.model_status)
    write_final_probabilities_and_ids_atomically(posterior, seeds)
    write_offline_point_only_visual_debug_and_numeric_qa()
```

**Important correction to the pseudocode:** soft-graph strain terms can be used as a **trajectory prior** in `refine_temporal_sequence` *or* can be used to approximate the marginalized evidence of that prior. They may not be introduced as independent new measured data a second time. Both `feature` residuals and `score` come from the same observed colored PCD; only the former enters the chosen likelihood. Keep the iterative objective mathematically consistent across changes to `graph` or `posterior`; if optimizing a variational ELBO, evaluate the full joint ELBO, including discrete-label entropy, not just a conditional trajectory objective while claiming global monotonicity. Candidate state changes and probability changes may have different valid objectives; log them separately if a common certified ELBO has not been derived.

### 14.1 Explicit optimization state and convergence

`outer` iterations alternate continuous trajectories and soft label inference. Within a continuous subproblem, use a fixed predicted Gaussian/process prior and evaluate nonlinear feature/visibility measurements at trial states. A valid convergence declaration requires **all** of: stationarity or a first-order residual certificate for continuous variables, no unresolved objective-descent step, stability of the discrete posterior/ELBO under the chosen inference approximation, finite/PSD covariance, and acceptable numerical tail/feature-approximation errors. Record any non-differentiable change in support neighborhood or visibility active set. No convergence based solely on maximum-iteration count or frozen particle positions produced by a zero score.

### 14.2 Fast and slow information must be distinguished statistically, not by an arbitrary frequency cutoff

A stationary physical point with noisy observations can show large frame-to-frame *apparent velocity*. Use measured XYZ and RGB temporal covariance and the process model in the likelihood. Low-amplitude coherent rigid motion near a hinge can have magnitude below a single-frame noise standard deviation. Consequently, implement a synthetic oracle where (i) the stationary observations jitter with independent/temporally correlated XYZ and RGB noise and (ii) near-hinge moving points have sub-noise displacement but are part of a support whose other points move visibly. A successful model gives reliable seeds from the informative region and **does not force** the near-hinge points static; their final recovery is deferred to joint-based Stage 4.

### 14.3 Model selection versus maximizing data fit

A moving hypothesis with more free parameters can almost always fit finite observations at least as well as a static hypothesis. The test is **integrated evidence** or a separately defended posterior approximation. Never use `max likelihood (moving) > max likelihood (static)` as sufficient evidence. Similarly, the minimum of a target density's potential is not a posterior membership probability. Record prior odds and all normalizing constants; evaluate competing explanations on the **same frames/cameras and observation support**. If using holdout frames/points for independent evidence checks, keep their source IDs disjoint from the fitted factors and make clear they are tests, not extra independent data assimilated twice.

---

## 15. Running and validating the implementation — command contract

The following commands are **to be implemented by Codex**; they are *not asserted to exist in the audited base commit*. Use the real scene path of the installed dataset (replace placeholders with actual paths). Do not create output folders in the repository.

```bash
# 0: confirm checkout and baseline
cd /path/to/RGBDiekf
git rev-parse HEAD
git status --short
python -B -m unittest discover -s tests -t .

# 1: targeted newly added mathematical/CUDA tests
python -B -m unittest discover -s tests -p 'test_svgd_*.py' -v

# 2: validate config and inventory ONLY; do not allocate production results
python -B tools/run_svgd_support.py \
  --scene /path/to/USB_100109 \
  --config configs/svgd_support.json --device cuda --dry-run

# 3: first 3 frames, full masked input, with explicit output directory
python -B tools/run_svgd_support.py \
  --scene /path/to/USB_100109 \
  --config configs/svgd_support.json \
  --device cuda --frame-limit 3 \
  --debug full --output /path/to/SSD_results/svgd_usb_3f

# 4: independent visual/numerical QA
python -B tools/qa_svgd_support_debug.py \
  --run /path/to/SSD_results/svgd_usb_3f

# 5: CUDA performance and correctness profile
python -B tools/profile_svgd_support_cuda.py \
  --run /path/to/SSD_results/svgd_usb_3f

# 6: after ALL gates, full scene with no frame truncation and a NEW folder
python -B tools/run_svgd_support.py \
  --scene /path/to/USB_100109 \
  --config configs/svgd_support.json \
  --device cuda --debug light \
  --output /path/to/SSD_results/svgd_usb_full
```

Do not use an existing run folder for a new run, and do not resume from a legacy four-pass run as though the first-stage observation model matched. Every output reports the **actual** git SHA, original source IDs/hash, validated scene-time source, noise assumptions, feature-map seed, numeric and model errors, and completeness.

---

## 16. Reporting format to return after implementation

Create `reports/svgd_support_implementation_and_validation.md` and a machine-readable `reports/svgd_support_validation.json` (if and only if Codex runs the corresponding tests). They must contain:

1. Actual checkout SHA, working-tree status, changed-file list and source hashes. Detail every change to an existing file and prove that legacy defaults are unchanged.
2. Exact equations implemented, with mapping to `module:function`; name every approximation and every calibration/sampling assumption. In particular, distinguish `Lambda` (target bandwidth), `H` (Stein bandwidth), `R` (measured sensor noise), `Gamma` (feature covariance), and `Q` (temporal process).
3. Independent oracles: maximum/median numerical error for `log p`, score, Stein force, repulsion sign, full feature Jacobians/JVP/VJP, temporal covariance, rigidity covariance, and posterior model odds. Include CPU-versus-CUDA parity and FP64 versus mixed-precision differences.
4. Actual scene/frame/camera counts; stable original IDs; number of samples used in **inference** versus **display**; mask, visibility and new/unrepresented counts. Report missing frames or unsupported features as failures, not zeroes.
5. Whole-sequence objective per accepted/rejected iteration, temporal motion and model-evidence terms, uncertainty and posterior semantics; no claim of convergence on cap exhaustion.
6. Motion-support evidence with seed precision/coverage and false-positive static contamination. Distinguish **GT post-hoc** numbers from inference. Report near-hinge unresolved points rather than counting them as stationary successes.
7. CUDA memory/time profiles at 1k, 10k and near-full clouds, exact-versus-approximate kernel error, bottlenecks, and runtime on the actual 8GB/32GB hardware if available; no extrapolated speed claims without stated assumptions.
8. Browser QA: screenshots and exact array/marker/source-ID checks; proof that changing viewer subsampling and camera toggles cannot alter inference results; no external JavaScript/CDN dependencies.
9. Clearly stated final status per gate: `PASS`, `FAILED`, `BLOCKED_MODEL`, `NOT_RUN`. If a model has only a motion-evidence score and not a defensible posterior, report **`PROBABILITY_NOT_YET_ESTABLISHED`** and do not relabel it as a probability.
10. Exact reproduction commands using the implemented CLI; no full-batch workload until prior gates pass. Only after Phase 1 is validated, propose a **separate** adapter to the existing rigid/joint stages. No fabricated downstream joints/axes.

---

## 17. Required decisions and research limitations: do not conceal them

The mathematical theory **does not uniquely specify** the target density's spatial/RGB smoothing, observation feature map, visibility/exposure mechanism, normalized prior over unknown supports, covariance of temporally resampled point-cloud features, or computational posterior approximation. Codex must implement declared model choices with tests and report sensitivity; **do not invent hardcoded parameters or silently convert undecidable cases to zero motion**. The most important limitations to certify are:

- **Particle meaning:** physical surface points used in restricted Stein transport are not SVGD samples of a common unknown trajectory posterior. Original SVGD KL-descent theorem does not certify physical motion.
- **Unknown support:** support/rigidity is jointly inferred, not an input; a rigid graph may include static points or cross-body edges with constant distance.
- **Insufficient appearance:** RGB flicker and view changes can produce motion forces, and homogeneous or repetitive texture cannot identify tangential material motion.
- **Visibility:** source masks are hard constraints but do not give full 3D visibility; an occluded point is not measured by its occluder. Changing visible support needs a correct observation event likelihood.
- **Sample density:** normalization by N and M is not a physical surface-area correction. Multiple cameras observing the same area must not generate false evidence solely from duplication.
- **Shared source information:** frame-0 XYZ/RGB uncertainty is common to all later explanations, and pairwise distance residuals share the noisy anchor. Treating each frame/edge as independent systematically overstates certainty.
- **Fitted transport is not evidence:** the more flexible moving model can fit noise better. Only properly integrated model likelihoods/normalized approximate posteriors support probabilistic labels.
- **Unknown motions:** the initial moving seeds should be precise enough for subsequent rigid-motion recovery. Near-hinge points can be unresolved; later joint-based hypothesis evaluation is responsible for them.
- **Computational approximation:** dense SVGD costs O(N²) and naive target scores O(NM), both unacceptable for 100k-point scenes. Acceleration must come with error certificates and parity tests, not arbitrary spatial neighborhoods or hidden downsampling.
- **Limited scope:** Stage 1 does not solve arbitrary numbers of joints, optimal joint axes, final SE(3) paths, full missing point reconstruction, or complete static/dynamic assignment. Those are later stages of the agreed pipeline.

## 18. Completion criterion

This task is complete **only when** the opt-in Phase-1 code (a) runs the genuine masked XYZ–RGB observations on CUDA for validated full sequences, (b) shows correct score/Stein/feature/rigidity operators against independent CPU oracles, (c) makes provisional transport and whole-sequence likelihood agree on an audited objective, (d) distinguishes static jitter from coherent motion without forcing near-hinge points static, (e) produces a **defensible, approximation-labeled** per-initial-point moving-support posterior and reliable seeds, (f) passes the new point-only browser QA, (g) preserves every legacy default and numerical baseline, and (h) reports all nonconvergence, missing data, and uncalibrated assumptions rather than treating a partial run as success.

**If one gate fails, stop and report the exact failing invariant, original point IDs, frame/camera context, input evidence, error magnitudes and code location, plus the theory equation and deviation-register ID or `NONE`. Do not repair the symptom with another uncalibrated threshold, invented part label, GT axis or opaque regularizer.**

### External references for the implementation audit

- [Liu and Wang (2016), SVGD](https://arxiv.org/abs/1608.04471) — Stein identity and kernel transport, NOT a proof of point-cloud physical correspondence.
- [Original SVGD Python implementation](https://github.com/dilinwang820/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py) — small dense numerical reference; target score is supplied by caller; do not copy dense all-pairs into production.
- [Audited RGBDiekf HEAD](https://github.com/SamueleMara/RGBDiekf/commit/d0b470b75e9ec9181f7561e1a80d2909fb75c7e5) — exact repository snapshot for this document.
- `Stein_SVGD_Moving_Support_Pipeline.pdf` — accompanying mathematical design discussion, including caveats about target-density choice, approximate model evidence and entire-sequence rigidity.



## 19. Final theory-parity sign-off (overrides any ambiguous completion wording above)

Implementation acceptance is **not** a subjective claim of “matching the theory.” It requires the entire equation-to-code matrix and all mandatory TF.6 oracles to be complete; strict-mode production code to implement the **selected** mathematical model without unauthorized alterations; GPU acceleration to be an auditable equivalent computation; model and numerical approximations to be named, quantified and, where non-equivalent, explicitly authorized; and the actual full-sequence Stage-1 output probabilities to retain valid normalization and honest uncertainty semantics. Every implementation deviation must appear in both deviation registers, with mathematical justification and its effects on `p_dynamic` and moving-seed selection. Any unapproved non-equivalent deviation or missing probability law means **`BLOCKED_THEORY_FIDELITY`**, not `COMPLETE`. A case that is correctly implemented but scientifically unsuccessful must be separately marked `PHYSICAL_MOTION_NOT_RECOVERED`. No guess, silent fallback or attractive visualization can override those statuses.
