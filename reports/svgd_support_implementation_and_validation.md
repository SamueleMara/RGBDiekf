# SVGD-assisted moving-support Stage 1 — implementation and validation (2026-11-03)

Contract: `CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md`. Theory: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`.
Frozen contract: `docs/SVGD_THEORY_TO_CODE_CONTRACT.md` (generated from `estimator/svgd_contract.py`). Machine-readable parity:
`reports/svgd/theory_parity_summary.json`. Deviations: `reports/svgd/theory_deviation_register.{md,json}`. Preflight: `reports/svgd/preflight_audit_2026-11-03.md`.

## TF.7 completion report

```text
CHECKOUT_COMMIT: d0b470b75e9ec9181f7561e1a80d2909fb75c7e5 (+ uncommitted working tree: new files only, tools/run_status.py +20 lines)
THEORY_REFERENCE: PDF sha256 33ebd5ff…; .tex absent -> 36 contract rows mapped by content to PDF eqs (1)-(44) (TEX_UNAVAILABLE_LABELS_MAPPED_BY_CONTENT)
SELECTED_MODEL_CONTRACT: estimator/svgd_contract.py sha256 b671731f… -> docs/SVGD_THEORY_TO_CODE_CONTRACT.md
EXACT_EQUATIONS_IMPLEMENTED: 23 PASS_THEORY_EXACT (E01-E03, E05-E09, E12-E14, E18-E22, E27-E30, E33-E35)
EQUIVALENT_CUDA_OPTIMIZATIONS: 6 (OPT-01 tiled streaming log-sum-exp in centred matmul form, OPT-02 tiled all-pairs Stein in matmul form,
                               OPT-03 matrix-free JVP/VJP, OPT-04 exact block-Thomas temporal solve, OPT-05 Sherman-Morrison Omega, OPT-06 CUDA float64)
NUMERICAL_APPROXIMATIONS: 1 (DEV-0009 bandwidth-statistic subset)
MODEL_APPROXIMATIONS: 5 (DEV-0001 V_t = all, DEV-0002 i.i.d. Gamma, DEV-0006 no B_t, DEV-0007 structured Omega, DEV-0008 conditional objective)
NON_EQUIVALENT_DEVIATIONS: 4 (DEV-0001, DEV-0002, DEV-0006, DEV-0008)
UNAPPROVED_DEVIATIONS: 10 (DEV-0001..DEV-0010, all PROPOSED_NOT_AUTHORIZED, reachable only with --svgd-theory-fidelity experimental)
BLOCKED_OR_UNSPECIFIED_EQUATIONS: 11 (E04 TC-0001; E10, E11, E15, E16, E17, E25, E26 NOT_DEFINED; E23, E24, E32 rely on unauthorized deviations)
UNTESTED_EQUATIONS: 1 (E36 stages 2-4: out of Phase-1 scope)
ALL_OPERATOR_PARITY_TESTS: PASS — 67/67 tests/test_svgd_*.py; worst: log p 2.2e-9 abs (far queries, |log p| ~ 1e5), score 1.6e-12 rel,
                           Stein vs Liu & Wang 4.3e-14, RTS 1e-15 m, Omega MC 0.7 %, kernel FD 8e-12
ALL_POSTERIOR_PROBABILITY_TESTS: PASS on tiny exact models (evidence vs 120^3 Gauss-Hermite, Laplace 1.9e-5 nats, 94-hypothesis enumeration);
                                 production posterior NOT_RUN: BLOCKED_MODEL_UNSPECIFIED
CUDA_TESTS: PASS — RTX 4070 Laptop 8 GB, float64, TF32 off: max rel CPU vs CUDA 4.4e-14 (score), 8.3e-15 (Stein), 3.2e-14 (temporal solve)
WHOLE_SEQUENCE_VALIDATION: NOT_RUN (only a 2-frame Blade GPU smoke, stopped after STEIN_PROPOSAL; strict mode blocks at TARGET)
VISUAL_DEBUG_QA: PASS on synthetic runs (4 runs, 50-62 checks each, 0 failed; screenshots + sha256 in RUN/debug/qa/); NOT_RUN on real data
LEGACY_REGRESSIONS: baseline 441 tests 7 pre-existing failures (listed in preflight); legacy code untouched except tools/run_status.py branch
OVERALL_THEORY_FIDELITY: BLOCKED_MODEL_UNSPECIFIED
NEXT_UNRESOLVED_MATHEMATICAL_ITEM: E04/TC-0001 (fixed vs marginalized reference colour) and target_density.Lambda_X / Lambda_C (PDF (8), section 14)
                                   -> they block TARGET in strict mode; then H_X/H_C, psi/B/D, Gamma_t, p(V | X, C), B_t, p(m, Z), the moving-motion prior.
PHYSICAL_MOTION: PHYSICAL_MOTION_NOT_RECOVERED on the synthetic Gate-4 toys (below), under the unauthorized experimental model
```

## 1. What was built (all new files; legacy defaults unchanged)

| file | content |
|---|---|
| `scene/pointcloud_sequence.py` | strict masked point-only `PointCloudFrame` (all mask-positive valid-depth pixels, all cameras, source triple unique), canonical catalogue, increment noise study (PDF (4)) |
| `estimator/svgd_target_score.py` | operator A: exact normalized XYZ×RGB mixture (8), responsibilities (9), XYZ score (11), reference-colour marginalization (TC-0001 option) |
| `estimator/svgd_particle_operator.py` | operator B: restricted physical-particle Stein field (13)-(15), global 1/N, attraction/repulsion separate, ablation switches |
| `estimator/svgd_feature_measurement.py` | operator C: declared RFF feature map ψ (seeded, hashed), means (17)-(18), matrix-free JVP/VJP (21), normalized Gaussian likelihood (19)-(20), exact-kernel MMD oracle (22) |
| `estimator/svgd_temporal.py` | whole-sequence [x, v] prior (16)/C4 with actual timestamps (reusing `point_kalman_cuda`), exact block-Thomas solve, PCG with certificates |
| `estimator/svgd_rigidity.py` | operator D: exact kNN candidate graph, distance histories (26), Jacobian (37), Ω (38) exact-dense + structured production form, D3 likelihood, (25) check |
| `estimator/svgd_visibility.py` | point-only 3D ray-ordering visibility DIAGNOSTIC (law not defined: never in a likelihood) |
| `estimator/svgd_support_posterior.py` | operator E: linear-Gaussian evidence, Laplace, Bayes rule, coherent partition enumeration, point marginals, E5 seed rule; production → BLOCKED |
| `estimator/svgd_support_run.py`, `tools/run_svgd_support.py` | opt-in runner (strict default / experimental), stage lifecycle, common-objective acceptance, immutable artifacts |
| `estimator/svgd_contract.py`, `tools/svgd_theory_parity.py` | frozen contract, deviation/conflict records, strict validator; audit regenerating docs + reports |
| `utils/svgd_support_output.py` | atomic checksummed writes + artifact ledger |
| `debug/svgd_support_debug.py`, `debug/assets/projective_svgd_{support,numerical}.html`, `tools/qa_svgd_support_debug.py` | 3D + numerical pages reusing `projective_common.js`; headless-Chrome QA |
| `tools/profile_svgd_support_cuda.py` | CUDA timing/memory profile with CPU parity |
| `configs/svgd_support.json`, `configs/svgd_support_experimental.json` | strict (gaps explicit) and experimental (PROPOSED deviations) model declarations |
| `tests/test_svgd_*.py` (10 files, 67 tests), `tests/svgd_fixtures.py` | oracles, mutations, invariances, runner, CUDA, visual |

Scale separation (report item 2): `Λ_X, Λ_C` target bandwidth (E01) ≠ `H_X, H_C` Stein bandwidth (E05) ≠ `R` measured sensor noise (`noise_calibration.json`) ≠ `Γ_t` feature covariance (E11) ≠ `Q` white-acceleration process noise (E18).

## 2. Why strict mode blocks (and must)

The theory names but does not specify Λ, H, ψ, Γ_t, p(V | X, C), B_t, p(m, Z), the moving-motion prior and Ω_model (PDF §14). Strict mode
(`configs/svgd_support.json`) therefore runs INIT (masked clouds, canonical IDs, noise study, accounting) and stops at TARGET with
`BLOCKED_MODEL_UNSPECIFIED` plus `BLOCKED_THEORY_CONFLICT TC-0001`, each with a calibration path. In experimental mode
(`configs/svgd_support_experimental.json`) the pipeline runs INIT → TARGET → STEIN_PROPOSAL ⇄ TEMPORAL_REFINEMENT → RIGID_GRAPH under
my PROPOSED choices (DEV-0001..0010) and stops at MODEL_EVIDENCE: no `p_dynamic.npy`, no seeds; `motion_score_uncalibrated.npy` (max provisional
displacement, m) is labelled UNCALIBRATED_SCORE everywhere.

## 3. Independent oracles (report item 3) — measured, see `theory_parity_summary.json` for every row

| operator | oracle | worst measured | tolerance |
|---|---|---|---|
| log p, responsibilities, score (A) | SciPy per-component densities, explicit softmax; 5-point FD; 81³ grid integral; 30³ Gauss–Hermite colour marginal | 2.2e-9 abs log p (far query, |log p|~1e5); 1.6e-12 rel score; FD 1.8e-8; integral 4e-15 | 1e-11 rel, 1e-6 FD |
| Stein field (B) | dense NumPy all pairs; Liu & Wang `svgd.py` direction (fixed h); kernel FD; mutations (1/degree, no repulsion, 5-NN truncation) all caught | 4.3e-14 vs Liu & Wang; FD 8e-12 | 1e-12 / 1e-11 |
| features (C) | exact Gaussian-kernel MMD (D = 40000: within 6/√D); ⟨Ju,w⟩ = ⟨u,Jᵀw⟩ 1e-13; FD 1e-7; SciPy NLL 5e-13 | | |
| temporal (C4) | autograd dense Hessian; independent Kalman + RTS smoother; SciPy Markov density | RTS 1e-15 m; normalizer to 10 digits | 1e-9 m |
| rigidity (D) | FD Jacobian; 2·10⁵-sample Monte Carlo Ω; dense inverse/slogdet | MC 0.7 %; SM exact to 1e-11 | 3 % MC |
| evidence (E) | 120³ Gauss–Hermite; 161² quadrature (nonlinear rotation); naive enumeration of 94 coherent hypotheses | Laplace 1.9e-5 nats; enumeration 1e-12 | 1e-8 / 0.05 nats |
| CPU vs CUDA float64 | every operator | ≤ 4.4e-14 rel (p95 ≤ 1.1e-14) | 1e-10 |

Measured invariances: camera loading order — Stein + target exact to 1e-18 m, full run 1.8e-8 m on 2.6 mm motion (Krylov-amplified roundoff in the capped
PCG, cond(L) ~ 1e10); duplicate source IDs removed by bookkeeping (invariant), a genuinely new camera exposure changes the 1/M_t density (declared law);
display cap never changes any inference array (byte-identical .npy/.npz).

Two real numerical findings fixed during the work: `torch.cdist`'s matmul path makes exactly coincident points non-coincident (exact mode now used for
every NN); batched cuSOLVER `eigvalsh` fails on 34k 3×3 matrices (replaced by the repository's Jacobi solver).

## 4. Gate 4 toys (report items 5-6): GT post hoc only, experimental model

Synthetic 2-camera masked RGB-D, 4 frames at 30 Hz, 911 canonical points (177 on the moving plate), 3 outer iterations:

| scenario | static drift median / p90 | moving true disp. | moving error median | recovered (median) | convergence |
|---|---|---|---|---|---|
| (a) static, 1 mm depth noise | 0.14 / 0.22 mm | — | — | — | STALLED_NOT_CONVERGED |
| (b) static, RGB flicker σ = 0.03 | **2.3 / 6.0 mm** | — | — | — | ITERATION_CAP |
| (c) translation 15 mm | **4.9 / 8.4 mm** | 15 mm | 7.2 mm | 52 % | ITERATION_CAP |
| (d) rotation about hinge | **3.7 / 6.3 mm** | 6.0 mm | 4.7 mm | 8 % | ITERATION_CAP |
| (e) two parts, (f) intermittent, (g) variable cameras | NOT_RUN (fixture has one plate) | | | | |

Reading: the accepted objective is monotone in every case and the model does not chase depth jitter (a). It does follow RGB flicker (b: failure
criterion (v) of PDF 13.5), and genuine motion is only partly recovered while static points are contaminated (c, d: failure criteria (iii)/(iv)) —
consistent with DEV-0001 (V_t = all), DEV-0004 (object-scale H) and DEV-0006 (no B_t). On the occlusion fixture 97 % of hidden static points have a
score pulling them toward the occluding plate; the visibility diagnostic flags all of them OCCLUDED. Status: **PHYSICAL_MOTION_NOT_RECOVERED**.
No near-hinge result exists at runner level; on the tiny exact model a sub-noise hinge point stays unresolved (γ ∈ (0.05, 0.95)) while its movers get γ > 0.99.

## 5. Real data and CUDA cost (report items 4, 7)

2-frame Blade smoke (`blade_103706`, 8 cameras, N = 34 369, M_t 34 354..34 369, all masked samples, experimental, stop after STEIN_PROPOSAL):
H_X^½ = 118 mm (median heuristic → object-scale coupling); the Stein proposal was REJECTED by the common objective after 20 backtracks.
Exact float64 cost after the matmul rewrite: target score 2.9 s / evaluation, Stein field 1.6 s / frame, peak 0.69 GB (was 14 s / 13.5 s / 1.05 GB before).
Synthetic profile (`reports/svgd/cuda_profile_1k_10k.json`): 10k points — score 0.29 s, Stein 0.15 s, RFF mean 13 ms, temporal solve 24 ms.
Quadratic extrapolation (not measured): USB (N ≈ 96k) ≈ 25 s score + 13 s Stein per frame per outer iteration → ≈ 40 min per outer iteration over 60
frames, ≈ 5 h at the 8-iteration cap; 32 GB hardware not available. Near-full profile: `--sizes 100000` (command below), not run here.

## 6. Gate status

| gate | status |
|---|---|
| 0 checkout / TF0 / masks / timing | PASS (back-projection oracle exact, MISSING_ACQUISITION_TIMING enforced, non-binary masks rejected) |
| 1 target score | PASS |
| 2 Stein operator | PASS (incl. Liu & Wang parity, attraction-only collapse vs full-Stein MMD gain, bounded memory) |
| 3 feature observation | PASS for operators; Γ model BLOCKED_MODEL |
| 4 whole-sequence transport | FAILED physically (static drift under flicker, partial recovery); operators/monotonicity PASS |
| 5 rigidity | PASS for operators; Ω_model and P_ζ BLOCKED_MODEL (DEV-0007/0010) |
| 6 model evidence | PASS on tiny exact models; production BLOCKED_MODEL → PROBABILITY_NOT_YET_ESTABLISHED |
| 7 visibility / overlap | BLOCKED_MODEL (law undefined); diagnostic PASS, failure mode exposed |
| 8 visual / CUDA / resources | PASS (synthetic QA, CUDA parity, 1k/10k profile); near-full profile NOT_RUN |
| 9 real pilots | NOT_RUN (Gates 4-7 not passed; only a 2-frame smoke) |

## 7. Commands (user-launched; see the chat reply)

Strict and experimental 3-frame pilots, QA, profile and the parity audit are listed in the reply that accompanied this report.

## 8. Response to audit `RGBDiekf_SVGD_Commit_b62c7cee_Technical_Audit_2026-11-03.md` (same day, uncommitted on top of b62c7ce)

Scope rule followed: diagnose before any model change (audit order item 2); every model law the audit asks for (items 1, 3–6) is written as a
**proposal awaiting approval** in `docs/SVGD_THEORY_AMENDMENT_PROPOSAL_2026-11-03.md` (decisions D1–D7), not implemented as inference.
Statuses are machine-readable in `estimator/svgd_contract.AUDIT_FINDINGS` and `reports/svgd/theory_parity_summary.json` (`audit_b62c7cee_findings`).

| finding | status | what changed |
|---|---|---|
| F1 rigidity never constrains transport | BLOCKED_AWAITING_THEORY_APPROVAL | proposal A4 (support labels incl. unknown U, SE(3) random-walk nuisance, CRP partition prior, variational ELBO coupling); runner unchanged |
| F2 positions-only Stein proposal | DIAGNOSED + optional fix | per-backtrack ΔE_obs/ΔE_time for both families, consistency residual, ⟨∇E_obs,P⟩, ⟨∇E_time,P⟩ (E38); exact conditional velocity lift E37 = DEV-0011, opt-in (`velocity_lift`), strict refuses it unless authorized |
| F3 Stein vs likelihood descent | DIAGNOSED | cos(Φ, −∇E_obs) per frame and stacked, cos(P, −∇E_total), lifted variant (STEIN_DIAGNOSTICS gate) |
| F4 visibility / B_t | BLOCKED_AWAITING_THEORY_APPROVAL | proposals A2 (normalized ray-ordering visibility with occlusion / free-space / off-foreground events, camera-stratified features) and A3 (Beta–Gaussian unexplained mixture) |
| F5 Γ uncalibrated | BLOCKED_AWAITING_THEORY_APPROVAL | proposal A5 (traceable decomposition + held-out χ² coverage) |
| F6 object-scale H_X | DIAGNOSED | `tools/audit_svgd_stein_rejection.py` (separate process, GT post hoc): exact moving/static source split of K and Φ (`source_mask`, parts add to 1e-12); proposal A6 |
| F7 Ω "conservative" | CORRECTED | DEV-0007 now "bias sign UNDETERMINED", conclusion UNDETERMINED; oracle: dropping ρ = +0.8 vs −0.8 cross-covariance gives opposite log-det bias |
| F8 no production posterior | BLOCKED_AWAITING_THEORY_APPROVAL | proposal A7; `support_posterior` still BLOCKED, no p_dynamic / seeds |
| F9 runtime/QA gates | PARTIAL | profile adds Γ covariance + Cholesky, objective gradient, GN apply, PCG, velocity lift, 34k default size; runner records `debug_write_seconds`; 34k/near-full and real-scene QA commands handed over |

Labelling fixes (audit "important distinctions"): the parity summary now states `scope_of_operator_parity` (algebra + small fixtures, not end-to-end),
`end_to_end_model_fidelity = NOT_ESTABLISHED`, `workload_acceptance = NOT_ESTABLISHED`; failure-revealing tests (occlusion, camera duplication,
bias sign, flat-plane information, Occam) carry `test_semantics: FAILURE_REVEALING: PASS = the failure mode was detected`; each Gate-4 scenario has a
`physical_outcome` under a declared toy criterion (static drift ≤ 0.5 mm; recovered ≥ 0.9): (a) NO_FALSE_DRIFT, (b) FALSE_DRIFT_FAILED,
(c) and (d) PHYSICAL_MOTION_NOT_RECOVERED. Trajectory fractions are not seed precision/recall (audit note): no seed metrics exist until A7.

Measured on the synthetic translation fixture (3 frames, outer 1, experimental): ⟨∇E_obs, P⟩ = −2.6e5 (Φ is an E_obs descent direction,
cos(Φ, −∇E_obs) = 0.49), ⟨∇E_time, P⟩ = 0 at the stationary state, yet at η₀ the positions-only proposal has ΔE_time = +6.5e6 (second order: inconsistent
velocities) and ΔE_obs = +1.8e3 (overshoot); the exact velocity lift reduces ΔE_time to +5.0e4 (130×). Post hoc, 21 % of static points have a Stein
field dominated by GT-moving sources (median static-receiver kernel mass from moving sources 0.15 %, p90 21 %). The Blade decomposition is a
user-launched diagnostic (commands in the chat reply); it has not been run.

Tests: 71 svgd tests (was 67) pass; new oracles: velocity lift vs dense autograd minimization (1.7e-16 rel), source-mask split, bias-sign oracle,
runner decomposition consistency + post-hoc audit on the synthetic oracle. CUDA profile 1k/10k re-measured while another job held the GPU at 100 %
(contended; e.g. target score 0.68 s at 10k vs 0.29 s uncontended earlier): `reports/svgd/cuda_profile_1k_10k.json`.

## 9. Blade Stein-rejection diagnosis (user-launched 2026-11-03 18:08/18:09, `svgd_blade_stein_audit_{none,conditional_temporal_lift}_*`)

Blade, 2 frames, 8 cameras, N = 34 369 (3 218 GT-moving, post hoc), experimental config. Both runs identical up to the proposal family.

- **No proposal accepted, with or without the velocity lift**: 21 backtracks, η from 3.0e-4 down to 2.8e-10 m²; ΔE_obs < 0 from the second backtrack on, but ΔE_time dominates at
  every step (η₀: ΔE_obs +135, ΔE_time +2.73e9; smallest: −4.3e-5 vs +2.5e-3). ⟨∇E_total, P⟩ = −1.5e5 < 0, so only a step ~26 further halvings smaller (sub-nm) would pass:
  the backtracking cap, not the sign, ends the search.
- **Cause = frame-0 anchor, not velocities (F2 is not the Blade mechanism)**: exact offline split at η₀: anchor ½|ηΦ₀|²_{Σ⁰⁻¹} = 2.725e9, process 0→1 = 3.3e6; the
  lift can only change velocities (2.728e9 → 2.725e9). 100 % of the anchor cost lies along the smallest eigen-direction of Σ⁰: the ray depth, σ = 69 nm, because the
  measured depth noise of this noiseless render is the float32 quantization floor (variance 4.7e-15 m², quantum 2.4e-7 m); lateral σ ≈ 2.0 mm. Stein moves frame-0
  particles by a median 34 µm although the frame-0 feature residual is identically zero (canonical = frame-0 samples, ∇E_obs,0 = 0, cos = 0 at frame 0).
- **F3**: cos(Φ, −∇E_obs) = 0.05 overall (frame 1: static 0.001, moving 0.134), but post hoc cos(Φ, true displacement) = **0.999** on GT-moving points at frame 1: the Stein
  direction is physically right on the blade while the RFF feature gradient (B = H = 118 mm, DEV-0005) is nearly orthogonal to it.
- **F6**: moving receivers take a median 21.8 % of their kernel mass from static sources; 15.6 % of static points have a frame-1 Stein field dominated by moving sources.
- Browser QA on the real scene: PASS, 47/47 checks, both runs. Run cost: target score 2.9 s/eval, Stein 1.6 s/frame, peak 0.69 GB allocated / 1.0 GB reserved, debug 0.3 s.
- CUDA profile (uncontended, `svgd_cuda_profile_20261103_1809.json`): 34k — score 2.83 s, Stein 1.58 s, Γ covariance 0.71 s, Cholesky 0.05 s, 10 PCG 0.58 s;
  100k — score 24.2 s, Stein 13.6 s, Γ 2.1 s, 10 PCG 1.7 s; peak ≤ 0.58 GB allocated / 1.09 GB reserved; CPU vs CUDA ≤ 7e-14.

### 9b. 10-frame Blade runs (user-launched 18:14 positions-only / 18:16 lift; stop after STEIN_PROPOSAL)

- Every proposal rejected in both runs (21 backtracks, η 3.0e-4 → 2.9e-10 m²). At η₀ the exact E_time split is anchor 2.817e9 vs process 1.04e7 over all 9 intervals
  (99.6 % anchor); the lift lowers ΔE_time by 0.25 % only (2.827e9 → 2.820e9). ⟨∇E_obs, P⟩ = −2.26e6, ⟨∇E_time, P⟩ = 0: same mechanism as 2 frames.
- Prismatic blade, 14.16 mm per frame (GT post hoc), KDE spacing s_t ≈ 2.6 mm. Because nothing is accepted every frame is proposed from the stationary state, and the
  Stein direction on GT-moving points degrades with the true displacement: cos(Φ, true) median 0.999 (14 mm), 0.876, 0.720, 0.786, −0.182 (71 mm), −0.099, 0.597,
  0.426, 0.325 (127 mm); p10 negative from frame 2. Large displacements leave the KDE basin (PDF 13.1 control 3).
- cos(Φ, −∇E_obs) ≤ 0.062 in every frame (F3); moving receivers take 21.8 % of kernel mass from static sources in every frame and the static fraction dominated by
  moving sources rises 15.6 % → 17.5 % (F6). QA PASS 56/56 both runs. Cost 123 s: target 2.9 s × 20, Stein 1.6 s × 10, peak 0.79 GB.
