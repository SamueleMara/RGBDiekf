# RGBDiekf SVGD Stage 1 — critical code/report audit and next Codex actions

**Audit date:** 2026-11-03  
**Repository:** https://github.com/SamueleMara/RGBDiekf  
**Committed HEAD verified:** `b62c7ceeb300c852a58f34dd748d5a1ce26d987a` — `SGVT implmentation`  
**Base commit of implementation report:** `d0b470b75e9ec9181f7561e1a80d2909fb75c7e5`  
**Basis:** committed source at `b62c7cee` inspected through GitHub, together with uploaded Codex transcript, `theory_parity_summary.json`, `theory_deviation_register.{json,md}`, `SVGD_THEORY_TO_CODE_CONTRACT.md`, CUDA profile and preflight audit. **No tests or full real-scene runs were independently executed in this review.**

## Executive decision

**The operator-level implementation is promising, but the Stage-1 scientific goal has NOT been achieved.** Codex's classification `BLOCKED_MODEL_UNSPECIFIED` is appropriate. The accepted *experimental* implementation does not produce `p_dynamic`, moving seed IDs, calibrated `rho_ij`, or an authorized whole-sequence posterior. It also fails its physical-motion toy cases. Do not integrate its provisional motions or uncalibrated scores with the established rigid/joint stages.

**Important distinctions:**

- **Operator parity is not end-to-end model fidelity.** The 23 exact-model entries/67 passing tests establish algebra and small fixtures, not full-sequence motion or probabilities.
- **CUDA numerical parity is not workload acceptance.** The timings are CPU/CUDA arithmetic and throughput observations for specific synthetic sizes and a 2-frame Blade smoke, not full 60-frame performance.
- **A 'PASS' on a failure-revealing fixture** (RGB flicker, occlusion, insufficient motion) means the test correctly detected the failure, not that motion recovery passed.
- **The reported 7 legacy failures are a baseline observation**, not proof of new regressions; verify source-matched reruns before asserting either regression or complete safety.

## Factual status from the report

| Category | Evidence | Verdict |
|---|---|---|
| Git commit | `b62c7cee` includes new SVGD modules, tests, debug assets and reports | COMMITTED |
| Exact operators | 23 theory-contract entries; dense NumPy, finite differences, Liu–Wang, RTS, quadrature | PASSED REPORTED ORACLES |
| New tests | 67/67 pass; no skips | PASSED REPORTED TESTS |
| Float64 CUDA parity | max reported relative error 4.4e-14 across compared operators | PASSED REPORTED ORACLES |
| Complete theory | 11 blocked equations, 10 unauthorized deviations, one theory conflict | **BLOCKED** |
| Whole-sequence physical motion | 4-frame toy: 52.3% translation fraction, 7.8% hinge fraction, substantial stationary drift | **FAILED** |
| Full real sequence | 2-frame Blade smoke only, Stein proposal rejected | **NOT VALIDATED** |
| Production posterior | `support_posterior()` returns blocked; `rho` is NaN; no `p_dynamic` or seeds | **NOT IMPLEMENTED** |
| Existing-stage integration | Independent opt-in `tools/run_svgd_support.py`; existing `train.py` modes unchanged | **ISOLATED**, which is desirable now |

### Reported physical-motion measurements

| 4-frame synthetic case | Static median drift (mm) | Moving true median displacement (mm) | Estimated recovered fraction (median) | Convergence |
|---|---:|---:|---:|---|
| Static, 1 mm depth noise | 0.136 | 0 | N/A | stalled |
| Static, RGB flicker | 2.283 | 0 | N/A | iteration cap |
| Translation | 4.915 | 15.0 | 0.523 | iteration cap |
| Hinge rotation | 3.718 | 6.021 | 0.0777 | iteration cap |

These *trajectory* metrics are not moving-seed precision/recall or posterior calibration; even a low median recovery fraction may coexist with a small usable moving core, so evaluate seed spatial spread and SE(3) identifiability separately before making a hard impossibility claim.

## Critical source-level findings

### F1 — Rigidity is diagnosed *after* transport; it never constrains transport

**Evidence:** `estimator/svgd_support_run.py:_stages`, approximately lines 293–407 at the pinned HEAD. The inference objective `SequenceObjective(terms, temporal)` has `E_obs + E_time` only. The `RIGID_GRAPH` stage executes later and writes distance/strain statistics, `rho` all NaN. No same-support variable `Z`, graph marginal, nor conditional shared rigid hypothesis is used to update provisional motion. Hence the intended loop *appearance transport -> rigid coherence -> retransport -> probability* is **not actually present**.

**Scientific consequence:** a consistent deformation of a moving part is not preferred over a collection of uncorrelated free-point motions by any rigidity likelihood. The spatial Stein kernel is not equivalent to a rigid-body constraint, and the measured 118 mm kernel can couple a moving and stationary body. A four-frame hinge failure is therefore unsurprising even if all Stein algebra is exact.

**Required action:** Before further throughput work, define an actual joint/alternating sequence objective over trajectories and uncertain same-rigid-support relations, with a mathematically specified *different-support* likelihood, a valid joint prior over coherent partitions, and a motion prior conditional on support hypotheses. Compare with and without the rigidity term on the **same** synthetic fixtures. Do not impose a GT partition or a known hinge axis.

### F2 — Stein proposal changes positions without compatible velocity updates

**Evidence:** `estimator/svgd_support_run.py:_stages`, Stein acceptance loop: `Sc = S.clone(); Sc[:, :, :3] += eta*Phi`. The state includes `(x,v)` and `estimator/svgd_temporal.py:TemporalPrior` penalizes `s_t - F_t s_{t-1}` with a white-acceleration process density. The proposal leaves all `v` unchanged.

**Consequence:** nonzero time-varying displacement increments increase `E_time` because the proposed position history is not accompanied by consistent velocity changes. The line search may reject a useful observation-directed Stein proposal. **This is a concrete mechanism to audit, not yet a demonstrated sole cause of Blade's rejection.**

**Required diagnostic:** For each proposed/backtracked step log `ΔE_obs`, `ΔE_time`, the initial and proposed consistency residual `Δx_t - Δt v_{t-1}`, and the directional derivatives `<∇E_obs, [Φ,0]>`, `<∇E_time, [Φ,0]>`, `<∇E_total,[Φ,0]>`. Compare with a *conditional temporal lift*: for a fixed proposed `δx`, solve for the `δv` that minimizes the **same stated temporal quadratic**, using exact block algebra. This may be an optimization-only proposal change, but cannot be called the identical original SVGD map without recording the distinction and verifying the same statistical target.

### F3 — Stein descent and likelihood descent are not equivalent

**Evidence:** `svgd_particle_operator.stein_field()` uses `g_t = ∇_x log p_t(x,c)` (observed-cloud KDE); `_stages` accepts the proposed full-sequence trajectory **only if** `E_obs(X)+E_time(S)` decreases. `E_obs` is instead the finite-feature XYZ–RGB likelihood. `E_rigid` and discrete support priors are absent.

**Consequence:** a field ascending the per-frame KDE density is not guaranteed to descend the RFF Gaussian feature likelihood plus temporal prior. A rejected Stein step is therefore not evidence that CUDA or the Stein sign is wrong; the objective gradients may simply disagree. Quantify cosine/dot products between `Φ`, `-∇E_obs`, and `-∇E_total` at each frame and for the stacked trajectory. Preserve the distinction: **restricted physical-particle SVGD is a heuristic proposal**, not a posterior-SVGD KL descent theorem.

### F4 — Experimental visibility and new-surface assumptions are knowingly invalid

**Evidence:** DEV-0001 uses `V_t = all initial points`, while `estimator/svgd_visibility.py:ray_visibility` is diagnostic **only**. DEV-0006 omits unexplained/newly exposed support `B_t`. The fixture reports **97% of hidden static points have a score towards the occluding plate**, even though the visibility diagnostic successfully detects them.

**Consequence:** this is not fixed by changing the bandwidth alone. The likelihood is evaluating particles that should not be compared and forces observed newly revealed geometry to be explained by canonical particles.

**Required action:** Declare a normalized, differentiable or piecewise-conditional visibility/occlusion observation model with explicit event accounting and an unexplained-support term; only then include it in inference. Check per-camera ray-ordering and mask-hard constraints. Distinguish occlusion from off-foreground/free-space contradictions. Never score hidden material against an occluder's RGB–D surface. Do not turn invalid observations into a zero residual or accept steps merely because observations disappeared.

### F5 — Feature-noise model is not calibrated

**Evidence:** DEV-0002 `Γ_t = Cov(ψ(Q_t))/M_t + Cov(ψ(P_0))/N`, frozen at stationary initialization. It assumes independent samples and ignores shared source pixel, camera, sampling, flicker and temporal correlations. Observed 2.3 mm static drift under RGB flicker is a direct warning, though not a causal proof of Γ as the sole culprit.

**Required action:** Construct a traceable covariance accounting for reference-colour uncertainty, RGB flicker and sensor/sampling correlations. Test on stationary repeated captures for false-motion coverage and on moving textured and textureless scenes for identifiability. Don't silently inflate Γ; compare residual covariance/whitening to independent held-out measurements and document any unidentifiable components.

### F6 — `H_X` is an object-scale rather than part-local interaction

**Evidence:** DEV-0004 uses a global median-distance heuristic with `H_X^1/2 = 117.95 mm` on Blade. The field applies all-to-all interactions across initial particles. Distinct moving and static supports may therefore exchange motion guidance.

**Required action:** Measure cross-boundary `K_ij` mass and whether the sign/magnitude of Stein attraction for static points is dominated by moving-point contributions. Report spatial+colour kernel mass by GT support **only in post-hoc analysis**, not inference. Propose and justify separate `Λ_X`, `H_X`, and feature `B_X` using physical camera resolution, acquisition density and supported multiscale hypotheses; do not equate them by default. Numerical approximation can preserve exact all-pairs sum through tiled CUDA computation.

### F7 — Rigidity covariance is *not guaranteed conservative*

**Evidence:** DEV-0007 replaces `Ω = J P_ζ J^T` by a diagonal-per-time plus shared-reference rank-one covariance from prior-propagated per-point marginals, dropping cross-point and temporal covariances. The report calls this conservative because prior marginals are broader than posterior marginals.

**Correction:** Dropping a cross-covariance can **increase or decrease** a difference's variance. For `d = x_i-x_j`, `Var(d)=Var(x_i)+Var(x_j)-2Cov(x_i,x_j)`. Suppressing positive correlation inflates variance; suppressing negative correlation deflates it. Temporal residuals have analogous effects. Consequently the likelihood bias is **not signed in general**, regardless of the individual marginal sizes.

**Required action:** Change the register's `conservative` assertion to `bias sign undetermined`; derive covariance of the actually used trajectory estimator, maintain frame-0 shared uncertainty, then compare each approximated Bayes factor / rigidity likelihood with dense exact tiny-problem oracles. Do not publish calibrated `ρ_ij` while Ω is unverified.

### F8 — Exact tiny posterior code is not a production posterior

**Evidence:** `estimator/svgd_support_posterior.py:support_posterior` returns `BLOCKED_MODEL_UNSPECIFIED` when laws are missing; even if no parameters are missing, it returns `BLOCKED_THEORY_FIDELITY` because production evidence is not implemented. `coherent_hypotheses(n)` enumerates all partitions and static/moving labels, and only tiny models are tested. `select_seeds()` rejects nonprobabilistic semantics.

**Consequence:** No `p_dynamic`, no estimated pair probabilities `ρ_ij`, no high-confidence seeds, and no handoff to rigid stage. User's central required output is absent. A sigmoid of `||Φ||`, `max ||Δx||`, or `χ²` cannot legitimately replace it.

**Required action:** Specify a scalable posterior approximation over motion/support hypotheses that *preserves transitivity or encodes coherent partitions*, includes an explicit unknown/unobserved alternative and evidence-complexity penalty, and yields calibrated marginals under its declared generative model. Establish independent exact small-model parity, calibration and held-out noise tests. A population with no useful observation must return its prior, not fabricated certainty.

### F9 — Physical acceptance and runtime gates remain open

**Evidence:** Gate-4 motion fails and no production full sequence ran. RTX 4070 Laptop (8 GB), float64: score 0.286 s and Stein field 0.147 s at 10k; Blade 34k 2.9 s and 1.6 s respectively. The full USB runtime is a quadratic extrapolation, not a measured 60-frame benchmark. CUDA kernels are PyTorch tiled matrix multiplications, not custom Triton; `--svgd-kernel-backend triton` is not proof of a Triton backend.

**Required action:** Reproduce 1k, 10k, 34k and then near-full memory/timing benchmarks, peak *allocated and reserved* memory, forward/backward scans, feature covariance factorization, gradient, PCG and rendering. Do not run long full-scene jobs until physical and visibility gates pass. Test the browser QA on a real scene. Avoid optimizing the wrong physical objective.

## Recommended order for Codex (do not authorize all DEV items en masse)

1. **Reconcile the theory itself**: select and document fixed reference-colour observation versus a shared latent true-colour model (TC-0001), including across-frame dependence. Derive and explicitly approve normalized `p(V|X)`, new-surface `B_t`, support prior `p(m,Z)`, moving-motion prior, and feature observation covariance `Γ_t`. Use the PDF as theory baseline; formally amend it if needed, not a silent code decision.
2. **Diagnose the Blade Stein rejection with a numerical decomposition**, before any model change: score `g`, Stein `Φ`, separate `ΔE_obs / ΔE_time`, velocity lifting, all-pairs kernel cross-boundary mass, RFF feature gradient alignment, and observation support changes. Make one deterministic 2-frame small real/one synthetic oracle case.
3. **Implement visibility and newly revealed geometry consistently** in the single distributional observation likelihood, with hard foreground masks and no one-to-one material correspondence.
4. **Implement true sequence-wide coupled rigidity inference**: uncertain pair/partition variables influence transport; compare fully coupled vs post-hoc graph, and test two simultaneously moving bodies and moving/static contact boundaries.
5. **Calibrate noise and latent reference colour** with multi-camera/temporal correlations and the correct shared source treatment. Only use parameter rules whose evidence and sensitivity are reported; no arbitrary `n_sigma`/bandwidth hidden in run code.
6. **Implement scalable normalized support evidence** (not an uncalibrated displacement score), define unknown hypothesis, and recover posterior `P_i(dynamic)` with calibrated uncertainty and a safe seed-selection rule.
7. **Validate the actual goal**: high-precision moving-seed identification and their spatial coverage/rigid-pose observability, not merely median displacement fraction. Include hinge, two moving parts, intermittent motion, textureless surfaces, RGB flicker, occlusion, changing sampling density, and no-motion controls. Keep GT exclusively for post-hoc evaluation.
8. **Only after these gates pass**, optimize throughput (selected sparse CUDA operators if their approximation error is bounded/declared), full-cloud multiframe runtime, real Blade/USB 60-frame and multi-camera visual-debug QA.

## Explicit policy for the next Codex pass

- Do not modify legacy `train.py` or existing four-pass estimators as a shortcut.
- Do not treat a working algebraic unit test as evidence of physical recovery.
- Keep the exact Stein operator as the numerical reference when changing optimization strategy.
- Do not claim `100% THEORY_PARITY` if the prior, visibility, feature covariance, support posterior, or observation sampling laws remain unspecified.
- Every changed equation must update `estimator/svgd_contract.py`, the equation-to-code table, the deviation register, and its independent oracle. Inexact approximations require explicit authorization, a quantified bias/error study, and a named gate.
- Do not export nonprobabilistic `motion_score_uncalibrated.npy` as `p_dynamic.npy` or as downstream ground-truth moving-core IDs.
- Do not assume unknown physical parts, joints, axes or pivots in Stage 1.
- Keep Stage 1's final handoff limited to calibrated per-point moving-support probabilities and trustworthy high-precision seed IDs; provisional trajectories remain internal/debug.

**Source of reviewed code:** permalink prefix `https://github.com/SamueleMara/RGBDiekf/blob/b62c7ceeb300c852a58f34dd748d5a1ce26d987a/` followed by the paths above. The report is source-derived; proposed remedies and critique are independent analysis, not claims of tests executed here.
