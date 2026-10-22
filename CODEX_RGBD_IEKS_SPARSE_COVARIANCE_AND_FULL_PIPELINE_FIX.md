# CODEX — MANDATORY CORRECTION: sparse bidirectional RGB-D IEKF/IEKS, covariance consistency, and complete reconstruction

**Priority:** P0. **Implement and test, do not merely describe.** This file supersedes contradictory requirements in the earlier `CODEX_IMPLEMENT_RGBD_BIDIRECTIONAL_IEKS_EXACT_STATE_SPACE.md`, `EXACT_IEKF_STATE_SPACE_FORMULATION.md`, and `CODEX_CORRECTION_SPARSE_RGBD_IEKS.md` where those instructions lead to dense global covariance, exponential full-scene enumeration, incorrect posterior certainty, incorrect RTS smoothing, or repeated reuse of measurements. Preserve the correct state-space and RGB-D ray-likelihood mathematics. **There is no 3DGS rendering anywhere in this task.**

## 0. Scope, sources, and the acceptance contract

The algorithm is one **bidirectional RGB-D, rigid-motion, probabilistic geometry-fusion estimator**. Input: a multiview RGB-D image sequence, calibrated cameras, and timestamps. Output: one canonical (possibly incompletely observed) point/surfel model, shared rigid-body poses and twists for all frames, point-existence/part probabilities, RGB/depth predictions, visibility explanations, correctly characterized uncertainty, and convergence diagnostics.

Read and reconcile these sources before editing code:

1. `theory_sparse_iekf_3dgs(2).pdf`, especially original Section 11 (matrix-free information equation) and Section 12 (rigid reduction): **sparse linear algebra is fundamental**. The original Gaussian renderer itself is NOT used.
2. `RGBD_Bidirectional_IEKS_Theory_and_Convergence_State_Space.pdf`: RGB-D first-hit sensor model, alternating posterior/ELBO, bounded convergence statements, and explicit state-space addendum.
3. `EXACT_IEKF_STATE_SPACE_FORMULATION.md`: 12D `SE(3)` pose-plus-spatial-twist dynamics and derivative conventions.
4. This file: **authoritative computational and covariance correction**. Report any remaining incompatible requirements rather than silently changing the mathematics.

**Do not claim** that a local Gaussian IEKF solves unrestricted multimodal Bayesian inference exactly. "Exact" in the sparse continuous solver means an exact representation of the chosen linearized operator with controlled numerical solve accuracy; **it does not mean exact global enumeration of millions of discrete labels**. The theory's deterministic variational discrete updates are admissible only when their chosen family and ELBO are explicitly identified. Do not secretly replace the global objective with nearest-neighbor ICP or a mean-probability first-hit renderer.

**Not permitted:** allocating `O(N_points^2)` covariance/information matrices, enumerating `3^N` configurations in the full reconstruction path, arbitrary per-point motion states, one rigid transform for every point, fixed four-point catalogues, three-frame production runs, heuristic novelty/depth residual thresholds, fake inverse passes, synthetic covariance values described as calibrated, declaring `RESOURCE_LIMIT` from an unnecessary algorithm a successful result, using ground-truth joints/part annotations in estimation.

The full supplied SAPIEN-style dataset has 60 times and eight cameras (640×480 images), including RGB, metric depth, camera parameters, and possibly masks. The loader must process all available frames and cameras by default and stream data; benchmark subsets are allowed *only in explicitly named tests*. Use actual acquisition timestamps when available; if unavailable, **require a user-configured `--dt-seconds`** rather than inventing the frame rate. Masks are optional foreground-selection observations, **never moving-part labels**; joint-value CSV/part annotations are evaluation-only.

## 1. Independent audit first: do not patch a broken covariance pipeline blindly

Before code changes, create `reports/covariance_audit.md` with a table of the actual existing files and functions implementing the following, including call sites and whether they are present, approximate, missing, or erroneous:

- Initial state mean and **initial covariance/information**; gauge constraints.
- Transition function and analytic `F_k,G_k`; derivation and discretization of `Q_k` as a function of actual `dt_k`.
- Camera, RGB, depth, no-return, and calibration uncertainties and their units.
- Per-frame prediction, IEKF nonlinear correction, curvature and covariance update.
- Handling of inter-body cross-covariance and source-frame map–motion covariance.
- New point initialization, probability of existence/part, source observation bookkeeping.
- Forward-to-backward tangent conventions, RTS smoother gain and covariance.
- Joint trajectory–map information matrix, matrix-free operators, Schur solves, marginal covariance queries.
- Outer-loop relinearization, noise/measurement reuse, objective acceptance and convergence.
- Numerically stable SPD treatment and which reported uncertainties are *conditional*, *marginal*, or *variational*.

Do not assume files named `ekf.py`, `smoother.py` or `covariance.py` perform the equations correctly: inspect their implementation and test it.

## 2. Required state, gauge, and error convention

For every **moving** rigid component `b=1,...,B` at time `k`:

```
s[b,k] = (T[b,k] in SE(3), nu[b,k] in R^6)    # translation-first world/spatial twist
error[b,k] = (delta_xi[b,k], delta_nu[b,k]) in R^12
T = Exp(hat(delta_xi)) @ T_nominal            # LEFT pose perturbation
nu = nu_nominal + delta_nu
```

The static body `b=0` is world-fixed (`T[0,k]=I`, no stochastic twist). The full per-frame dynamic state has **`12*B` error coordinates**. Maintain off-diagonal correlations between different moving bodies if rays, motion constraints, or shared map elements couple them.

Canonical map: time-invariant point/surfel variables `m_i` with canonical position `p_i`, RGB appearance `a_i`, finite spatial support/orientation and, when modeled, support uncertainty. Discrete variables `z_i=(existence_i,part_i)` are inferred separately; do not quietly insert them into the continuous EKF as Euclidean pose variables.

### 2.1 Gauge fixing

A canonical-world gauge is required. For a component whose canonical coordinates are defined at `k=0`, `T[b,0]=I` is a **coordinate definition**, not an observed zero-uncertainty physical measurement. If that component is initially invisible, its physical trajectory remains uncertain; choosing coordinates must **not manufacture information** about an unobserved object.

- Eliminate gauge coordinates from the solve (preferred) or impose mathematically exact equality constraints via reduced coordinates/null-space projection.
- Do not invert singular gauge modes; do not hide them behind `P=1e-6*I` or large diagonal penalties.
- Gauge conventions apply consistently to inverse-transported late points, pose covariance, map covariance and evaluation.
- If choosing `T[b,k_ref]=I` for another reference time, transform both state variables **and their full error covariance/information** with the corresponding Jacobian.
- Publish an observability/rank diagnostic that distinguishes unobservable directions from numerical instability.

## 3. Nonlinear state equation and its exact first-order linearization

For each body define `Delta_k=t[k+1]-t[k]`, `A_k=Exp(hat(Delta_k*nu[b,k]))` and process noise `w=(w_T,w_nu)` with a declared local covariance `Q_local[b,k]`:

```
T[b,k+1]  = Exp(hat(Delta_k*nu[b,k])) @ T[b,k] @ Exp(hat(w_T))
nu[b,k+1] = nu[b,k] + w_nu
w ~ N(0, Q_local[b,k])
```

For the left-pose error convention, **translation-first** Lie coordinates:

```
F[b,k] = [[Ad(A_k), Delta_k * J_left_SE3(Delta_k*nu[b,k])],
          [0_6x6, I_6]]
G[b,k] = [[Ad(T_pred[b,k+1]), 0_6x6],
          [0_6x6, I_6]]
```

where `T_pred[b,k+1]=A_k @ T[b,k]`, and `J_left_SE3` is the true SE(3) left Jacobian, not the SO(3) Jacobian or the identity. Derive/test `F,G` by finite differences with **nonzero rotations, translations and twists**.

```
s_pred[k+1] = f(s_filt[k], w=0)
P_pred[k+1] = F[k] @ P_filt[k] @ F[k].T + G[k] @ Q_local[k] @ G[k].T
```

The last two lines are permitted as **dense operations only on the small `12B × 12B` conditional dynamic-state covariance**. Cross-body blocks are not disposable unless mathematically proven zero in that model. Do not allocate a `(12B+3N)×(12B+3N)` covariance.

### 3.1 Process-noise time scaling: a critical missing detail

Do **not** choose the same discrete `Q_k` for different `Delta_k` merely because its shape is `(12,12)`. Two legitimate configurations are possible and must be unambiguous:

**A. Explicitly specified discrete process:** user supplies the full SPD/PSD `Q_local[k]` *at each time interval*, with units and provenance. Transform with `G` exactly as above.

**B. Continuous-time stochastic dynamics:** user supplies a continuous noise spectral density `Q_c` and injection `L`, then discretize for the stated local error dynamics with

```
Q_discrete(dt) = integral_0^dt exp(A*tau) L Q_c L.T exp(A.T*tau) d tau
```

and map the resulting covariance into the correct left-pose/right-process-noise chart. Implement a verified Van Loan method or accurate deterministic quadrature; test against the known Euclidean constant-velocity / white-acceleration case

```
Q_euclidean(dt) = q * [[dt^3/3 * I, dt^2/2 * I],
                        [dt^2/2 * I, dt * I]]
```

in `[position,velocity]` order. **This is a verification limiting case, not permission to replace the declared SE(3) process with Euclidean dynamics.** If using white-acceleration noise, note the position/velocity cross-covariance is nonzero. Do not set `Q_Tnu=0` without a justified model.

**Important:** The process equation above already fixes a discrete right-pose-noise model. To use a continuous-time acceleration model, derive its **equivalent** local `Q_local` and document how the noise charts are transported. Do not blindly add `Q_discrete` to `G Q_local G^T` twice.

## 4. Calibrated uncertainties: initialization and units are part of the model

Create a validated typed configuration `config/noise_model.yaml` plus a strict parser. Required fields: uncertainty model and provenance for RGB, depth, pixel sampling, optional no-return, process noise (continuous or discrete), camera calibration if uncertain, and priors on the initial twist and component/geometry hypotheses. A field must be marked as either `calibrated`, `estimated-from-independent-data`, or `explicit-user-assumption`; do not label an assumption as measured.

If critical parameters are unavailable, fail with `MISSING_NOISE_MODEL` and an actionable missing-field report **before a production estimation run**. An example config with visibly uncalibrated numerical values is fine for smoke tests only; tests must never imply those values are intrinsic to the dataset. It is acceptable to fit physically meaningful sensor-noise parameters from independent calibration data; do not infer a calibrated noise law merely from an uncalibrated reprojection residual distribution contaminated by motion error.

### 4.1 First observation -> 3D covariance

From pixel `u,v`, depth `d`, intrinsics `fx,fy,cx,cy`:

```
y_cam = [d*(u-cx)/fx, d*(v-cy)/fy, d]^T
J_uvd = [[d/fx, 0, (u-cx)/fx],
         [0, d/fy, (v-cy)/fy],
         [0, 0, 1]]
Sigma_y_world = R_wc J_uvd Sigma_uvd J_uvd.T R_wc.T
```

`Sigma_uvd` must contain actual depth/pixel sampling uncertainties and any measured correlation. If camera extrinsics are uncertain, include a **shared** extrinsic nuisance state or its proper cross-correlated marginal effect; do **not** add the same extrinsic variance independently to every point and then treat all points as mutually independent. If extrinsics are conditioned on fixed calibration, state that clearly and omit extrinsic uncertainty only in that conditional posterior.

Separate covariance roles: point localization covariance (meters²), RGB covariance (radiometric units²), and depth-sensor conditional pixel covariance. Do not use a 3D PCD covariance as though it were the full 4D `[RGB,depth]` camera noise matrix; do not use a single scalar for all cameras/depths if the declared noise law is heteroscedastic. Zero/no-return depth is a categorical event, never a Gaussian depth=0 observation.

### 4.2 Initial dynamic covariance `P_0`

- Explicitly specify and validate the distribution of **unconstrained initial dynamic variables**, especially twist `nu[b,0]` and relative component pose when not gauge-fixed.
- A gauge-fixed coordinate has **zero dimension in the stochastic state**; do not set its variance to an arbitrarily tiny positive number. An initially unobserved component must not acquire a sharply known trajectory by gauge convention.
- Construct joint `P_{0,ss}` including cross-body blocks if priors or shared calibration create correlations. Only factor the prior if independence is actually an assumption.
- A noninformative/improper prior is a *singular information model*, not `P=1e20 I` without numerical consequences. Use the constrained/null-space sparse representation or a documented proper prior; refuse unsupported covariance inversion.
- Distinguish **initialization proposal** `s_init` (e.g. odometry from images) from **prior statistical information** `P_0`: an initialization algorithm run on the **same RGB-D frames** is not an independent prior. Either model all shared data in the factor graph with proper dependence or use initialization only as a nominal linearization point without adding its information a second time.

Log eigenvalues/rank/condition numbers of each small prior/posterior block. Initialize uncertainty in consistent tangent coordinates using tested change-of-reference Jacobians.

## 5. RGB-D output distribution and measurement covariance

Do NOT substitute an easy point-reprojection loss for the complete output probability in the convergence objective.

For world point `x_i(k)=T[b_i,k]p_i`, camera point `q_i=R_c x_i + t_c` and ray `r=(k,c,u)`, the projected pixel and depth are:

```
u_i=pi(K_c,q_i), d_i=(q_i)_z
```

A **fixed known visible correspondence** has a conditional `[RGB(3),depth(1)]` Gaussian observation with covariance `R_{r,i}`. The full occlusion/unknown-association output, however, is the normalized first-hit mixture (including valid-return and no-return events):

```
eta_ri = existence_i * coverage_ri
v_r,i_j = eta_r,i_j * product_{a<j}(1-eta_r,i_a)
v_r,empty = product_a(1-eta_r,i_a)
L_r(y) = v_r,empty f_r,empty(y) + sum_j v_r,i_j f_r,i_j(y)
```

All `f` must be proper **normalized** densities/masses; ordering uses increasing *predicted* depth and is rebuilt after accepted nonlinear steps. `coverage` is the calibrated finite-support surfel/pixel overlap, not an arbitrarily truncated infinite-support Gaussian. A surface sample modeled as a mathematical point of zero image area cannot directly produce a complete full-image first-hit probability law.

Use the **full score**, differentiating projection, depth, surface support, coverage and first-hit ordering within an active visibility region:

```
g_r(theta) = sum_j gamma_rj [grad log(v_rj) + grad log(f_rj)]
gamma_rj = (v_rj*f_rj(y)) / L_r(y)
```

Do not omit `grad log(v_rj)` or treat the responsibilities `gamma` as constants when computing the **exact Hessian** of the observed-data likelihood. Numeric derivatives must be verified for cases with overlapping/occluding surfels.

The Gaussian conditional covariance `R` cannot be added independently to an extra RGB/depth reprojection objective built from the *same ray*, or the measurements get counted twice. Cross-pixel and cross-camera independence must be an explicitly stated **conditional** sensor-model assumption. If images were filtered/downsampled in a way that creates correlated noise, model the correlation or state exactly which conditional likelihood is used; do not present an independence assumption as measured fact.

## 6. Covariance update of the forward IEKF: conditional vs full posterior

Let `s_minus` and `P_minus` be the prior at time `k` **in a specified left-error tangent chart**. Holding the map and a declared discrete variational distribution `q(z)` fixed, define

```
E_k(delta) = 0.5 * delta.T @ solve(P_minus,delta)
             - E_q [ sum_{all rays at k} log L_r(y_r | s_minus boxplus delta, m, z) ]
```

The correct local score and Hessian are

```
g = solve(P_minus,delta) - grad_delta(ell_k)
B_true = inv(P_minus) - Hessian_delta(ell_k)
```

where inverses represent **solves**. Build information via ray-local analytic JVP/VJP/HVP; no full Jacobian and no global dense covariance. A Fisher/PSD metric is allowed for the nonlinear *step direction* if documented as a curvature approximation and paired with the **complete score** and the **actual objective** for Armijo acceptance.

```
(B_direction + lambda_LM * D) Delta = -g
```

The prediction prior is **fixed throughout all IEKF relinearizations at that frame**. Never replace it with the prior computed after a previous iteration of the same measurement. Recompute coverage/depth order when a nonlinear candidate step is accepted.

### 6.1 Posterior covariance MUST NOT contain optimizer damping

`lambda_LM D` is a numerical trust-region/optimization device, not sensor evidence. **Do not write**

```
P_plus = inverse(B_true + lambda_LM*D)  # WRONG as probabilistic covariance
```

After convergence, evaluate and transport the **undamped** local posterior information in the final accepted state chart. If using a Laplace approximation and `B_true` is SPD on the observable subspace,

```
P_plus_local = solve(B_true, I_small)
```

with correct retraction/tangent transport. If `B_true` is indefinite, there is no valid SPD Laplace covariance at that point: continue valid optimization, use a **declared** positive-semidefinite Fisher/local curvature approximation for an explicitly labeled approximate filter covariance, or report failure/unidentifiability. Do not silently add diagonal jitter and call it physical information. A small numerical factorization stabilization can be used only with a logged bound and an error check demonstrating it does not materially change the covariance.

In the **special linear-Gaussian conditional measurement** case, implement and test the stable Joseph form

```
K = solve_based_gain(P_minus,H,R)
P_plus = (I-KH) P_minus (I-KH).T + K R K.T
```

and verify against `(P_minus^{-1}+H.T R^{-1}H)^{-1}`. That identity does **not** justify treating the full occlusion mixture as one additive Gaussian measurement.

### 6.2 No incorrect "covariance reduction" with new unobserved points

A point with no informative visibility/correspondence provides **zero (or correctly computed null) information** on a motion direction. A candidate initialized from one source observation cannot independently revalidate its own source pose. Confirm numerically that inserting an unobserved candidate into the map does not magically reduce `P_ss` or drive an innovation covariance below its physical measurement bound.

## 7. The most critical correction: shared map/motion cross-covariances

A small forward IEKF with the map held fixed computes **`P(s | Y, m_fixed, q_fixed)`**, not the marginal `P(s | Y)` of the joint map–motion posterior. Both are useful, but different. A small-body RTS pass on these conditional covariances **is not an exact joint RTS smoother in the presence of uncertain canonical points shared by multiple times**.

For a local factor with motion Jacobian `H_s` and map Jacobian `H_m`, predicted observation covariance of a Gaussian joint state is:

```
S = H_s P_ss H_s.T + H_m P_mm H_m.T
  + H_s P_sm H_m.T + H_m P_ms H_s.T + R
```

Dropping either `P_mm` or `P_sm` creates overconfident state updates. **Do not** replace the two cross terms by zero merely because constructing dense `P_sm` would be too large.

### 7.1 Sparse information representation (authoritative)

Maintain the linearized joint information operator over trajectory `s[0:K]` and canonical geometry `m`:

```
A = [[A_ss, A_sm],
     [A_ms, A_mm]]
```

where `A` comes from the original initial prior, all transition factors, all ray-local likelihood factors, and geometric priors. **Never allocate dense `A` or dense `A^{-1}`.** The exact action on `(v_s,v_m)` is the sum of locally supported factor JVP/VJP (or curvature HVP) operations, plus sparse prior/temporal terms.

For an SPD linearized information operator and invertible mapped `A_mm` on the retained subspace:

```
S_s  = A_ss - A_sm solve(A_mm,A_ms)
rhs  = b_s - A_sm solve(A_mm,b_m)
solve(S_s,delta_s) = rhs
solve(A_mm,delta_m) = b_m - A_ms delta_s
```

Implement `S_s @ v` **matrix-free**. If `A_mm` has cross-point structure from shared ray/support priors, do NOT pretend it is block diagonal. Use sparse solves/PCG with tested preconditioners. If points are genuinely independent conditional on `s`, blockwise inversions may be exact for *that* explicitly demonstrated factorization.

- PCG only on SPD operators; if a full Hessian is indefinite use a documented SPD information approximation for steps or a symmetric-indefinite solver, with actual-objective acceptance.
- Report relative linear residuals, factorization diagnostics, condition estimates, PSD failures and time/memory usage.
- LM/Levenberg damping used for the search direction is **not automatically part of the final posterior precision**.
- Marginal pose covariance in the linearized joint Gaussian model is `P_ss_marginal = (A_ss-A_sm A_mm^{-1} A_ms)^{-1}`; **conditional** covariance is `P_ss_conditional=A_ss^{-1}` (in the full trajectory/motion partition, with appropriate fixed map). These are not equal except in special decoupled cases.
- Never explicitly form either inverse for a large problem. Compute selected blocks or diagonal entries through **verified** sparse linear solves (`A x=e_i`, Schur solves), or return an information operator with on-demand marginal queries. Distinguish selected exact inverse blocks from diagonal/block-Jacobi approximations.

### 7.2 Source-frame point initialization and cross-covariance

A newly revealed point observed at time `s`, hypothetically assigned to moving body `b`, is initialized by inverse transport:

```
p_i^(b) = R[b,s].T @ (y_world - t[b,s])
x_i(k) = T[b,k] @ p_i^(b)
```

For small left-pose perturbation `delta_xi=(delta_rho,delta_phi)` and world observation `y`, the first-order Jacobians are

```
J_y = R[b,s].T
J_s = -R[b,s].T @ [I_3, -skew(y)]      # 3x6, evaluated consistently at y=T p
```

For a source measurement *independent of the motion prior* before its update,

```
P_pp = J_y Sigma_y J_y.T + J_s P_ss J_s.T
P_ps = J_s P_ss
```

and for **two points** `i,j` inverse-warped using the SAME uncertain source pose,

```
P_pi_pj = J_s_i P_ss J_s_j.T  (+ any shared measurement/calibration covariance)
```

Thus even independent depth pixels become **correlated canonical points** through the same uncertain pose. Do not store each point with its own 3×3 covariance and then assume the resulting points are independent. If `Cov(y,s)` is nonzero (e.g. the source measurement already updated that pose), include

```
P_pp += J_y Cov(y,s) J_s.T + J_s Cov(s,y) J_y.T
P_ps += J_y Cov(y,s)
```

These formulas describe **linearized covariance initialization**, not a license to double-count the source measurement: prefer registering the raw source ray **once** as a factor involving `p_i` and `s_s`, and use inverse transport strictly as an initialization proposal. Do not add an independently weighted point prior derived from the same RGB-D source ray *and* retain that ray factor, unless the two are correctly jointly modeled.

The **candidate's existence and part-assignment uncertainty** is non-Gaussian and cannot be summarized solely by a 3×3 covariance. Keep probabilities `q(e_i,b_i)` and component-conditioned canonical hypotheses `p_i^(b)` separately, with their conditional covariances. No unjustified averaging across different part frames.

### 7.3 Repeated late-point observations

Points first seen at a late frame generate information about earlier poses only via shared transformations, valid cross-frame associations/visibility likelihoods, and the sparse joint factors. If they were occluded earlier under the current geometry, their absence is **not** a zero-depth residual. Conversely predicted front-of-depth surfaces must contribute their valid free-space contradiction probability. Keep original source observation in the likelihood exactly **once** per pixel across all outer passes.

## 8. Covariance and the backward smoother: correct the charts and the shared map

For a **conditional-on-map** Markov Gaussian trajectory with the specified left pose errors, the (locally chart-consistent) RTS recursion is

```
C_k = P[k|k] @ F_k.T @ solve(P[k+1|k], I)
delta_next = Log_SE3(T[k+1|K] @ inv(T[k+1|k])) plus twist difference
s[k|K] = retract(s[k|k], C_k @ delta_next)
P[k|K] = P[k|k] + C_k (P[k+1|K] - P[k+1|k]) C_k.T
```

**Do not copy this covariance formula naively for non-Euclidean, differently centered errors.** Before subtraction, transport `P[k+1|K]` from the smoothed-pose chart into the predicted chart; after updating `s[k|K]`, transport the resulting covariance to the final smoothed chart by the derivative of the retraction/coordinate change. Record the coordinate Jacobians used. Small-angle approximation requires a quantified test.

- Use stable linear solves, not `np.linalg.inv(P_pred)` or dense global inverse.
- Store `P[k|k]`, `P[k+1|k]`, `F_k`, `s_pred`, `s_filt`, and **the chosen chart** for every step.
- Preserve cross-body covariance blocks.
- When a shared uncertain map exists, the above pose-only RTS gives a **conditional** smoother. To obtain joint/marginal smoothing uncertainty, use the sparse joint information system of Section 7, optionally carrying out block elimination; compare this result to dense augmented-state Kalman/RTS on tiny tests. Do not label a conditional RTS result `P_joint`.
- Re-running a backward pass does NOT mean applying the RGB-D images as another independent set of measurements. Both passes operate on the **same** likelihood factors.
- Do not feed smoothed `P[k|K]` back as an independent `P_0` at the next outer iteration. Every outer iteration starts from the original declared prior and original factors, changing only nominal linearization/variational parameters.

## 9. Geometric discovery and uncertainty propagation: no arbitrary threshold

### Stage A: motion

Freeze the current geometry and the declared discrete posterior family `q(z)`. Process **all frame/camera RGB-D rays** in the selected full-data objective using analytic full first-hit likelihood scores and sparse information updates. Estimate the pose and twist sequence, and run conditional RTS or equivalent sparse trajectory smoothing. Accept updates using the **same actual ELBO** as Stage B.

### Stage B: backward geometric refinement

Freeze the shared motion trajectory estimate **for that conditional geometry update**. Evaluate later-sourced or currently unexplained observed RGB-D rays under existing geometry and all modeled component labels; propose inverse-warped candidate canonical surfels and reproject them into earlier and nonconsecutive frames:

```
x_i(k) = T[b,k] @ inv(T[b,s_i]) @ y_i(s_i)
```

Use the same normalized first-hit RGB-D sensor likelihood (including valid return, no return, occlusion, and predicted free-space contradictions). Update existence/part probabilities **deterministically and probabilistically**, and refine canonical points/support variables while accounting for their motion uncertainty via the joint sparse factors. Treat a first appearance as a proposal, not automatic confirmed geometry.

- **No `residual > tau` for births**, no variance-clipping to manufacture evidence, no fixed quota such as `--max-candidates 4` in the real reconstruction.
- Geometry candidates may be lazily materialized from the finite sequence of observed samples, provided the inference/prior model and objective remain mathematically unchanged. **Lazy activation is a computational index, not a residual gate**. If the catalogue expands in a way that changes the prior/model, the fixed-catalogue convergence theorem does not apply automatically; either define the global catalogue before inference with lazy storage or provide a separate valid expansion objective argument.
- Posterior `q(z)` may be a **declared** deterministic variational family. Its updates must improve the same ELBO, not substitute a mixture evaluated only at mean existence probabilities. Provide exact discrete enumeration only in tiny oracle tests.
- Treat geometric surfaces first observed once as potentially unidentifiable and preserve the resulting posterior uncertainty.

## 10. Correct outer-loop objective, convergence and stopping

Use one global objective `F(q,s,m)` derived from **all original RGB-D rays once**, the normalized sensor model, the original process/initial priors, the map/complexity prior, and entropy of the declared discrete posterior `q`. Log all terms separately. In every cycle:

```
load original prior and factor definitions             # never turn previous posteriors into independent priors
Stage A: optimize motion/twists given q,m              # sparse IEKF/IEKS; full likelihood score
  accept only if F does not decrease
Stage B: infer/update canonical geometry with s fixed # reverse visibility + source-once handling
  accept only if F does not decrease
Stage C: update discrete q in its declared family      # deterministic posterior/variational update
  accept only if F does not decrease
repeat until objective + gradient / KKT convergence criteria hold
```

- Use explicit numerical convergence tolerances for **optimization**, documented independently of **physical point/motion thresholds** (which must not exist).
- A fixed `--outer 2` is a smoke test, not a convergence criterion. Detect and report stalled line search, information singularity, unobservable motions, variational-family uncertainty and numerical breakdown.
- Posterior Gaussian covariance/credible regions are **local** and do not establish a global posterior mode; discrete mixture uncertainty remains separate. If reporting a marginal over mixture modes, use the law of total covariance in a consistent tangent frame:

```
mean = sum_z q(z) mean_z
P_total = sum_z q(z) [P_z + (mean_z-mean)(mean_z-mean).T]
```

for local Euclidean coordinates (use manifold-consistent transport for poses). For large coupled discrete models where this marginal cannot be evaluated exactly, label uncertainty as conditional or variational: no false exactness.
- The convergence theorem guarantees monotone objective values under its assumptions and stationary accumulation points only under additional regularity/sufficient-progress assumptions; **not** exact recovery of all occlusions, global MAP, or convergence of true scene geometry from unobservable data.

## 11. Required source layout and API

```
main.py                         # --input --output; all 60x8 by default; time/noise config
rgbd/dataloader.py              # streaming RGB-D, validity/no-return, calibration and timestamps
rgbd/config.py                  # validated explicit noise/priors/provenance, no hidden defaults
rgbd/lie.py                     # left SE3 error, Exp/Log, adjoint, F/G, tangent transport
rgbd/uncertainty.py             # depth->3D, continuous->discrete Q, SPD/rank, covariance charts
rgbd/geometry.py                # shared canonical surfels; birth proposals and ownership
rgbd/visibility.py              # normalized first-hit ray likelihood and score
rgbd/operators.py               # sparse factor graph JVP/VJP/HVP, streaming ray incidence
rgbd/ekf.py                     # actual forward prediction and IEKF update, small 12B cov
rgbd/smoother.py                # chart-correct conditional RTS and comparison with joint smoother
rgbd/joint_information.py       # sparse trajectory/map precision, Schur, marginal queries
rgbd/backward_refinement.py     # inverse-warped late geometry, visibility/existence evidence
rgbd/posterior.py               # explicit exact-tiny/variational-full modes and ELBO
rgbd/outer_loop.py              # coordinate ascent, same full-data objective, convergence
rgbd/outputs.py                 # map/pose files, RGB-D predictions, confidence qualifiers
rgbd/diagnostics.py             # covariance/observability/memory reports
scripts/run_full_scene.sh       # existing conda env, configurable SSD output
config/noise_model.example.yaml # TEST EXAMPLE ONLY: uncalibrated until user supplies values
tests/                           # mandatory tests enumerated below
reports/covariance_audit.md     # per-file forensic report from section 1
```

Public interfaces to implement and test:

```python
load_dataset(input_path, camera_subset=None, frame_subset=None, dt_seconds=None)
validate_noise_model(config, dataset) -> NoiseModel
se3_predict(state, P_filter, dt, noise_model) -> (state_pred, F, G, P_pred)
ray_logprob_and_score(ray, contributors, motion, canonical_map, discrete_state)
apply_sparse_joint_information(v_motion, v_map) -> (Av_motion, Av_map)
query_marginal_covariance(variable_ids, chart) -> SelectedCovariance
forward_iekf(factors, priors, map_state, q) -> FilterTrace
backward_rts(trace, fixed_map=True) -> ConditionalSmootherTrace
joint_motion_map_refinement(factors, current_state, map_state, q) -> JointRefinement
backward_geometry_discovery(observations, motion, map_state, q) -> ProposalAndPosterior
outer_loop(original_priors, dataset, config) -> CompleteResult
```

Do not claim that `forward_iekf` conditional covariances are joint marginals. Source-frame late-point updates must be traceable to original ray factor IDs.

## 12. Mandatory numerical tests (not just shape tests)

**12.1 State/process and time:** finite-difference `F,G` on nontrivial SE3; `Q(dt)` Euclidean limiting test; different `dt` and zero-time limit; cross-body covariance propagation; gauge nullspace invariance.

**12.2 Measurement covariance:** depth+pixel Jacobian finite differences, PSD and unit checks; RGB-depth cross-correlation case; missing/no-return case; uncertain shared extrinsic generates cross-point covariance.

**12.3 IEKF:** linear-Gaussian Joseph/information posterior equality; same observation processed twice in IEKF iterations leaves final result unchanged; varying LM damping changes convergence behavior but **not** final undamped posterior covariance for the same local optimum; consistent covariance transport between left tangent charts; indefinite Hessian must not yield fabricated SPD posterior.

**12.4 Sparse-vs-dense reference:** compare matrix-free `A @ v`, gradient, PCG solution, sparse Schur solve, selected inverse covariance blocks, and joint marginal pose covariance against a dense double-precision oracle for a tiny linearized problem. Assert relative residual and maximum covariance-block error using declared numerical tolerances. Assert no default allocation with dimension scaling like `3N × 3N` or `12B+3N` dense.

**12.5 Shared pose-map covariance (essential):** initialize two late points from the same uncertain pose; verify `Cov(p_i,p_j)=J_si P_s J_sj^T` is nonzero. Compare tiny dense *augmented-state* smoothing to sparse joint smoothing. Verify pose **conditional** covariance is smaller than or equal to pose **marginal** covariance in PSD order; verify dropping cross terms fails an assertion. Repeat when measurement noise is correlated with the source pose.

**12.6 Backward RTS:** compare all filtered/smoothed means and cross-body covariance blocks to exact dense Kalman/RTS on a small linear Gaussian system; compare nonlinear manifold smoother to finite-difference/fixed-gauge batch linearization; require chart transport under nonzero rotational correction.

**12.7 Visibility/birth:** verify two overlapping surfels with different depths, valid/no-return and empty-ray normalization; a point hidden at earlier frames is **not** penalized by foreground RGB; a point predicted in free space yields negative evidence; a new surface seen only once stays uncertain; no update from fully uninformative occlusion; reusing source observation does not artificially shrink covariance; nonconsecutive frame links work.

**12.8 Posterior semantics:** compare exact discrete posterior with a declared variational approximation in tiny cases; verify mean-field is **never** labeled exact; ELBO accepts/rejects using all modeled observation terms; gradient finite differences include visibility coverage and empty-ray terms.

**12.9 Full dataset integration:** enumerate every RGB-D camera/time; validate world/camera image-down convention using multi-view reprojection; use no GT joint values or part labels; check that all pixels/rays contribute once to the production objective (streaming is allowed). Export motion trajectories, corrected canonical map, reprojected RGB/depth and occlusion images, CSV objective traces, covariance qualifiers, rank/condition report, memory peak and runtime. Do **not** substitute a 3-frame or four-point demonstration.

**Failure assertions that must be in CI:** making `P_plus` depend on LM damping, dropping `P_sm`, counting a source depth ray twice, treating static gauge as small variance, resetting `Q` independently of `dt` under continuous noise, pretending `P_conditional=P_marginal`, allocating dense global covariance, and using an unnormalized ray mixture must all cause tests to fail.

## 13. Logging, checkpointing and resource safety

- Stream sensor inputs; use memory-bounded ray tiles and finite-support exact culling. Read large arrays incrementally. Dense arrays are acceptable only for small per-frame rigid-state blocks and tiny reference tests.
- Checkpoint posterior family, map catalogue/proposals, motion trajectory, linearization charts, optimizer state, original prior ID, observation factor IDs, and noise-model configuration; support deterministic resume without counting rays twice.
- At every outer cycle write: `ELBO_total`, individual likelihood/prior/entropy terms, motion/map/posterior stage change, physical residual distributions, rank of observed motion directions, solve residual norms, filter covariance summary, **joint marginal covariance query** summary, number of current geometry hypotheses (not a hard novelty threshold), frame/camera coverage and memory.
- Distinguish numerical failure (`NUMERICAL_BREAKDOWN`), incomplete calibration (`MISSING_NOISE_MODEL`), genuinely unobservable geometry (`UNIDENTIFIABLE`), and optional tiny-oracle enumeration limit (`ORACLE_RESOURCE_LIMIT`). Do not use global enumeration limit to fail the full scalable sparse path.

## 14. Nonnegotiable definition of done

1. `main.py` genuinely runs loader -> Stage A IEKF -> backward RTS -> Stage B backward geometry discovery -> joint sparse refinement -> discrete posterior -> common objective convergence, without invoking 3DGS.
2. Correct 12D/body state and explicit `f,F,G,Q`, initial state/covariance and calibrated output noise are implemented and tested.
3. Dense joint covariance/precision and global discrete-scene enumeration are absent from production code.
4. Source observation is used **exactly once**; shared uncertain pose induces retained motion–map and map–map information.
5. Conditional filtering covariances, joint/marginal covariances, variational/discrete uncertainties are clearly differentiated and numerically tested against tiny oracle solutions.
6. Posterior covariance is not contaminated with optimization damping or fabricated priors; gauge and singular modes are handled transparently.
7. The same full-data objective is used in Stage A, Stage B and outer-loop acceptance. Correctly qualified monotonicity/stationarity diagnostics, not a claim of global reconstruction completeness.
8. The real dataset runs all frames/cameras in the default configuration, writing all outputs to the requested external SSD; do not use evaluation GT for inference.
9. Deliver actual code changes, a real full-scene launcher, full command lines, `reports/covariance_audit.md`, numerical test output and honest unresolved limitations. Do not stop at a document or a toy demo.

**If two mathematical requirements cannot simultaneously hold (e.g. unrestricted exact million-variable discrete posterior and polynomial complexity), expose the conflict explicitly and implement the source-theory-permitted sparse continuous solution with a declared deterministic probabilistic discrete inference family. Never conceal an approximation, invent a calibration, or sacrifice the sparse framework to satisfy an unnecessary exponential enumerator.**
