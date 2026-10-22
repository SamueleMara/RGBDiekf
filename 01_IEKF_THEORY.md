# FRAMEWORK A — Analytical Iterated Error-State EKF and RTS RGB-D SLAM

**MANDATORY FOREGROUND-MASK ADDENDUM:** The binary foreground masks are a hard admissible-measurement and predicted-visible-point feasibility constraint. Read and implement `CODEX_PARALLEL_IEKF_UKF_HARD_FOREGROUND_MASKS.md` before using this document. No background pixel can initialize SOLVED/UNSOLVED, be matched, enter an IEKF/UKF measurement, or be fused into geometry. Occluded solved points remain valid. Identical policy in both pipelines.


**Independent theory.** The entire solved/unsolved reconstruction algorithm is *exactly* `00_COMMON_SOLVED_UNSOLVED_SPEC.md`. This document derives its sole inner motion solver: **analytical IEKF**. It does not call a UKF or dynamically switch filters.

## A1. Problem, conditional measurements and Bayesian state

At frame `k`, let there be `B` currently accepted rigid groups. The conditional Gaussian motion state for group `g` is

\[
s_{g,k}=(T_{g,k},\nu_{g,k}),\quad T_{g,k}\in SE(3),\quad \nu_{g,k}\in\mathbb R^6.
\]

Its 12D error state is `delta s=(delta xi, delta nu)`. The full frame error state dimension is `n=12B` (not a function of the number of points). `SOLVED` contains fused persistent point IDs and their accepted or provisional motion associations. `UNSOLVED[k]` is solely an indexed set of raw observations. A static trajectory is simply `T[g,k]=I`; no static label is a state variable.

For translation-first `xi=(rho,phi)` define `xi^ = [[phi]x,rho;0,0]`, `T=Exp(delta xi^) Tbar`, `nu=nubar+delta nu`. The gauge is a group's reference frame `T[g,s_ref]=I`; its point coordinates are in that frame. A truly static group need not be predeclared and its estimated motions may approach identity naturally.

Conditional on an external association `a=(i,k,c,u)` with a solved canonical 3D point `p_i`, the observed vector is

\[
z_a=(u_x,u_y,D_{c,k}(u),I_{c,k}(u)^\top)^\top\in\mathbb R^6,
\quad
z_a=h_a(s_k,p_i)+\epsilon_a,
\]

\[
h_a=\big(\pi_c(X_a)^\top,\;[X_a]_z,\;a_i^\top\big)^\top,
\quad X_a=R_{cw,c}(R_{g,k}p_i+t_{g,k})+t_{cw,c},\quad
\epsilon_a\sim\mathcal N(0,R_a).
\]

`R_a` describes target-pixel association/localization, depth and RGB uncertainty, and its cross terms where applicable. A matched-pixel RGB residual is an **association / appearance** factor: with point appearance fixed, its pose derivative is zero. If true direct photometric pose refinement is wanted, additionally/alternatively use warped-image residual `I_k(pi(X))-a_i` with analytical `nabla I` and covariance propagated from raw pixels, not a second independent reuse of the RGB pixel. There are no normals or explicit surface elements. The sparse projector estimates visibility from depth ordering before applying the conditional measurement. Nonmatching observations stay in `UNSOLVED` and do not generate fake residuals.

**Meaning of exactness:** `H` is the exact analytical derivative of the *specified conditional* output; the iterated EKF is still a local Gaussian approximation to nonlinear Bayesian filtering. Correspondences are estimated outside the filter and may be ambiguous.

## A2. Nonlinear dynamic state equation and process covariance

Use spatial/world twist `nu=(v,omega)` and the same left perturbation chart throughout:

\[
\boxed{\begin{aligned}
T_{g,k+1}&=\operatorname{Exp}(\Delta t_k\widehat\nu_{g,k})T_{g,k}\operatorname{Exp}(\widehat w^T_{g,k}),\\
\nu_{g,k+1}&=\nu_{g,k}+w^\nu_{g,k}.
\end{aligned}}
\]

`w=(wT,wnu)` has the documented model covariance `Q_k`. The right-applied pose process noise is mapped into the left error chart. Define `phi=Delta t * nubar`, `A=Exp(phi^)` and `Tpred=A Tbar`. Translation-first adjoint is

\[
\mathrm{Ad}_{(R,t)}=\begin{bmatrix}R&[t]_\times R\\0&R\end{bmatrix}.
\]

With the left group Jacobian defined by `J_l(phi) = integral_0^1 Ad_{Exp(s phi^)} ds`, the first-order state transition is

\[
\boxed{F_k=\begin{bmatrix}\operatorname{Ad}_{A}&\Delta t_k J_l(\phi)\\0&I_6\end{bmatrix}},\qquad
G_k=\begin{bmatrix}\operatorname{Ad}_{T_{pred}}&0\\0&I_6\end{bmatrix}.
\]

Compute (for all groups, including any cross-block covariance)

\[
\boxed{P^-_{k+1}=F_kP_k^+F_k^\top+G_kQ_kG_k^\top.}
\]

If using a continuous-time white-acceleration model, discretize its spectral-density covariance with the actual `dt`, including pose/twist cross terms. Do not replace it with independent diagonal position and velocity noise without model justification. `P0` and `Q` numerical assumptions must come from the approved existing configuration and be labeled as assumptions. The chosen gauge is fixed before inversion.

## A3. Analytical RGB-D output Jacobian

Let `x=R_g p_i+t_g` and `X=R_c x+t_c` in camera coordinates. With `e3=(0,0,1)^T`, `pi(X)=(f_x X_x/X_z+c_x, f_y X_y/X_z+c_y)`:

\[
J_{pose3}=R_c\begin{bmatrix}I_3&-[x]_\times\end{bmatrix},\qquad
J_\pi(X)=\begin{bmatrix}f_x/X_z&0&-f_x X_x/X_z^2\\0&f_y/X_z&-f_yX_y/X_z^2\end{bmatrix}.
\]

The conditional output derivative wrt `delta s=(delta xi,delta nu)` is

\[
\boxed{H_{a}=\begin{bmatrix}J_\pi J_{pose3}&0_{2\times6}\\e_3^\top J_{pose3}&0_{1\times6}\\0_{3\times6}&0_{3\times6}\end{bmatrix}}\in\mathbb R^{6\times12}.
\]

For the alternative *warped* RGB residual `r_RGB=I_k(pi(X))-a_i`, its pose derivative is `grad I_k(pi(X)) J_pi J_pose3` (sign depends on whether Jacobian is defined for `r` or `h`). For source-acquired points `T_k T_s^{-1} y_s`, both `T_k` and `T_s` contribute analytically when jointly refined: differentiate the relative transform rather than freezing the uncertain source pose. For `p_i` uncertainty, `J_p = R_c R_g`; incorporate its conditional residual covariance or retain sparse point-pose information factors.

No finite differences are permitted in the production analytical Jacobians. Verify them by finite differences in tests, including camera world/convention transforms and left chart perturbations.

## A4. Prediction, iterated correction and covariance

For the fixed predicted state `(s^-,P^-)`, build the deterministic current correspondence and visibility set outside the IEKF. At iteration `ell`, define `delta_ell = s_ell boxminus s^-`, observed residual `r_ell=z-h(s_ell)` and current `H_ell = dh/d delta` (with chart transport of prior terms if the chart changes).

For a fixed linearization chart the Gauss-Newton form of the IEKF correction is

\[
\boxed{\left[(P^-)^{-1}+H_\ell^\top R^{-1}H_\ell\right]\Delta_\ell
=H_\ell^\top R^{-1}r_\ell-(P^-)^{-1}\delta_\ell.}
\]

The prior `(s^-,P^-)` is **fixed for all inner relinearizations**. The measurement is used once even if a 10-step IEKF is used to converge. Apply a Lie retraction and optionally backtracking against the *actual fixed-association conditional MAP objective*, then reproject and relinearize. External matching changes may trigger a new well-defined conditional problem, but a trial update must not compare scores from different matches without invoking the same global objective.

Measurement information can be accumulated without constructing `H`:

\[
A=(P^-)^{-1}+\sum_{a\in E_k}H_a^\top R_a^{-1}H_a,\quad
b=\sum_{a\in E_k}H_a^\top R_a^{-1}r_a-(P^-)^{-1}\delta.
\]

For one group `A` is `12x12`; cross-group factors produce sparse `12x12` blocks. For a joint state with map variables, expose `Hv` / `H^T v` or Schur-eliminate the sparse map blocks. No dense `N_points x N_points` or `N_pixels x N_pixels` arrays. Uncertainty **after** the accepted update must be computed with the undamped local posterior information `A`, with correct tangent-chart transport. Levenberg damping used for step acceptance must not artificially increase posterior precision. A Joseph-form covariance update with the equivalent linearized `(H,R)` is a small-case reference test. Do not call this exact marginal covariance when map uncertainty and point-pose coupling are omitted.

## A5. Backward motion smoothing (IEKF branch only)

After one forward filtered run save `(s^+_k,P^+_k)`, `(s^-_{k+1},P^-_{k+1})` and `F_k`, with reference tangent frames. Compute Rauch-Tung-Striebel gain

\[
C_k=P^+_k F_k^\top(P^-_{k+1})^{-1}.
\]

Using tangent errors expressed in the appropriate charts:

\[
\delta_k=C_k\big(s_{k+1|K}\boxminus s^-_{k+1}\big),\quad
s_{k|K}=s^+_k\boxplus\delta_k,
\]

\[
P_{k|K}=P^+_k+C_k(P_{k+1|K}-P^-_{k+1})C_k^\top,
\]

with coordinate transport to the updated tangent. This is the first-order Lie-group RTS smoother; it does not discover points. For map poses correlated with geometry, the small conditional state smoother must not be labeled the exact joint map marginal smoother: use retained sparse joint factors or selected Schur-complement queries.

## A6. SAME solved/unsolved outer loop, no filter mixing

Use the algorithm in `00_COMMON_SOLVED_UNSOLVED_SPEC.md` **without modification**:

1. Frame-0 multiview points form `SOLVED` (group affiliation initially provisional); all other valid RGB-D observations start in `UNSOLVED`.
2. Forward for every `k`: predict pose/twist + covariance, move only solved points, sparse associate RGB-D, run analytic IEKF correction, retain unmatched observations.
3. Backward RTS motion smoothing of the *same* poses/covariances.
4. Late-to-early reverse discovery: transport unsolved points under each accepted group `T[g,k] T[g,s]^{-1}`; compare backward visibility, RGB and depth. Propose new group only from coherent multiframe correspondence; its initial rigid fit is a deterministic initializer for IEKF.
5. Fuse accepted points into one canonical point per physically matched location, retaining provenance and shared pose uncertainty.
6. Progressive chronological frontier: for `s=1..K`, solve `UNSOLVED[s]` against earlier frames, update affected trajectories via this IEKF, and refresh associations.
7. Repeat full outer sweeps while the common global objective actually improves.

No named static body, no normals, no UV/texture atlas, no TSDF, no dynamic 3DGS. No UKF fallback anywhere in this framework.

## A7. Convergence: valid statement, not an overclaim

For a fixed finite map, group structure and association set, if the measurement function is locally continuously differentiable with locally Lipschitz Jacobian, declared `R` and gauge-fixed prior are positive definite on the estimated tangent subspace, and an Armijo/damped IEKF line search ensures sufficient decrease, accepted conditional MAP objective values decrease and have a finite limit when bounded below. Under standard boundedness/nondegeneracy and sufficient-descent conditions, accumulation points are stationary for that *fixed-association local objective*. This does not establish global registration or scene discovery correctness.

The complete outer loop changes data association and point/group existence discretely. If every accepted outer change is evaluated on the **same comparable full sequence negative log-posterior including a proper unassigned-observation and model-complexity term**, its accepted values are nonincreasing and converge as scalar values when bounded below. Stationarity across discrete changes and guaranteed recovery of all occlusions do not follow. Outlier, invalid depth, missing cameras and genuinely invisible geometry may remain `UNSOLVED`.

## A8. Computational bounds and falsifiable tests

- Runtime: roughly `O(sum_k active_matches_k * C_pose + K * solve(12B))` per sweep, plus sparse GPU matching and backward point-discovery passes.
- Memory: `O(N_solved + max_raw_frame + K (12B)^2 + sparse_map_factors)`, not `O(K*C*W*H*B)` persistent state.
- Required tests: left-chart `F,G` vs finite differences; `H` vs finite differences; correct predicted depth and camera conventions; IEKF vs linear Gaussian analytic posterior; Joseph/reference covariance; fixed prior across inner iterations; RTS vs dense batch reference on small problem; shared pose covariance after late-point insertion; both temporal directions; empty/occluded/no-return behavior; group splitting from initial frame geometry; long-sequence complete forward-backward-frontier loop; monotone global score for every accepted outer edit.

**Deliverable:** Theory-compliant standalone analytical IEKF motion module, its own RTS smoother, and the shared solved/unsolved outer algorithm. All outputs go to `results_iekf/` and are labeled **IEKF** only.
