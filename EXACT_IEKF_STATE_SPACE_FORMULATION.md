# Mandatory mathematical contract: complete state-space RGB-D IEKF/IEKS

> **Status:** This section fills a genuine omission in the earlier Codex contract and theory note. It is **normative**: implement these equations, with the stated Lie-coordinate and noise conventions. The original theory specified the motion process and the first-hit observation distribution separately; it did **not** state one complete EKF state/output system, derive its full 12-by-12 state transition, or make the output-likelihood versus conditional-reprojection distinction sufficiently explicit. This addendum does so. It is not 3DGS and must not be replaced by a pose-only batch optimizer.
>
> **Mathematical limitation:** An IEKF is, by definition, a local Gaussian filtering approximation for nonlinear measurements. For an occlusion mixture with unknown correspondences its **exact posterior is not generally Gaussian**. Therefore an implementation cannot simultaneously claim a globally exact posterior and a 12-dimensional Gaussian IEKF update. Preserve exact finite-catalogue discrete marginalization where computationally possible and explicitly label the continuous IEKF as the local approximation already assumed by the theory. Do not mislabel a mixture of posteriors as an exact single Gaussian.

## 1. Indices, frames, state variables, and gauges

- `k=0,...,K`: **all** 60 acquisition times; use actual `dt_k=t[k+1]-t[k]` or a user-specified uniform interval if timestamps are absent.
- `c=0,...,C-1`: all 8 calibrated cameras. World-to-camera transform `G_c=(R_c,t_c)` is fixed; all depth values are metric.
- `b=1,...,B`: unknown moving rigid components in a finite modeled component universe. `b=0` is static world geometry, with `T[0,k]=I` for every `k`.
- `i=1,...,N`: canonical surface candidates sampled from the **complete RGB-D sequence**, including points first seen late. Their source-time label only initializes an estimate; it is **not** a second measurement.
- For moving component `b` at time `k` define `T[b,k] in SE(3)` (canonical-to-world), and a **world/spatial twist** `nu[b,k]=(v,omega) in R^6`, translation first.

The moving-state vector (manifold-valued) is

```math
s_k = \big((T_{1,k},\nu_{1,k}),\ldots,(T_{B,k},\nu_{B,k})\big),
\qquad \dim(\delta s_k)=12B.
```

The fixed-in-time canonical map is

```math
m=\{p_i\in\mathbb R^3,\;a_i\in\mathbb R^3,\;S_i,\;n_i\}_{i=1}^{N},
```

where `S_i` is a finite oriented surface support whose pixel coverage comes from the RGB-D sampling geometry (a mathematical point of zero area **cannot** define a normalized full-image occlusion model). The discrete scene state is `z=(e_i,b_i)_i`: point existence and rigid-part ownership. The first-hit indicator `h_r` is a **conditional auxiliary latent variable marginalized analytically in (O6)**. For a complete-data augmentation, use the joint state `(z,h)` and its complete-data factors `v_{r,h_r} f_{r,h_r}` **instead of** the marginalized `L_r`. Never retain `h_r` in `z` while simultaneously treating `L_r` as its complete-data likelihood. Discrete variables are inferred in Stage B, **not** smuggled into the Gaussian dynamic state as real numbers.

Use `T[b,0]=I` as a gauge for each moving component **only if the map coordinates are defined at `t0`**; this gauge does **not** assert that the component was observed at `t0`. Alternatively use a single explicitly specified reference `T[b,kref]=I` and transform its entire canonical map accordingly. Gauge must be fixed before inverting information matrices.

## 2. Error-state convention (required, not configurable by accident)

For `xi=(rho,phi) in R^6`, use translation-first wedge

```math
\widehat\xi=\begin{bmatrix}[\phi]_\times&\rho\\0&0\end{bmatrix}.
```

The **left-multiplicative, world-frame pose error** and additive twist error are

```math
T_{b,k}=\operatorname{Exp}(\widehat{\delta\xi}_{b,k})\bar T_{b,k},
\qquad \nu_{b,k}=\bar\nu_{b,k}+\delta\nu_{b,k},
\qquad
\delta s_{b,k}=\begin{bmatrix}\delta\xi_{b,k}\\\delta\nu_{b,k}\end{bmatrix}\in\mathbb R^{12}.
```

Retraction: `bar_s boxplus delta_s = (Exp(delta_xi^) bar_T, bar_nu+delta_nu)`. Local subtraction: `s boxminus bar_s = (Log(T bar_T^{-1})^vee, nu-bar_nu)`. **Every derivative, covariance and RTS discrepancy must use these same tangent charts.**

For `T=(R,t)` and translation-first coordinates, implement

```math
\operatorname{Ad}_T=
\begin{bmatrix}R&[t]_\times R\\0&R\end{bmatrix},\qquad
\operatorname{ad}_{(v,\omega)}=
\begin{bmatrix}[\omega]_\times&[v]_\times\\0&[\omega]_\times\end{bmatrix}.
```

The left group Jacobian is defined (no sign ambiguity) as

```math
J_l(\phi)=\int_0^1 \exp(\tau\operatorname{ad}_\phi)\,d\tau
=\sum_{n=0}^\infty\frac{\operatorname{ad}_\phi^n}{(n+1)!},
\quad
\operatorname{Exp}(\widehat{\phi+\epsilon})
=\operatorname{Exp}(\widehat{J_l(\phi)\epsilon})
\operatorname{Exp}(\widehat\phi)+O(\|\epsilon\|^2).
```

Use a robust closed form or a convergent Lie-series evaluation with correct small-angle limits. Never approximate `J_l` by identity away from a numerically verified small-angle regime.

## 3. **State dynamic equation**: exact declared discrete process and its EKF linearization

**Nonlinear process equation**, for each moving component:

```math
\boxed{\begin{aligned}
T_{b,k+1}&=\operatorname{Exp}(\Delta t_k\widehat\nu_{b,k})
T_{b,k}\operatorname{Exp}(\widehat w^T_{b,k}),\\
\nu_{b,k+1}&=\nu_{b,k}+w^\nu_{b,k},\\
\begin{bmatrix}w^T_{b,k}\\w^\nu_{b,k}\end{bmatrix}&\sim
\mathcal N(0,Q^{\rm local}_{b,k}).
\end{aligned}}
\tag{D1}
```

`w^T` is a **right/body-frame** pose perturbation applied *after* the deterministic left spatial-twist step, as stated in the original theory; `w^nu` is a world-frame additive twist innovation. This is not an inertial dynamics law and does **not** enforce a joint axis yet; it is the declared constant-spatial-twist *stochastic prior*. Do not silently replace it with `T += nu * dt` or pose finite differences.

At the nominal state define `phi=dt_k * bar_nu[b,k]`, `A=Exp(phi^)` and `bar_T_next=A bar_T`. Prediction:

```math
\boxed{\bar T_{b,k+1|k}=A\bar T_{b,k|k},\qquad
\bar\nu_{b,k+1|k}=\bar\nu_{b,k|k}.}
\tag{D2}
```

**Derived 12-by-12 first-order error transition** under the left-pose/right-process-noise conventions:

```math
\boxed{
\delta s_{b,k+1}=F_{b,k}\delta s_{b,k}+G_{b,k}w_{b,k}+O(\|\delta,w\|^2),
}
\tag{D3}
```

```math
\boxed{
F_{b,k}=\begin{bmatrix}
\operatorname{Ad}_{A}&\Delta t_k\,J_l(\phi)\\
0_{6\times6}&I_6
\end{bmatrix},\qquad
G_{b,k}=\begin{bmatrix}
\operatorname{Ad}_{\bar T_{b,k+1|k}}&0\\
0&I_6
\end{bmatrix}.
}
\tag{D4}
```

**Derivation, which must be preserved in code comments/tests:** perturb `T_k` on the left. Holding `nu` fixed, `A Exp(delta_xi)T = Exp(Ad_A delta_xi) A T`. Perturbing the twist, `Exp((phi+dt delta_nu)^) = Exp((J_l(phi) dt delta_nu)^) A + O(delta_nu^2)`. Finally `bar_T_next Exp(w_T^) = Exp((Ad_bar_T_next w_T)^) bar_T_next`. Therefore the upper blocks of `F` and `G` are exactly those in (D4) at first order. The lower blocks follow from `nu_next=nu+w_nu`.

If the process covariance `Q_k` is **defined directly in output left-error coordinates**, set `G=I` only after transforming `Q_k=G Q_k^local G^T` consistently; otherwise this changes the physical model.

For all moving bodies, stack body blocks (including any modeled inter-body process correlation) and predict

```math
\boxed{
P_{k+1|k}=F_kP_{k|k}F_k^\top+G_kQ_k^{\rm local}G_k^\top.
}
\tag{D5}
```

Cross-body covariance is **not** automatically block diagonal after joint RGB-D updates. Store full cross-block information when the observation/correspondence model couples bodies. The process noise specification and units must come from a stated model/configuration, not secretly invented numerical coefficients.

## 4. **Output dynamic equation**: two *different* mathematically legitimate forms

### 4.1 Conditional point-to-RGB-D reprojection, when identity/visibility are conditioned on

For a candidate `i` owned by `b_i`, define its world and camera coordinates:

```math
x_{i,k}=R_{b_i,k}p_i+t_{b_i,k},\qquad
q_{i,c,k}=R_{cw,c}x_{i,k}+t_{cw,c}=(X,Y,Z)^\top,
```

```math
\hat u_{i,c,k}=\pi_c(q_{i,c,k})=
\begin{bmatrix}f_xX/Z+c_x\\f_yY/Z+c_y\end{bmatrix},
\qquad \hat d_{i,c,k}=e_3^\top q_{i,c,k}.
```

For a *specified* valid association and visible surface, the nonlinear residual and output relation are

```math
\boxed{
r_{i,c,k}(s_k,m)=
\begin{bmatrix}
I_{c,k}(\hat u_{i,c,k})-a_i\\
D_{c,k}(\hat u_{i,c,k})-\hat d_{i,c,k}
\end{bmatrix},\quad
r_{i,c,k}=0+\epsilon_{i,c,k},\quad
\epsilon_{i,c,k}\sim\mathcal N(0,R_{i,c,k}).
}
\tag{O1}
```

This registration equation relies on image interpolation; it is **not a complete generative model for pixels**, cannot be evaluated as a surface correspondence for an occluded point, and must not replace the full output likelihood (O5) below. If association instead refers to a known *fixed pixel* `u`, the corresponding conditional additive measurement is

```math
\boxed{y_{k,c,u}=h_{k,c,u}(s_k,m,i)+v_{k,c,u},\quad
h_{k,c,u}=\begin{bmatrix}a_i\\\hat d_{i,c,k}\end{bmatrix},\quad
v_{k,c,u}\sim\mathcal N(0,R_{k,c,u,i}).}
\tag{O2}
```

This conditional output is four-dimensional `[RGB(3), depth(1)]`. It has instantaneous dependence on **pose** but not on `nu`; twist becomes observable across time through `F`. Conditional on fixed `m`, `i`, `u` and visibility, its Jacobian uses the depth row from (O3), while RGB pose derivatives of a fixed point color are zero. The pixel **coverage probability** changes with pose and must be differentiated in the full likelihood (O5). Do not confuse the fixed-pixel conditional Jacobian with the image-sampling Jacobian below.

For a left pose perturbation let

```math
J_x=\begin{bmatrix}I_3&-[x_{i,k}]_\times\end{bmatrix},\quad
J_q=R_{cw,c}J_x,\quad
J_\pi=
\begin{bmatrix}
f_x/Z&0&-f_xX/Z^2\\
0&f_y/Z&-f_yY/Z^2
\end{bmatrix}.
```

The **analytical image-sampling residual Jacobian**, ordered `[pose(6),twist(6)]`, is

```math
\boxed{
H^{\rm reproj}_{i,c,k}=\begin{bmatrix}
\nabla I_{c,k}(\hat u)J_\pi J_q &0_{3\times6}\\
(\nabla D_{c,k}(\hat u)J_\pi-e_3^\top)J_q&0_{1\times6}
\end{bmatrix}.
}
\tag{O3}
```

The **fixed-pixel generative depth-mean Jacobian** is

```math
\boxed{
H^{\rm fixed-ray}_{i,c,k}=
\begin{bmatrix}0_{3\times6}&0_{3\times6}\\
e_3^\top J_q&0_{1\times6}\end{bmatrix},
}
\tag{O4}
```

with an additional map-point block `∂d/∂p_i=e_3^T R_c R_b`; the projected support/visibility derivative is separate. The conditional covariance `R` is the actual joint RGB-depth covariance, not a hand-picked scalar weight or a second loss counted on top of the ray likelihood.

### 4.2 **Actual full-frame RGB-D output dynamic equation**, including occlusion, new surfaces, and no-return rays

For every camera pixel ray `r=(k,c,u)`, the output `y_r` is either a valid value in `R^4` or a separate `no-return` symbol. The **full nonlinear measurement law** is a probability distribution, rather than a single additive Gaussian equation:

```math
\boxed{y_r\mid (s_k,m,z)\sim L_r(\cdot\mid s_k,m,z),\quad
p(Y_k\mid s_k,m,z)=\prod_{c,u}L_{(k,c,u)}(y_{k,c,u}\mid s_k,m,z).}
\tag{O5}
```

Define the calibrated pixel-response overlap `c_{r,i}(s_k,m)∈[0,1]` of the projected surface support `S_i` with pixel `u`; fix an ordering `i_1,...,i_M` by positive increasing predicted depth on the current smooth active region; define `eta_{r,i}=e_i c_{r,i}`. The first-hit and empty weights are

```math
v_{r,i_j}=\eta_{r,i_j}\prod_{a<j}(1-\eta_{r,i_a}),\qquad
v_{r,\emptyset}=\prod_{a=1}^M(1-\eta_{r,i_a}),\qquad
v_{r,\emptyset}+\sum_{j=1}^M v_{r,i_j}=1.
```

For each covered candidate let `f_{r,i}(y)` be the **proper joint RGB-D sensor distribution** centered on `(a_i,d_{i,r})` for a valid return, with its own specified no-return probability mass. Let `f_{r,empty}` be a **proper** background/previously-unexplained surface distribution including valid RGB-D and no-return outcomes. Then

```math
\boxed{
L_r(y\mid s_k,m,e,b)
=v_{r,\emptyset}f_{r,\emptyset}(y)
+\sum_{i=1}^{N}v_{r,i}f_{r,i}(y).
}
\tag{O6}
```

This is the **exact stated output equation for the finite RGB-D first-hit model**. For valid depth one possible specified branch is `f_{r,i}(y)=N_4(y; [a_i,d_{i,r}]^T,R_{r,i})` times a properly normalized return probability. A `D=0` observation is not `D=0` Gaussian data; it is the no-return outcome. Conditional independence over rays is an explicit model assumption, not an assertion that RGB-D errors are inherently independent.

The output for a full time step is `Y_k={y_{k,c,u}: all cameras/pixels}` and `h_k` is the **vector of ray-distribution parameters** `(v_{r,i}, a_i,d_{i,r}, sensor-noise parameters)` defining `p(Y_k|s_k,m,z)`. Its dimension grows with the number of observed rays. **There is no globally valid additive equation `Y_k=h_k(s_k)+v_k` with Gaussian `v_k` after occlusion/association has been marginalized.** Codex must implement (O5)–(O6), not silently flatten the mixture into independent Gaussian reprojection residuals.

## 5. Exact analytical *score* for the full output distribution

For one observed ray, write `L_r=Σ_{j∈{empty,i}} v_{r,j} f_{r,j}` and the **conditional** first-hit responsibility

```math
\gamma_{r,j}=\frac{v_{r,j}f_{r,j}(y_r)}{L_r(y_r)},\qquad
\sum_j\gamma_{r,j}=1.
```

For any tangent coordinate `theta` of pose, twist, map or local geometric support, with fixed depth-order active set:

```math
\boxed{
\nabla_\theta\log L_r
=\sum_j\gamma_{r,j}
\left(\nabla_\theta\log v_{r,j}
+\nabla_\theta\log f_{r,j}(y_r)\right).
}
\tag{S1}
```

In particular, for `eta_i=e_i c_i` and non-degenerate `0<eta_i<1`:

```math
\nabla\log v_i
=\frac{\nabla\eta_i}{\eta_i}
-\sum_{j\prec i}\frac{\nabla\eta_j}{1-\eta_j},\quad
\nabla\log v_{\emptyset}
=-\sum_j\frac{\nabla\eta_j}{1-\eta_j}.
\tag{S2}
```

Boundary cases `eta=0/1` must be evaluated through stable prefix-product differentiation / limiting derivatives, **not** by division by zero or adding an undocumented epsilon to probabilities. For a valid-return Gaussian branch with fixed covariance,

```math
\nabla_\theta\log f_{r,i}
=J_{\mu,i,\theta}^\top R_{r,i}^{-1}(y_r-\mu_{r,i}),\quad
\mu_{r,i}=[a_i;d_{i,r}].
\tag{S3}
```

If covariance and/no-return probabilities depend on pose or map, include their derivatives too. `∇c_{r,i}` must include the pose-dependent **pixel/surfel overlap** (and any surface-normal/support parameters); `∇d_{i,r}` uses (O4). Do **not** calculate the full-output gradient using (O3) alone, because that omits `∇log v` and does not differentiate the generative ray density correctly.

For any discrete scene configuration `z=(e,b)`, use the score conditional on that state, with the first-hit `h_r` **already marginalized inside** `L_r`. In Stage A, with a **fixed exact discrete-scene posterior** `q^r(z)` and map `m^r`, the expected log-likelihood score is

```math
g_k(s_k)=\mathbb E_{q^r(z)}
\sum_{c,u}\nabla_{s_k}\log L_{(k,c,u)}
\big(y_{k,c,u}\mid s_k,m^r,z\big).
\tag{S4}
```

An alternative explicit complete-data model over `(z,h)` uses `E_{q(z,h)}[log v_{r,h_r}+log f_{r,h_r}]` and must **not** additionally multiply by or add `log L_r`; it is an alternative EM representation of the same observed-data model. The expected score shown in (S4) is **not** generally the score of a ray mixture formed by replacing binary `e_i` and part labels with independent posterior means. The global `q(z)` may have coupled assignments. If exact marginalization is infeasible, raise `RESOURCE_LIMIT` in the required exact mode rather than using such a substitution silently.

## 6. Forward IEKF: prediction, correct measurement update, covariance

At frame `k`, given the fixed geometry/map, discrete posterior for this outer iteration, predicted state `bar s^-_k` and covariance `P^-_k`, define the local tangent `delta` about **that fixed predicted state** and maximize the conditional posterior:

```math
\boxed{
E_k(\delta)=\tfrac12\delta^\top(P^-_k)^{-1}\delta
-\mathbb E_{q^r(z)}\sum_{c,u}
\log L_{(k,c,u)}(y_{k,c,u}\mid \bar s^-_k\boxplus\delta,m^r,z).
}
\tag{K1}
```

Let `ell_k(delta)` denote the expected log-ray-likelihood. The **actual** gradient and curvature are

```math
\nabla E_k=(P^-_k)^{-1}\delta-\nabla\ell_k(\delta),\qquad
\nabla^2E_k=(P^-_k)^{-1}-\nabla^2\ell_k(\delta).
\tag{K2}
```

Every iterated solve uses the **same `P^-_k` and the same original frame-k images**. One valid damped local Newton-information iteration is

```math
\boxed{
\left[(P^-_k)^{-1}-\nabla^2\ell_k(\delta_l)+\lambda_l D_l\right]\Delta_l
=-\nabla E_k(\delta_l),\qquad
\delta_{l+1}=\delta_l+\alpha_l\Delta_l.
}
\tag{K3}
```

Choose `lambda_l D_l` to make the local system positive definite and use an Armijo/backtracking acceptance test evaluated on the **actual nonlinear** `E_k`, with active ray coverage/ordering rebuilt after accepted updates. A positive-definite Fisher/information metric may substitute for the Hessian only when it is **explicitly labeled a curvature approximation**; it must still multiply the full score (S4). Implement analytic local second derivatives or the mathematically specified information operator, not an image-reprojection-only surrogate.

At the converged local correction, retract `bar s^+_k=bar s^-_k boxplus delta*`. For conditional additive Gaussian observations with known correspondences and fixed H, the usual IEKF/Gauss–Newton special case is

```math
\boxed{
\left[(P^-_k)^{-1}+H_l^\top R^{-1}H_l\right]\Delta_l
=H_l^\top R^{-1}(y-h(s_l))-(P^-_k)^{-1}\delta_l,
}
\tag{K4}
```

and the Joseph covariance update `P^+=(I-KH)P^-(I-KH)^T+K R K^T` with `K=P^-H^T(HP^-H^T+R)^-1`. **(K4) is not the full finite-mixture ray update** and must never be presented as its complete equivalent.

For the mixture (O6), a single Gaussian filter returns a **local Gaussian approximation**, with covariance obtained in a consistent final tangent chart from the positive-definite local posterior information (e.g. `P^+≈[∇²E_k(delta*)]^-1` where that Hessian is SPD, with chart transport as needed). This covariance is not the exact globally multimodal posterior variance. Record posterior uncertainty semantics and matrix conditioning. For Stage A's coupling to uncertain geometry, obey Section 8 below.

## 7. Actual backward smoother and outer iteration

Store every `(s_{k|k-1},P_{k|k-1},s_{k|k},P_{k|k},F_k)` and run an RTS backward pass from `K` to `0`. In locally consistent tangent charts:

```math
C_k=P_{k|k}F_k^\top(P_{k+1|k})^{-1},
\tag{R1}
```

```math
\boxed{
\hat s_{k|K}=\hat s_{k|k}\boxplus
C_k\big(\hat s_{k+1|K}\boxminus\hat s_{k+1|k}\big),
}
\tag{R2}
```

```math
\boxed{
P_{k|K}=P_{k|k}
+C_k\left(P_{k+1|K}-P_{k+1|k}\right)C_k^\top,
}
\tag{R3}
```

where all terms are transported into the same tangent charts before addition, and `P^-` inverse means a stable linear solve. Explicitly test against a linear-Gaussian batch smoother, including block cross-covariances. For nonlinear IEKS, reset to the same *original* initial state/process prior and original images after each relinearization, not to the previous posterior as an additional independent prior. Armijo-accept outer changes using the **same global ELBO** as defined in the theory.

## 8. Map/pose covariance, late points and the separated Stage B

A point first observed at source frame `s` is initialized under component `b` by

```math
p_i^{(b)}=T_{b,s}^{-1}y_{i,s}^{3D},\qquad
x_{i,k}^{(b)}=T_{b,k}T_{b,s}^{-1}y_{i,s}^{3D}
\quad \forall k\in\{0,\ldots,K\}.
\tag{B1}
```

When `p_i=g(y_i,s_s)`, linearize `delta p_i=J_y delta y_i+J_s delta s_s`; for independent acquisition measurement noise and predicted motion error,

```math
P_{p_i}=J_yR_yJ_y^\top+J_sP_{s_s}J_s^\top,\quad
P_{p_i,s_s}=J_sP_{s_s};
\tag{B2}
```

if they are dependent, include the cross terms. This is **not** a new independent observation of the source frame. Every later/earlier ray contributes only once through (O5); absence behind the foreground has no extra penalty. Stage B freezes all `s_k`, uses (B1) for candidate testing in **every earlier and later frame**, and computes exact posterior existence/part evidence under the fixed finite universe. It then updates canonical map parameters by the same complete ray likelihood. Stage A resumes with the refined map and all information/cross-correlations.

**Non-negotiable distinction:** A pose-only, `12B`-dimensional IEKF conditioned on a fixed map cannot by itself provide the *joint* motion–map posterior claimed by the theory. Exact first-order Gaussian motion/map coupling requires either (a) augmenting the dynamic state by static map variables `m` with dynamics `m_{k+1}=m_k`, `Q_m=0` and preserving the full joint covariance, or (b) algebraically exact batch marginalization/Schur elimination retaining pose–map cross-information. An implementation using only (K1) with fixed uncertain map values must label its motion covariance **conditional on the map**, not the full marginal motion covariance. Verify dense augmentation against the selected marginalization on a small linear test.

## 9. Main-loop contract, exact test names, and what must not be claimed

The required control flow is

```text
load all RGB-D frames, all cameras, metric calibration
build the fixed finite candidate universe from ALL valid depth observations
initialize shared candidate map and existence/part hypotheses, gauge and priors
repeat outer iteration:
    Stage A: map and q fixed
        forward IEKF: for k = 0,...,K perform (D1)-(D5), (O5)-(O6), (S1)-(S4), (K1)-(K3)
        backward RTS: k = K-1,...,0 perform (R1)-(R3)
        relinearize same original measurements and optimize motion under full ELBO
    Stage B: trajectories frozen
        each later-source candidate is inverse-transported to all earlier frames using (B1)
        evaluate complete (O5)-(O6) visibility and RGB-D/no-return evidence
        update candidate geometry and exact discrete posterior without residual thresholds
    evaluate the SAME full-data ELBO after each accepted substep
    stop only when actual objective and KKT/gradient residual conditions hold
export one shared canonical geometry, motion history, full prediction images, and uncertainty
```

Implement tests named at least `test_F_SE3_left_pose_right_noise_finite_difference`, `test_G_right_process_noise_adjoint`, `test_rgbd_projection_and_O3_jacobian`, `test_fixed_ray_O4_jacobian`, `test_first_hit_normalization_and_occlusion`, `test_full_ray_score_S1_against_finite_difference`, `test_conditional_ekf_gaussian_special_case`, `test_ieks_RTS_against_batch_linear`, `test_source_pose_map_cross_covariance`, `test_backward_revealed_surface_with_occlusion`, `test_no_double_count_on_relinearization`, and `test_full_60x8_manifest`. All unit tests must exercise nonzero rotation, translation, and nonzero twist; identity-only tests are insufficient.

**Interpretation:** The equations above are the missing EKF state/output specification. They are sufficient to define the dynamic state and nonlinear output distribution of the declared RGB-D framework. They do **not** uniquely specify calibrated sensor priors or process-noise numbers (not present in the supplied theory/data); require explicit model configuration, or estimate/calibrate parameters from independent evidence. The convergence theorem applies only under the PDF's stated assumptions, not globally or to an intrinsically multimodal posterior collapsed into a Gaussian.
