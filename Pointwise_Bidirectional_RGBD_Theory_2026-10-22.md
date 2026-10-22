---
title: "Pointwise Bidirectional Masked RGB-D Reconstruction"
subtitle: "Rigorous Theory: One Initial Forward Filter, Event-Driven Backward Discovery, Independent Two-Sided Point Filtering"
author: "Mathematical Specification"
date: "22 October 2026"
lang: en-US
---

# Scope, assumptions and methodological boundary

This is the **revised** mathematical specification for point-only RGB-D motion and geometry discovery. It supersedes the per-point Gaussian-orientation state and the fixed-$K$ rigid grouping of earlier specifications. The user-defined algorithm consists of **one large initial forward filter** for all frame-0 foreground points followed by **independent per-new-point backward and forward filters** anchored at the actual first discovered unmatched observation. Previously solved point trajectories are immutable during discovery. Part clustering, rigid-group transforms and joint identification are deliberately deferred.

The data comprise $C$ synchronized, calibrated cameras at acquisition times $\tau_0<\cdots<\tau_T$. At time $t$, camera $c$ records an RGB image $I_{ct}$, camera-axis metric depth $D_{ct}$, binary whole-object mask $M_{ct}$, and valid-return mask $V_{ct}$. Intrinsics $K_c$ and world-to-camera extrinsics $G_c=(R_c,t_c)$ are fixed and known. A genuinely usable geometric observation has $M_{ct}(u)=1$, $V_{ct}(u)=1$, $D_{ct}(u)>0$, and finite depth. RGB outside $M$ and depth outside this set provide no foreground measurement; **no synthetic background depth exists**.

The word *IEKF* here means **iterated extended Kalman filtering** (relinearized measurement MAP), *not* an invariant Lie-group EKF. The present point state is Euclidean. When discussing Maurer et al. (2025), the correct conclusion is that their full-order Lie-group reset is **not required** for this $\mathbb R^6$ state; it becomes relevant only if a later rigid-body model on $SE(3)$ is introduced.

# Mathematical point and observation definitions

Let the immutable raw observation identifier be

$$a=(t,c,u),\qquad u=(u_x,u_y).$$

Write its geometric measurement as

$$z_a=(u_x,u_y,d_a)^\top,\qquad d_a=D_{ct}(u)>0.$$

For the ideal pinhole camera, backprojection into world coordinates is

$$
y_a=R_c^\top \left[d_a K_c^{-1}\begin{bmatrix}u_x\\u_y\\1\end{bmatrix}-t_c\right].\tag{1}
$$

Camera distortion and aligned depth conventions must be accounted for by the calibrated projection/backprojection rather than ignored. If the measured observation covariance is $R_a$ in pixel/pixel/metre units, define the propagated *world* covariance

$$\Sigma_a=J_{b,a}\,R_a\,J_{b,a}^{\top},\qquad J_{b,a}=\frac{\partial y_a}{\partial z_a}.\tag{2}$$

With independent pixel coordinate and depth errors, $R_a$ may be diagonal, but $\Sigma_a$ is generally **anisotropic** and may be non-diagonal. The same observation must not appear as two independent factors, once in $(u,v,d)$ and once in XYZ.

An initially solved point is a geometric sample of foreground observations at $t=0$, with a stable point identity $i$. All valid foreground measurements at $t=0$ must be represented; multi-camera duplicates can be fused if both the operation and source-observation provenance remain explicit. The initial set is $\mathcal S_0$ with cardinality $N_0$.

# Minimal free point state: why exactly six dimensions

For each point $i$ at frame $t$, define

$$
\boxed{s_{it}=\begin{bmatrix}x_{it}\\v_{it}\end{bmatrix}\in\mathbb R^6},\qquad
\boxed{P_{it}\in\mathbb S^6_{+}}.\tag{3}
$$

$x\in\mathbb R^3$ is the world-coordinate position, and $v\in\mathbb R^3$ is its world-coordinate time derivative. There is no intrinsic orientation of an XYZ/RGB point. A rigid part may rotate, yet each constituent point describes a curved position trajectory with no need for its own rotational state. The full 12D Gaussian state $(\mu,R,v,\omega)$ from the original 3DGS IEKF included rotation because an anisotropic Gaussian footprint has orientation; those variables are not observable or necessary in direct point reprojection.

The six-dimensional state is minimal for the *chosen first-order constant-velocity dynamical model*. If one discarded velocity, a memoryless position-only motion model would be different; it would not retain the current temporal-velocity estimation property. No artificial $SE(3)$ motion is imposed on one point.

# Physical-time stochastic motion model

Assume the continuous stochastic model

$$\frac{d}{dt}\begin{bmatrix}x\\v\end{bmatrix}
=\underbrace{\begin{bmatrix}0&I_3\\0&0\end{bmatrix}}_{A}
\begin{bmatrix}x\\v\end{bmatrix}
+\begin{bmatrix}0\\I_3\end{bmatrix}w_a(t),\qquad
\mathbb E[w_a(t)w_a(t')^\top]=Q_c\,\delta(t-t').\tag{4}$$

$Q_c\succeq0$ is the acceleration spectral-density matrix, with consistent units. In addition, an optional independently declared direct position diffusion $Q_x\succeq0$ can be retained from the original implementation. For duration $h=\tau_{t+1}-\tau_t>0$,

$$F(h)=e^{Ah}=\begin{bmatrix}I_3&hI_3\\0&I_3\end{bmatrix}.\tag{5}$$

The *exact discretization of this linear stochastic process* is

$$
\boxed{s^-_{t+1}=F(h)s_t^+},\qquad
\boxed{P^-_{t+1}=F(h)P_t^+F(h)^\top+Q_f(h)}.\tag{6}
$$

To derive the acceleration covariance, compute

$$Q_f(h)=\int_0^h e^{A\lambda}
\begin{bmatrix}0&0\\0&Q_c\end{bmatrix}
 e^{A^\top\lambda}\,d\lambda
=\int_0^h\begin{bmatrix}\lambda^2Q_c&\lambda Q_c\\\lambda Q_c&Q_c\end{bmatrix}d\lambda.$$

Therefore,

$$
\boxed{Q_f(h)=\begin{bmatrix}
\frac{h^3}{3}Q_c+hQ_x&\frac{h^2}{2}Q_c\\
\frac{h^2}{2}Q_c&hQ_c
\end{bmatrix}.}\tag{7}
$$

The off-diagonal covariance blocks are not optional: removing them changes the underlying stochastic model. All $P$ and $Q$ blocks are maintained as complete $6\times6$ matrices, symmetric and numerically positive semidefinite. No per-frame reset of $P$ is permitted.

For frame-0 point $i$, use $s^+_{i0}=(p_i,0)$ and a covariance with $P_{xx,i0}$ derived from source depth/backprojection and $P_{vv,i0}$ expressing unknown initial velocity. The nominal value $v=0$ is a starting estimate, **not** an assumption that points are stationary.

# Direct calibrated RGB-D measurement model

For world position $x$, camera coordinates are $q_c=R_cx+t_c=(X,Y,Z)$. In an ideal pinhole camera,

$$
\boxed{h_c(s)=\begin{bmatrix}f_xX/Z+c_x\\f_yY/Z+c_y\\Z\end{bmatrix},\quad Z>0.}\tag{8}
$$

The observation $z_a=[u,v,d]^\top$ is only admissible for valid foreground pixels. Its residual (observation minus prediction) is $r_a(s)=z_a-h_{c(a)}(s)$. The analytical Jacobian is

$$
\boxed{H_c(s)=
\begin{bmatrix}
 f_x/Z&0&-f_xX/Z^2\\
 0&f_y/Z&-f_yY/Z^2\\
 0&0&1
\end{bmatrix}
R_c\begin{bmatrix}I_3&0_3\end{bmatrix}.}\tag{9}
$$

With calibrated distortion, differentiate the *implemented* distortion-aware projection; the resulting $H$ must match central finite differences. The instantaneous velocity columns are exactly zero: velocity becomes informed through the $P_{xv}$ temporal cross-covariance, not through fabricated instantaneous velocity measurements.

The channel RGB is reserved for *correspondence affinity* and record-level diagnostics. Unless a calibrated differentiable photometric surface model is explicitly built, RGB is **not** an extra continuous-state likelihood factor for a lone XYZ point.

# Association to real measurements and visibility

At each frame, project predicted SOLVED positions into camera rays and compare them with *raw valid foreground* pixels. A nearest-neighbor candidate set is generated in world XYZ using a scalable spatial index; candidate validity also requires projected ray, image bounds and measured-depth visibility compatibility. Pure 3D nearest-neighbor distance cannot distinguish coincident but occluded surfaces.

For one proposed world-XYZ match $i\leftrightarrow a$,

$$\Delta x_{ia}=y_a-\hat x^-_{it},\quad
S^x_{ia}=P^-_{xx,it}+\Sigma_a,\quad
\boxed{d_{ia}^2=\Delta x_{ia}^\top(S^x_{ia})^{-1}\Delta x_{ia}.}\tag{10}$$

Equivalently, for the direct measurement coordinates, use $S^z=HP^-H^\top+R_a$ and $d^2=r^\top(S^z)^{-1}r$. These are **alternative scores** for the same raw observation, not additive independent likelihoods. Under known associations, correctly specified Gaussian measurement noise and the linearization, a chi-square reference with three geometric dimensions is informative; after nearest-neighbor selection and occlusion gating, its exact calibration is not guaranteed. Plot empirical residual distributions and inspect the actual match/unmatch separation. If the user-expected margin is absent, report that openly.

Match assignment must enforce exclusive ownership of raw observation IDs and preserve an unmatched outcome. Predicted-only positions with high uncertainty are not new sensor measurements. Point sampling has finite spacing, so nearest-point coverage must be audited against sampling density; do not introduce an invented alpha/coverage renderer.

# Euclidean iterated Kalman measurement update

Given prediction $\bar s=s_t^-$, covariance $P^-$, and a fixed accepted correspondence set $\mathcal A$, minimize the conditional nonlinear negative log posterior

$$
\boxed{E(s)=\frac12(s-\bar s)^\top(P^-)^{-1}(s-\bar s)
+\frac12\sum_{a\in\mathcal A}r_a(s)^\top R_a^{-1}r_a(s).}\tag{11}
$$

At iteration $k$ with current nominal $s^{(k)}$, linearize $h(s^{(k)}+\delta)\simeq h(s^{(k)})+H^{(k)}\delta$. The quadratic normal system is

$$\boxed{
\Lambda^{(k)}\delta=b^{(k)},\quad
\Lambda^{(k)}=(P^-)^{-1}+\sum_a H_a^\top R_a^{-1}H_a,
}\tag{12}$$

$$
\boxed{b^{(k)}=\sum_a H_a^\top R_a^{-1}r_a(s^{(k)})-(P^-)^{-1}(s^{(k)}-\bar s).}\tag{13}
$$

Use $6\times6$ Cholesky/LDLT with numerical diagnostics, update $s^{(k+1)}=s^{(k)}+\alpha\delta$, and accept only a trial which decreases the *actual* $E$. Damping or trust region is an algorithmic stabilization, not an additional physical measurement. **Prediction and prior stay fixed across all inner iterations.** The posterior covariance is approximated at the accepted solution $s^*$ as

$$\boxed{P^+\simeq\left[(P^-)^{-1}+\sum_a H_a(s^*)^\top R_a^{-1}H_a(s^*)\right]^{-1}.}\tag{14}$$

This is a final-linearization Laplace/Gauss--Newton covariance, not an exact posterior for nonlinear projection or unknown correspondences. With no matched observation, set $s^+=s^-$ and $P^+=P^-$. Since the state is Euclidean and updates are additive, its covariance chart does not change: **no Lie-group Jacobian reset**.

# Optional separate six-dimensional UKF

If a UKF is provided, it is an *alternative pipeline* with exactly the same 6D point state, same association semantics and same event-driven discovery controller. For $n=6$ and declared $\alpha,\beta,\kappa$, set $\lambda=\alpha^2(n+\kappa)-n$ and sigma points

$$\chi_0=\bar s,\qquad
\chi_{j,\pm}=\bar s\pm\big[\sqrt{(n+\lambda)P^-}\big]_j,\quad j=1,\dots,6.\tag{15}$$

The 13 sigma points are sent through the nonlinear stacked measurement function over all accepted camera correspondences; use prescribed unscented weights $W^m,W^c$:

$$\bar z=\sum_kW_k^m h(\chi_k),\quad
S_z=\sum_kW_k^c(h(\chi_k)-\bar z)(\cdot)^\top+R,$$

$$P_{sz}=\sum_kW_k^c(\chi_k-\bar s)(h(\chi_k)-\bar z)^\top,
\quad K=P_{sz}S_z^{-1},$$

$$s^+=\bar s+K(z-\bar z),\quad P^+=P^--KS_zK^\top.\tag{16}$$

Use stable covariance factorization, actual assembled measurement noise (including correct camera independence assumptions), and avoid double-counting per camera. For the linear process equation, the usual $F P F^\top+Q_f$ is already exact; no process sigma-point overhead is needed. UKF does not become an IEKS/smoother, does not mix with IEKF iterations, and uses the same bidirectional branch initialization and discovery semantics.

# Reverse-oriented stochastic factor for discovery

A point discovered at frame $d$ is anchored to the *real measured point* $y_{a_d}$ and a neutral velocity prior. To estimate its missing trajectory toward earlier frames, define the backward transition by the inverse of the forward linear model, *without changing the physical meaning/sign of velocity*:

$$\boxed{F_r(h)=F(h)^{-1}=\begin{bmatrix}I_3&-h I_3\\0&I_3\end{bmatrix}.}\tag{17}$$

Consider the forward process residual on an edge $(t,t+1)$,

$$e_f=s_{t+1}-F(h)s_t,\quad e_f\sim\mathcal N(0,Q_f).$$

The algebraically equivalent backward residual is

$$e_r=s_t-F_r(h)s_{t+1}=-F_r(h)e_f.$$

Consequently,

$$
\boxed{Q_r(h)=F_r(h)Q_f(h)F_r(h)^\top
=\begin{bmatrix}
\frac{h^3}{3}Q_c+hQ_x&-\frac{h^2}{2}Q_c\\
-\frac{h^2}{2}Q_c&hQ_c
\end{bmatrix}.}\tag{18}
$$

The two quadratic edge penalties are exactly equal:

$$\boxed{e_f^\top Q_f^{-1}e_f=e_r^\top Q_r^{-1}e_r.}\tag{19}$$

The proof follows from $e_r=-F_r e_f$ and $(F_r Q_f F_r^\top)^{-1}=F_r^{-\top}Q_f^{-1}F_r^{-1}$. If $Q_f$ is singular, state and test this identity on its supported subspace using appropriate generalized inverses, or configure strictly positive-definite discretized process covariance.

**Scope of the claim:** Equations (17)--(19) specify an equivalent *reoriented process factor*. They do **not** assert that $P(s_t\mid s_{t+1},\mathcal Y_{0:t})=Q_r$ for an arbitrary initial prior or after past observations. In this implementation, new-point backward and forward branches are both anchored at the same discovery posterior; each assimilates the observations on *its own* temporal side only. No exact global RTS smoother is claimed.

# Exact new-point event: its discovery observation is an anchor, not a trajectory

Let $a_d=(d,c,u)$ be the first unmatched real valid foreground observation found by the reverse scan. Allocate fresh `point_id=j` and world measured location $x_{jd}=y_{a_d}$. Initialize the six-dimensional anchor posterior

$$s_{jd}^{\mathrm{anchor}}=(y_{a_d},0),\quad
P_{jd}^{\mathrm{anchor}}=\operatorname{diag}(\Sigma_{a_d},P_{vv,0}).\tag{20}$$

The position block is already informed by the anchor observation, and therefore **the same anchor likelihood must not be applied again**. If multiple genuinely distinct camera observations are fused at this frame, each raw ID is used exactly once. Record all anchor measurements and provenance.

The two runs are:

$$\boxed{\text{Backward: } d\to d-1\to\cdots\to0},$$

$$\boxed{\text{Forward: } d\to d+1\to\cdots\to T}.\tag{21}$$

Both branches start from *independent copies of the exact same anchor posterior*. Earlier frames apply $(F_r,Q_r)$ and their real available RGB-D measurements; later frames apply $(F,Q_f)$ and real measurements. Unobserved frames perform prediction-only propagation. Combine by concatenating backward results for $t<d$, exactly one anchor result for $t=d$, and forward results for $t>d$. This is the user's requested **full temporal reconstruction through two anchored filtering branches**, not a batch smoother and not an average of the two estimates.

The newly found point is now **SOLVED** after its actual optimization, even if some temporal states remain uncertain predictions. Its initial discovery point is not a demand for a second confirmation stage; it is the geometric condition from which the motion is inferred.

# One large initial optimization, then no old-point changes

Define $\mathcal S_0$ as all initial frame-0 points and $\mathcal X_i^{(0)}=\{s_{it},P_{it}\}_{t=0}^T$ their initially optimized forward tracks. For any discovery iteration $r\ge0$ and every already SOLVED point $i\in\mathcal S_r$,

$$\boxed{\mathcal X_i^{(r+1)}=\mathcal X_i^{(r)}\quad\text{bitwise (unchanged)}.}\tag{22}$$

For each new seed $j\notin\mathcal S_r$, create only its own $\mathcal X_j$ using equation (21), then append

$$\boxed{\mathcal S_{r+1}=\mathcal S_r\cup\{j\}}.\tag{23}$$

No old Kalman update or old covariance recalculation is allowed. New points change the observation matching/coverage bookkeeping and spatial indices; this is separate from re-estimating old motion.

The discovery controller takes the first unmatched foreground raw observation in deterministic order $t=T,T-1,\dots,0$, then camera index and pixel index. After each successfully optimized new point, it **restarts the scan at T** against the expanded frozen map; this is not an optimization rerun. Every success commits its source raw observation ID to ownership, so under a finite observation catalogue and correct transactional ID handling, at most finitely many distinct successful discovery seeds exist. Failures require a model-version-specific attempted registry to prevent infinite identical retries.

# Occlusion, missing parts and the limits of inference

An entirely unseen-at-$t=0$ moving part is admissible. Its first unmatched later observation is a valid point anchor, and the filter estimates that point independently in both temporal directions without any known part motion. Earlier or later frames where it remains invisible yield *predicted* states, with growing covariance. The method cannot infer a uniquely determined physical trajectory where data never constrain velocity or position; **the honest output is an uncertain prediction, not invented evidence**.

A novel seed might geometrically coincide with a previously solved point due to correspondence noise or finite sampling. Exclusively owned source observations, camera-ray compatibility and measured nearest-residual separation control duplication; no guarantee of perfect novel-surface detection can be derived from a 3D proximity margin without assumptions on noise, calibration and sample spacing. The algorithm still obeys the user's one-seed-one-bidirectional-fit schedule rather than inventing extra group hypotheses.

# Complexity and information bookkeeping

With $N_0$ initial points, $T+1$ frames and $M$ subsequently discovered point seeds, the number of **state-filter passes** is exactly one vectorized initial forward run plus $M$ separate two-sided per-point runs. Their filtering arithmetic scales as $O((N_0+M)T)$ in state dimension for fixed correspondence count; the full reverse rescan of all foreground observations and kNN indexes adds additional matching cost. There is no global $6(N_0+M)$ dense solve, and no all-frame Gaussian rasterizer.

Active state/covariance storage in float32 uses $6+36=42$ floats per point, 168 bytes per active point, excluding workspaces and metadata. Time histories require chunked/file-backed arrays for large $N_0T$. Storing old tracks once avoids repeatedly optimizing or rewriting them. Only new point tracks are appended. For $M$ new points, a strict one-at-a-time, rescan-from-T procedure can still be expensive in matching operations; correctness takes precedence for this first implementation, and caches can accelerate scans only with equivalence tests.

The Gaussian covariance approximation after selecting correspondences excludes discrete matching uncertainty and correlations introduced by common camera calibration or shared observation modeling. Label this explicitly. NIS distributions, observed-only residuals and covariance diagnostics are indispensable to reveal failure modes rather than overclaim exactness.

# Falsifiable consistency and convergence statements

**Exactly defined invariants:**

1. Every real foreground frame-0 observation contributes to the initial solved map (up to explicitly documented duplicate fusion).
2. Every newly solved point has one real discovery observation and one 6D two-sided optimization anchored there.
3. Raw observations have unique identities and are assimilated at most once into any point's trajectory at each branch/time.
4. No existing solved state/covariance is changed during discovery; test bitwise equality.
5. A processed new point is appended immediately, and the unmatched scan restarts at the final frame.
6. The covariance increases correctly under prediction-only dynamics, and no per-frame covariance initialization/reset occurs.
7. There is no moving-part count $K$, group motion assignment, Gaussian rotation or Lie reset in this point stage.
8. Every numerical optimization event produces inspectable numerical and actual-data visual diagnostics.

**What is not proved:** correctness of unknown point correspondences under arbitrary occlusion; recovery of unobserved velocity from a single point measurement; global convergence of the nonlinear IEKF; optimality of incremental greedy discovery; equivalence of two-sided anchored filtering to global smoothing; discovery of all hidden surfaces without sufficient RGB-D observations.

# Reference mapping and relation to previous theory

- Source implementation: `RigEuler3DGS/estimator/process.py`, `covariance.py`, `sparse_iekf.py`, `masked_observation.py`, `tools/run_iekf_scene.py`. Reuse only the Euclidean translational filtering principles and camera/mask discipline; the prior Phase-1 runner resets covariance and imposes supernodes, both forbidden here.
- Earlier reference: *Parallel Masked RGB-D IEKF/UKF Fixed-K Theory/CUDA v3* (21 October 2026). The current revision **removes fixed-K part-clustering, group poses in the point state, 12D Lie pose variables, group-weighted motion and global IEKS as the discovery engine**, retaining direct RGB-D, uncertainty and sparse-GPU derivations.
- Finn G. Maurer, Erlend A. Basso, Henrik M. Schmidt-Didlaukies, Torleiv H. Bryne, *Equivalence of Left- and Right-Invariant Extended Kalman Filters on Matrix Lie Groups*, arXiv:2506.01514v1, 2 June 2025. Its correct full-order covariance-reset result concerns Lie-group pose-state filters. In the new Euclidean 6D point IEKF, a Lie-group reset is not applicable.

**Normative companion:** `CODEX_EXACT_POINTWISE_BIDIRECTIONAL_DISCOVERY_2026-10-22.md` contains the executable operational interface, exact schedule, mandatory debug artifacts, source safety, and acceptance tests.
