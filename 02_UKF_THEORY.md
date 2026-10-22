# FRAMEWORK B — Independent Lie-Group Unscented Kalman Filter and Unscented Smoother for RGB-D SLAM

**MANDATORY FOREGROUND-MASK ADDENDUM:** The binary foreground masks are a hard admissible-measurement and predicted-visible-point feasibility constraint. Read and implement `CODEX_PARALLEL_IEKF_UKF_HARD_FOREGROUND_MASKS.md` before using this document. No background pixel can initialize SOLVED/UNSOLVED, be matched, enter an IEKF/UKF measurement, or be fused into geometry. Occluded solved points remain valid. Identical policy in both pipelines.


**Independent theory.** Use precisely the solved/unsolved outer algorithm defined in `00_COMMON_SOLVED_UNSOLVED_SPEC.md`. In this version the inner motion estimator is **only an augmented Lie-group UKF**; the backward temporal pass is **only an unscented RTS smoother**. There is NO IEKF fallback, analytical measurement-Jacobian step, or hybrid selector.

## B1. Same observations, state and geometry as Framework A

At each time `k`, with `B` currently supported rigid groups, use

\[
s_k=\{(T_{g,k},\nu_{g,k})\}_{g=1}^B,\quad T_g\in SE(3),\quad \nu_g\in\mathbb R^6,\quad \dim(\delta s_k)=n=12B.
\]

The canonical map `SOLVED` consists of persistent XYZ/RGB point samples and their rigid-group ownership. `UNSOLVED[k]` holds raw observation IDs not currently explained. They are NOT components of the UKF state. All transformations may be identity; there is no special static label or separate static-motion estimator.

Use the same translation-first tangent chart as the IEKF derivation:

\[
s\boxplus(\delta\xi,\delta\nu)=(\operatorname{Exp}(\widehat{\delta\xi})T,\nu+\delta\nu),\quad
s_1\boxminus s_2=(\operatorname{Log}(T_1 T_2^{-1})^\vee,\nu_1-\nu_2).
\]

The state output for a matched raw RGB-D point observation is identical to Framework A:

\[
z_a=(u_x,u_y,D_{c,k}(u),I_{c,k}(u)^\top)^\top,
\]

\[
h_a(s_k,p_i)=(\pi_c(X_a)^\top,[X_a]_z,a_i^\top)^\top,\quad
X_a=T_{cw,c}T_{g,k}p_i.
\]

Raw localization/depth/RGB observation covariance `R_a` and association rules are **exactly the same**. A raw RGB residual informs correspondence and shared appearance, but fixed point color has no local derivative with respect to pose; the UKF likewise cannot infer pose from a state-independent RGB output. To use direct photometric pose evidence, substitute the explicitly specified warped-image output `h^photo=I_k(pi(X))`, and propagate shared raw-pixel measurement covariance; never count the same RGB sample twice.

## B2. Dynamic equation: nonlinear constant-twist process

The physical dynamic model is IDENTICAL to Framework A:

\[
\boxed{\begin{aligned}
T_{g,k+1}&=\operatorname{Exp}(\Delta t_k\widehat\nu_{g,k})T_{g,k}\operatorname{Exp}(\widehat{w^T_{g,k}}),\\
\nu_{g,k+1}&=\nu_{g,k}+w^\nu_{g,k}.
\end{aligned}}
\]

`w=(wT,wnu)` is an independent process-noise vector in the declared right-pose/linear-twist noise chart, with covariance `Q_k`, discretized from the same approved physical assumptions and actual `dt_k` as Framework A. **No analytical state-transition Jacobian is used for UKF prediction.**

### Augmented sigma-point prediction

Let the state error dimension be `n=12B`, process-noise dimension `q=12B` for the stated process. Construct an augmented zero-mean error `eta=(delta s,w)` with covariance

\[
P_{aug}=\operatorname{diag}(P_{k|k},Q_k)\in\mathbb R^{(n+q)\times(n+q)}.
\]

Choose and log valid sigma-point parameters `(alpha,beta,kappa)` with `n_a=n+q`, `lambda=alpha^2(n_a+kappa)-n_a`, `n_a+lambda>0`. Generate `2 n_a+1` deterministic sigma points with exact Cholesky (or PSD eigendecomposition if necessary), using

\[
\eta_0=0,\quad\eta_j=+\sqrt{(n_a+\lambda)P_{aug}}_j,\quad
\eta_{j+n_a}=-\sqrt{(n_a+\lambda)P_{aug}}_j.
\]

Weights: `wm0=lambda/(n_a+lambda)`, `wc0=wm0+1-alpha^2+beta`, other weights `1/[2(n_a+lambda)]`. Retraction maps each state perturbation onto `SE(3)`; apply each process-noise perturbation through the **nonlinear** dynamic equation, producing `s_{k+1}^{(j)}`. Compute their weighted Lie/Karcher mean using consistent left logarithms and reconstruct

\[
P_{k+1|k}=\sum_j w_j^c\,\delta s_{k+1}^{(j)}\delta s_{k+1}^{(j)\top},\quad
\delta s_{k+1}^{(j)}=s_{k+1}^{(j)}\boxminus\bar s_{k+1}.
\]

If a sigma-point weight choice produces a numerically indefinite covariance in a strongly nonlinear regime, report the failure and use a documented sigma scheme with positive covariance weights. Do not silently clip large negative eigenvalues and call them calibrated uncertainty. A stable explicit option is `alpha=1`, `kappa=0`, `beta=2`, giving `lambda=0`, `wm0=0`, `wc0=2`, and all non-central weights positive. Numerical sigma parameters govern the deterministic cubature approximation, **not motion/outlier thresholds**.

### Cross-covariance for backward smoothing

From the **same augmented prediction sigma set**, save

\[
P_{k,k+1}=\sum_j w_j^c (s_k^{(j)}\boxminus \hat s_{k|k})(s_{k+1}^{(j)}\boxminus\hat s_{k+1|k})^\top.
\]

Here `s_k^(j)` is the state component of the augmented sigma point (excluding independent noise), and cross-covariance includes the exact sigma-based nonlinear mapping, not `P F^T` borrowed from EKF.

## B3. Nonlinear matched-point UKF correction without a huge measurement covariance

The outer matcher fixes a set `E_k` of raw RGB-D associations and visibility hypotheses for one conditional update. **Generate fresh state sigma points from the predicted `(s^-,P^-)`** (dimension `n`, count `L=2n+1`), using the same manifold chart and declared positive covariance weights.

Evaluate every conditional matched-point output `h_a(s^(j),p_i)` at each sigma point. If the same match becomes invalid under significant sigma-point variation (occlusion, leaving camera bounds, ordering switch), do not pretend the conditional smooth measurement model remains valid: defer that factor to the outer association/visibility calculation or explicitly model the visibility event probabilistically. A discontinuous hard z-buffer is NOT a smooth continuous-valued UKF output.

For active observation `a`, let `hbar_a=sum_j wm_j h_a(s^(j))`, `e_a=z_a-hbar_a` and sigma-centered columns

\[
Z_a[:,j]=\sqrt{w_j^c}\left(h_a(s^{(j)})-\bar h_a\right)\in\mathbb R^{d_a},\quad
A[:,j]=\sqrt{w_j^c}\left(s^{(j)}\boxminus\bar s\right)\in\mathbb R^n.
\]

Because the covariance weights are chosen nonnegative, the `sqrt(wc)` representation is real. Concatenate `Z_a` vertically into an `m x L` operator `Z` (NEVER materialize a huge persistent m-by-L tensor) and residual `e` into `m`. For the **joint batch** nonlinear unscented approximation:

\[
S=R+ZZ^\top,\quad C_{sz}=AZ^\top,\quad
\Delta s=C_{sz}S^{-1}e,\quad
P^+=P^--C_{sz}S^{-1}C_{sz}^\top.
\]

### Exact Woodbury computation for the UKF's sigma-defined covariance

With positive definite declared observation covariance `R`, accumulate **streamingly**:

\[
\boxed{M=I_L+\sum_a Z_a^\top R_a^{-1}Z_a,\qquad
b=\sum_a Z_a^\top R_a^{-1}e_a.}
\]

These block sums assume the selected raw measurement factors are independent *conditional on the state/map*. For overlapping interpolated pixels or known correlated depth errors use a sparse/shared raw-noise `R^{-1}` operator instead of the block sums. No covariance may be silently treated as diagonal when raw observations are reused.

By Woodbury:

\[
\boxed{\Delta s=A M^{-1}b,\qquad
P^+=P^- - A\big(I_L-M^{-1}\big)A^\top.}
\]

These equations are **algebraically exact for the chosen UKF sigma-point approximation**. They do NOT form the `m x m` innovation matrix `S`, any `m x m` covariance or an `m`-dimensional dense inverse. Solve the small `L x L` matrix `M`; for one group `n=12`, `L=25`, even if hundreds of thousands of matched raw observations are processed. With multiple coupled groups `n=12B` and `L=24B+1`. Use the same conditional observation likelihood as Framework A and check the accepted update against that nonlinear score; damping/line-search may be used for numerical acceptance but **is not an IEKF derivative update**.

The UKF posterior covariance is the sigma-based conditional approximation. A decrease in the objective does not imply this Gaussian covariance is exact for unknown data association or uncertain map points. Shared point-source pose correlations must be represented through retained sparse map-motion information factors or clearly distinguished conditional covariance outputs.

## B4. Unscented backward temporal smoother, not EKF RTS

For each `k=K-1...0`, use the UKF's *saved sigma-point cross-covariance* `P_{k,k+1}` and the already computed predicted covariance `P^-_{k+1}`:

\[
\boxed{G_k^{UKF}=P_{k,k+1}(P^-_{k+1})^{-1}.}
\]

Then smooth in the selected Lie tangent chart:

\[
\hat s_{k|K}=\hat s_{k|k}\boxplus\left[G_k^{UKF}\left(\hat s_{k+1|K}\boxminus\hat s_{k+1|k}\right)\right],
\]

\[
P_{k|K}=P_{k|k}+G_k^{UKF}\big(P_{k+1|K}-P_{k+1|k}\big)(G_k^{UKF})^\top.
\]

Transport covariance when changing the retraction center. This is a sigma-point approximation to nonlinear smoothing, not the analytically linearized RTS update. Do not run an unrelated independent reverse UKF and average poses. All later-observation corrections must act on the SAME shared group trajectories.

## B5. Exactly the shared solved/unsolved outer algorithm

Execute `00_COMMON_SOLVED_UNSOLVED_SPEC.md` verbatim:

1. Fused frame-0 points initialize `SOLVED`. All other raw foreground observations remain `UNSOLVED`.
2. For `k=1..K`, UKF-**predict** the low-dimensional shared group states; transport solved points; sparse RGB-D match; UKF-**correct** conditionally; record unassociated observations.
3. Unscented backward RTS smooth the complete shared trajectory using `P_{k,k+1}` computed from augmented sigma points.
4. Late-to-early discovery from unsolved raw observations: transport under `T[g,k]T[g,s]^-1`, compare raw depth ordering/RGB, reassign to an existing group or propose a new rigid group whose motion is inferred by the **same UKF method**.
5. Fuse accepted cross-frame point matches into one canonical map, without cloning frame/camera depth pixels.
6. Progressive frontier `s=1..K`: resolve unsolved observations against `0..s-1`, revise affected trajectories with UKF prediction/correction and unscented smoothing.
7. Repeat accepted passes under the unchanged common full-sequence score until the outer loop has no further accepted improvement.

There is **no IEKF in this project**. Group identity trajectories are simply valid UKF estimates near `I`, not a different solver.

## B6. Convergence — what is and is not proved

The UKF sigma transform approximates the nonlinear predicted mean/covariance. The resulting update is **not generally a Gauss-Newton descent direction**, and a standard UKF correction need not monotonically decrease a nonlinear MAP objective. Therefore **there is no honest blanket theorem that repeated UKF measurement updates converge to a MAP stationary point**.

A deterministic proposal acceptance rule against the same proper conditional negative log posterior as Framework A can enforce nonincrease of the accepted score. The outer loop accepts every map/group/association edit only after its full comparable sequence score improves, so the sequence of accepted scalar scores is monotone and converges if bounded below. This does **not** prove UKF posterior convergence, global association recovery, or stationary points unless additional descent and regularity hypotheses are independently verified. This limitation must be stated in every output/theory report.

The sigma-point sampling rule is deterministic, not exact integration for arbitrary nonlinear functions. Its algebraic Woodbury computation above is exact **for those sigma-point moments** and does not change the approximation class of the UKF.

## B7. Complexity, verification, outputs

One-group correction uses only `L=25` sigma points and a `25x25` low-rank innovation solve. Augmented 12-state/12-noise prediction uses `49` sigma points. For `B` jointly coupled groups, dimensions scale with `12B` and `24B+1`; state-space size is independent of number of depth pixels. GPU uses shared point transport and sparse matching, then batched sigma-point projections and reductions into `LxL` matrices. Retain `O(N_solved + current_frame_raw + K(12B)^2 + sparse_map_factors)` persistent memory.

**Mandatory tests:** sigma points reproduce a known mean/covariance; zero-noise constant twist agrees with `SE(3)` propagation; UKF vs exact linear Gaussian Kalman update; low-rank Woodbury correction and covariance vs explicitly formed small `S`; matching and visibility tests; correlated-pixel observation noise regression; unscented RTS vs linear-Gaussian RTS on linear fixtures; late-observed geometry back-transports and updates the same shared trajectory; multiple rigid groups split by multi-frame evidence; full forward/backward/frontier run; memory does not scale quadratically in number of measured pixels.

**Deliverable:** Independent stand-alone UKF inner estimator, independent unscented temporal smoother, and the identical solved/unsolved outer SLAM logic. All outputs under `results_ukf/` labeled **UKF** only.
