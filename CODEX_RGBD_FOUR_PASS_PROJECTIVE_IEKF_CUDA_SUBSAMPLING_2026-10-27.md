# CODEX — AUTHORITATIVE RGB-D-ONLY BIDIRECTIONAL FREE → RIGID IEKF IMPLEMENTATION

**Revision:** 2026-10-27 — replaces and corrects `CLAUDE_CORRECT_RGBD_PROJECTIVE_IEKF_THEORY_AND_REBUILD_2026-10-27.md` and *supersedes* all prior splat/image-energy/point-association implementation contracts for the new production path.

**Repository to modify:** `/home/samuelemara/Joint3DGS/RGBDekf` / [SamueleMara/RGBDiekf](https://github.com/SamueleMara/RGBDiekf). The last source inspected for this contract is commit `34131bcdcedbbd02f6118f1a766fd8394bed3b3e`. **Inspect the actual current local HEAD and worktree before making edits; do not assume that a later checkout is identical.**

**Reference for *organization only*:** [RigEuler3DGS](https://github.com/SamueleMara/RigEuler3DGS) at `481b2cf2d5602932286948cf14114f9caee4de3c`. Reuse its sequence of stages, coherent-motion reasoning, SE(3) mathematical conventions where relevant, separate computational and coherence supernodes, and visual/numerical debugging patterns. **Do NOT port its Gaussian rasterization, alpha compositing, normal-operator image optimizer, Gaussian states, loss function, or posterior approximations merely because they are in the reference.**

## 0. The user's exact algorithm, stated in one place

**Input:** synchronized, calibrated, masked **RGB-D images** from all available cameras at frames `0 ... T` (e.g. eight cameras, frames 0–59). Form or load one genuine measured **colored XYZ point cloud at frame 0** and one independently measured **colored XYZ point cloud at frame T**. Keep intrinsics, metric depth scale, extrinsics and real acquisition timestamps. The foreground mask is a hard inclusion constraint.

**Run exactly these four independent directional filters:**

| Pass | Canonical point cloud | Processing order | State being filtered | Measurement |
|---|---|---|---|---|
| F1 — free forward | measured points from frame 0 | `0 → 1 → … → T` | one independent `XYZ + velocity`, 6D state/covariance **per original point** | direct projection of transported points into *actual* next-frame RGB-D images |
| B1 — free backward | measured points from frame T | `T → T−1 → … → 0` | distinct 6D states/covariances for **different** endpoint points | same direct measured RGB-D sampling |
| F2 — rigid forward | frame-0 moving core selected from completed F1 trajectories | `0 → 1 → … → T` | one *shared* SE(3) rigid pose + spatial twist (12D tangent/covariance) | same direct RGB-D measurement, chained through rigid-motion Jacobian |
| B2 — rigid backward | frame-T moving core selected from completed B1 trajectories | `T → T−1 → … → 0` | a separate 12D rigid pose/twist state, anchored in frame T | same direct RGB-D measurement |

**Between F1/B1 and F2/B2:** independently discover coherent rigid moving cores from the completed free trajectories, using RigEuler's fixed-canonical representative graph. **After F2/B2:** extract/validate a time-invariant revolute/prismatic joint axis and time-varying coordinate `q(t)`; subsequent constrained joint filtering and point-support completion are separately gated downstream steps, not substitutes for the four filter runs.

**Non-negotiable exclusions:**

- **NO frame-to-frame point correspondences**: no matching frame-0 points to frame-1 point IDs, no KNN, ICP, point-to-point or point-to-plane surface assignments, feature matching, optical flow, or association/controller layer in Phase 1 or the rigid passes. Any eventual comparison between endpoint clouds comes *after* both transports and physical inference; it must not be an input to the filters.
- **NO splatting or synthesized RGB-D images**: no compact splats, footprint distributions, normalized color/depth contributions, opacity, alpha, visibility taper weights, background renderer, accumulated coverage field or differentiable RGB-D rasterizer. You may **project points into actual cameras** and use an ordinary depth-buffer index solely to determine whether a projected point is geometrically hidden.
- **NO image-energy optimization**: remove the current Student-t/coverage-weighted rendered-image energy, IRLS, normal-equation PCG/trust-region solver and scene-label Gibbs image-energy optimizer **from the new production path**. Standard IEKF correction is computed from sensor innovations, Jacobians, `R`, Kalman gain and covariance, not from an unrelated minimizer.
- **NO physical rigidity in free Phase 1**: points remain independent 6D Kalman states. Supernodes may index/cache/batch GPU computation but **must not constrain their state increments** to a common 12D rigid-group tangent. Only F2/B2 impose rigid movement after coherence.
- **NO Gaussians**, meshes, surfels, normals, learned 3D representations, invented points at unseen surfaces, GT motions/joints/labels in inference, nor setting a target moving fraction.
- **No invented motion guarantees:** flat/textureless/occluded geometry may be locally unobservable. Preserve growing uncertainty and prediction-only states rather than introducing fake constraints or pretending to track perfectly.

**Vocabulary rule:** call the operation **projective image sampling** or **RGB-D innovation**. It is *not a correspondence search*. Pixel coordinates derive from the **current predicted 3D state**, and the sampled RGB-D values are measured fields. The 3D back-projection is a geometric check along that predicted ray, **not an independently matched 3D point**.

---

## 1. Frame ownership, coordinates and endpoint initialization

Use one world frame shared by all calibrated cameras. For camera `c`, let

\[
X_c=R_{cw}\,x_w+t_{cw},\qquad T_{cw}=\begin{bmatrix}R_{cw}&t_{cw}\\0&1\end{bmatrix}.
\]

Calibration convention: `T_cw` **world→camera**. If the source file stores `T_wc`, invert *once during loading* and persist both matrices. For camera intrinsic matrix `K_c`, no undocumented resizing, axis flip, transposition or millimetre→metre conversion. Distorted images require either calibrated rectification applied consistently to RGB, depth and masks or the correct distortion projection and Jacobian; state which and test it.

At **frame 0**, construct canonical `P_0 = {(p_i^0, color_i^0, original_source_id_i)}` using **only valid, foreground-masked real measured depth samples**, calibrated back-projection, and immutable source provenance `(endpoint, camera, pixel, depth)`. Do the analogous construction at **frame T** as a *different* canonical `P_T`; their sizes and point IDs can differ. If the input already provides a measured canonical endpoint cloud with validated metadata, reuse it rather than resynthesizing it.

Do **not** union/merge different cameras using an implicit cross-frame correspondence algorithm. Within an endpoint, each source pixel that yields a valid measured point can retain its own point identity; duplicate multiview samples may be flagged geometrically for density/measurement reuse control but do not fabricate a point identity across time.

The forward and backward systems each keep a stable point ID from their own endpoint for all processed frames. The order stored on disk is **original chronological indexing**; processing order is a separate manifest field. A backward result `states[0]` means *frame 0 estimates of frame-T canonical points*, not the frame-0 canonical cloud.

Record at minimum: `canonical_frame`, `canonical_xyz`, `canonical_rgb`, `original_source_ids`, `point_id`, `processing_order`, `original_frame_indices`, `timestamps_seconds`, `camera_geometry_hash`, `mask_hash`, `subsample_manifest`, `subsampled_to_full_index` and actual selected point count.

### 1.1 Back-projection is measured geometry, not matching

Let the predicted position project to measured image coordinates `(u,v)` in camera `c`; define

\[
 b(u,v)=K_c^{-1}[u,v,1]^\top,\quad
 p_c^{\mathrm{meas}}(u,v)=D_{c,k}(u,v)\,b(u,v),
\]
\[
 x_w^{\mathrm{meas}}(u,v)=R_{cw}^{\top}\big(p_c^{\mathrm{meas}}(u,v)-t_{cw}\big).
\]

For the *same predicted ray*, `X_c = z_pred b(u,v)`, so

\[
 \boxed{\ x_w^{\mathrm{meas}}(u,v)-x_w^{\mathrm{pred}}
   =R_{cw}^{\top}b(u,v)\,[D_{c,k}(u,v)-z_{\mathrm{pred}}].\ }
\]

This identity is decisive: measured back-projected XYZ minus projected/predicted XYZ is **rank-one ray-depth information**, not three independent Cartesian measurements. **Implement back-projection and show the equality in debug/tests**, but do not add its three coordinates and the same scalar depth measurement as four independent geometric channels. That double-counts depth and spuriously constrains tangential motion.

---

## 2. Direct measured RGB-D observation model — exact local equations

For every currently estimated point `x_i`, each camera `c` and actual observation time `k`, project

\[
X_{ic}=R_{cw}x_i+t_{cw}=(X,Y,Z)^\top,
\quad u=f_x X/Z+c_x,\quad v=f_y Y/Z+c_y.
\]

The projection derivatives are

\[
J_{\pi}=
\begin{bmatrix}
 f_x/Z &0&-f_xX/Z^2\\
 0&f_y/Z&-f_yY/Z^2
\end{bmatrix},\quad
J_u=J_{\pi}R_{cw},\quad J_Z=e_3^\top R_{cw}.
\]

**Read the measured images at those coordinates**, via GPU-validity-aware bilinear sampling:

\[
 I^{\mathrm{meas}}=I_{c,k}(u,v),\quad D^{\mathrm{meas}}=D_{c,k}(u,v).
\]

Original point RGB `a_i` is **fixed** and used as a photometric reference; it is never trained. Define **zero target measurement** and the implicit projective observation function

\[
\boxed{h_{ic}(s_i)=
\begin{bmatrix}
 I_{c,k}(\pi(T_{cw}x_i))-a_i \\[2pt]
 D_{c,k}(\pi(T_{cw}x_i))-Z_{ic}
\end{bmatrix}\in\mathbb R^4,\qquad z_{ic}^{\mathrm{target}}=0.}
\]

At IEKF iterate `s_i^(j)`, the actual innovation is

\[
\boxed{r_{ic,j}=z^{\mathrm{target}}-h_{ic}(s_i^{(j)})
=\begin{bmatrix}
 a_i-I_{c,k}(u_{ic,j},v_{ic,j})\\
 Z_{ic,j}-D_{c,k}(u_{ic,j},v_{ic,j})
\end{bmatrix}.}
\]

Here `I` and `D` refer to the **measured frame**, never a rendered output. The RGB residual has three channels; the depth residual one scalar. No residual to a selected frame-1 3D point is formed.

Let `∇I(u,v)∈R^(3×2)` and `∇D(u,v)∈R^(1×2)` be derivatives of the **measured, validly interpolated image fields**. Then

\[
\boxed{H_{ic,j}=\frac{\partial h_{ic}}{\partial s_i}
=\begin{bmatrix}
 \nabla I(u,v)J_{\pi}R_{cw}&0_{3\times3}\\
 \nabla D(u,v)J_{\pi}R_{cw}-e_3^\top R_{cw}&0_{1\times3}
\end{bmatrix}\in\mathbb R^{4\times6}.}
\]

**Test the sign** against the defined `r=−h`; changing the sign of `H` without changing the innovation invalidates the filter. Derivatives at discontinuous depth, mask edges or occlusion changes do not exist in the required smooth sense: mark those samples invalid/ambiguous, never differentiate across foreground/background or zero-depth holes.

### 2.1 Measurement covariance and absence of invented certainty

For each valid point/view, define a positive definite physically justified covariance `R_ic` in `(RGB^3,depth)` units. The **default** may be a calibrated diagonal `diag(σ_R²,σ_G²,σ_B²,σ_D(Z)²)` for speed; annotate calibrated-versus-assumed noise and RGB appearance/model mismatch. Optional color-to-depth correlated noise or calibration covariance must be handled explicitly or documented as an approximation. No Student-t IRLS, no normalized splat coverage factor, no appearance model learning.

Depth units are metres. If bilinear interpolation uses raw noisy image samples, consider variance of interpolation as `Σ w_j² σ_j²` plus any correlated/acquisition model term; **do not interpolate invalid neighbours as zero**. If including image-coordinate uncertainty, use the correct propagation through sampled gradients, e.g. `R_eff=R_sensor+[∇I;∇D]Σ_uv[∇I;∇D]^T`, and account for resulting channel correlations. Keep the baseline deterministic and verifiable.

Observability: depth typically constrains the surface-normal/ray-depth component; RGB gradients can constrain tangential components if they exist. Blank texture, depth discontinuity and self-occlusion may leave directions unobservable. Log per-point/aggregate information eigenvalues, do not shrink covariance in unsupported directions due to duplicated measurements.

### 2.2 Valid measurement and visibility policy (NOT correspondences)

Use **only** synchronized images/cameras at `k` for each directional step, and only source pixels in the hard foreground. A measurement candidate requires:

1. Finite point state and `Z>near>0`, projected `(u,v)` inside a valid **four-neighbour bilinear footprint** (unless an explicitly tested alternative interpolation is selected).
2. All neighbours contributing to RGB/depth/gradients satisfy valid depth and foreground mask. Never smear across zero depth, mask outside, disocclusion, or a visible depth break. Measured gradient sampling must use only well-defined gradients on valid support.
3. Ordinary **point-only camera visibility**: build a sparse per-pixel nearest-depth table from projected current XYZ points *for that frame and camera*. Use it solely to mark points behind the predicted front surface; it is not a learned renderer or a point-to-point matching operation. Do not construct footprint splats or color/depth images from the point cloud.
4. Compare predicted depth to measured depth only to identify likely **external occlusion** with uncertainty-aware geometry. A point with inconsistent depth must not be silently declared visible or automatically deleted. Explicitly report whether it is occluded, possibly mispredicted or ambiguous; avoid a hard small motion/depth-distance gate that would suppress the very motion being estimated.
5. If multiple predicted points reuse effectively the **same measured pixel sample**, do not treat them as independent high-confidence sensor readings. Resolve duplication with a deterministic front-most-per-pixel-per-camera sparse index (or an auditable correlated measurement model), and retain the remaining points as prediction-only for that camera. **This is intra-cloud visibility/measurement reuse control, not inter-frame point correspondence.** Other cameras can still measure them.
6. When no camera yields valid data for a point, **perform prediction only** and carry `P^-` unchanged by measurements. Mark observation/occlusion reason explicitly.

No GT mask semantics, no background observations, no point IDs from later frames, no KNN searches and no propagation based on an assumed future correspondence.

---

## 3. Free Phase 1: genuine independent 6D point IEKF, run twice

Let the point `i` at frame `k` have

\[
s_{i,k}=\begin{bmatrix}x_{i,k}\\v_{i,k}\end{bmatrix}\in\mathbb R^6,
\qquad P_{i,k}\in\mathbb S_+^6.
\]

The point's identity is its own endpoint sample, transported throughout the sequence. **No explicit motion-part assignment, no rigid group, no correspondence to another point cloud.**

### 3.1 Prediction and noise, including reverse time

For `Δt=t_k−t_previous` (signed real-time interval):

\[
F(\Delta t)=\begin{bmatrix}I_3&\Delta t I_3\\0&I_3\end{bmatrix},
\quad s^-_{i,k}=F(\Delta t)s^+_{i,previous}.
\]

For white acceleration noise spectral density `q_a` in units `m²/s³`, set `τ=|Δt|>0` and

\[
\boxed{Q(\Delta t)=q_a
\begin{bmatrix}
\tau^3 I_3/3&\operatorname{sgn}(\Delta t)\tau^2 I_3/2\\
\operatorname{sgn}(\Delta t)\tau^2 I_3/2&\tau I_3
\end{bmatrix}.}
\]

Then `P^-=F P^+F^T+Q`. Validate PSD for both forward and reverse time. **Do not insert negative `Δt³` into a forward noise formula**. The backward filter is independently initialized at `T`; it is not the algebraic inverse of the forward posterior.

Initialize `x_i` using the **measured endpoint depth** and its camera-calibration covariance. Initialize `v_i` with mean 0 unless justified by endpoint-only observations, and a defensibly broad prior covariance. No GT velocities, no direct use of future frames. Carry posterior covariance from frame to frame; no covariance reset at the start of each frame.

### 3.2 IEKF iteration — fixed prior and actual Kalman correction

At each frame, for each point, collect **only its valid camera observations**, up to 8 cameras × 4 channels per camera. In one IEKF inner iteration `j`:

\[
S_{i,j}=H_{i,j}P^-_i H_{i,j}^T+R_{i,j},
\quad K_{i,j}=P^-_i H_{i,j}^T S_{i,j}^{-1}.
\]

The corrected state is

\[
\boxed{s_i^{(j+1)}=s^-_i+K_{i,j}
\left[r_{i,j}+H_{i,j}\left(s_i^{(j)}-s^-_i\right)\right].}
\]

Here `r_{i,j}=−h(s_i^(j))`, `H=∂h/∂s`, `s^-` and `P^-` are **fixed through all IEKF iterations**. Reproject into the actual next-frame RGB-D observations and re-evaluate the samples/Jacobians each iteration. **Do not** sequentially apply the same image as multiple independent Kalman observations. Stop on an explicit state-increment criterion + innovation consistency (or numerical limit); report lack of convergence, invalid samples and weak information honestly.

**Fast equivalent information-form solve for each independent point (not a new global optimizer):**

\[
\Lambda^-_i=(P^-_i)^{-1},\qquad
A_{i,j}=\Lambda^-_i+\sum_{c\in\mathcal V_{i,j}}H_{ic,j}^TR_{ic,j}^{-1}H_{ic,j},
\]
\[
 b_{i,j}=\sum_{c\in\mathcal V_{i,j}}H_{ic,j}^TR_{ic,j}^{-1}r_{ic,j}
  -\Lambda^-_i(s_i^{(j)}-s^-_i),\qquad
\delta s_{i,j}=A_{i,j}^{-1}b_{i,j},
\]
\[
\boxed{s_i^{(j+1)}=s_i^{(j)}+\delta s_{i,j}.}
\]

This **is** the same IEKF gain update, algebraically, when the same linearization and Gaussian covariance are used; verify equivalence against the explicit `K` formula on dense small fixtures. It is ideal for CUDA because **`A` is only 6×6 per point** and camera terms accumulate without creating a giant `4C×4C` innovation matrix. Use `torch.linalg.solve`/Cholesky solves, not explicit inverses in production. The information form is a Kalman computation, **not an invented image energy**.

### 3.3 Posterior once, Joseph form, no hidden cross-point assumption

At the **final accepted linearization** `H,R`, update the covariance **once**:

\[
\boxed{P_i^+=(I-K_iH_i)P_i^-(I-K_iH_i)^T+K_iR_iK_i^T.}
\]

The equivalent linearized independent-point information covariance is

\[
P_i^+=(\Lambda^-_i+H_i^T R_i^{-1}H_i)^{-1}.
\]

Use the information matrix to compute `P+` efficiently in the batched implementation and **verify it against an independently computed full Joseph reference** in float64 tests; for production implement the Joseph-equivalent computation or a stable Joseph expression. Symmetrize only floating-point noise; do **not** silently clamp significant negative eigenvalues or reset/inflate covariance to hide failures. Record covariance calibration and PSD measures.

Per-point independent covariance is a deliberately **conditional approximation** if points share sensor pixels or systematic calibration noise. Address repeated pixel use as in §2.2; document remaining correlations. **Do not** claim the collection of `6×6` marginals equals a fully coupled world covariance.

If zero valid views: `s^+=s^-`, `P^+=P^-`. If a reprojected sample changes/disappears during an iteration, update validity, do not secretly reuse an out-of-mask measurement. **At a final iterate with zero valid views, also set `s_plus=s_minus`—not merely `P_plus=P_minus`.** Do not use external point-to-point targets as a fallback.

### 3.4 Computational supernodes — **never** restrict Phase-1 motion

RigEuler's Phase-1 correction used a 12-DoF rigid-like supernode tangent, while its distinct coherence graph used 192 representative nodes. **That Phase-1 rigid-like tangent is NOT required for the requested free pointwise IEKF** and previously caused an overly restricted free update. In this new formulation:

- Computational supernodes are optional **scheduling / memory / GPU batch indices** (for spatial tiles, block allocation, chunked accumulation, deterministic processing), not physical or correction-state groups.
- Preserve each point's own `s_i∈R6`, `P_i∈R6×6`, gain and update. Do **not** map `G zeta` into point motion; do not share pose/twist increments with neighbors.
- Coherence supernodes are constructed **after completing** the free trajectory. Default fixed 192 nodes, graph kNN 12 and held-out rigidity; their graph membership must not be altered to make a weak Phase-1 outcome pass.
- A grouping speedup is admissible only if output state/covariance matches an ungrouped baseline to floating-point tolerance on identical input data.

### 3.5 The two free runs are not a smoother

Run `F1` from `P_0` toward `T`, and **independently** run `B1` from `P_T` toward `0`. The endpoint point clouds are not assumed equal or registered by point ID. No backward-state initialization from the forward run, no forward/backward posterior multiplication, no Rauch–Tung–Striebel smoother crossing unrelated point identities.

**Output independently** for each direction: original point IDs, canonical XYZ/RGB, predicted/posterior XYZ and velocity over time, per-point covariance/observability/visibility, per-frame per-camera diagnostics, and `processing_order`/timestamps. Store original-frame-indexed results.

---
## 4. Motion coherence — independent for each completed free pass

Only **after** F1 and B1 have finished is the motion of their original endpoint points available. Run coherence **independently per endpoint**, in its own original anchor gauge. Do not use GT part IDs, GT joints, image-energy fitting, recovered feature matches or correspondences to build the graph.

RigEuler's validated branch (commit `481b2cf2`) is the structural reference:

1. Build a second deterministic canonical-XYZ representative partition of **192 nodes by total count** (`min(192,N)` if fewer points), separate from Phase-1 GPU scheduling batches. Farthest-point seeds, canonical Lloyd assignment, fixed node IDs. Keep seed, parameter hash and point→node mapping.
2. For representative node `a` at time `k`, aggregate the **actually transported original point tracks**: `y_a(k) = Σ_{i∈a} w_i(k) x_i(k) / Σ w_i(k)`. The pinned baseline uses uniform weights. If the RGB-D-only implementation uses observation support or uncertainty weights, treat it as an *explicitly named adaptation*, and compare to the unweighted pinned result. Never make an occluded node artificially stationary by overwriting its predicted trajectory with its last measurement.
3. Construct one symmetric **canonical** k-nearest-neighbour graph (`k=12`) on representative XYZ at the anchor. Never rebuild the graph from deformed positions every frame.
4. Measure temporal edge-length preservation: `l_ab(k)=||y_a(k)-y_b(k)||`, `z_ab(k)=log((l_ab(k)+eps)/(median_t l_ab(t)+eps))`, and `C_ab=sqrt(Σ w_k z_ab(k)^2 / Σ w_k)`. The edge is coherent under rigid translation/rotation, irrespective of motion magnitude.
5. Reuse the graph-constrained candidate hierarchy, held-out Kabsch rigidity checks, complement-relative articulation validation, and seed selection. In the pinned reference, a qualifying seed **must have a nonempty graph boundary** and meaningful relative rigid evidence; the whole connected graph may not be selected as an initial articulated body.
6. Save **seed-core nodes separately from recruited nodes**. Optional recruitment tests whether each remaining trajectory is better explained by the seed's single fitted rigid sequence than by remaining at the canonical location; never hide a selected 183-node near-entire seed under a final 192-node count. Keep `MOVING_RIGID`, `STATIC_COMPATIBLE`, `INCOHERENT_OR_AMBIGUOUS` distinct.
7. A coherent moving core is **not guaranteed**. If evidence is insufficient, output `UNRESOLVED`/`NO_IDENTIFIABLE_MOVING_CORE` with candidate diagnostics; do not force half the scene, all the scene, a minimum moving percentage or a GT-derived cutoff.

The earlier **offline parity audit** (`tools/audit_coherence_parity.py`, `reports/coherence_reference_parity.md`) already established equivalence of the pinned extractor and local version on identical *old* trajectories. **Do not rewrite this verified extractor to paper over Phase-1 deficiencies.** Run it on the new measured-RGB-D projective IEKF trajectories, and preserve the pinned/raw comparison. If the adaptation changes weighting, require separate results for each weighting policy.

**Outputs, per direction:** canonical node centers, point membership, time-varying node XYZ, observation-support fractions, edge indices and strains, rejected/qualified candidates, initial seed core, refinement list, final labels, held-out rigid-vs-identity evidence, selected moving endpoint point IDs, and a 3D/temporal debug viewer.

---

## 5. Phase 2 — the **second, explicitly rigid**, bidirectional RGB-D IEKF

This is NOT Kabsch fitting of Phase-1 trajectories as the final result, and is NOT continued free-point filtering. A single coherent moving core now shares one SE(3) motion for the entire run. The **RGB-D images remain the actual measurements**; the Phase-1 trajectories only supply core membership and an admissible initial motion/twist estimate. Do not feed the old free trajectory back as if it were an independent sensor measurement.

### 5.1 Two distinct rigid canonical cores and processing orders

- **F2**: selected point IDs are from canonical frame-0 `P_0`. Set rigid pose `T_0=I`, estimate a broad initial 6D twist prior (e.g. zero, or motion suggested by the already-completed free trajectories with correctly documented uncertainty). Process frames `1,...,T` using all actual camera images.
- **B2**: selected point IDs are from independent canonical frame-T `P_T`. Set rigid pose `T_T=I`, initialize separately, process `T−1,...,0`. Do not copy F2's canonical point IDs, filtered pose or covariance into B2.
- Keep the other endpoint points as **static at their own canonical XYZ for visibility diagnostics** if evidence supports that status; ambiguous points do not become static/moving by fiat. The explicit rigid filter's measurement update is driven by the current core's projected points. Other foreground surfaces can cause valid geometric occlusions but are not “background-rendered.”

### 5.2 State, prediction and covariance

Let the rigid state be

\[
\mathcal X_k=(T_k,\xi_k),\qquad T_k\in SE(3),\ \xi_k\in\mathbb R^6,
\]

with **12×12 tangent covariance** `P_k`. Fix the pose/twist convention explicitly: use **spatial/world left pose perturbation** `T' = Exp(δτ^) T` and spatial twist with `[v,ω]` ordering throughout (or document a different convention and modify *every* Jacobian accordingly). Canonical point `p_i` is transported by

\[
\boxed{x_i(k)=R_k p_i+t_k.}
\]

For interval `Δt` signed in physical seconds:

\[
\boxed{T_k^- = \operatorname{Exp}(\Delta t\widehat{\xi_{previous}^{+}})T_{previous}^{+},\qquad
\xi_k^-=\xi_{previous}^{+}.}
\]

For small errors `δ=[δτ;δξ]`, derive the 12×12 discrete error transition for that precise convention. For the spatial left convention, the standard first-order structure is

\[
F_k=\begin{bmatrix}
\operatorname{Ad}_{E}&\Delta t\,J_l(\Delta t\xi)\\
0&I_6
\end{bmatrix},\quad E=\operatorname{Exp}(\Delta t\widehat{\xi}),
\]

with the `SE(3)` left Jacobian `J_l` and adjoint consistent with `[v,ω]`. **Check by finite-difference perturbations on SE(3)**, in both directions; do not blindly trust a named library helper or mix left/right Lie error conventions. Propagate `P_k^-=F_kP^+_{previous}F_k^T+Q_k^{rigid}` with a documented positive-semidefinite integrated white-acceleration process noise, including pose–twist cross blocks and signed-time treatment. Report time units (`m/s`, `rad/s`, process spectral density).

### 5.3 Rigid RGB-D measurement and exact chain-rule Jacobian

For each moving core point `p_i`, at iterate `T_j`, calculate `x_i=T_j p_i` and **project that XYZ into the next actual RGB-D cameras**, exactly as in §2. Sample actual `I_k,D_k`; form the same four-channel innovation, with **no synthesized cloud image**.

For spatial **left** pose correction `T'=Exp(δτ^)T`,

\[
\frac{\partial x_i}{\partial\delta\tau}
=\boxed{B_i=\begin{bmatrix}I_3&-[x_i]_{\times}\end{bmatrix}\in\mathbb R^{3\times6}.}
\]

Hence the rigid observation Jacobian is

\[
\boxed{H^{rigid}_{ic}=
\begin{bmatrix}
 H^{point}_{ic,x}\,B_i&0_{4\times6}
\end{bmatrix}\in\mathbb R^{4\times12},}
\]

where `H_point_x` is the first 3 columns of the *same measured-image* Jacobian from §2. (Use a different lever arm only if explicitly changing to a body-centered perturbation and rederiving the entire model.)

Aggregate all valid core-point/camera observations into a **single 12D rigid Kalman measurement**. Use the frozen-prior iterated correction analogous to §3.2, but with a Lie-group prior error and retraction:

- `δτ = Log(T T_minus^-1)` is the left relative pose error; `δξ=ξ−ξ_minus`.
- Use the nonlinear prior-chart error and **its explicit left-chart Jacobian** at current iterate:

  \[
  e_j=\begin{bmatrix}\operatorname{Log}(T_j(T^-)^{-1})\\\xi_j-\xi^-\end{bmatrix},\qquad
  L_j=\frac{\partial e_j(\operatorname{Exp}(\delta\tau)T_j,\xi_j+\delta\xi)}{\partial(\delta\tau,\delta\xi)}\Bigg|_0
  =\operatorname{blkdiag}(J_l(e_{j,pose})^{-1},I_6).
  \]

  For the chosen spatial left convention `J_l` is the standard SE(3) left Jacobian and `e_pose=Log(T_j (T^-)^-1)`; compare both this derivative and its `Log` branch behavior with finite differences, including backward time.
- Recompute actual RGB-D projections and Jacobians for each current rigid iterate. The **Kalman-equivalent** 12D local information solve is

  \[
  \boxed{A_j=L_j^T(P^-)^{-1}L_j+\sum_{i,c}(H_{ic,j}^{rigid})^TR_{ic,j}^{-1}H_{ic,j}^{rigid},}
  \]
  \[
  \boxed{b_j=\sum_{i,c}(H_{ic,j}^{rigid})^TR_{ic,j}^{-1}r_{ic,j}
                  -L_j^T(P^-)^{-1}e_j,\qquad
  \delta_j=A_j^{-1}b_j.}
  \]

  Apply `T_{j+1}=Exp(δτ_j^) T_j`, `ξ_{j+1}=ξ_j+δξ_j`. `P^-` is fixed during IEKF iterations; no repeated assimilation of the same video frame. Accumulate camera/point terms with GPU reductions; solve one small SPD `12×12` system on device, with no rendered-image optimization.
- At the **final accepted** pose/twist compute `H_final`, `L_final`, and the one-frame local posterior via `P_final=(L_final^T (P^-)^-1 L_final+H_final^T R^-1 H_final)^-1`, which is the Joseph-equivalent covariance in the **final pose tangent chart**. Verify it against an independently computed local-chart Joseph expression using `P_prior_local=L_final^-1 P^- L_final^-T`. **Do not apply a second reset if the posterior was already computed directly in the final tangent chart.**
- If instead the implementation computes a posterior *before* moving the nominal pose and then stores it *after* the accepted left retraction `T_new=Exp(δτ^)T_old`, it **must reset** that pre-retraction covariance to the new chart. For the stated left-error convention `e_old=Log(T_true T_old^-1)`, `e_new=Log(Exp(e_old) Exp(-δτ))`; its local differential at the accepted correction is `J_l(δτ)` on the pose-error block. Thus transform with `J_reset=blkdiag(J_l(δτ),I6)` and `P_new=J_reset P_old J_reset^T` (with any additional cross-chart terms dictated by an explicitly different twist convention). Verify `J_reset` numerically; **choose only ONE of the two covariance-chart policies and report it**. An algebraically PSD Joseph formula in the wrong Lie chart remains incorrect.
- An occluded/unsupported core point contributes no update; update must never pull static points into the moving core. Store pose, twist, full 12×12 covariance, core projected measurements and rejection reasons per frame.

A shared rigid transform is the **physical constraint of Phase 2**, unlike Phase 1. Point motion must be *exactly rigid* by construction within that core, without any per-point independent correction after applying `T_k`.

### 5.4 Failure modes and stage gates

If F1/B1 coherence cannot identify a moving core, F2/B2 for that direction are **not run** and are marked `SKIPPED_UNRESOLVED_COHERENCE`, not silently seeded using GT or all points. If the rigid state is locally unobservable, keep uncertainty, and do not declare a known SE(3) pose merely because finite-dimensional code returns a number.

Require independent synthetic tests of static, pure translation, arbitrary axis rotation, combined 6DoF, varying angular speed, large baselines, occlusion and both processing directions. Verify exact `||x_i(k)-x_j(k)||` invariance for every selected core point pair (within float tolerances), analytic-vs-numerical Lie Jacobians, Kalman gain, covariance Joseph + reset, and measured-image overlay. No global splat objective anywhere in this module.

---

## 6. After the four filters: one fixed joint geometry, then optional constrained filtering

These are **downstream tasks**; do not use them to correct free/rigid Phase-1/2 outputs or to force motion-coherence success.

**Joint reduction:** For each directional rigid trajectory, compute relative transforms in its canonical gauge, e.g. `ΔT_k=T_{k+1} T_k^-1`, corresponding spatial twists `Log(ΔT_k)/Δt` with correct sign/branch handling. Generate both **prismatic and revolute** fixed-axis hypotheses, plus a static/no-identifiable-joint alternative. Determine identifiability from observed motion and uncertainty, not a prescribed object name or GT. Verify that the axis line is constant through time, even when joint speed or motion direction changes.

For a fixed unit direction `e` and axis point `a` (the axis line is independent of the chosen `a→a+λe`):

\[
T_{prismatic}(q)=\begin{bmatrix}I_3&q e\\0&1\end{bmatrix},
\qquad
T_{revolute}(q)=\begin{bmatrix}R_e(q)&(I_3-R_e(q))a\\0&1\end{bmatrix},
\]

with `R_e(q)=Exp(q[e]_×)`. A chosen reference-pose convention may require a constant rigid offset: identify and persist it, rather than confusing the joint axis with the world's origin. Estimate `q(t)` and, where identifiable, `qdot(t)`; report confidence and degeneracy (e.g. near-zero rotation or far-away revolute axis mimicking translation). A weakly excited joint is `UNIDENTIFIABLE`, **not** automatically revolute or prismatic.

**Optional subsequent constrained IEKF** (do not merge this with the four required passes): once a fixed joint geometry has been accepted, filter state `(q,qdot)` with standard `2×2` Kalman prediction and the **same sampled actual RGB-D** measured-image residual, chained through

\[
 H^{joint}_{ic}=H^{point}_{ic,x}\frac{\partial x_i(q)}{\partial q}
\quad\text{(zero velocity columns for instantaneous image measurement)}.
\]

For prismatic `∂x/∂q=e`; for revolute `∂x/∂q=e×(x(q)-a)` in the chosen world gauge. Joint geometry parameters `a,e` remain fixed during this q-filter; if geometry uncertainty is ignored, state that approximation and separately quantify identification confidence. Do not fit a new axis at each frame or replace Kalman correction by comparing rendered full-scene images.

**Support completion:** On endpoint points not already in the rigid moving core, compare **real measured projective RGB-D consistency over time** for (i) stationary identity and (ii) transport by the accepted time-invariant joint. Use positive observations and correct geometry; if the point is hidden/unobservable, remain `AMBIGUOUS`/`OCCLUDED`. Do not label all unsupported points as static or moving, do not use global Gibbs pseudo-energy, point-splat pixel flips, or information from GT. A support label must be backed by measured-frame evidence on visible frames, not mere failure to falsify a hypothesis.

**Bidirectional consistency:** F1/B1 and F2/B2 have **different endpoint point IDs**. Put rigid/joint results in the same calibrated world coordinates and compare physical axis line, direction up to orientation sign, time-aligned displacement/velocity, and static/moving coverage. Do not create cross-frame point correspondences while building the filters. Do not fuse two filter posteriors as if the two runs had independent images; their image evidence overlaps. Possible endpoint linking/registration is *optional downstream evaluation* only.

---

## 7. CUDA-first implementation — correctness before kernels, maximum measured throughput

### 7.1 Computational model and tensor shapes

Everything inside each video frame's Kalman prediction and update must execute on CUDA (`--device cuda`); the 4070 8 GB must be a supported target, with 5090 32 GB optional for large scenes. CPU is a small-fixture reference/fallback. **Avoid host synchronization per point or per PCG iteration**: the new pointwise implementation must not use PCG at all.

Suggested GPU-resident tensors (batch `N` is **selected canonical endpoint points**):

| Tensor | Shape | Meaning |
|---|---|---|
| `x0`, `rgb0` | `[N,3]`, `[N,3]` | immutable canonical measured XYZ and RGB |
| `state`, `state_minus` | `[N,6]` | world XYZ, physical-time velocity |
| `P`, `P_minus` | `[N,6,6]` | pointwise covariance, *not group shared* |
| `cam_R`, `cam_t`, `intrinsics` | `[C,3,3]`, `[C,3]`, `[C,4]` | calibrated cameras on GPU |
| `image_rgb`, `image_depth`, `mask` | `[C,H,W,3]`, `[C,H,W]`, `[C,H,W]` (or ragged camera tensors) | actual frame measurements |
| `active_point`, `camera_id`, `uv`, `z` | `[M]`, `[M]`, `[M,2]`, `[M]` | compact **valid view projections**, `M≤NC` |
| `innovation`, `H`, `R` | `[M,4]`, `[M,4,6]`, `[M,4,4]` or `[M,4]` variances | independent valid sampled view terms |
| `A`, `b`, `delta` | `[N,6,6]`, `[N,6]`, `[N,6]` | batched 6D Kalman information update |
| `T_rigid`,`twist`,`P_rigid` | `[4,4]`, `[6]`, `[12,12]` | one global shared moving-core state per direction |

`M` is *number of valid projected point/camera samples*, **not** number of point-to-point correspondences. Never allocate dense `[N, pixels]`, `[N,N]`, `[N,6N]`, `[M,M]` or a full `[pixels,point_contributors]` splatting structure. A 4×4 measurement covariance per sample is optional; the diagonal case is preferred when justified and efficiently implemented.

### 7.2 Exact CUDA dataflow per free frame

Use this orchestration, with CPU only for loading images/checkpoint boundaries:

```python
# This is algorithmic pseudocode: implement against actual repository APIs.
# No frame-to-frame point matcher and no renderer is called.
image_gpu = transfer_actual_rgb_depth_masks_and_camera_calibration(frame_k)
dt = signed_time_seconds[k] - signed_time_seconds[previous]
s_minus, P_minus = gpu_cv_predict(state_plus, P_plus, dt, process_noise)
sj = s_minus
for j in range(max_iekf_iterations):
    projected = project_world_points_to_all_cameras(sj[:, :3], cameras_gpu)
    visible = sparse_projective_visibility(projected, image_gpu.masks, image_gpu.depth)
    measurements = sample_ACTUAL_rgbd_and_gradients(projected, visible, image_gpu)
    # r=-h(sj), H=dh/ds, R=sensor noise; no later-frame point IDs anywhere.
    r, H, R, point_ids, camera_ids = build_direct_projective_terms(measurements, sj, rgb0)
    A, b = gpu_accumulate_point_information(P_minus, sj, s_minus, r, H, R, point_ids)
    delta = gpu_batched_spd_solve(A, b)  # [N,6], one 6×6 solve per point
    snext = sj + delta
    if gpu_convergence_condition(sj, snext, valid_mask=measurements.valid):
        sj = snext
        break
    sj = snext
# ONE final consistent measurement linearization and posterior update:
projected, measurements, r, H, R = relinearize_final_state(sj, image_gpu, rgb0)
P_plus = gpu_joseph_equivalent_posterior(P_minus, H, R, point_ids)
state_plus = sj
# Unobserved point rows carry P_minus without a measurement correction.
```

**Important:** `A,b` in the pseudocode must include the **fixed prior and prior-chart correction term** from §3.2, not just camera observations. Do not implement a generic residual descent or let a small regularizer replace physical measurement noise. Handle invalid frames and changing sampled validity explicitly: when relinearization has no valid measurements for a point, **roll its state back to `s_minus` and retain `P_minus` for that frame** (or explicitly mark an unaccepted correction and rerun a valid final linearization); do not keep an unsupported shifted state while claiming a prediction-only covariance. Do not claim convergence based on stale observations.

### 7.3 GPU/sparse speedups to implement and benchmark

- **Batch projection**, FOV and hard-mask tests across points; batch by camera if image sizes differ. Keep calibrated camera matrices on GPU throughout a run.
- Use `torch.nn.functional.grid_sample`, a custom CUDA kernel, or Triton **only to sample the true RGB-D fields/gradients at projected `(u,v)`**; it must not rasterize a synthetic image. Gather four neighbors with mask validation; inspect sign/orientation of coordinates and interpolation derivatives independently.
- Precompute measured image spatial gradients / validity maps once per frame on GPU. Keep their validity and units consistent with the interpolation derivative; using inconsistent Sobel edges as if they were exact bilinear derivatives is an approximation requiring a stated test.
- Build **sparse active `(point_id,camera_id)` rows**, one small observation per visible point and camera. Avoid pointless full-image computation except preparation of measured image gradients/masks.
- Accumulate per-camera `HᵀR⁻¹H` and `HᵀR⁻¹r` into `[N,6,6]`, `[N,6]` using `index_add` / `scatter_add` / sorted segmented reductions, preserving all camera contributions. Choose stable deterministic reductions for numerical oracle tests. **Do not reduce different points to one physical group parameter**.
- Compute batched `6×6` Cholesky/solve, and one final batched covariance update. If `torch.linalg.cholesky_ex` flags numerical failure, emit row IDs and spectrum; use a specified float64 retry/diagnostic or fail loudly, not silent bogus PSD floor.
- For the rigid 12D system, accumulate all visible core-point contributions on GPU, reduce to one `12×12` information system and solve on device; avoid a per-pixel least-squares optimizer or CPU inversion.
- Profile and eliminate Python per-point loops, repeated `.cpu().numpy()`, `.item()`, `.tolist()`, `torch.cuda.synchronize()` in the inner update, repetitive `torch.nonzero` host synchronization, repeated kernel launches for trivial scalar expressions, large intermediate Jacobian copies and CPU-side eigenanalysis per point/frame. Record aggregated counters GPU-side and transfer **once at diagnostic boundaries**.
- Prefer fast contiguous layouts and tuned chunk sizes; benchmark PyTorch fused vectorization before writing custom C++/CUDA/Triton kernels. `torch.compile`/CUDA Graphs are allowed only when measured, stable with variable valid-measurement counts, and numerically equivalent.
- Compare float32, float64 and mixed precision **on the real GPU**. The previous system spent excessive CUDA time on tiny float64 batched GEMMs. A preferred candidate is float32 for image sampling/projection and float32 or float64 for the very small covariance/solve as dictated by PSD/calibration tests. No unsupported blanket claims that float32 is “exact.”
- Budget memory for an 8 GB RTX 4070; test ≥34k and ≥146k endpoint points. Avoid retaining full `[frames,N,6,6]` GPU tensors. When memory approaches the configurable cap, chunk **independent point filters** without changing their mathematical update. Retain only current/prior state and camera tensors on GPU.
- Use CUDA events/`torch.profiler` for *real* latency, memory peak and synchronized frame totals. Benchmark CPU preprocessing, camera loading, GPU transfer, projection, validity, sampling, gradients, information accumulation, 6×6 solves, covariance and debug I/O **individually**. Never optimize by turning off required cameras or masks.

### 7.4 Minimal GPU entry points to implement

The new code should have testable functions equivalent to:

```python
project_points_cuda(xyz_world, camera_params)                    # [C,N,2], [C,N], analytic J projection
sample_measured_rgbd_cuda(images, depths, masks, uv, valid)       # actual values/gradients, [M] compact rows
projective_terms_cuda(projected, measured, canonical_rgb, noise)  # r,H,R,valid/reason, no synthesized image
predict_points_cuda(s_plus, P_plus, dt, qa)                       # [N,6], [N,6,6]
update_points_iekf_cuda(s_minus, P_minus, measured_frame, ...)    # frozen prior, iterated update, final Joseph
update_rigid_iekf_cuda(T_minus, xi_minus, P_minus, core, frame, ...) # 12D Lie update, final covariance reset
```

Use thin Python orchestration; optimize kernels after a pure PyTorch CUDA reference is numerically proven. In a custom CUDA kernel, every sensor datum must be traceable to a measured `(camera,frame,u,v)` sample, not a rendered pixel.

---
## 8. Optional deterministic endpoint point-cloud subsampling — user-selected factor

The purpose is to make a **quick run computationally cheaper without changing the estimator's equations**. This feature applies **at both independent endpoint canonical clouds before F1/B1 start**, not after the filter, not by changing the image measurements, not by reducing the camera set, and not by dropping time frames.

### 8.1 CLI and exact count semantics

Expose, prominently, at minimum:

```text
--pcd-subsample-factor 1              # integer >=1; 1 = OFF, retain all original points
--pcd-subsample-method spatial-stratified   # other supported: seeded-uniform (diagnostic baseline)
--pcd-subsample-seed 0
--pcd-subsample-boundary-preserve true      # optional toggle
--pcd-subsample-report true                # coverage report per canonical endpoint
```

For each endpoint cloud containing `N_full` **real measured** points, factor `f` selects a target

\[
\boxed{N_{target}=\left\lceil\frac{N_{full}}{f}\right\rceil,\qquad
 \text{actual reduction}=N_{full}/N_{selected}.}
\]

Examples: with `f=4`, a 34,369-point canonical endpoint targets **8,593** points; a 145,973-point endpoint targets **36,494**. Report the **actual** selected count, not merely the requested factor. Forward and backward endpoints may have different `N_full` and will be sampled **independently**.

`--pcd-subsample-factor 1` must produce the exact full canonical set and IDs without any reorder (zero-behaviour-change test). For `f>1`, select a deterministic subset of **existing canonical point IDs**, preserving their original measured XYZ, RGB, source pixel/camera and initial covariance. **Do not replace groups by voxel means, interpolate synthetic colors or invent new 3D points.** The new selected IDs are stable through the entire directional free→coherence→rigid chain.

### 8.2 Spatially smart, label-free selection

The default implementation should avoid global random thinning that erases thin moving parts and articulation boundaries:

1. Estimate spatial voxel occupancy from **canonical XYZ only**, not future trajectories or GT part labels. Derive voxel scale automatically to approximate target count; support overflow/underflow with a deterministic follow-up selection.
2. Stratify candidate point IDs by voxel / coarse spatial tile and available endpoint camera-source provenance. Sample representatives from original XYZ points without merging them.
3. Preserve representation near *measured* frame-0/T foreground silhouettes, measured depth discontinuities and RGB edges, **using endpoint images only**; do not use actual moving/static labels or later video frames. This is preferential **sampling coverage**, not a new physical model or segmentation.
4. Avoid destroying very small spatial components simply because the majority of points lies on a large flat surface; monitor component-size and voxel coverage. If exact target count conflicts with retaining every boundary sample, enforce reproducible priority and document what was dropped instead of silently retaining an arbitrary larger set.
5. Enforce camera-source and image-grid coverage diagnostics so a single densely sampled camera/view does not eclipse other observed surfaces. Support homogeneous source clouds without camera provenance by falling back to pure spatial stratification.
6. Do **not** use farthest-point sampling with an unbounded `N²` distance matrix or Python quadratic loops; implement voxel hashing/sorted key grouping on CPU once or CUDA, with O(N log N) or better memory and runtime for 100k–350k points.

The inference uses **all original RGB-D image pixels as fields** during projective sampling: subsampling reduces *transported points*, never fabricates a lower-resolution sensor or hides a camera. Keep full raw endpoint clouds available as separate metadata, without transporting or assigning state to the omitted points during that run.

### 8.3 What is and is not preserved under downsampling

- Pointwise Kalman update for **retained points** uses identical `F,Q,h,H,R` and produces mathematically consistent state/covariance on that subset. Because projected cloud density changes, self-occlusion, pixel reuse and sample support can differ; **do not promise a full and sampled run are numerically identical**.
- For large factor, the 192-node coherence partition will have fewer points per node, possibly fragment a thin moving core or leave it unobservable. Keep the coherence graph definition explicit and report effective representative occupancy, discarded regions, boundary sampling and seed/recruited component sizes. Never tune coherence thresholds to “repair” an aggressive sampling artifact.
- Do not project estimated motion back to untracked omitted canonical points as if they had independent IEKF trajectories or covariances. At a **later**, explicitly separate joint-support stage, an accepted rigid/joint motion may be evaluated on the original full endpoint cloud, but such derived positions are *model-implied*, not directly free-filtered estimates.
- Do not confuse the point-cloud subsample factor with image pyramid scale, camera count, time subsampling, visibility radius, computational-group size or coherence representative count. Each has its own option/manifest.

### 8.4 Required comparative experiments

At least `f = 1,2,4,8` on blade `blade_103706`, plus `f=1,2,4` on a ≥145k-point scene if memory permits. Use exactly the same cameras, timestamps, noise and process settings. Record:

- initialization and subsampling runtime; `N_full`, `N_selected`, per-camera source distribution, retained voxel and boundary coverage;
- per-frame wall/CUDA time and maximum GPU allocated/reserved memory; number of projected/valid/occluded/sampled measurement rows per camera;
- innovation distributions (RGB, metric depth), Jacobian rank/conditioning, position/velocity covariance eigenvalues and NIS summaries;
- number of prediction-only points, effective visible scene coverage, and failures on thin/fast moving regions;
- coherence seed selection, candidate pool, recruited nodes, `UNRESOLVED` status and held-out rigidity (do not use GT to drive this);
- **post-hoc-only** synthetic GT-based endpoint-trajectory, rigid pose and joint-axis accuracy when available;
- speed/accuracy tradeoff; do not declare a factor “optimal” from one favorable scene.

The full/factor-1 estimator is **the correctness reference**; smaller factors are clearly labeled acceleration options, not silently substituted defaults. A failed sampled run is reported as such, not changed to `COMPLETE` for convenience.

---

## 9. RigEuler-quality visual + numerical debugging, using ONLY measured RGB-D and point clouds

**Use RigEuler's diagnostic organization and visual clarity, NOT its Gaussian rendering content**. Every displayed color/depth image is **actually recorded RGB-D**, and every 3D geometry element is an existing colored XYZ point / trajectory / estimated rigid transform / joint axis. Do not show simulated splat RGB, synthetic opacity, Gaussian ellipsoids/scales, rendered coverage, “photometric loss” plots, per-point correspondence arrows toward matched next-frame IDs, or a fabricated fully dense predicted depth image.

Build a single offline index (`debug/index.html`) linking standalone per-stage dashboards and machine-readable numeric logs. Include **both forward and backward directions** and a frame scrubber in original chronological numbering. Carry precise coordinate and provenance labels in plot legends, including which cloud is canonical in each panel.

### 9.1 Required interactive visual pages

**A. `free_forward_3d.html` and `free_backward_3d.html`**

- Interactive world-space 3D point cloud: immutable canonical endpoint (muted), predicted XYZ before measurement (second color), filtered posterior XYZ after measurement (main color). Option to render RGB-colored retained points, measured point density, camera frusta, 3D world axes, and static camera positions.
- Frame slider, play/pause, camera selection, point-selection-by-index, free velocity arrows (decimated for readability), per-selected-point trajectory lines over the complete processing sequence. **The line connects one canonical point's own propagated time states**, not matches to different frames.
- Foreground and visibility color layers: valid measured sample, excluded mask, invalid depth, projected outside FOV, intra-cloud occluded, measured-depth occluded, no useful RGB texture/weak observability, prediction-only. Each layer has a legend and factual counts.
- Optional uncertainty visualization: position `3×3` covariance ellipsoids for a small **selected set** of point states, correctly scaled by eigenvectors and sigma, plus velocity uncertainty; these are **Kalman uncertainty glyphs for XYZ points**, never 3D Gaussian splatting primitives or shape/opacity parameters.
- Display selected point ID, canonical source pixel, current `x,v`, predicted/filtered `x`, `diag(P)`, number of valid cameras, depth/RGB innovation, sample coordinates and status on hover.

**B. `free_forward_rgbd.html` and `free_backward_rgbd.html`**

- Camera selector, original-frame slider, and **two raw measured panels**: the actual RGB frame and the actual **metric depth frame** (with labeled colorbar in metres), plus the actual binary foreground mask. Source filenames and calibration/timestamp displayed.
- Overlay **projected initial, predicted and posterior point positions** as distinct marker shapes/colors. “Initial” is the untransported canonical endpoint projection; “predicted” is the process forecast; “posterior” is after the actual-image IEKF. Draw predicted→posterior 2D displacement arrows **only** (state correction arrows are allowed; do not point arrows to supposedly matched frame-k points).
- For each visible projected marker, display **sampled actual RGB**, immutable canonical RGB, **sampled actual metric depth**, predicted camera-Z depth and innovation values. Show a 2D scatter/marker heatmap of `Z_pred−D_meas` on **valid projected point samples only**; empty pixels stay empty. No dense fake prediction at unsampled pixels.
- Numeric/visual validity layers: FOV, mask, four-neighbour validity, depth discontinuity, projected-cluster nearest-depth visibility, measured depth blocker, reused-pixel handling, no observation. Show invalid marker reasons with colors and per-camera counts.
- Show calibrated camera axes and original/endpoint image extent; verify projected markers fall on real measured structures. Clicking a projected point synchronizes selection in the 3D viewer.

**C. `coherence_forward.html` and `coherence_backward.html`**

- Canonical 192-node (or documented smaller) representative graph with actual fixed kNN edges; edge strain heatmap and time slider. Identify valid sample support vs prediction-only segments.
- Visually distinguish **seed core**, **recruited nodes**, **static-compatible** and **ambiguous** nodes. Show their corresponding canonical XYZ points in a linked 3D view; no color label based on GT during inference.
- Candidate hierarchy, persistence, boundary edges, held-out Kabsch rigidity, complement-relative versus world-frame evidence, seed→final growth, and explicit unresolved status.
- Keep a comparison view of pinned/raw coherence and any explicitly enabled RGB-D support weighting, without concealing disagreement.

**D. `rigid_forward_3d.html`, `rigid_backward_3d.html`, and `rigid_rgbd.html`**

- Show initial rigid-core cloud and current **exactly rigid** transformed core, all actual static-compatible endpoint points in a separate neutral layer, known cameras and observed RGB-D images.
- Visualize filtered `T_k`, spatial twist vector and physically meaningful 6D pose covariance on frame timeline. A representative set of local coordinate triads follows the rigid body, making translation, rotation and direction clear.
- Overlay rigidly moved point projections onto **measured** RGB and metric depth. Show same true measured sample residuals, validity and sample counts as free Phase 1. This distinguishes free vs constrained motion using the **same observation model**.
- Display an explicit numerical rigidity check, `max_{i,j}| ||x_i(k)−x_j(k)||−||p_i−p_j|| |`, and failures. Separate full-scene context from moving-core points; no “background image” rendered from the static cloud.

**E. `joint_forward.html`, `joint_backward.html`, and `bidirectional.html`**

- On colored XYZ geometry show the world-space *fixed* revolute-axis line (one point plus unit direction, with uncertainty), or fixed prismatic direction; overlay time-varying `q(t)`, `qdot(t)`, measured rigid pose and joint-predicted poses.
- Explicit plots of axis-line drift/identification residual, pose/twist uncertainties, excited vs unobservable motion intervals, and whether candidate is `PRISMATIC`, `REVOLUTE`, `STATIC` or `UNIDENTIFIABLE`.
- Independent endpoint point clouds can be shown together in shared calibrated world coordinates; distinguish them visually. Compare axis geometry across directions without point-to-point connections. Show regions only one endpoint observes and any ambiguous support.
- Foreground RGB and depth overlays are source data only; display actual point projections and valid sampled sensor innovations.

**F. `numerical_diagnostics.html` + simple CSV/JSON**

For every directional run provide timelines and summary distributions of the exact quantities in §9.2; include per-camera and per-frame filters, frame scrubber, and direct navigation to suspect frames. Avoid requiring browser console inspection to see errors. The HTML must function offline with local copies of approved JS dependencies and contain no externally fetched images or models.

### 9.2 Required machine-readable per-frame/per-camera numerical diagnostics

Persist `run_manifest.json`, `numeric/frame_XXXX.json` and `numeric/timeline.csv` for each of F1/B1/F2/B2. Record at least:

| Category | Explicit fields |
|---|---|
| Provenance | input scene, point-cloud canonical endpoint, source SHA/hash, code commit, camera indices and calibration hash, original frame, processing frame/order, actual timestamp/`Δt`, subsampling factor/seed/selected IDs |
| Processing | wall time; CUDA elapsed ms for transfers/projection/visibility/sampling/gradients/Jacobian/info accumulation/solve/covariance; GPU peak allocated/reserved; host→GPU transfer bytes; num iterations and termination reason |
| Sensor | camera index; total foreground pixels; total points; projected positive-Z/in-FOV; mask-valid bilinear samples; actual depth-valid; rejected at discontinuities; self-occluded; external-occluded; duplicate-pixel suppression; measurement rows accepted; unique points observed; prediction-only count |
| Innovation | signed metric depth innovation quantiles (metres), RGB difference quantiles (normalized or 0–255 explicitly), NIS quantiles; per-camera counts; number of outlier/invalid/projective discontinuity flags; **measured RGB/depth sample provenance** |
| Filter update | predicted→posterior displacement/velocity quantiles; per-camera information contribution; Kalman gain/operator norms; posterior/prior covariance position/velocity eigenvalue quantiles; PSD failures, minimum eigenvalue and condition number |
| Free-point state | canonical point IDs; residual norm/rank information; posterior velocity (m/s); component-wise uncertainty and physically unobservable directions; no fake group pose |
| Coherence | representatives, graph edges, seed size, recruitment count, candidate ranks, node observation support, held-out consistency and ambiguity |
| Rigid state | `T_k`, spatial twist (`m/s`,`rad/s`), full 12×12 posterior diagnostics, Lie-reset Jacobian checks, rigid-core pair-distance error |
| Joint | hypothesis type, fixed axis line/direction, `q,qdot`, confidence/degeneracy, forward/back axis comparison and time convention |

**Mandatory high-detail numerical trace** for a small configurable deterministic set of original point IDs (`--debug-point-ids`) at selected frames (`--debug-frames 0 1 2 5 10 ...`): world predicted/current/posterior XYZ+velocity; per-camera `(u,v,z)`, actual sampled `(RGB,D)`, foreground/validity; `r[4]`, `H[4,6]`, `R[4,4]`, Kalman information block, `K` or implicit-gain check, raw IEKF correction and `P-/P+`, and finite-difference deviation. Generate a similar rigid 12D debug trace. **This is numerical evidence that Kalman equations, not a hidden optimizer, were used.**

GT metrics (motion distance/axis error/support precision etc.) may appear only in an explicitly separate `evaluation_gt/` namespace **after all inference outputs are written**. Never feed GT into filtering, camera validity, coherence candidate selection, joint selection, speed-ablation sampling or runtime optimization.

### 9.3 Visual-debug acceptance tests — screenshots are part of validation

- Programmatically render/screenshot representative pages at least at frames 0,1,5,10, last for both directions. **Actually inspect** the screenshots against original RGB/depth/mask arrays and known camera geometries; a browser opening without an error is not enough.
- The RGB underlay must equal the input frame values (modulo named display scaling); depth must have the correct metric scale; mask may not include excluded background pixels. Verify a few selected overlaid projected coordinates numerically against `π(T_cw x)` from independent CPU code.
- Move sliders and camera selectors and verify that both 2D and 3D points, numeric summaries and original-frame labels really change; do not present static HTML placeholders.
- For backward pages at original frame 0, explicitly show that canonical XYZ is *from frame T*; if the viewer instead calls frame 0 canonical, fail.
- Debug generation should never alter filter states, save excessive full-resolution per-frame point screenshots by default, or transfer all `[T,N,6,6]` GPU matrices every IEKF iteration.

---
## 10. Surgical code migration, matched to the inspected RGBDiekf source

Start with `git status --short`, `git rev-parse HEAD`, inspect every listed module and make an independent `source_snapshot`/branch before editing. The following are actual files at inspected commit `34131bcd`; use the user's current worktree as source of truth if it has newer changes. **Do not modify/erase archived old results or silently claim that an old splat codepath implements the new model.** Prefer a clearly named *new production implementation* and leave the former experiments under a clearly marked legacy entry point.

| Inspected file / function | Current behaviour established by source | Mandatory new production behaviour |
|---|---|---|
| `pointwise_rgbd/rgbd_point_renderer.py`: `_candidates`, `render_view`, `Linearization` | Projective footprint splats, normalized contributor RGB-D, visibility taper, per-pixel coverage and synthetic-image Jacobians | **Never import for new production passes.** Replace with simple projection/sampling kernels and actual measured RGB-D samples. This file may remain **legacy only**. |
| `pointwise_rgbd/image_observation.py`: `student_t`, `FrameObservation.terms`, `.pixel_energy`, `.apply_W` | Student-t and coverage-tempered energy, IRLS weights, negative-mask and constructed rendered-image terms | Raw masked frame accessor; measured-image gradients and `projective_terms` with 4-channel `r`, analytic `H`, physical `R` and rejection reason codes. No energy or IRLS anywhere in four passes. |
| `pointwise_rgbd/image_iekf.py`: `linear_system`, `trust_region_solve`, `_frame_step` | `Gᵀ(HᵀWH+Λ)G`, Steihaug/PCG and rendered-image optimization | Free **independent per-point** batched 6D IEKF (fixed prior) with GPU observation-information accumulation, direct Kalman equivalent solve, and posterior Joseph. No grouped physical correction `G ζ`. |
| `pointwise_rgbd/phase1_covariance.py`, `restricted_posterior.py` | alternative block-information and restricted Joseph models for grouped splat updates | New production **per-point** `6×6` covariances under a clearly declared pointwise-independence approximation. Proper signed physical-time `F,Q` and final Joseph per frame; archive old covariance modes for reproducibility. |
| `pointwise_rgbd/supernodes.py`: `GroupBasis` | canonical rigid-like 12D grouped correction tangent | Do not use `GroupBasis.expand()/reduce()` in *free* update. Optionally reuse canonical spatial indexing for GPU batching; keep independent coherence partition. |
| `pointwise_rgbd/image_phase1.py`: `endpoint_catalogue`, `run` | measured endpoint samples followed by footprint radius and per-frame `render()`/`frame_step()` | Reuse verified measured endpoint extraction and provenance; **after optional deterministic subsampling**, call free 6D projective CUDA IEKF separately for `first` and `last`. Save original chronological indexing and valid-measurement status. |
| `pointwise_rgbd/image_coherence.py`, `supported_coherence.py`, `rigeuler_ref/coherence_extraction.py` | saved representative graph, fixed kNN, reference candidate selection, image-support variants | Keep pinned coherence logic and existing parity tests; replace renderer-derived `point_observed` input with actual projective measurement validity. Maintain separate raw-versus-weighted audit. |
| `pointwise_rgbd/rigid_image_iekf.py`: `small_state_step`, `run` | shared pose/twist via rendered-image measurement energy | Keep relevant Lie state math only after checking conventions; implement F2/B2 shared rigid **projective RGB-D IEKF**, 12D gain, Lie retraction, final covariance reset. |
| `pointwise_rgbd/joint_image.py`: `reduce`, `refine`, `run` | joint reduction + rendered-image candidate fitting/energy testing | Reduce from F2/B2 rigid trajectories; fixed-axis geometry, then optional `q,qdot` projective IEKF, no rendered-scene loss. |
| `pointwise_rgbd/image_support.py`: splat removal/flip energy and block ICM | point-label MAP based on full synthesized scene's pseudo-likelihood | Separate downstream observed projective consistency for fixed joint/static hypotheses; retain ambiguous/occluded labels. |
| `pointwise_rgbd/image_visualization.py`, `rigeuler_debug.py` | visual pages often based on rendered images/covariance output | Rebuild readers to show actual RGB-D frames, point projections, covariance glyphs, coherence graphs, filtered rigid poses and joint axes. Preserve useful reference UX and original-frame sliders. |
| `tools/run_rgbd_image_iekf.py` | CLI defaults to `rigeuler_block_information`, `RenderPolicy`, `footprint_scale`, old rendered-image pipeline | Introduce `tools/run_rgbd_projective_pipeline.py` with the exact four passes and factor flag. Make the new entry point unambiguously recommended in README; mark old runner `LEGACY_RENDERED_SPLAT_BASELINE`, or redirect only after full regression tests. No mixing snapshots from incompatible models. |

**Suggested new modules** (names may be adjusted to actual repo architecture but their duties MUST remain distinct):

```text
pointwise_rgbd/projective_camera.py          # world-to-camera, projection, inverse, J_pi, calibration integrity
pointwise_rgbd/projective_sensor.py          # GPU measured RGB/D/mask sampling, masked gradients, variance
pointwise_rgbd/projective_visibility.py      # FOV, projected-point z-buffer, measured-depth blocker, pixel reuse
pointwise_rgbd/projective_terms.py           # r,H,R, camera/sample provenance, validity; no image renderer
pointwise_rgbd/point_kalman_cuda.py          # F,Q, pointwise batched 6D information IEKF, Joseph
pointwise_rgbd/free_projective_run.py        # F1/B1 endpoint lifecycle, GPU transfers, outputs
pointwise_rgbd/canonical_subsample.py        # deterministic factor, spatial stratification, exact original-ID mapping
pointwise_rgbd/rigid_kalman_cuda.py          # F2/B2 12D Lie IEKF, rigid point Jacobian, covariance reset
pointwise_rgbd/joint_projective.py           # downstream rigid reduction + optional constrained q IEKF
pointwise_rgbd/projective_debug.py           # measured-image overlays, 3D points and numeric viewers
tools/run_rgbd_projective_pipeline.py        # staged CLI, both directions, safe resume, hashes
tools/profile_rgbd_projective_cuda.py        # CUDA events, bottleneck/memory/precision benchmarks
tests/test_projective_camera.py              # independent sensor geometry oracle
tests/test_projective_measurement.py         # finite-difference h,H; strict mask/visibility
tests/test_point_kalman_cuda.py              # dense K / information equality + Joseph
tests/test_rigid_kalman_cuda.py              # Lie propagation/retraction/J/reset
tests/test_canonical_subsample.py            # exact factor-1 and deterministic factor >1
tests/test_four_passes.py                    # both independent endpoint gauges/directions
```

Avoid introducing misleading permanent filenames called `render` for point projections. A function that projects `XYZ` to `(u,v,Z)` is a projection; it should not compute synthetic RGB or synthetic depth over an image grid.

### 10.1 Exact CLI contract and example commands

Expose the following names (choose stable names, document any compatibility aliases):

```text
--scene PATH                     # calibrated RGB-D dataset root
--output PATH                    # new, nonexisting output root; no overwrites
--device cuda|cpu                # CUDA production; CPU oracle/test
--direction forward|backward|both  # default both
--stage free|coherence|rigid|joint|support|all
--stop-after free|coherence|rigid|joint|support
--fps FLOAT / --timestamps-file PATH / --dt-seconds FLOAT
--noise-config PATH              # physically declared RGB/depth/process noise
--pcd-subsample-factor INT        # default 1, >=1
--pcd-subsample-method spatial-stratified|seeded-uniform
--pcd-subsample-seed INT          # default 0
--pcd-subsample-boundary-preserve true|false
--cuda-backend torch|triton       # torch reference first; Triton only if implemented and verified
--covariance-precision float32|float64|mixed
--point-chunk-size INT           # deterministic bounded GPU memory
--coherence-count 192 --coherence-knn 12 --coherence-folds 5
--max-iekf-iters INT
--debug none|light|full
--debug-frames 0,1,2,5,10,30,59
--debug-point-ids comma-separated-ids
--save-state-interval INT
--resume-from PATH               # read-only completed artifacts with exact hashes; no model mixing
```

Normal full-resolution run (both independent endpoint clouds and both rigid passes, all cameras):

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
SCENE="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/blade_103706"
python tools/run_rgbd_projective_pipeline.py \
  --scene "$SCENE" \
  --output output/projective_blade_full \
  --device cuda --direction both --stop-after rigid \
  --pcd-subsample-factor 1 --debug full
```

Faster ablation (factor 4, **same** actual RGB-D observations/cameras):

```bash
python tools/run_rgbd_projective_pipeline.py \
  --scene "$SCENE" \
  --output output/projective_blade_factor4 \
  --device cuda --direction both --stop-after rigid \
  --pcd-subsample-factor 4 --pcd-subsample-method spatial-stratified \
  --pcd-subsample-seed 0 --debug light
```

Focused mathematical gate, free pass only for a short scene prefix (do not run full sequence first):

```bash
python tools/run_rgbd_projective_pipeline.py \
  --scene "$SCENE" --output output/projective_blade_free_smoke \
  --device cuda --direction forward --stop-after free \
  --pcd-subsample-factor 1 --debug full
```

**These are implementation-target commands:** Codex must create/verify the new CLI before claiming they execute. Do not display them as commands already supported by current GitHub HEAD.

### 10.2 Runtime/storage policy

Use separate folders `free_forward/`, `free_backward/`, `coherence_forward/`, `coherence_backward/`, `rigid_forward/`, `rigid_backward/`, `joint/`, `bidirectional/`, `debug/`, `performance/`, `evaluation_gt/`. Keep each stage's source hashes, inputs and output-status manifest. Do not run an unresolved rigid phase with a fabricated core. Honor `--stop-after` and allow strict read-only resume of completed earlier stages **only if point-ID subsample mapping, calibration, timestamps, cameras, noise and implementation hashes match**.

Store current/next GPU state only; write per-frame compact numeric diagnostics. For expensive per-point arrays use optional chunked snapshots / memory maps with bounded intervals, not full `[T,N,6,6]` GPU history or a giant compressed archive copied every iteration. Persist enough complete trajectories for coherence (XYZ `[T,N,3]`, optional velocity/precision information), but do not require full 6×6 covariance at every frame unless `--debug full` or `--save-full-covariance` is requested. Source RGB-D and masks remain on disk; no need to duplicate all original sensor frames into the output directory. Visual debug can load actual input frames by retained source path/hash.

---

## 11. Mandatory independent numerical tests and acceptance gates

**Perform in order. Do not relabel a failure as a pass or suppress tests. Do not tune using the blade GT as an optimizer target.** Each gate must emit the exact command, code revision, input and test logs.

### Gate A — measured RGB-D geometry and sample provenance

- CPU double-precision and GPU projection/back-projection roundtrips with known calibrated cameras; include nontrivial extrinsic rotations and translations, unequal `fx/fy`, noncentral principal point, and millimetre-to-metre scaling tests.
- Verify `x_meas_w−x_pred_w = R_cw^T K^-1[u,v,1](D_meas−Z_pred)` numerically (no fake tangential constraints). On a planar, textureless RGB-D image the depth-only observation Jacobian must not imply arbitrary 3D lateral certainty.
- Test analytic `J_pi`, measured RGB bilinear gradient, measured depth bilinear gradient and the **complete** `4×6` Jacobian `H` against independent finite differences in smooth, fully foreground-valid image interiors. Test sign under `r=−h`.
- Masks with invalid/zero depth in any of four neighbors, out-of-FOV points, behind-camera points, foreground/background edges and occlusion must produce **no contaminated measurements**. Sample source pixel values must agree with the actual input arrays; independent CPU sampling oracle.
- Two projected points sharing a pixel may not automatically count as two independent measured-depth observations; verify the declared duplicate/visibility policy and exact observed-point count. Test an intentionally missing whole camera and a fully occluded point.

### Gate B — true 6D IEKF numerical equivalence (CPU oracle before GPU optimization)

- Dense small linear Gaussian toy: arbitrary SPD `P^-∈R6×6`, known `H,R,r`, nonzero `x_j−x^-`: reproduce textbook iterated Kalman `K`, `x_new` and Joseph `P+` independently. Compare against information-form `A,b` to double precision without an optimizer.
- Nonlinear observed RGB-D synthetic field with known smooth gradients and actual point trajectories: finite-difference check `H`; verify one-step correction sign, convergence behavior and truthful NIS. Test one camera, eight cameras, reverse-time `Δt<0`, changing visibility and no valid measurement (strict prediction-only).
- Verify that a depth-only textureless planar scene remains unobservable tangentially; covariance cannot contract laterally without corresponding valid RGB gradients/multiview information. Verify RGB texture gives tangential constraints where the measured image gradients actually support them.
- Verify projected points never receive an update from mask-background or non-visible pixels. Verify individual points evolve independently except explicitly documented sensor shared-pixel approximations; no hidden `Gζ` group rigidity, no `point-to-point` lookup.
- Compare full double-precision CPU oracle, vectorized PyTorch CUDA and (if implemented) Triton on identical fixture states; report state relative/absolute errors, gain, NIS and PSD. No silent precision regressions or skipped CUDA test presented as a pass.

### Gate C — forward and backward *free* point-cloud transport

- Use short known-motion RGB-D sequences with a fixed calibrated camera rig: static, translated textured plane, near-planar depth scene, multiple independently moving foreground parts, one point temporarily occluded, and a sequence reversing motion. **All observations are actual simulated RGB-D fields**, not a splat renderer's own self-consistent predictions; synthetic GT is quarantined until output written.
- Verify distinct canonical `P_0` and `P_T`, independent start states/covariances, exact original-time indexing, signed velocity units and proper backward `Q` PSD. Time slider must show two *different endpoint point populations* transported toward the opposite endpoint.
- For real blade, inspect frames 1,2,5,10 from identical input RGB-D to locate the **first motion-tracking divergence**. Do not judge the new direct model by the old splat-image energy. Report true observable residuals, motion estimates, uncertainty, stationarity, missed measurements and real runtime.
- Try no global rigid motion assignment: points can move differently. A production run may be `INCOMPLETE_UNOBSERVABLE` when direct image fields provide insufficient information; do not insert correspondences or GT to force success.

### Gate D — GPU equivalence, bottlenecks and subsampling

- CUDA profiling for factor-1 blade and larger 145k-point scene, with all eight cameras: ms by stage and peak allocated/reserved MB, kernel summary, host synchronization, CPU preprocessing and data copying; include an independent CPU reference on small fixtures.
- Factor `1` yields bitwise-identical selected IDs and no difference in the canonical point catalogue. Factor `2`, `4`, `8` selects the exact target count of **real original point indices** (or reports why impossible), deterministic with same seed; the two anchors select separately. No camera, frame or mask reduction.
- Compare original and downsampled 6D IEKF observability/innovation/covariance, coherence graph occupancy and held-out rigid validation; explicitly quantify the trade-off. Faster but physically invalid runs do not count as a successful acceleration.
- A large scene must either complete within configured GPU memory or fail with a precise bottleneck allocation and safe point-chunk fallback; no silent OOM avoidance by disabling required observations.
- Measure any new custom CUDA/Triton kernel against the pure PyTorch CUDA reference; retain a fallback and a documented build path. No GPU performance claims without actual measured timings.

### Gate E — coherence, independent for F1 and B1

- Preserve the previously validated pinned RigEuler representative graph/seed/refinement code; match CPU reference on identical synthetic tracks. Compare raw vs optional supported tracks explicitly.
- Save all candidates, selected **seed** and subsequent **recruited** nodes; distinguish stationary core selection from moving selection; fail or mark unresolved if the full graph or static majority is selected erroneously rather than modifying tests/thresholds.
- Verify neither GT nor joint geometry enters this stage. If Phase 1 produces fragmented nonrigid tracks, report as upstream limitation; do not “fix” it by forcing a part-size prior.

### Gate F — forward and backward shared rigid 12D IEKF

- Full 12×12 Lie-group prediction and covariance on CPU and CUDA; test spatial `Exp/Log`, adjoint, left Jacobian, signed-time process noise, 3D-to-SE(3) point Jacobian, fixed-prior iterated update and covariance chart reset with finite differences.
- Test exact pairwise core-distance invariance for all rigid frames; all camera samples originate from measured next-frame RGB-D and use the same `H_point_x` as the free filter.
- Verify Phase-2 filtered pose is a new Kalman estimate from actual RGB-D, not a Kabsch copy of free trajectories or a renderer optimization. The static foreground remains distinct from the moving core; occluded core points contribute no false measurements.
- Forward rigid model uses `P_0` core and backward model `P_T` core; neither may refer to the other direction's point IDs. Independently inspect 3D poses, actual RGB-D overlays, innovation/NIS and full rigid covariance.

### Gate G — joint and support (only after F1/B1/F2/B2 accepted)

- Recover fixed revolute/prismatic geometry on independent analytic sequences; verify constant axis under variable speed, correct `q(t)` sign/time alignment, discriminability vs static/no joint, and report degeneracy/uncertainty. Test optional constrained `q,qdot` IEKF against independent small Kalman oracle.
- Support completion labels actual endpoint points from measured RGB-D evidence and keeps occluded points ambiguous. No GT mask leak, no rendered-image Gibbs pseudo-likelihood and no implicit point matching across endpoints.
- Bidirectional comparison uses a common calibrated world frame without multiplying overlapping-image posteriors or inventing cross-endpoint IDs.

### Gate H — real visual and numerical debugging

- Screenshot and inspect real RGB-D plus overlays in both directions at selected frames. Verify numeric sampling/backprojection/projection at selected real point IDs and the displayed actual data at source pixel coordinates.
- Prove sliders, camera selector, point IDs, labels and per-frame displayed covariance really change, and that backward point population is canonical at `T`. Provide linked PNG screenshots and machine-readable per-point IEKF trace, not just an “HTML opened successfully” message.
- No synthesized splat RGB/D in any new inference/debug page. GT-based plots separate under `evaluation_gt/`.

---

## 12. Working discipline and definition of completion

This is an **implementation request**, not a request to propose another approach. Codex should create the new code **incrementally**, adhering to the gate order below; this prevents another apparently successful joint-axis result from masking a wrong Phase-1 measurement filter.

1. **Audit current HEAD and save baseline**. Produce `reports/projective_rgbd_migration_audit.md` with concrete existing imports/functions/sources to retire, source hash, worktree status and CUDA hardware. Do not overwrite the prior good/bad experimental result folders.
2. **Stage A:** Implement measured sensor geometry and sampling (`projective_camera`, `projective_sensor`, visibility) and validate Gate A, including actual source-image debug. No filter, coherence, or joint modifications at this step.
3. **Stage B:** Implement batched CPU/CUDA independent **6D IEKF** + physically dimensioned signed process noise + Joseph covariance, using those actual measured RGB-D samples, not the old `render`. Validate Gate B and first-frame visual/numerical trace.
4. **Stage C:** Implement **F1 and B1** with independent canonical point clouds, real timestamps, all cameras, hard foreground masks and separate outputs. Test Gate C. Do not proceed while the direct RGB-D innovation/Jacobian is wrong or the filter still calls a splat function.
5. **Stage D:** Add deterministic user-selected **point-cloud subsampling factor**, GPU optimization, profiling and comparisons; validate Gate D without changing the accepted measurement or filter equations. Keep `factor=1` as production reference.
6. **Stage E:** Connect completed free tracks to the already audited independent 192-node RigEuler-style coherence graph; preserve initial core/recruitment diagnostics and validate Gate E without tweaking model selection to match GT.
7. **Stage F:** Implement distinct **F2 and B2** shared rigid pose/twist IEKFs, using the **same** measured projective RGB-D innovation and chain-rule Jacobians, with Lie covariance reset. Test Gate F before joint code.
8. **Stage G:** Implement/convert downstream fixed-joint reduction, optional q-filter, support completion and honest bidirectional comparison. Test Gate G; do not claim their correctness from unit tests of the previous splat-based pipeline.
9. **Stage H:** Complete the RigEuler-quality **point-only measured RGB-D visual and numerical debug**, full forward/backward interactive pages, screenshots and numeric traces; evaluate Gate H throughout the stages rather than tacking visuals on at the end.
10. **Final integrated run**: after gates pass, run a named real blade and at least one revolute and one static synthetic sequence with *both free directions and both rigid directions*, first factor 1 and then a documented factor 4 comparison. Provide measured wall time/GPU time, peak VRAM, per-stage model status, uncertainty validity, first failing gate and debug index. Preserve separate predictions and post-hoc evaluation results.

### Required final report

Produce `reports/projective_four_pass_cuda_implementation.md`, containing:

- exact commit IDs/base checkout and status of changed files; full source files/functions replaced or added;
- the final **actual** equations implemented for free `F,Q,r,H,R,K,P+` and rigid `Exp,Ad,F,H_rigid,K,Joseph/reset`, with exact sign and coordinate conventions;
- proof **no production dependency** calls `rgbd_point_renderer.render`, `FrameObservation.terms`, splat coverage, Student-t image energy, old `GroupBasis` physical correction, or cross-frame point matching;
- explicit canonical point counts, selected sampled ID maps for both endpoints, separate execution orders and arrays;
- CPU double oracle + CUDA results, Jacobian finite-difference errors, Joseph PSD/NIS and all-mask/occlusion tests;
- full GPU profiling tables for the 34k and large scene, subsampling accuracy/speed/memory trade-offs for factors 1/2/4/8;
- free forward/backward and rigid forward/backward results, each with actual observed-image projections, uncertainty, timings and failure status;
- coherent core selection with seed/refinement, independent joint hypotheses and common-gauge bidirectional axis checks;
- verified working links/paths for 3D point, measured RGB/depth overlay, coherence, rigid, joint and numeric debug pages, with representative inspected screenshots;
- a candid **Implemented / Tested / Passing / Failing / Not implemented** table. **Do not write “complete” if any mandatory directional Kalman pass has no valid moving core or fails a gate.**

**Do not** spend additional time repairing the retired rendered-image objective, maintaining a fake-Gaussian “feature” to make RGB-D equations fit, reintroducing point correspondence, or tuning a final joint-axis score to disguise weak free tracking. If an actual RGB-D-only projective IEKF cannot observe a particular motion, state the mathematical limitation and numerical evidence rather than adding an unauthorized sensor/model.

**One-line invariant:** The only moving geometry in the four filters is the **original measured endpoint XYZ** transported by either independent free-point states or a single shared rigid SE(3) state; the only observations are the **next actual masked calibrated RGB-D frames** sampled at the transported points' projections; every correction is a mathematically defined **Kalman** correction with valid covariance.
