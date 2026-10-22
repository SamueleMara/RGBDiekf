# CODEX — REBUILD `RGBDekf` AS THE DIRECT, SPARSE, BIDIRECTIONAL RGB-D IEKF REQUESTED BY THE USER

> **Priority P0. This supersedes all earlier Codex files wherever they introduce an eager observation catalogue, enumerate scene configurations, treat each RGB-D pixel as a permanent map variable, allocate one map copy per body, or substitute an enormous joint optimizer for the sparse IEKF.**
>
> **Task:** Modify the *actual* repository `/home/samuelemara/Joint3DGS/RGBDekf`. Do not deliver a fresh unrelated prototype. Inspect and edit its real source, run its tests, and show a genuine end-to-end result on the supplied RGB-D dataset. **Never claim success because a mathematical unit test passes while the real reconstruction does not run.**
>
> The central algorithm is intentionally simple: **(1) construct one compact surface model from initially visible, masked RGB-D points; (2) analytically estimate its rigid motion through the entire multiview RGB-D sequence using a sparse IEKF; (3) compare its projection against later RGB/depth observations; (4) take *surfaces genuinely not represented by the current model* and project them backward under the same estimated rigid motions; (5) fuse them into the SAME canonical surface model; (6) repeat forward IEKF and backward geometry refinement until the common RGB-D objective ceases to improve.** Point appearance/disappearance is a visibility question, not permission to allocate one latent 3D point for every observation in every image.

---

## 0. Diagnose the implementation that must be replaced

The user's report of the present repository describes:

- `rgbd/geometry.py:56`: **197,919,833** independent depth-derived proposals, one per valid pixel, including unmasked environment points and duplicate observations of the same surface.
- Two copies of XYZ/RGB and fixed support-edge hypotheses per proposal; initialization stacks all frames while moving poses are still identity.
- `rgbd/objective.py:82` and `rgbd/operators.py:40`: dense-length map gradients and trial vectors over roughly **2.375 billion map scalar unknowns**. Matrix-free multiplication does NOT make this representation sparse or acceptable.
- A renderer iterating through all proposals for each image *and* every discrete scene configuration.
- `rgbd/posterior.py:23`: five hard-coded whole-scene patterns, including artificial periodic label patterns. No geometric part discovery.
- A run that did not complete even one full objective evaluation.

**The Codex report is the available evidence; verify against the actual files before editing.** The desired remedy is **not** float32 conversion, chunking the 197.9-million-proposal catalogue, using a bigger SSD, or a more elaborate scene enumeration. **Replace the state and data flow.**

### Delete these architectural choices

1. `proposal = every (frame, camera, valid depth pixel)` as a persistent variable. Observation pixels are measurements, not map elements.
2. `xyz_and_rgb[N_observations, N_parts, 6]`. There is one canonical `xyz[N_map,3]`, `rgb[N_map,3]`, surface normal/support, uncertainty, and part probability per **fused** surface element. Uncertain part ownership does not justify cloning all geometry once per part.
3. All `3**N`, whole-scene enumeration, all-absent/all-static/all-moving patterns, and index-based part patterns. These are NOT required by the user's original sparse IEKF.
4. Any residual or Jacobian vector with dimensionality proportional to `num_frames * num_cameras * num_pixels * num_parts` as a **persistent state**.
5. Rendering the complete map independently for each hypothetical global scene configuration.
6. Any fake "backward refinement" that merely reverses an array, without transporting late RGB-D surfaces backward and checking earlier visibility.
7. Any claim of “exact inference” for an exponential discrete posterior, or “proved global reconstruction convergence.”

**Keep/reuse only mathematically correct code**: camera loader and its verified conventions, `SE(3)` routines, analytical Jacobians, small-state IEKF/RTS utilities, lossless ray-local operators, and tests. Verify every reused routine, rather than assuming it is correct.

---

## 1. The EXACT requested geometric interpretation

Let \(k=0,\ldots,K\) be acquisition times, \(c\) calibrated cameras, \(u\) pixels, and

\[
Y_{c,k}(u)=\left(I_{c,k}(u),D_{c,k}(u),M_{c,k}(u)\right).
\]

Here `M` is the **dataset foreground/object mask when available**. Use it as the observation domain. It is NOT ground-truth moving-part segmentation. A mask that includes both the fixed object base and movable section is correct.

A single canonical geometric surface map is

\[
\mathcal M=\{(p_i,a_i,n_i,S_i,\Sigma_i,\pi_i)\}_{i=1}^{N},
\]

where `p_i` is a 3D point in the component's reference coordinates, `a_i` is RGB, `n_i` is a local surface normal, `S_i` is a finite observed pixel/surfel support, `Σ_i` is geometric uncertainty, and `π_i[b]` is its ownership probability over static and moving rigid components. `N` is the number of **fused persistent surface samples**, not the number of observations across time.

Static part `b=0`: \(T_{0,k}=I\). Dynamic part(s) `b=1,...,B`: \(T_{b,k}\in SE(3)\). Start with `B=1` for the simplest one-joint scene, support configurable `B` without creating copies of the whole map. The rig is calibrated and fixed (if extrinsics are time-dependent, honor that explicitly).

Conditional on ownership \(b_i=b\), the ONLY dynamic position is

\[
\boxed{x_{i,k}=T_{b,k}p_i.}\tag{1}
\]

For a point measured at source time \(s\) and later reprojected to time \(k\):

\[
\boxed{x_{i,k}=T_{b,k}T_{b,s}^{-1}y_{i,s}.}\tag{2}
\]

Equation (2) works identically forward and backward. **No `xyz[i,k]` state, no independent per-point velocity, no independent trajectories for observations.** Only the shared rigid transformation moves each component.

A late-observed point may have multiple *temporary* component hypotheses. Do not allocate a separate permanent `p_i[b]` for every single original pixel. Resolve/fuse compatible hypotheses using the measurements; keep uncertainty when not identifiable.

### Non-negotiable initialization

1. Read *the first actual frame* across all available cameras; use foreground masks.
2. Backproject measured depth to world coordinates with the correct camera calibration/metric depth.
3. **Fuse overlap between cameras immediately**, in the common world/reference frame, accounting for observation footprint and covariance. Do not retain independent duplicates of identical surface patches just because different cameras produced them.
4. Store this as `canonical_map`. It may be incomplete; this is intentional.
5. The first-frame map is a measurement-based initialization, not an immutable truth. Its uncertainty comes from depth + calibration. Do not repeatedly reuse the same source image as both an independent map prior and a measurement update.
6. Do **not** stack all future RGB-D frames before motion estimation. They are not in the same body reference coordinates until motion is estimated.

### Fusion of surface samples (deterministic; no arbitrary novelty threshold)

- A depth pixel is a sample of a *surface patch*, not an infinitely thin point with no support. Derive initial support size from camera pixel angular footprint and measured depth (and surface orientation when available); do not choose a free-space metric radius unrelated to the sensor.
- Use a camera/local geometric index (voxel hash with cell size derived from the footprint, or `cKDTree`) to retrieve only nearby candidate surfels. The index is a data structure, not a second point catalogue.
- For an observation `y` and compatible surfel `p`, evaluate a measurement likelihood using its position, plane normal, RGB and propagated measurement covariance. Evaluate `new-surface` as the alternative under an explicitly declared prior. Decide MAP association deterministically or retain local posterior weights. **No `||y-p|| > hand_chosen_threshold`, no index modulo segmentation.**
- Fuse multiple observations into one persistent surface element using uncertainty-weighted information accumulation **only conditional on a fixed motion/correspondence hypothesis**. Retain the shared pose uncertainty separately (see §8); do not count correlated observations as independent.
- Prevent uncontrolled map growth via physical surface fusion and footprint coverage, **not by a silent `max_candidates=4/10000` cap**.
- Fusing two samples of *different* parts just because they overlap in one configuration is incorrect: require cross-time geometric/appearance/rigid-motion consistency.

---

## 2. EXACT low-dimensional rigid-motion EKF state and process

For every moving rigid part use the **12-dimensional pose + twist state** (translation-first twist):

\[
\boxed{s_{b,k}=(T_{b,k},\nu_{b,k}),\qquad \delta s_{b,k}=(\delta\xi_{b,k},\delta\nu_{b,k})\in\mathbb R^{12}.}\tag{3}
\]

Use a left-multiplicative pose error \(T=\operatorname{Exp}(\widehat{\delta\xi})\bar T\), additive twist error, and **one consistent tangent convention everywhere**.

Nonlinear state equation:

\[
\boxed{\begin{aligned}
T_{b,k+1}&=\operatorname{Exp}(\Delta t_k\widehat\nu_{b,k})T_{b,k}\operatorname{Exp}(\widehat w^T_{b,k}),\\
\nu_{b,k+1}&=\nu_{b,k}+w^\nu_{b,k}.
\end{aligned}}\tag{4}
\]

Set \(\phi_k=\Delta t_k\nu_{b,k}\), \(A_k=\operatorname{Exp}(\widehat\phi_k)\). The first-order error-state matrices under this convention are

\[
\boxed{F_{b,k}=\begin{bmatrix}\operatorname{Ad}_{A_k}&\Delta t_kJ_l(\phi_k)\\0&I_6\end{bmatrix},\qquad G_{b,k}=\begin{bmatrix}\operatorname{Ad}_{\bar T_{b,k+1|k}}&0\\0&I_6\end{bmatrix}.}\tag{5}
\]

\[
\boxed{P^-_{k+1}=F_kP_k^+F_k^\top+G_kQ_kG_k^\top.}\tag{6}
\]

`P_k` is **12B × 12B**, not a covariance over all map points. The full body cross-covariance is allowed: measurements may couple bodies. Gauge-fixed reference poses are not estimated as free variables.

Time: read `fps`/timestamps from the **actual selected scene directory** (not from a different uploaded sample); \(\Delta t_k\) from timestamps or `1/fps`. If absent, support an explicit override; never invent playback FPS. Noise: use the user-approved initial guesses from the earlier documented config, with units and provenance. Do not invent missing calibration as measured fact.

### Motion initialization must not be silently all-zero forever

An identity first pose is a valid gauge, but setting every future moving pose to identity with no observed-motion initialization can leave the estimator in a degenerate local optimum.

- Initialize candidate relative rigid motion from **measured cross-frame RGB-D correspondences**, with fixed multiview calibration: e.g. deterministic optical-flow/feature correspondences followed by weighted rigid Procrustes/SE(3) fitting; use local geometrical coherence to generate a static and a moving hypothesis. Preserve all derived measurements as *initializers*, not pseudo-ground-truth factors.
- Do not use the dataset's joint-angle CSV, true part labels, simulated object poses, or dynamic-point ground truth for inference. They are evaluation-only.
- If visual data do not determine a nonidentity motion, report the observability problem rather than fabricating movement or replacing the IEKF with ground truth.
- If more than one moving component is configured, initialize distinct candidate rigid motions from geometric evidence, not alternating array indexes.

---

## 3. DIRECT RGB-D OUTPUT EQUATION, NOT A 3DGS RENDERER

For surfel point `i` associated with moving body `b`, and camera `c`, define

\[
q_{i,c,k}=R_{cw,c}(R_{b,k}p_i+t_{b,k})+t_{cw,c},\qquad
\hat u_{i,c,k}=\pi_c(q_{i,c,k}),\quad \hat d_{i,c,k}=e_3^\top q_{i,c,k}.\tag{7}
\]

For a predicted **visible** surface with a compatible observed patch:

\[
\boxed{r_{i,c,k}=\begin{bmatrix}I_{c,k}(\hat u_{i,c,k})-a_i\\D_{c,k}(\hat u_{i,c,k})-\hat d_{i,c,k}\end{bmatrix}.}\tag{8}
\]

For image sampling with a left world-pose perturbation, let \(x=R_bp_i+t_b\),
\[
J_x=[I_3,-[x]_\times],\quad
J_\pi=\begin{bmatrix}f_x/Z&0&-f_xX/Z^2\\0&f_y/Z&-f_yY/Z^2\end{bmatrix}.
\]

Then

\[
\boxed{H_{\mathrm{pose}}^{RGB}=\nabla I(\hat u)J_\pi R_{cw}J_x,\quad
H_{\mathrm{pose}}^{D}=(\nabla D(\hat u)J_\pi-e_3^\top)R_{cw}J_x.}\tag{9}
\]

The instantaneous twist columns are zero; twist is informed through temporal state coupling. The canonical point Jacobian follows the same chain with \(J_x\) replaced by \(R_b\); source-time reverse transport requires the chain rule through both endpoint transformations. Verify all using finite differences at nonzero rotation/translation and irregular \(\Delta t\).

**Visibility is physically determined by depth ordering, not by applying a pixel RGB residual to an occluded point.**

- Project only map elements whose finite measured support intersects the current image. Use a deterministic z-buffer of *oriented surfel supports* or the equivalent local first-hit ray model. This is simple surfel point projection, NOT Gaussian splatting or 3DGS alpha compositing.
- At any pixel, nearer measured depth can explain why a predicted deeper surface is hidden. Hidden points yield no spurious RGB correction.
- Predicted geometry in valid measured free space gives a contradiction likelihood; not a forced cross-surface match.
- No-return, out-of-FOV and missing depth are distinct cases and must be modeled as such.
- For uncertain relative depth, calculate depth-ordering probabilities via the specified noise distribution, e.g.

\[
P(z_{\mathrm{candidate}}>D_{\mathrm{measured}})=
\Phi\left(\frac{z_{\mathrm{candidate}}-D_{\mathrm{measured}}}{\sqrt{\sigma_z^2+\sigma_D^2}}\right).\tag{10}
\]

Use the **same** RGB-D/no-return/visibility sensor likelihood in forward correction and backward geometry evidence. RGB residuals apply only when the candidate is visible and sampled at a compatible surface.

### One direct projection per frame and camera

For each frame and camera: project the current **fused canonical map once** at its estimated part poses; resolve depth/visibility per pixel/tile; gather sparse reprojection residuals; accumulate a **small 12B-state** information update. **Do not scan the entire map again for five fabricated global scene assignments.** Use GPU/CPU tiled point projection, range-limited/support-culling, and batch reductions. All foreground observations must be accounted for across the run, with streaming; no all-frame ray tensor or 984 image render cache in RAM.

---

## 4. SPARSE FORWARD IEKF — EXACT USER INTENT

During Stage A, **hold the canonical map fixed**. Optimize only the common rigid motions. Use real observed RGB/depth residuals and the source IEKF process equations.

For each frame:

1. Propagate `T`, `nu` and small `P` using (4)–(6).
2. Generate projected visible map contributions from the current pose.
3. For each valid foreground observation supported by the map, evaluate (8), noise covariance `R`, visibility responsibility, and analytical `H` from (9).
4. **Accumulate** ray-local score and information without storing the full Jacobian:

\[
A_k=P_{k|k-1}^{-1}+\sum_{r\in\mathcal R_k}H_r^\top W_r H_r,\qquad
b_k=\sum_r H_r^\top W_r r_r.\tag{11}
\]

Use consistent residual sign: with (8) as `observed - predicted`, the linear residual derivative shown in (9) is `∂r/∂xi`; the right-hand side of a least-squares normal system is **\(-\sum H^\top Wr\)**. Equivalently redefine `H=∂prediction/∂xi` and use \(+\sum H^\top Wr\). **Pick one convention, validate it by objective reduction, and do not silently mix them.**

5. Iterate the **same** frame prediction/prior with a damped IEKF update and reproject/relinearize after each accepted correction; do not count the same measurements twice as independent updates.
6. Compute posterior covariance from the **undamped** local information at the accepted linearization, with tangent-chart transport as needed; numerical LM damping is not Bayesian information.
7. Save predicted/filtered means, \(P^-_k,P^+_k,F_k\), and the motion score.

The full output distribution may be a visibility/association mixture; if optimizing that, use its *correct analytic score and local information* rather than pretending fractional responsibilities automatically yield an exact Gaussian EKF covariance. The user requested deterministic formulas, NOT global discrete enumeration. An exact local first-hit marginalization is permitted; avoid unproven claims about the global discrete posterior.

**Acceptable small dense objects:** `12B × 12B` pose/twist covariance or normal matrix. **Forbidden:** any dense `N_map × N_map`, `(12B + 3N_map)^2`, `N_observations × N_observations`, or a vector of all proposed points at all timestamps.

---

## 5. TRUE BACKWARD RTS MOTION REFINEMENT

After the complete forward pass, perform an actual backward smoother over the same state trajectory:

\[
G_k^{RTS}=P_{k|k}F_k^\top(P_{k+1|k})^{-1},\tag{12}
\]

\[
\hat s_{k|K}=\hat s_{k|k}\boxplus
G_k^{RTS}(\hat s_{k+1|K}\boxminus\hat s_{k+1|k}),\tag{13}
\]

\[
P_{k|K}=P_{k|k}+G_k^{RTS}(P_{k+1|K}-P_{k+1|k})(G_k^{RTS})^\top.\tag{14}
\]

In a Lie-group implementation, transport the local errors/covariances between tangent charts correctly. An RTS smoother is not equivalent to iterating a list backward and never updating earlier states. The smoother must actually change earlier estimates when later evidence is added. Verify against a small independent linear-Gaussian batch posterior.

**Map uncertainty:** pose-only RTS conditional on a frozen map is not the same as joint marginal uncertainty. For joint covariance queries use sparse motion–map factors (§8); do not report conditional covariance as full map-marginal covariance.

---

## 6. TRUE BACKWARD GEOMETRY DISCOVERY — THE MAIN ALGORITHM

**Stage B runs AFTER Stage A motion is estimated; freeze its rigid motions while proposing and fusing geometry.** This is what makes incorrect motion and missing surface distinct estimation problems.

1. Start at the final frame (`k=K`). Project the **current transported canonical map** into every final-frame camera to produce `RGB_pred`, `Depth_pred`, `coverage`, visibility and the RGB-D evidence per valid foreground pixel.
2. For valid observed foreground depth that is *not explained by the current projected map*, evaluate a **new-surface** hypothesis using the same normalized RGB-D likelihood, not an arbitrary distance threshold. `not explained` = posterior preference for a new surface over all existing compatible surface explanations (or retain a continuous novelty responsibility; do not require a hard gate).
3. Backproject the observed location \(y_{c,k}(u)\). For each geometrically plausible part `b`, form the canonical hypothesis

\[
\boxed{p_{new}^{(b)}=T_{b,k}^{-1}y_{c,k}(u).}\tag{15}
\]

4. Reproject **that SAME point** into all earlier frames using

\[
\boxed{x_{new,j}^{(b)}=T_{b,j}T_{b,k}^{-1}y_{c,k}(u),\quad j<k.}\tag{16}
\]

5. Evaluate, in *each calibrated camera* in each earlier frame, whether it (a) matches an observed surface in RGB/depth, (b) was behind a measured closer surface, (c) lies in genuinely observed free space (contradiction), or (d) was out of view/unmeasured (uninformative). Use covariance-normalized likelihoods, not `depth > τ`.
6. Combine the evidence across times **without counting a source pixel twice**. Use the original raw images once in the common objective. Store posterior existence and part probabilities. A point observed once and occluded everywhere else may remain ambiguous; do not invent its motion.
7. **Fuse** accepted/supported observations and repeated views into the SAME canonical map. The map is one set of surface elements, with uncertainty/provenance. Later cameras and frames observe and refine the SAME surfel; do not create a new persistent surfel for each subsequent pixel.
8. Continue reverse scan `K,K-1,...,0` to discover surfaces first visible at intermediate times (not only those visible in the last frame). Use camera overlap fusion at each source time.
9. Maintain contribution of existing points; newly discovered points must not overwrite earlier ones or form a separate incompatible "backward map." At the end there is one canonical map and one motion trajectory.
10. Re-enter the **forward** IEKF with the enlarged/fused map, then RTS, then backward discovery again. All outer iterations use the same raw measurements, not the previous posterior as an independent prior.

### New vs old is local surface coverage, not new per-pixel catalogue

Do NOT create one proposal for *every valid depth pixel* and then hope a posterior will prune it. Iterate observed pixels/patches in a streamed geometry-discovery pass, query the projected map for coverage/compatibility, and **immediately associate/fuse observations with existing geometry**. Only genuinely unrepresented surface regions contribute new persistent surface elements. Aggregate spatially overlapping evidence from cameras/frames in a bounded local working buffer, then commit fused surfels. No global eager `observations × parts` tensor.

### Deterministic inference, not fabricated scene patterns

- Infer per-surface probability \(\pi_i[b]\) from the likelihood that its projected RGB-D trajectory belongs to part `b`, plus the fixed declared ownership prior. A deterministic update is

\[
\pi_i^{new}(b)=\frac{\pi_i^{prior}(b)L_i(b)}{\sum_{b'}\pi_i^{prior}(b')L_i(b')},\tag{17}
\]

where `L_i(b)` integrates RGB-D matches, depth occlusions, and observed free-space contradictions across all frames where information is available. Correctly handle correlation when shared poses are uncertain; do not assume 984 views produce independent evidence about a single uncertain transform.
- Segmentation **must be geometry/motion-derived**, not a fixed pattern of catalogue indices. Unknown points can retain soft membership until more evidence arrives.
- The motion pass may use conditional MAP ownership or appropriately marginalized responsibilities, but changes to assignments are performed as a **separate outer-loop operation**, not hidden inside the pose update.
- Avoid claiming that a product of per-point ownership probabilities is the *exact full coupled scene posterior* when occlusion or shared map uncertainty couples them. Exactness applies to the chosen local calculation and linearized solve, not arbitrary exponential global inference.

---

## 7. OUTER LOOP, COMMON OBJECTIVE, AND STOPPING

```text
load complete sequence metadata, cameras, masks, timing
validate calibration / RGBD / noise initialization
M = FUSE_RGBD_ACROSS_CAMERAS(frame=0, foreground_only=True)
T, nu, P = INITIALIZE_RIGID_MOTION_FROM_RGBD_CORRESPONDENCES(M, sequence)
repeat outer = 0,1,...:
    freeze(M)
    for k = 0 .. K:
        IEKF_predict(T, nu, P, dt[k])
        project_and_zbuffer(M, T[k], cameras_at_k)
        IEKF_update_from_visible_RGBD_residuals_analytically()
    RTS_smooth_from(K to 0)
    freeze(T_smooth)
    for k = K .. 0:
        project_and_zbuffer(M, T_smooth[k], cameras_at_k)
        evaluate_unexplained_measured_surface_likelihoods()
        inverse_transport_late_observations_via_same_T()
        test_earlier_visibility_and_RGBD_consistency()
        update_existence_and_part_probabilities()
        fuse compatible surface observations into M
    evaluate_same_full_sequence_RGBD_objective_and_motion_prior()
    accept/reject or damp update against that objective
    save trajectory/map/diagnostics/checkpoint
until a documented numerical stationarity criterion is satisfied
```

No fixed number of outer passes is a substitute for checking convergence. A `--max-outer` **safety ceiling** is allowed and must report `NON_CONVERGED` when reached; do not call reaching the ceiling convergence. A numerical objective/gradient tolerance for solver stopping is not a geometric motion/novelty threshold.

### Objective and mathematical honesty

Optimize the same observed-image model throughout:

\[
\mathcal J(\mathcal M,T,\pi)=
-\sum_{k,c,u\in\text{valid ROI}}\log p(I_{c,k}(u),D_{c,k}(u)\mid\mathcal M,T_k,\pi)
+\tfrac12\sum_k\|s_{k+1}\boxminus f_k(s_k)\|_{Q_k^{-1}}^2
+E_{\text{map prior}}.\tag{18}
\]

Use one ray/sample likelihood per raw RGB-D measurement, including valid foreground no-return/occlusion terms. Keep the map-complexity/existence prior explicit so zero-residual one-point-per-pixel overfitting is not automatically rewarded. For optional pixel subsampling **only in a diagnostic test**, account for sampling weights; production shall stream full selected foreground observations.

For fixed map/associations, a damped IEKF update accepted against the actual nonlinear stage objective has standard local descent properties. When the map expands, do **not** misapply a theorem for fixed state dimension; accept changes against the common posterior/description-length objective with explicit priors and report its value. No claim of global physical recovery or unique segmentation under permanent occlusion.

---

## 8. UNCERTAINTY: SMALL POSE COVARIANCE + SPARSE MAP COUPLING

**Never store the inverse of a giant joint covariance.**

- For each point initialized from measured depth \(y\) under pose \(T_s\), propagate measurement and pose uncertainty into canonical coordinates:

\[
\delta p=J_y\delta y+J_s\delta s,\quad
P_{pp}=J_yR_yJ_y^\top+J_sP_{ss}J_s^\top,\quad
P_{sp}=P_{ss}J_s^\top,\tag{19}
\]

with appropriate cross terms when measurement and pose are correlated. This initialization is **not** an independent prior if the same source RGB-D measurement remains included in the estimator; avoid double counting.
- For two new surfels sharing the same uncertain source pose, their errors are correlated through that shared pose. **Do not construct a huge full matrix to represent it.** Preserve the common pose variable and its sparse factors; marginal/cross-covariance queries follow from information solves.
- Joint linearized information has sparse blocks:

\[
\begin{bmatrix}A_{ss}&A_{sm}\\A_{ms}&A_{mm}\end{bmatrix}
\begin{bmatrix}\delta s\\\delta m\end{bmatrix}
=\begin{bmatrix}b_s\\b_m\end{bmatrix}.\tag{20}
\]

- If geometry is conditionally independent given the shared body poses and ownership, \(A_{mm}\) consists of small per-surfel blocks; otherwise its extra sparse off-diagonals must be retained. Eliminate surfel blocks with a Schur complement or apply the joint operator matrix-free. Do not pretend \(A_{mm}\) is diagonal after adding explicit spatial coupling.
- Implement `H @ v` and `H.T @ w` by accumulating contributions only for projected **visible/local-support surfels and current-body state blocks**. Never construct a global Jacobian.
- Permit small dense `12B × 12B` filtering covariance for each time and sparse temporal smoothing factors. For full joint/marginal uncertainty, use sparse selected solves; label `P_motion|map` versus `P_motion marginal` honestly.
- Noise initialization must use the earlier user-approved assumptions unless independent calibration is available. Verify symmetry, PSD/PD, units, process-noise time scaling, and Joseph-form consistency in linear tests. Never treat LM damping as measurement information.

---

## 9. PERFORMANCE CONTRACT: WHAT REAL SPARSITY MEANS

Dataset size can be hundreds of millions of foreground **observations** over time. This does **not** imply hundreds of millions of persistent geometry variables.

Required asymptotic behavior (for `N_map` fused surfels, `F` frames, `C` cameras, `R` observed foreground pixels over all frames, `B` bodies):

- Persistent map: \(O(N_{map})\) geometry + local uncertainty + ownership, **independent of F and C after fusion**.
- Motion state: \(O(12BF)\), per-step small covariance at most \(O((12B)^2)\).
- Observation pass: streaming \(O(R+FC\cdot N_{map,projected})\) up to index/render complexity, not `R × (number of full-scene configurations)` and not `R × N_map` brute-force nearest neighbors.
- Temporary buffers: limited to one/tiled images, active projected points, local correspondences, and small accumulated information matrices.
- CPU/GPU: use NumPy/SciPy/OpenCV/Open3D/PyTorch already in `rigeuler3dgs`; 8GB GPU acceleration may be useful but correctness may NOT require the entire sequence/map on GPU.
- Do not allocate any array with leading dimension `N_observations × N_parts`. Do not build dense matrices in point-count dimensions. Use float64 for small inference matrices and as needed for stable reductions; float32 surface storage is allowed when backed by measured numerical error analysis.
- Support checkpoint + resume and progress indicators for frame/camera/tile, true IEKF iterations, geometry fusion, and per-outer-iteration global objective.

**Preflight must log:** number of frames/cameras, foreground vs background valid pixels, point count after fusing frame-0 cameras, current N_map and newly fused additions, projected visible coverage, RAM/GPU peak. Abort with an informative error before allocations that exceed the configured budget; do not present `RESOURCE_LIMIT` from an intentionally wrong global catalogue as a valid completion.

---

## 10. REQUIRED SOURCE MODULES AND EXACT RESPONSIBILITIES

Implement in the current `RGBDekf` tree (modify actual current files, preserving correct utilities):

```text
main.py                       CLI, full-scene run, restart/checkpoint
rgbd/dataloader.py            all cameras/frames/masks, fps/timestamps, calibration, metric depth
rgbd/geometry.py              compact canonical SurfelMap, local index, fuse/merge, no eager catalogue
rgbd/projection.py            project surfels + finite supports, z-buffer, RGB/depth predicted images
rgbd/lie.py                   SE(3) Exp/Log/Ad/J_l/F/G in one convention
rgbd/ekf.py                   actual recursive 12B pose/twist IEKF; stream ray factors
rgbd/smoother.py              RTS backward means/covariances
rgbd/visibility.py            depth ordering / RGB-D/no-return likelihood; analytic local derivatives
rgbd/backward_refinement.py   novel visible surface evidence; reverse transport; canonical fusion
rgbd/associations.py         deterministic local surfel ownership/existence probabilities
rgbd/operators.py            sparse/matrix-free pose+map JVP/VJP, optional Schur information queries
rgbd/outer_loop.py           stage separation, common objective, acceptance, convergence
rgbd/outputs.py              map PLY, trajectories, RGB/depth comparison and coverage metrics
rgbd/evaluation.py           optional GT evaluation; strictly separate from estimation
```

**Recommended minimal APIs** (use equivalent tested signatures where necessary):

```python
load_scene(path) -> SceneSequence
frame_observations(scene, k, c, foreground=True) -> RGBDFrame
build_initial_map(scene, frames=(0,)) -> SurfelMap
fuse_observations(map, measurements, T_fixed, part_hypotheses) -> FusionStats
project_map(map, body_poses, camera) -> ProjectionAndVisibility
predict_state(s, P, dt, Q) -> Prediction
iekf_update(pred, projected_map, rgbd_image, noise) -> FilterState
smooth_trajectory(filtered, predicted, transitions) -> SmoothedTrajectory
backward_discover(map, T_smoothed, scene) -> (UpdatedMap, EvidenceStats)
run_outer_loop(scene, cfg) -> ReconstructionResult
```

`backward_discover()` MUST use actual reverse projection (15)–(16) and measured depth-order/RGB evidence. `project_map()` MUST be called on the **fused map**, not on each raw pixel ever observed. `main.py` MUST coordinate all frames, not 0/middle/last by default.

---

## 11. MANDATORY TESTS: NO MORE SMALL-SCENE-ONLY SUCCESS CLAIMS

### A. Correctness on synthetic, transparent geometry

1. Two calibrated cameras see the same planar surface at frame 0. Initial fused map contains one physical surface representation, not two persistent copies per observed image.
2. One rigid object translates/rotates; a known rigid motion should make RGB-D reprojected residuals vanish within synthetic sensor precision.
3. Analytical residual Jacobians (9), process `F/G` (5), projection and backward-transport derivatives match central finite differences at nonzero rotation and nonuniform frame intervals.
4. A moving drawer exposes a previously hidden colored surface. Forward map cannot explain it; backward discovery inserts/fuses it into the SAME canonical model; its earlier projections lie behind foreground depth.
5. Inject a candidate point projected *in front* of measured depth where free space is known. Backward evidence must reject/decrease its probability relative to the correct explanation.
6. Surface observed in frames 0 and K only: correct transport across missing middle observations; absence due to occlusion must not create RGB residuals.
7. Two-frame repeat observations of one surfel must **not** create two persistent map elements. Two different body surfaces overlapping in one frame must **not** be fused if other frames distinguish them.
8. Moving/static segmentation responds to rigid-motion evidence and not to pixel index, source frame, mask shape, or hand-coded pattern.

### B. Sparse IEKF / covariance

9. Small linear Gaussian reference: forward EKF + backward RTS match batch posterior; correct tangent transport in nonlinear case.
10. `H @ v` and `H.T @ w` pass the adjoint identity; per-ray local sparsity matches an explicit tiny Jacobian.
11. Schur elimination matches full small joint solve, including pose–point cross terms. New surfels sharing a pose produce nonzero cross-information (implicit representation acceptable).
12. Covariances remain PSD; prediction grows appropriately with process noise and Δt; Joseph-form check; no covariance reduction merely because a previously unobserved point was added.
13. No source-frame observation is counted twice; re-running outer iterations with unchanged state/data cannot falsely shrink covariance.
14. No dense global matrices, no arrays indexed by *all observation pixels × part hypotheses*, no `3**N` scene enumeration in real-scene mode.

### C. Full real-data acceptance (mandatory)

15. Load entire user-provided fridge scene **from its actual folder**, all discovered frames and cameras, original foreground masks, and timing. Report correct counts. The USB sample is only a separate regression fixture.
16. Complete at least **one actual forward IEKF pass, one actual RTS backward pass, one actual geometry-discovery reverse pass, and one subsequent forward IEKF pass** on the real scene. Partial initialization/indexing is NOT success.
17. Confirm `N_map` initially reflects fused foreground surface coverage and changes through actual geometry fusion, not `sum(all valid pixels over all frames)`. Give `N_map`, map size and memory at every outer iteration.
18. Export one canonical fused map PLY; one shared pose/twist trajectory; RGB/depth prediction-versus-observed images at early, middle, final frames; visible/occluded/new-surface maps; forward/backward motion and geometry residual curves.
19. Evaluate one common full-data objective for each accepted outer step. Report initial and final values, global convergence status, and real runtime/memory. Do NOT treat a selected-three-frame test as production validation.
20. If full run does not finish, identify the exact functional blocker, show a failing test, and do not claim a completed reconstruction.

### Dataset and evaluation guards

- The USB archive previously supplied has `metadata/cameras.json`, `rgb/cam_*/frame_*.png`, `depth_npy/...`, masks and metadata. Discover formats from the **real selected input folder** rather than hardcoding only USB layout.
- Read FPS or timestamps from that folder. Use initial noise assumptions from the prior approved document; log source/units.
- `frame_values.csv`, joint labels, joint axes and ground-truth part cloud files are **evaluation-only**. Never use them for initialization, labels or motion correction.

---

## 12. EXECUTION COMMAND, EXPECTED REPORT, AND NO-FURTHER-QUESTIONS POLICY

Build a reproducible launcher, with CLI paths properly quoted (the SSD path includes spaces):

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf
python main.py \
  --input "/ABSOLUTE/PATH/TO/fridge_SCENE" \
  --output "/media/samuelemara/Extreme SSD/Samuele/rgbd_iekf_results/fridge" \
  --all-frames \
  --all-cameras \
  --use-foreground-masks \
  --mode bidirectional-sparse-rgbd
```

CLI flag names may differ only if documented in `--help` and tested. Do not create guessed scene paths or require the user to edit Python source to select input/output. By default run all available frames and cameras. An explicit debug subset flag is acceptable **only** if the output and report say DEBUG.

When done, Codex must deliver:

1. The code diff for the actual repository (not a prose description alone).
2. Exact main command and config values; calibration/timing/noise provenance.
3. Before/after architecture comparison: original proposal count and allocation versus fused map size, storage and peak memory.
4. Numerical correctness test results, especially Jacobians, EKF/RTS covariance, source-frame nonduplication and backward-occlusion tests.
5. Full-real-scene stage logs and outputs; per-stage and per-outer objective values.
6. Honest limitations: unobservable geometric components cannot be uniquely recovered; nonlinear IEKF is local inference; no unsupported global-convergence claims.

**Do not ask the user for FPS if it is in the selected folder. Do not demand independently calibrated noise when explicit initial assumptions have already been authorized. Do not fabricate parameter values. Do not stop to re-implement an exponentially large global posterior. Begin by replacing the eager catalogue and do the real forward/backward pipeline.**

---

## Final architectural check (must be answered YES in the implementation report)

- [ ] Is there exactly **one compact fused canonical surface map**, not one map variable per image pixel per frame?
- [ ] Are real foreground masks applied to the observed RGB-D domain?
- [ ] Do all copies of a surface seen by different cameras/frames fuse into that common map under estimated rigid motion?
- [ ] Is the motion state **12B per frame** with `T,nu` and actual analytic IEKF prediction/update?
- [ ] Does every motion evaluation use actual image/depth projection and ray-local Jacobians?
- [ ] Does the backward stage actually reverse-transport newly revealed geometry and check previous occlusions/free space?
- [ ] Are missing geometry and motion corrected in **separate alternating stages**?
- [ ] Is there one temporal RTS smoother and one common full-sequence objective?
- [ ] Are covariance correlations retained implicitly through sparse joint factors, rather than ignored or materialized densely?
- [ ] Do static/moving labels come from geometric evidence, not fixed index patterns?
- [ ] Does the complete real-scene run finish stages and produce outputs, not just tests or initialization?

**If any box is unchecked, this does NOT implement the requested algorithm.**
