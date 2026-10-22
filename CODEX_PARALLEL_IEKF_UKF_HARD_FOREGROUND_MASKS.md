# MANDATORY CORRECTION — HARD FOREGROUND MASKS FOR BOTH INDEPENDENT RGB-D SLAM PIPELINES

**Status: normative, supersedes weaker mask wording in all earlier IEKF/UKF instructions.** Apply to `main_iekf.py` **and** `main_ukf.py`, and to every shared preprocessing, point matching, forward filtering, backward discovery, point fusion, and global-objective code path. Do not create a separate segmentation subsystem. **Keep the two inner filters mathematically independent**; the common outer solved/unsolved loop is unchanged.

## 0. The requirement — no ambiguity

The calibrated binary mask `M[k,c,u]` is a **hard specification of which observed pixels belong to the object/foreground**. It is not a semantic part label and not a moving/static label.

**Invariants:**

1. **Only** pixels with `M[k,c,u] == 1` and valid positive finite depth can initialize candidate 3D points, form an accepted geometric/RGB-D correspondence, support an IEKF or UKF motion measurement, be added to the solved map, or remain in `UNSOLVED[k]`.
2. A pixel with `M[k,c,u] == 0` is **never** a foreground target or source and **never** contributes RGB/depth *matching* residuals or information to the inner filter. Never create background map geometry. Background depth may be read **only as occlusion/free-space evidence**, per Section 3, without turning it into a foreground match.
3. A model point that is **predicted visible** in a particular camera/frame cannot be accepted at a projected pixel marked background. This is a **hard feasibility constraint on candidate poses and new map associations**, not a low-weight soft penalty. The visibility qualification is essential: an occluded point may project over background without becoming an observed foreground surface.
4. A missing or occluded SOLVED point remains in SOLVED; it is not converted to UNSOLVED merely because its current projection has no foreground match. `UNSOLVED` indexes **unclaimed foreground measurements**, not all missing predictions.
5. Perform these rules identically in the IEKF and UKF programs, for the same camera/time calibration, binary masks and observed pixel IDs.
6. Never dilate/erode/blur/inpaint foreground masks, change mask boundaries by a chosen margin, or use a guessed mask probability. Preserve genuine binary support and visibility handling. Nearest-neighbor sampling only for integer mask membership; do not bilinearly interpolate the mask.

If a dataset mask is not aligned to RGB/depth resolution, fail with a clear error (or use *documented, calibration-correct, nearest-neighbor* image registration when explicit camera registration is provided). Do not simply resize incompatible mask imagery to make code pass.

## 1. Input and mathematical notation

For each camera `c`, timestamp `k`, pixel `u`:

\[
 M_{ck}(u)\in\{0,1\},\quad I_{ck}(u)\in\mathbb R^3,\quad
 D_{ck}(u)\in\mathbb R\cup\{\mathrm{invalid}\}.
\]

Canonical valid measured foreground pixel set:

\[
\boxed{\Omega_{ck}^{F}=\{u:\ M_{ck}(u)=1,\ D_{ck}(u)>0,\ D_{ck}(u)\text{ finite}\}.}
\tag{M1}
\]

Corresponding allowed measured points (common world coordinates):

\[
\boxed{Y_{ck}^{F}=\{T_{wc}[D_{ck}(u)K_c^{-1}\widetilde{u}]:u\in\Omega_{ck}^{F}\}.}
\tag{M2}
\]

`M=1` and invalid depth is **foreground with no usable point observation**, not background or an allowed depth match. Retain its mask value separately for silhouette tests; treat unknown depth according to the declared invalid-return model, never as a 3D point.

Frame-0 initialization:

\[
\boxed{\mathrm{SOLVED}_0=\mathrm{FuseAcrossCameras}\big(\bigcup_c Y_{c0}^{F}\big).}
\tag{M3}
\]

For `k>0`, `UNSOLVED[k]` contains only raw `(k,c,u)` identifiers with `u in Omega^F` not owned by an accepted observation-to-point association; it never contains background observations.

Point `p_i` of group `g_i` predicts

\[
X_{ick}=T_{cw,c}T_{g_i,k}p_i,\quad
\hat u_{ick}=\pi_c(X_{ick}),\quad
\hat d_{ick}=e_3^\top X_{ick}.
\tag{M4}
\]

A proposed correspondence `(i,k,c,u)` belongs to the admissible set **only if**:

\[
\boxed{
 u\in\Omega_{ck}^{F},\quad
 \hat u_{ick}\text{ projects within the calibrated camera image},\quad
 \text{candidate is not known to be occluded at }(k,c).
}
\tag{M5}
\]

Additional RGB and depth compatibility, uniqueness, and assignment scores follow the *existing declared* raw RGB-D likelihood. Do not introduce fixed new residual gates.

## 2. What the masks mean for point-motion optimization

### 2.1 Hard observed-target constraint

Every raw target in both filters satisfies `(M == 1) and valid_depth`. For an accepted point-to-point association the raw observation is

\[
z_a=[u_x,u_y,D_{ck}(u),I_{ck}(u)^\top]^\top,\qquad u\in\Omega_{ck}^{F}.
\]

The smooth conditional reprojection function is unchanged:

\[
h_i(x)=[\pi_c(T_{cw,c}T_g p_i)^\top,e_3^\top T_{cw,c}T_g p_i,a_i^\top]^\top.
\]

In the IEKF, analytical Jacobians of `h` are computed only for **accepted masked-foreground association IDs**. In the UKF, each sigma point evaluates `h` for those **same fixed measured targets**, not a new background RGB/depth target. The outer matcher owns association changes; the inner filter does **not** invent new target pixels outside the mask.

### 2.2 Hard visibility/silhouette feasibility (not a mask-gradient loss)

Let `V_ick(T,map,raw_depth)` be the deterministic visibility status of a predicted model point in camera `c` at frame `k`, based on calibrated frustum, predicted z-buffer, and the observed first-hit depth: `visible`, `occluded`, or `unknown`.

- `visible`: predicted object point is first-hit and is not hidden behind another modeled point or a nearer observed foreground/background depth.
- `occluded`: there is credible closer observed/model geometry, so the model point must **not** be forced to match foreground RGB or mask at its projection.
- `unknown`: depth invalid/no-return, outside camera, or ambiguous visibility; **no hard negative visibility constraint** may be invented.

The hard feasible pose set is

\[
\boxed{
\mathcal C_{g,k}=\left\{T_{g,k}\in SE(3):\,
 V_{ick}=\mathrm{visible}\ \Longrightarrow\
 M_{ck}(\operatorname{pixel}(\hat u_{ick}))=1
\ \text{for every relevant }i,c\right\}.
}
\tag{M6}
\]

This is a constraint on **predicted visible foreground geometry**, not a demand that *every* canonical point be observed in every camera. The model point is allowed to disappear through occlusion or leave the field of view.

Evaluate the mask at a precisely documented discrete pixel-center/rasterization convention, with no tunable erosion/dilation, dilation radius or signed-distance margin. Points have discrete projected pixel ownership; do not invent normals, surfels, meshes, splat support or Gaussian rendering to enforce this.

**Important numerical and physical detail:** a binary mask does not directly tell whether a 3D point is hidden behind the real background. If an unmasked background pixel has a valid closer depth measurement, this may explain an occluded model point; the background depth can enter *visibility checking only*. If the visibility relationship is ambiguous within the declared depth uncertainty, return `unknown` and do not hallucinate a contradiction. No background RGB/depth is ever an accepted foreground tracking observation.

Define the objective for a candidate group state `x` as

\[
J_k(x)=J_{\mathrm{prior},k}(x)+J_{\mathrm{matched\ foreground},k}(x)
+\iota_{\mathcal C_{g,k}}(x),
\quad \iota_{\mathcal C}(x)=\begin{cases}0&x\in\mathcal C,\\+\infty&x\notin\mathcal C.\end{cases}
\tag{M7}
\]

The binary indicator is nonsmooth. **Never differentiate masks as if they were continuous images** and never add a heuristic soft silhouette weight. Instead, the *outer matcher* fixes measurement correspondences and the *inner solver* computes normal analytical IEKF or sigma-point UKF proposals; reject a posterior candidate if it violates (M6), or perform a deterministic feasible line search/relinearization (IEKF). The UKF runs as an independent UKF, not as an IEKF fallback. A UKF sigma point is an uncertainty probe, **not an accepted physical pose**; sigma points must not be clipped independently to the mask, since doing so corrupts the unscented moment calculation. A UKF posterior candidate is checked for feasible visible geometry before acceptance.

If no feasible update exists, retain the previous state and report an unresolved association/visibility conflict. Do not silently project points onto mask borders or force a pose that satisfies impossible masked correspondences. Normal small Gaussian covariance after conditioning on a hard nonlinear feasible set is only a local approximation; label it as such.

### 2.3 Hard mask intersection in projective matching

On GPU, matching is restricted to the indices in `Omega^F`:

```python
mask01 = exact_binary_mask(mask_image)  # 0/1; reject other values
foreground_xyz_valid = mask01 & isfinite(depth) & (depth > 0)
# all source map initialization, target lookup, and unsolved lists use foreground_xyz_valid
# proposed raw observation u accepted only when foreground_xyz_valid[u] is true
```

Store a per-frame bitset of `Omega^F`. Background must not be enumerated as candidate target bins. `match_rgbd(...)` takes the mask explicitly and asserts every produced raw target ID is foreground. Accumulate filter factors only after these checks. The unassigned state applies only to foreground raw observations.

## 3. Exact depth/occlusion interaction

For a predicted point at pixel `u` and depth `z`, if the measurement at `u` has depth `d`:

1. `M(u)=1` and valid depth: an associated foreground observation may update motion **if the point is the predicted/observed visible correspondence**. A closer foreground observation may instead occlude it.
2. `M(u)=0` and valid depth: the background point **must not** initialize geometry or become a filter residual. If it demonstrably lies closer than the predicted model point, that point may be occluded and is unconstrained by this pixel. If the model point is the actually predicted visible first hit, the pose violates the hard mask (M6).
3. `M(u)=1` with invalid depth: no raw 3D target or depth residual. Its foreground label can still be used in silhouette feasibility when visibility is otherwise determined.
4. `M(u)=0` with invalid depth: no RGB-D match; mask exclusion remains hard for verified visible foreground points, but do not turn a no-return reading into measured free-space certainty.
5. Outside image: no observed target and no mask constraint from that camera; the point may still be observed in other cameras.

The visibility test must not use background pixels as a new geometric map or a moving/static part. Use actual calibrated depth ordering and declared depth uncertainty; never introduce a depth-sign epsilon as an arbitrary classification threshold. Depth-noise-based visibility probabilities may inform whether visibility is *known*, but must not turn mask-excluded pixels into admissible foreground observations.

## 4. Outer solved/unsolved loop — required handling

**Initialization:** use `Omega^F` at `k=0`, no background pixels. Fuse across frame-0 camera overlaps using declared correspondence uncertainty; no separate point per image pixel across all time.

**Forward:** predict/transport existing solved points, z-buffer, associate *only* with observed `Omega^F` pixels, compute IEKF/UKF corrections *only from those matched foreground observations*. Record unused IDs of `Omega^F` in `UNSOLVED[k]`; do not put negative-mask pixels into `UNSOLVED`. Check accepted visible points against mask feasibility. Keep occluded solved points in SOLVED.

**Backward discovery:** only sample candidates from `UNSOLVED[s]`, which are therefore masked foreground points. For each candidate group trajectory use `T[g,k] @ inv(T[g,s])` and test source and earlier RGB-D projections with the SAME foreground gating and mask feasibility. A candidate that would have been visible in a known background mask cannot be accepted under that trajectory. If it was occluded, its earlier invisibility is allowed. No reverse candidate from background.

**Frontier:** recompute `UNSOLVED[s]` as valid masked foreground observations not currently owned; verify that map growth only comes from these IDs; all accepted state/map changes satisfy the shared mask-constrained score.

**Global acceptance:** compare the same full-data, foreground-conditioned objective plus feasibility indicators and established priors. Never claim monotonicity if the matcher/visibility/point map changed while the reported objective used cached old masks, z-buffers, or correspondences.

## 5. GPU / CUDA contracts

- `load_masks_to_gpu(k,c)` gives contiguous `uint8`/`bool` arrays with exact binary values, indexed in the calibrated depth pixel coordinate system. Keep masks paired with images/camera IDs/frame indices.
- `gpu_backproject_frame0`: compact mask-positive, finite-depth indices **before** backprojection and any frame-0 fusion.
- `gpu_project_and_zbuffer`: project only SOLVED points; calculate first visible points and candidate masks; do not synthesize background map points.
- `gpu_match_sparse`: use mask-positive valid raw-target pixel bitsets for local correspondence, with `M=0` branches yielding **zero match entries**; support unmatched FG IDs.
- `gpu_accumulate_iekf`: segmented `H^T R^-1 H`, `H^T R^-1 r` over **only** valid matched FG edges. Build small per-group/block sparse information, not a matrix proportional to all raw pixels.
- `gpu_evaluate_ukf`: evaluate matching for the **same fixed foreground measurement associations** across the sigma points; never read raw background RGB/depth as a target; batched nonlinear reprojection and low-rank update, no dense measurement covariance.
- `gpu_backward_discovery`: candidate source indices **only** from mask-positive UNSOLVED; negative-mask projections are checked as visible/occluded/unknown and never treated as candidate matches.
- GPU unit-test kernel parity with reference CPU for masks at pixel edges, subpixel projections, out-of-image, depth-no-return, overlapping foreground/background, hidden geometry, and every view.

No all-frame depth-pixel catalogue. No mask-based CPU fallback over each pixel for every sigma point and every camera. No repeated full-scan background evaluation inside the inner filter.

## 6. Covariance, measurement independence and numerical integrity

- `M` is observed semantic foreground support, not an extra independent depth or color sample. Conditioning on `M` restricts association/feasibility; do not multiply the same raw RGB/depth likelihood twice because it appears in mask and point residual checks.
- If mask generation already depended on RGB/depth, a probabilistic mask-noise factor would be correlated with RGB/depth. Here use masks as **declared hard dataset constraints**, not independent extra Gaussian measurements.
- Image-space pixel sampling and depth noise assumptions must correspond to source raw pixels. Never count overlapping interpolated samples as independent.
- Covariance propagation remains filter-specific (IEKF analytical F/Q/Joseph/RTS, UKF sigma prediction/update/unscented smoothing), with correct pose–map initialization correlations and the same base noise assumptions.
- The hard feasible set is not differentiable and Gaussian covariance does not capture its entire truncated posterior near boundaries. Expose the accepted nominal pose, local filter covariance, mask feasibility status and unresolved ambiguity; do not claim exact constrained Bayesian covariance.

## 7. Mandatory validation and acceptance tests

**Tests that must pass for BOTH standalone pipelines:**

1. Binary mask parser accepts values `{0,1}` and `{0,255}` as exact binary conventions; inconsistent intermediate greys require an explicit input error, not silent thresholding.
2. RGB/depth/mask camera/time/resolution alignment verified. Missing mask fails by default (`require_foreground_mask: true`).
3. Frame-0 map only contains points from valid, positive-depth, masked FG pixels, and its provenance IDs satisfy `M==1`.
4. Every forward matched observation ID satisfies `M==1` and valid depth; no background point appears in IEKF info-factor lists or UKF stacked target vectors.
5. Every `UNSOLVED[k]` raw observation ID is masked foreground and has a valid point measurement; count identity: `FG_valid = matched_owned + unassigned` (with a unique-ownership definition).
6. Predicted visible point on a known mask-zero pixel is **rejected as infeasible**; same point hidden behind a nearer valid measurement is **not** rejected and receives **no** foreground RGB/depth correction.
7. Late-discovered foreground point transported backward remains feasible when initially occluded; a candidate motion that predicts an unoccluded foreground point on background is rejected.
8. Different masks in different cameras/time steps enforced separately; one view's background cannot silently overwrite another view's foreground measurement.
9. Mask boundaries are nearest-neighbor sampled under the declared pixel rule; no bilinear mask scores, morphology, or guessed margin.
10. IEKF candidate and UKF candidate final pose both checked against the **same** hard visibility/mask feasibility, with covariance maintained by their own filter mathematics.
11. Mask-zero RGB color perturbations cannot change the motion objective, accepted correspondences, or new-map source IDs. Background depth perturbations may change **occlusion classification only**, never become a tracking point/residual.
12. GPU and CPU implementations give matching masked FG selection, association, z-buffer/occlusion, information/moment reductions and reverse discovery on controlled examples.
13. Full run of **each** pipeline demonstrates: frame-0 masked map -> forward inner-filter motion -> filter-specific backward smoother -> actual backward discovery from masked UNSOLVED -> next forward pass; provide real counts and objective/feasibility traces.
14. Export `fg_mask_observed/`, `projected_point_valid/`, `projected_visible_fg/`, `matched_fg/`, `unmatched_fg/`, and `mask_constraint_violations.csv` (frame, camera, point, pose stage, visible/occluded/unknown status), along with the previously required poses, covariances and maps.

**Any run in which background points contribute to initialization, correspondence, filter correction, fusion, or UNSOLVED is a FAIL**, regardless of objective improvement or numerical test success.

## 8. Source-level requirements and work order for Codex

Audit every existing function that touches pixel selection, projected matching, objective construction, motion filtering, sigma-point reprojection, reverse discovery and fusion. Remove or rewrite any code that uses `depth > 0` without also checking the aligned foreground mask.

Suggested reusable shared functions:

```python
read_binary_foreground_mask(scene, frame, camera) -> Bool[H,W]
valid_foreground_depth_mask(rgbd_frame) -> Bool[H,W]
backproject_foreground_only(rgbd_frame) -> (xyz, rgb, raw_ids)
project_solved_points(group_poses, canonical_points, camera) -> (pixel, depth, point_id)
classify_visibility(projected, model_zbuffer, raw_depth, raw_valid) -> {visible, occluded, unknown}
associate_foreground_only(projected, raw_rgbd, fg_valid_mask) -> SparseAssociation
check_hard_mask_feasibility(projected, visibility, fg_mask) -> bool
get_unclaimed_foreground_ids(raw_fg_ids, ownership) -> IDs
```

The same `check_hard_mask_feasibility` and `associate_foreground_only` MUST be called by both `main_iekf.py` and `main_ukf.py` during their respective corrections and by the shared outer SLAM loop. No filter-specific masks or differing foreground enforcement.

Mandatory development order: parser/camera alignment -> frame0 map -> matching -> forward IEKF + UKF independent tests -> mask visibility constraints -> smoother -> backward discovery -> frontier -> complete real datasets. Validate a full cycle on a reduced diagnostic sequence first, then all actual timestamps and cameras for final acceptance.

## 9. Exact instruction to follow

> Treat this file as a mandatory amendment to `CODEX_IMPLEMENT_PARALLEL_IEKF_UKF_RGBD_SLAM.md`, `00_COMMON_SOLVED_UNSOLVED_SPEC.md`, and both filter theory documents. Enforce the binary foreground mask as a hard admissible-observation and predicted-visible-geometry constraint in *every* inner and outer operation. Preserve the same solved/unsolved algorithm and two independent IEKF/UKF filters. The mask contains no rigid-part or motion labels. Implement all tests above and do not claim a completed reconstruction until both complete pipelines satisfy them.
