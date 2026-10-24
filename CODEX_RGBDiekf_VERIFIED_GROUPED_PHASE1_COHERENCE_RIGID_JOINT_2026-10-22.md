# Codex execution contract — RGBDiekf: surface-aware supernodes → locally rigid RGB-D IEKF → coherence → one moving rigid trajectory → revolute/prismatic model selection

**Date:** 2026-10-22  
**Target:** `SamueleMara/RGBDiekf`, verified `main` commit `dd4426edcbac772b45a6f634b7c6c665b3651b8c` (reinspect the local checkout before editing).  
**Reference implementation:** `SamueleMara/RigEuler3DGS`, verified `main` commit `481b2cf2d5602932286948cf14114f9caee4de3c`.  
**Priority:** Implement the new pipeline as an **isolated, runnable and testable implementation**. Keep the existing independent 6D point tracker intact as a reproducible baseline. **POINT RECOVERY / NEW-POINT DISCOVERY IS PAUSED.**

## 0. Non-negotiable meaning of “XYZ-only”

1. The geometry to be estimated, evaluated and exported is **only the position** `x_i(t) ∈ R³` of each original observed XYZ/RGB sample. No per-point orientation, normal, Gaussian covariance/scale/opacity, per-point angular velocity, splatting or renderer is required.
2. Rigid motion of a *multi-point local neighborhood* cannot be represented by three translation parameters alone. It requires a shared `R_s(t) ∈ SO(3)` and `t_s(t) ∈ R³`. These are **latent group-level motion variables used to generate XYZ**, not per-point rotations. A group-level twist is permitted solely as a temporal dynamics state.
3. A group member's position is always

   `x_i(t) = R_s(t) p_i + t_s(t)` for `i ∈ G_s`, where `p_i` is that sample's immutable **canonical endpoint XYZ**. Enforce this **exactly at every frame**, not merely as a shared correction applied to otherwise free individual positions.
4. `T_s(0) = I` for a frame-zero forward run, `T_s(T) = I` for a final-frame backward run, with an explicit gauge at the selected endpoint. Export every resulting point trajectory in **chronological original-frame order**, independent of run direction.
5. A spatial supernode is a **computational hypothesis**, not a semantic part or an already validated rigid body. Never infer static/moving labels from mere spatial proximity or from internal distances that your own hard constraint forces to stay constant.
6. Preserve raw RGB-D measurement semantics: `valid = whole_object_mask & original_sensor_depth_validity & finite_positive_metric_depth`; use known intrinsics, world-to-camera extrinsics and real acquisition timestamps. Raw RGB is an **appearance cue for correspondence**, not an invented image-rendering residual.
7. Every initially valid foreground observation remains represented. Do not covertly downsample the estimated point catalogue, create a static PCD with a made-up voxel resolution, assume a fixed number of moving parts, use GT segmentations/joints, or declare prediction-only tracks to be motion evidence.
8. The output is **one hypothesized moving rigid part and a stationary competitor**, with a third `INCOHERENT_OR_AMBIGUOUS` class. Do **not** implement general `K`-body selection now. If no distinguishable moving part exists, output a justified `NO_COHERENT_MOVING_COMPONENT` and do not invent an axis.
9. Stop after comparison of prismatic and revolute joint models. Do **not** start point recovery, all-frame newly seeded tracks, or joint-constrained Phase 3.

## 1. Inspect and reuse the right source, without mixing architectures

### Source locations in RGBDiekf (verified on above commit)

| Existing file | Use in new implementation | What MUST change in new path |
|---|---|---|
| `pointwise_rgbd/dataset.py` | `Dataset`, real frame IDs/time, calibrated cameras, distortion guard | Add a direction-aware **iteration order** without permuting physical timestamps |
| `pointwise_rgbd/observations.py` | `samples()` measured `(u,v,Z)`, world XYZ, RGB, covariance, source pixel IDs | Initialize from first **or final** frame, all cameras, with source provenance; final PLY alternative only when explicitly supported |
| `pointwise_rgbd/projection.py` | Analytic projection and backprojection conventions | Add group-level Jacobian composition; the existing `project()` returns 3×6 for `[XYZ,v]`, **not** the required group-pose Jacobian |
| `pointwise_rgbd/association.py` | Candidate 3D kNN, 2D ray gate, depth gate, RGB Mahalanobis, exclusive measurement refs | Associate predicted **group-transported XYZ**; accumulate accepted residuals by `supernode_id`, no independent point IEKF updates |
| `pointwise_rgbd/config.py` | Documented sensor noise and authorized `1%` chi-square tail | Declare and log *new group-level pose/twist process priors*, dimensional units and uncalibrated status; don't quietly reuse the old 6D point `Q` for a 12D Lie state |
| `pointwise_rgbd/filter_iekf.py` | Fixed-prior nonlinear IEKF solve pattern, accepted-energy line search, SPD checks | Implement separate group-level manifold correction / covariance; do not modify old 6D estimator in place |
| `pointwise_rgbd/controller.py` | One-pass frame loop, output checkpointing | New initial-run entrypoint **must not call** `run_exact_pointwise_discovery()` / its recovery while-loop |
| `pointwise_rgbd/coverage.py` | Observed-to-transported geometric coverage for diagnostics **only** | Coverage ≠ exclusive filter measurement. Broad uncertainty does not certify physical reconstruction |
| `pointwise_rgbd/store.py`, `frozen.py` | Immutable history and source-identity ideas | Store group pose, twist, covariance, membership version, true observations, point XYZ, hashes; no overwrite of prior results |
| `pointwise_rgbd/visualization.py` | Baseline output only | New dedicated viewer, dark Three.js reference style, with explicit sample count and no hidden 512-point cap |

Reference **RigEuler3DGS** source to inspect and **adapt, not import wholesale**:

- `estimator/supernodes.py::build_supernodes`, `GroupedObservation`: farthest-point seeds, Lloyd membership, centroid graph, grouped Jacobian/adjoint.
- `estimator/sparse_iekf.py::iekf_frame_step`: `Gᵀ(HᵀR⁻¹H)G` grouped solve. No Gaussian rasterizer in RGBDiekf.
- `estimator/motion_coherence_support.py::{canonical_knn_graph,edge_strain}`: fixed canonical **inter-node** geometry and temporal strain.
- `estimator/coherence_extraction.py::{candidate_hierarchy,cross_validated_rigidity,select_moving_component,refine_from_core}`: graph-persistent candidates, held-out validation, complement-relative articulation, no target moving fraction.
- `estimator/rigid_part_runtime.py::{weighted_kabsch,kabsch_initialize,RigidState}`: common rigid initialization and constrained refinement.
- `joint_reduction/rigid_trajectory_reduction.py::{relative_increments,one_dof_diagnostics,fit_prismatic,fit_revolute_axis,fit_axis_point,refine_prismatic,refine_revolute}`: screw geometry and rank/conditioning diagnostics.

**Do not import:** Gaussian renderer, rasterizer, joint GT, fixed body labels, per-Gaussian orientation, opacity/alpha/silhouette equations, supernode CUDA kernels coupled to Gaussians, or the prior `rgbd_iekf` fixed-K architecture.

## 2. Precise forward/reverse endpoint preparation

### 2.1 Forward

`anchor_frame = 0`, `processing_order = [0,1,…,T]`, canonical `p_i` = each valid masked metric RGB-D backprojection in **all** anchor cameras. Preserve `(original_frame_id,camera_id,raw_pixel_id)` and RGB for each point. Identical-physical-surface samples from different cameras remain explicit unless a separate auditable multi-camera fusion is implemented. Canonical position is uncertain because the original depth was measured; declare if subsequent inference conditions on it rather than marginalizes it.

### 2.2 Backward

`anchor_frame = T`, `processing_order = [T,T−1,…,0]`. Initialize from **all** actual valid masked final-frame RGB-D observations. Optional `--anchor-source final-ply` may load actual final XYZ/RGB, but **only XYZ/RGB and safe coordinate metadata**. NEVER read part IDs, joint axes, motion labels or synthetic GT. It must still have a legitimate measurement-uncertainty treatment; absent metadata is logged as an assumption, not invented calibration. Keep both endpoint-source ablations independently named.

Run exactly the **same filter code** forward or backward with signed `h = time[next]-time[current]`. `h<0` in the backward run. Physical velocity/twist is defined in world seconds and must not be inverted just because array iteration is reversed. Derive and test the reoriented process covariance; do not reuse forward covariance blindly. Output files are indexed using actual chronological frame identities and a separate `processing_order` metadata field. These runs have **independent point IDs, supernodes and estimates**; do not pretend they have matched point identities or fuse them yet.

No recovery / seeding from any other frame.

## 3. Construct **computational** supernodes from the endpoint

### 3.1 Surface connectivity, before any motion inference

For each canonical point `i`, build local 3D kNN candidates. Permit graph edges only when actual evidence supports a common local surface:

1. Within the same camera, preserve adjacency in the source depth image (4- or 8-neighborhood) and reject edges that jump a measured depth discontinuity incompatible with the declared depth covariance.
2. Between cameras, accept a cross-camera edge only with consistent backprojected positions, calibrated projection and visibility/depth support; do not bridge occluding foreground and background layers merely because XYZ Euclidean distance is short.
3. Retain every original point, including isolated/unresolved points, with explicit membership status. No unexplained raw sample is silently dropped.
4. If a surface's adjacency cannot be established (occlusion, missing depth, untrusted cross-camera merge), leave components separate. This is preferable to irreversible over-grouping.

Describe the actual geometric/depth compatibility tests and noise covariance used. Do not add arbitrary fixed metric distance cutoffs disguised as physical calibration.

### 3.2 Geodesic farthest-point seeds and connected assignment

Run **deterministic** farthest-point seeding per connected surface component, preferably using shortest-path/geodesic distance on the observed-surface graph. Use a configurable *target computational occupancy* (e.g. RigEuler's reference 64 as an explicit benchmark starting point, **not** a physically correct default or a number of moving parts). Assign points to graph-connected seeds; refine centroids only if membership remains connected. Keep group IDs, centroid, radius, members, adjacency, original raw IDs and partition hash. No static/moving labels at this step.

A kNN/Voronoi grouping in Euclidean XYZ without surface connectivity is **not** acceptable near a door/cabinet contact or two closely stacked surfaces.

### 3.3 A partition is fixed within an optimization epoch

Membership `g(i)` and canonical local offsets `p_i-c_s` stay immutable during each full temporal filter run. The next stage may propose **measured-evidence-driven splits**. When membership changes, rerun the affected group(s) with an explicit new partition version across the sequence; do not mix positions generated by different group definitions while claiming exact temporal rigidity.

## 4. Phase 1 — Grouped RGB-D IEKF that *exactly* constrains XYZ

### 4.1 State and physical output

Per supernode `s`, maintain:

- Canonical endpoint XYZ `p_i`, never optimized independently inside this grouped run.
- `T_{s,k}=(R_{s,k},t_{s,k})∈SE(3)`, mapping endpoint XYZ → world XYZ.
- Optional/expected **group-level** physical spatial twist `ξ_{s,k}=(v_s,ω_s)∈R⁶` for time propagation.
- Local error state dimension `12 = [δpose(6), δtwist(6)]` per **group**, including a group-level 12×12 conditional covariance. (If implementing pose-only group propagation instead, state it clearly and supply a sound temporal model; do not call independent 6D point state “grouped”.)

For each point `i∈G_s` at every frame:

\[
\boxed{x_{i,k}=R_{s,k}p_i+t_{s,k}.}
\]

**Exact rigidity proof/test:** for every pair within a group and every frame,
`||x_i(k)-x_j(k)|| = ||p_i-p_j||` within floating-point tolerance. The equality must remain true across prediction, accepted update, resume, and backward run; not only within one IEKF iteration.

### 4.2 Prediction with physical timestamps

Choose and document a consistent **spatial left-invariant** convention. A possible nominal process:

\[
T^-_{s,k+1}=\operatorname{Exp}(h_k\hat\xi^+_{s,k})T^+_{s,k},\quad
\xi^-_{s,k+1}=\xi^+_{s,k},\qquad h_k=t_{k+1}-t_k.
\]

`h_k<0` in the reverse pass. Propagate the **full group pose/twist covariance**, including pose–twist cross blocks, with the analytically or numerically verified linearization of this Lie-group process and physically meaningful acceleration spectral densities. The reverse-time **process factor** must be explicitly derived/reoriented and tested, and not misrepresented as exact backward Bayesian smoothing. Maintain SPD and do the correct covariance chart reset after an accepted Lie-state retraction. Do not replace the posterior covariance with its initialization at every frame. Don't clone a 6D point `F,Q` and pretend it describes the group manifold.

At the anchor, `T=I` is the canonical gauge. Initial twist is zero-mean **uncertain**, not an assumption that the group is static. Conditioning on noisy canonical geometry versus treating it as uncertain must be documented; never re-assimilate the anchor observation twice.

### 4.3 Shared RGB-D measurement objective; **no renderer**

Given a predicted point XYZ, for each calibrated camera `c`, the observation model is

\[
 h_c(x)=\left(f_x X/Z+c_x,\;f_y Y/Z+c_y,\;Z\right),\quad
 (X,Y,Z)=R_{cw}x+t_{cw}.
\]

The matched real observation is `z=(u,v,D(u,v))`, covariance `R_{c,u,v}` from the existing sensor config. Keep the existing appearance-aware 3D kNN / depth-ray / color matching as a **candidate association step**. Color is used only to establish correspondence, not as a fabricated independent image likelihood on a point with no renderer.

**Prediction-to-pixel correspondence:** one raw observation ID must not be used by two independent filter factors. Several real cameras may measure a physical group. Match exclusivity is about actual IEKF update factors; **geometric coverage is many-to-one and separate**.

Let the left pose perturbation be `δτ=(δρ,δθ)` so that

\[
\delta x_i=\delta\rho- [x_i]_\times\delta\theta,
\quad G_i=[I_3\; -[x_i]_\times],
\quad H^{\text{group}}_{ic}=J_{\pi,c}R_{cw}G_i.
\]

(For a centroid-centered twist chart, substitute `x_i-c_s` and apply the corresponding translation/adjoint convention consistently. Do not combine the two charts.) The instantaneous measurement Jacobian for `[δpose,δtwist]` is `[H^{group}_{ic}, 0_{3×6}]`.

With fixed actual association IDs for the current accepted IEKF update, estimate the shared group correction by the nonlinear fixed-prior MAP objective

\[
E_s(T,\xi)=\tfrac12\|[\operatorname{Log}(T(T_s^-)^{-1}),\;\xi-\xi_s^-]\|_{(P_s^-)^{-1}}^2
+\tfrac12\sum_{(i,c,u)\in\mathcal A_{s,k}}\|z_{c,u}-h_c(Tp_i)\|_{R_{c,u}^{-1}}^2.
\]

Use fixed predicted prior across **all inner iterations**, analytic projection/Jacobian composition, accepted-energy line search, final-information covariance, and correct manifold covariance reset. Reject truly singular solves explicitly; record observability deficiencies and avoid claiming a physically measured group rotation with insufficient non-collinear correspondences. Do not treat the many member points' induced pose uncertainty as independent observations.

Per-point positional covariance is a **marginal derived from the shared group covariance**:

\[
P_{x_i,k}\approx G_i P_{\tau_s,k}G_i^\top,
\qquad P_{x_i x_j,k}\approx G_iP_{\tau_s,k}G_j^\top\ (i,j\in G_s).
\]

Canonical endpoint measurement uncertainty requires an additional **declared** term or conditioning assumption. The nonzero cross-point covariance is essential: it forbids naively counting 64 members as 64 independent prior poses in Phase 2 or the joint fit.

### 4.4 GPU & numerical execution

Batch observations/group factors where possible; accumulate group information for each distinct observation ID. Use bounded CUDA chunks suitable for the existing RTX 4070 8GB; keep FP64 covariance/normal equations if practical. Do not form a dense `(12S)²` covariance if groups are assumed conditionally independent. **But** make the conditional-block approximation explicit; shared data association/calibration uncertainties are not marginalized. Report GPU/CPU agreement and exact profile times.

### 4.5 Association refinement is not rigidity validation

When correspondences are wrong, rigidity may force an entire patch toward an incorrect target. Log observed matched pixel IDs, depth/ray/color costs, matched distances and residual maps. Optionally rerun association *between accepted outer iterations* and re-solve with a well-defined factor ownership transaction; never swap a different raw observation invisibly within a fixed-association IEKF energy evaluation.

## 5. CRITICAL — Validate/split groups crossing a physical joint

A fixed spatial partition **can be wrong**. A group that contains both cabinet and door points is physically incapable of following one rigid transform. Never silently hope that an IEKF residual penalty fixes that contradiction.

### 5.1 Independent measured evidence (not the group's self-generated trajectory)

For each group and frame, retain residual vectors and actual matched raw depth/RGB sample refs. Split those measurements into deterministic independent train/held-out sets by stable member IDs and, in a separate robustness check, by temporal/camera blocks. **Avoid near-duplicate multi-camera observations leaking across folds**: canonical sample origins and same-surface duplicates should be handled as a linked provenance group when appropriate.

Fit the best **single** group rigid motion to training measurements. Evaluate the *unchanged* held-out measurements. Detect structured, incompatible motion residuals rather than just a high total error: e.g. two graph-connected subpatches require different 3D displacements/rotations across multiple measured frames.

### 5.2 Competing split hypothesis

Propose a **surface-graph-constrained split** from TRAINING residual evidence only. Refit **two distinct group rigid trajectories** (each relative to its own immutable canonical points) on training observations; evaluate both one- and two-group hypotheses on exactly the same held-out real observations and original timestamps.

Approve splitting only when out-of-sample improvement remains identifiable after accounting for extra model freedom and uncertainty. Use a stated cross-validated likelihood / information criterion or a statistically justified likelihood ratio; do not insert arbitrary displacement or residual thresholds. If insufficient geometry, temporal observations or rank exists, return `UNRESOLVED_SPLIT`, not an invented boundary. Record full objective, group rank, held-out sample counts, proposed membership and reason.

Refit the two new groups from their canonical XYZ across the whole sequence. Version the partition and save audit/history of old vs new membership. Do **not** re-use old group pose covariances as if valid for a different membership, or claim old trajectory values were left unchanged while membership changed.

### 5.3 Termination

Continue only while valid split evidence exists. A declared compute budget is allowed for testing but **hitting it must be reported as incomplete**, not as mathematical convergence. No oscillating split/merge heuristic, no forcing a K, no semantic ground truth.

## 6. Coherence extraction AFTER local Phase 1

### 6.1 Use only genuinely informative frames

The input is `T_s(k)`-transported canonical supernode centroids (and their observation-backed support and group uncertainty). Critically, after hard local rigidity, **within-supernode distances are tautologically constant**. They CANNOT be used as evidence of static/dynamic separation. Use **between-supernode** canonical-neighbor edges and independent real measurement support.

Compute for an edge `(s,r)` with valid direct observation evidence:

\[
\ell_{sr}(k)=\|c_s(k)-c_r(k)\|,
\quad\bar\ell_{sr}=\operatorname{median}_{k\in\mathcal O_{sr}}\ell_{sr}(k),
\quad z_{sr}(k)=\log\frac{\ell_{sr}(k)+\epsilon}{\bar\ell_{sr}+\epsilon},
\quad C_{sr}^2=\frac{\sum_{k\in\mathcal O_{sr}}w_{sr,k}z_{sr}(k)^2}{\sum_{k\in\mathcal O_{sr}}w_{sr,k}}.
\]

Weights and `O_sr` come from **real matched updates** and declared group uncertainty; no long unobserved prediction-only segment can masquerade as evidence for rigidity or moving classification. Propagate uncertainty of centroid separation; do not assume within-group members are independent.

### 6.2 Candidate components, not forced binary segmentation

Follow the `RigEuler3DGS/estimator/coherence_extraction.py` v2 principles: sort cross-node graph edges by increasing strain; build a graph-constrained component hierarchy with safeguards against single bridge edges; evaluate size-stratified candidate components **without prespecifying a moving fraction**. Record internal-vs-boundary strain distributions.

Do **not** copy the function blindly: its original input treats every frame's trajectory as fully available, whereas new RGBDiekf needs direct-observation masks, per-group support rank and covariance-aware weighting. Inferred group motion with only a huge prediction covariance must not be treated like a measured centroid.

### 6.3 Identify articulation relative to its complement

For candidate component `S`, fit a rigid pose `T^K_k` to the *complement* `K` using observed, well-supported nodes. In candidate-relative coordinates

`y_i(k) = (T^K_k)^-1 x_i(k)`.

Compute on held-out observed nodes (or raw RGB-D measurements) the competing **identity** and **shared rigid** model errors:

\[
E_I^{cv}(S)=\sum_{k,i\in S_{val}}w_{ik}\|y_i(k)-p_i^{K}\|^2,
\quad
E_R^{cv}(S)=\sum_{k,i\in S_{val}}w_{ik}\|y_i(k)-T^{S}_kp_i^{K}\|^2,
\quad\eta_{cv}=1-\frac{E_R^{cv}}{E_I^{cv}+\epsilon}.
\]

The `p_i^K` are canonical coordinates expressed in the complement's **canonical reference frame**, so this comparison is in a consistent gauge; equivalently transform both predictions to world for scoring. If the complement itself is unobservable/incoherent, mark `REFERENCE_UNRESOLVED` and do not fabricate relative articulation.

Select a *confident coherent moving core* based on held-out improvement **relative to complement**. Once existence of articulation is established, identify which side is stationary in the known fixed-camera world frame by the held-out identity-vs-rigid error **per node-frame**, not total error that biases group size. If fixed-camera world-frame validity itself is questionable, report ambiguity.

### 6.4 Expand from the moving core to individual XYZ points

Fit `T^S_k` from the moving core only. For each other point, compare the moving and static models **using actual observation-supported frames** and without including that point in the transform used to classify it:

`E_i^M = Σ_{k∈O_i} w_{ik} ||y_{ik} - T^S_k p_i||²`,
`E_i^I = Σ_{k∈O_i} w_{ik} ||y_{ik} - p_i||²`.

Use the same real RGB-D measurement space for both predictions (prefer original `(u,v,Z)` with camera Jacobian/covariance) if its source associations are available; or measured world XYZ with proper propagated covariance. **Don't train/score on group-generated positions alone**: that would only retest the imposed group model. Because accepted observations can be correlated via associations and shared group pose, disclose the approximation or validate on genuinely held-out raw measurements.

Output exactly three classes on original point IDs:

- `MOVING_RIGID`: moving model supported, stable coherent local evidence.
- `STATIC_COMPATIBLE`: identity model supported.
- `INCOHERENT_OR_AMBIGUOUS`: insufficient measurements, invalid group, near-axis ambiguity, contradictory correspondences, or neither hypothesis justified.

Never force unidentifiable axis-adjacent points to be static or moving. Keep per-point model residuals, direct-observation count, evidence strength and originating group IDs. No motion magnitude cut; no target fraction; no GT.

## 7. Phase 2 — One shared moving rigid trajectory with a **fresh RGB-D refinement**

The selected `MOVING_RIGID` set, its canonical point XYZ/RGB, calibrated observations and physical timestamps are frozen for this stage. Initialize one `R_k,t_k` per frame with weighted Kabsch on only appropriately supported Phase-1 XYZ trajectories. **A Kabsch fit is an initializer, not the final estimate.**

Refine a single world rigid pose `T_k∈SE(3)` (plus a physically defined group spatial twist and covariance if using an IEKF), with point motion

\[
\boxed{x_{i,k}=R_kp_i+t_k\quad\forall i\in MOVING\_RIGID.}
\]

Use **original actual RGB-D measurements and the same matching/calibration model**, not just an SVD fit to Phase-1 outputs. Non-moving points must not be transported with the moving pose; ambiguous points must not be silently treated as measured static anchors. The same raw observation cannot be assimilated twice within the same filter update. Independent frame-0 and final-frame runs refine independently.

Report `poses[k]`, `twists[k]`, group pose covariance (conditional, chart declared), number of accepted distinct raw measurements, actual depth/pixel residuals and held-out rigid error. If Phase-1 support is insufficient for 3D rigid observability, output `RIGID_UNIDENTIFIABLE`, not random rotations.

## 8. Joint reduction — compare **revolute vs prismatic** with the same positions

Use the optimized Phase-2 rigid trajectory, actual timestamps and frozen moving support. Fit two separate one-DOF candidates:

**Prismatic:**

\[
R^P_k=I,\quad t^P_k=q^P_k e_P,\quad\|e_P\|=1,\quad x^P_{i,k}=p_i+q^P_ke_P.
\]

Estimate `e_P` from the translation sequence (weighted PCA / direct fit) and refine `q^P_k` against the free rigid poses, then measured positions. Pure translation does **not** identify a physical joint origin; any displayed base point is an explicitly chosen reference convention.

**Revolute:**

\[
R^R_k=\exp(q^R_k[e_R]_\times),\quad t^R_k=(I-R^R_k)a_R,\quad
x^R_{i,k}=a_R+R^R_k(p_i-a_R).
\]

Estimate fixed `e_R` from the finite rotation vectors, find the axis point using `(I-R_k)a_R≈t_k` with gauge `e_Rᵀ(a_R-c_0)=0`, unwrap `q^R_k` via consecutive relative rotations using **physical** timestamps, then refine geometry/`q_k` jointly in a small least-squares stage. Record axis rank, singular values, condition, extent-relative axis distance and pose fit residuals; a very distant axis can mimic translation and is a known degeneracy.

**Selection:** Score **both candidates on identical held-out physical `(point,frame,camera,pixel)` measurements**, using the same fixed noise model and the same association provenance. Primary comparison is held-out **position/pixel-depth prediction error** on the same evaluation set, not angular-versus-linear pose numbers in incompatible units. Also record normalized residuals, model complexity, identifiability, uncertainty, one-DOF rank-one twist consistency, and per-frame plots. Evaluate candidate-specific identifiability and measurement consistency independently. Select PRISMATIC when its nonzero translation direction is observable and its held-out residual is consistent, even if REVOLUTE has an unidentifiable finite axis or an axis at infinity. Select REVOLUTE when its finite axis direction and perpendicular axis location are observable and its held-out residual is consistent, even if PRISMATIC is degenerate or inconsistent. Never select a candidate merely because its competitor failed. When both candidates are identifiable and consistent, compare their identical-set held-out position-space evidence and select only if statistically distinguishable. If neither is identifiable/consistent, excitation is insufficient, motion is indistinguishable from measurement noise, or evidence cannot distinguish the candidates, output `JOINT_TYPE_UNRESOLVED`. A prismatic physical origin is inherently unidentifiable and is not a degeneracy; gauge the revolute axial-position freedom. This corrected policy is explicitly authorized by the user on 22 October 2026; test 11.4 is authoritative. No arbitrary motion thresholds, selection margins, or GT labels. Joint candidates **read** the Phase-2 pose history and never mutate it. No Phase-3 joint-constrained filter yet.

Keep all three geometries explicit: static identity, moving free SE(3), and prismatic/revolute candidates.

## 9. Results, diagnostics and required **real** visualization

Store distinct forward and backward run roots with source/hash/manifests, one `partition_version` per whole-series refit, full `group_poses[T,S,4,4]`, twists/covariance, exact per-point `xyz[T,N,3]` derived from group poses, per-point observed masks and original raw refs, label classes, candidate hierarchy, held-out errors, final shared rigid poses, both joint fits and status/uncertainty. No fabricated frame observations or numerical values.

- Numerical: group-size/spatial-connectivity distributions; group split/validation evidence and rank; time per stage; observed-vs-predicted counts; association and depth residual histograms; covariance trace/condition; cross-group strain; candidate held-out `E_I,E_R,η_cv` in both world and complement frames; moving/static/ambiguous counts; one-rigid-fit residual vs measured data; P/R held-out error and axis conditioning; forward/backward disagreement after both are in the same calibrated world gauge.
- Visual: reuse the **layout and interaction style** of user's provided `rigid_phase2_debug.html` (dark full-window Three.js viewport, persistent point buffers, right sidebar, bottom frame playback slider, layer checkboxes, responsive orbit/zoom and stable object-centered camera). Display actual full spatial group membership, validated/split patches, three point labels, direct measurement vs prediction-only, moving rigid transport, static points, residuals, revolute axis / prismatic direction, both candidate reconstructions. Never use hidden fixed 512/4096 point caps. Use explicit adjustable display LOD and report `displayed/total`; inference uses all points. Do not autofit bounds to huge prediction outliers: measured object is the default camera focus; outliers have a separate view/layer. Show every temporal frame with actual stored data, stream frame-specific buffers if necessary; no invented interpolation between the eight old sampled audit frames.
- Export overlay images for selected real cameras showing original RGB/mask/depth + transported predicted points + actual matched source pixels; display residuals and split proposals, not ground truth joint data.

## 10. Files to create (suggested; make minimal surgical adjustments only)

- `pointwise_rgbd/surface_supernodes.py`: source-connected endpoint partition and serialization.
- `pointwise_rgbd/group_pose.py`: exact SE(3) group XYZ motion, Lie tangent and Jacobians.
- `pointwise_rgbd/group_process.py`: signed-time group pose/twist prediction, covariance propagation and correct reset.
- `pointwise_rgbd/group_iekf.py`: batched group RGB-D IEKF, measurement ownership and fixed-prior solve.
- `pointwise_rgbd/group_partition_validation.py`: held-out one-vs-two rigid hypotheses, versioned refit.
- `pointwise_rgbd/group_coherence.py`: inter-group strain hierarchy and measured-support-aware moving/static/ambiguous labeling.
- `pointwise_rgbd/rigid_xyz_phase2.py`: selected support Kabsch init plus one shared rigid RGB-D IEKF.
- `pointwise_rgbd/xyz_joint_reduction.py`: independent P and R candidates with common held-out metric.
- `pointwise_rgbd/phase1_grouped_controller.py`: explicit forward or reverse schedule, checkpoint/resume, no discovery.
- `tools/run_grouped_rgbd_joint.py`: standalone CLI, `--anchor first|last`, `--anchor-source raw-rgbd|final-ply`, `--stop-after phase1|coherence|rigid|joint`, `--device cpu|cuda`, `--resume`, `--debug-numeric`, `--debug-visual`, `--output`.
- `tests/test_grouped_*.py`, including independent observation-backed fixtures and real-data smoke validation.

Do **not** delete or edit the old `pointwise_rgbd/{process6,filter_iekf,controller,discovery}.py` without an individually demonstrated, backwards-compatible bugfix. Do not change the user's validated static RGB-D loader or older masked static/3DGS pipeline. All new artifacts go into a **fresh output** directory.

## 11. Mandatory tests (ALL must be run and their real outputs linked)

1. **Exact rigidity:** three+ non-collinear points, arbitrary 3D translation/rotation, fixed within-group distances at every frame to floating tolerance, including prediction-only intervals, checkpoint/resume and both directions.
2. **Group observability:** one point -> rotation unobservable; two points -> axial spin unobservable; >=3 non-collinear -> full 6D local pose observable; report singular-value spectra; point-positions are only measured outputs.
3. **Static:** `T_k≈I`, no moving part or P/R axis fabricated; measure false-motion rate on held-out RGB-D.
4. **Translation:** group XYZ translates with constant and variable speeds; rigid Phase 2 matches GT *for tests only*; identifiable measurement-consistent prismatic chosen with correct direction, no fake origin, even when the revolute competitor has an unidentifiable finite axis or unbounded radius. Also test static/noise-indistinguishable motion, insufficient excitation and statistically indistinguishable candidates returning `JOINT_TYPE_UNRESOLVED`. Candidate failure alone never establishes its competitor.
5. **Rotation:** known off-origin rotation, stops/reversals, axes in 3D, angle wraps; revolute axis direction/point identifiable and fitted from held-out positions.
6. **Critical joint-boundary group:** two initially close surfaces move differently. Starting erroneous supernode is rejected/split based on **held-out real synthetic RGB-D observations**, both new groups refit across full sequence, no double-owned source pixels. Do not pass merely because a test hand-supplies correct group membership.
7. **Occlusion:** some group members disappear for tens of frames while others remain measured; group continues coherently with correct increasing uncertainty. Fully unobserved group does not become falsely well measured or labeled moving.
8. **No false coherence:** all internal distances are invariant by construction, so test that extractor ignores that tautological evidence and uses inter-supernode, held-out measurements. Deliberate prediction-only drift must NOT produce a moving label.
9. **Many-to-one coverage vs exclusive measurement:** exactly one raw pixel use per filter factor, no duplicate assimilation, coverage remains independent of owners.
10. **Lie math:** finite-difference group Jacobians, pose composition, signed physical dt, reverse process covariance factors, covariance retraction/reset; CPU/CUDA agreement and SPD.
11. **Held-out leakage:** disjoint training/evaluation raw measurements and physical-surface duplicate grouping; exact same evaluation set for identity/rigid/P/R comparisons.
12. **Two views/directions:** forward starts frame 0, backward starts frame T; original frame IDs preserved, chronological outputs, endpoint gauges correct and no shared GT labels.
13. **Impossible case:** insufficient rigid support or tiny motion -> `UNRESOLVED`, no arbitrary joint axis or moving fraction.
14. **End-to-end performance:** bounded real scene subset; time/memory, actual histograms and accurate 3D viewer vs source snapshots. Only after this run whole 328×3 sequence. Never claim end-to-end validation from only unit tests.
15. **Baseline preservation:** hashes of baseline code/old result inputs unchanged and existing legacy tests still pass.

## 12. Acceptance stages / working discipline

**Gate A (mathematics):** implement/validate endpoint partition, group transform/Jacobians, covariance propagation and unique observation factors. Demonstrate exact local rigidity and correct no-observation behavior before a full run.

**Gate B (boundary):** construct synthetic cabinet+door with near-touching endpoint surfaces; show wrong initial grouping is identified and refitted **from measured RGB-D**, not from ground-truth part labels.

**Gate C (coherence):** produce a detected moving core and three-way labels using inter-node, independent data. Demonstrate no false mover under static-only/global-drift experiments.

**Gate D (shared rigid):** compare independent-point baseline, grouped Phase 1 and Phase-2 shared rigid against the **same actual measured raw RGB-D factors**, with per-frame observed support.

**Gate E (joint):** fit P and R on the **same held-out point observations**, give physical residual units, axis conditioning and unresolved semantics.

**Gate F (forward/reverse):** test both endpoint directions independently, then compare physical predicted geometry, static/moving labels, and joint axes in the common world frame.

**Only after each gate passes**, undertake the full source sequence. If a gate fails, stop at that gate and report the root cause with a narrow patch proposal. Do not replace it with an arbitrary threshold or falsely report success.

## 13. Deliver to user

Report, in this order:

1. Exact committed git SHA plus `git status`, changed-file diff summary; mention if uncommitted.
2. A table mapping **each** section above to file/function and specific implemented test.
3. True test output, observed vs prediction-only support, and bounded real-data residuals/performance.
4. Partition map, within-group rigidity verification, split examples with real held-out evidence.
5. Moving/static/ambiguous memberships and supporting coherence plots (not GT labels).
6. Shared rigid trajectory and measured-depth/image diagnostics.
7. Both joint hypotheses' fitted axes, pose/point residuals and identifiability; explicit winner or unresolved status.
8. Working real Three.js HTML with complete frame control and honest display LOD; screenshot/browser verification, if available.
9. Clear remaining failures/assumptions. **Do not present a generated mock plot or a prediction-only segment as an actual measured result.**

---

**Core invariant:** The estimator must infer point XYZ motion from real RGB-D while **exactly preserving local rigid geometry within validated supernodes**, discover the moving support only from independent temporal evidence **between** supernodes, then fit one rigid trajectory and compare prismatic versus revolute models on the same held-out observed positions. No point recovery and no unnecessary per-point rotational degrees of freedom.
