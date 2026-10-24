# Codex corrective implementation contract — RGBDiekf motion discovery, RigEuler comparison, grouped RGB-D inference, and truthful debug

Date: 2026-10-22. **Read this in full before editing.**

## 0. Scope, baseline, and NON-NEGOTIABLE evidence rules

Target repo (user workstation): `/home/samuelemara/Joint3DGS/RGBDekf` (`SamueleMara/RGBDiekf`). Previously verified public `main`: `dd4426edcbac772b45a6f634b7c6c665b3651b8c`; **the grouped changes are UNCOMMITTED and thus NOT in public GitHub**. You must inspect local modified/untracked code, not assume GitHub `main` contains it. Reference: `SamueleMara/RigEuler3DGS`, relevant audited main `481b2cf2d5602932286948cf14114f9caee4de3c`; verify actual checked-out reference source before porting.

Read local files, including:

- `reports/grouped_2026-10-22_validation.md`, previous signed test outputs, complete current implementation contract; current `git status --short`, `git diff`, untracked new source files.
- `pointwise_rgbd/{surface_supernodes,group_pose,group_process,group_iekf,phase1_grouped_controller,group_partition_validation,group_visualization}.py` and `assets/group_viewer.html`, `tools/run_grouped_rgbd_joint.py`, all grouped tests, actual factor database.
- Working baseline `pointwise_rgbd/{controller,process6,filter_iekf,association,observations,projection,dataset,store}.py`; never change its behavior.
- RigEuler: `estimator/{supernodes,sparse_iekf,motion_coherence_support,coherence_extraction,rigid_part_runtime,rigid_part_se3}.py`, `joint_reduction/rigid_trajectory_reduction.py`, and `tools/kpart_pipeline.py::bootstrap_positions`.

**Goal:** one moving articulated rigid part versus stationary support, **XYZ-only point geometry**, no per-point orientation states, no GT motion or part labels. Phase 1 must discover motion without knowing part identity. Then select genuinely moving support, refine one shared free rigid trajectory from the raw RGB-D factors, and compare revolute/prismatic geometries. Run independent full-sequence forward and backward endpoint models once test gates pass. NO point recovery, no Phase 3 joint-constrained filter, no K-part generalization now. Do not modify old results, claim a mathematical unit test proves physical reconstruction, fabricate measurements or accept a failed gate silently.

**Important limitation of this external audit:** the two supplied result ZIPs contain trajectories, manifests, correspondence databases, JSON diagnostics and viewer HTML, but NOT the uncommitted implementation's `.py` sources or original Gate-B fixture. Descriptions below of current code are based on its signed runtime report and output; confirm exact coding faults by reading the LOCAL files before applying a change. Present code-level findings with line numbers after inspection.

## 1. Verified diagnostic facts that any new explanation must account for

### 1.1 Fridge, real RGB-D 25 Hz, prefix 0–7 only

- 184,274 raw initial samples, **4,401** rigid groups, actual target occupancy 64; real median occupancy 23, 864 singleton groups. Reverse prefix: 184,640 samples and 4,431 groups (anchor = frame 7, **not** full frame 327).
- By frame 7, only **84,819 / 184,274 (46.0%)** original points have directly accepted factor(s). **99,455** are prediction only; 1,304 groups get no accepted factors at frame 7.
- Independent check from saved ZIP: raw endpoint-0 to endpoint-7 nearest-neighbor surface p95 ≈2.0 mm, yet forward-estimated endpoint-to-observation surface NN p95 ≈107.8 mm; for prediction-only subset p95 ≈159.8 mm; among predicted-only points 30.7% exceed 50 mm. These are *NN surface distances, not identity correspondence or GT error*.
- No group/point is certified static simply because first eight observed surfaces overlap. The huge discrepancy demands investigation.
- Group pose initially follows 12-state process with **`P_{xi,0}=I_6` in physical `[m/s,rad/s]` units** and `Qc=0.01 I6` (uncalibrated initial guesses). Ignoring Lie-adjoint terms, one 0.04-second step implies **40 mm one-sigma translation uncertainty per axis from initial twist alone**. At 0.016949 s it implies 16.95 mm. This is a MODEL ASSUMPTION issue, not a verified software arithmetic bug. A broad prior may permit wrong matches, then unsupported groups propagate ballistically; check with recorded factors and cross-covariances.

### 1.2 Blade `blade_103706`, eight cameras, 59 Hz, frames 0–7 only

From the actual saved `grouped_blade_103706_test_first02.zip`, NOT just the report:

- 34,369 anchored raw observations, **582** groups, target occupancy 64, actual group occupancy median **54.5**, 13 singleton groups, 56 groups containing ≤3 points, **145 groups with <10 points**. Median group radius **31.7 mm**; 95th percentile radius 41.8 mm.
- Eight-frame interval is **0 to 0.118644 s**, not the complete scene. Final frame 7: **32,454/34,369 (94.43%)** direct-updated points; **562/582** groups with accepted factors; **514/582** groups with numerically rank-six measurement information. Thus the blade failure CANNOT be attributed mainly to zero observation support.
- Despite support, **68 / 1,243** supernode-adjacency edges change inter-centroid separation by >10 mm; **113 / 1,243** exceed 5 mm; maximum change 127.3 mm. Such edges may genuinely cross a joint; they are *not* automatically erroneous. Their location and physically coherent component structure must be tested.
- Group-centroid motion from first to frame 7: median ~0 mm, 90th percentile **8.76 mm**, 95th **23.56 mm**, 99th **93.25 mm**, max 145.25 mm. Accumulated group rotation: median 0°, 90th 4.81°, 95th 11.04°, 99th ~21°. Some fitted tracks have very large angular velocities, others remain identically anchored.
- Same accepted-factor final correspondence errors: pixel norm median ~0.021 px, p99 ~0.527 px; **depth residual p95 ~3.36 mm, p99 ~15.60 mm**; ~2.39% have final |depth residual|>10 mm. A very low accepted *pixel* residual is compatible with incorrect assignment/overfitting and DOES NOT validate physical motion. The hard matcher selects factors and the fit is evaluated on them.
- Exploratory-only probe: a threshold-selected **67-supernode** connected subset has frame-7 stationary-model centroid RMS ~40.78 mm, free rigid in-sample ~38.77 mm and held-out-group ~39.69 mm. This is **not** the approved moving-part extractor and is not an accepted moving support; it highlights that simply fitting a rigid motion to an arbitrary connected drifting subset explains little of its discrepancy.
- Eight-frame browser reports all frames read and ALL points displayable; the viewer lacks raw multiview observation layers and physical labels and its "measured ROI" is the endpoint-derived fixed AABB. Browser load/LOD tests do NOT certify diagnostic correctness.

### 1.3 Failing Gate B: known unresolved issue

The initial 200-point/10-frame synthetic cabinet–door fixture generates a **119/81** proposed residual partition. One side has **two disconnected canonical surface components**, so it is rightly rejected. Late in the sequence the other side has only one independent training observation and instantaneous rank 3. A deficient instantaneous rank does NOT alone disqualify a whole temporally observable component, but no accepted connected cut, replay, two-motion refit, or proper held-out comparison has been achieved. Keep this fixture and its assertion intact.

### 1.4 Practical conclusion

The outputs establish exact imposed **within-group** rigid transport. They do not establish valid grouping, coherent **between-group** motion, true correspondence recovery, adequate pose/twist noise, or a successful joint. The blade and fridge failures are different: fridge loses factors; blade has plentiful factors but weak cross-group motion coherence. A one-size-fits-all "reacquisition" patch will not fix both.

## 2. Theoretical design correction: what RigEuler actually does vs current grouped RGB-D

### 2.1 Two distinct meanings of "supernode"

RigEuler Phase-1 has per-Gaussian nominal geometric/motion states and pointwise prior blocks. On a frame's linearized IEKF update, it restricts the CORRECTION direction with `delta_x = G delta_z`, where for a group-centered 3D point

`delta x_i = delta t_s + delta theta_s × (x_i - c_s)`.

The grouped normal/information operator is `(Gᵀ A G) delta_z = Gᵀ b`, with `A = Lambda_prior + Hᵀ W H`, and the image residual derives from RGB/depth/alpha rendering of the entire set. A group is primarily an **optimization basis**, not a rigorously identified physical part. The reference often builds groupings from the predicted per-frame geometry when a fixed partition is not passed. Consequently its free trajectories can show internal inconsistency and the later coherence extractor is genuinely needed.

The new Codex grouped RGB-D Phase 1 instead keeps a PERSISTENT physical rigid pose and twist for every initially chosen patch, with `x_i,k = T_s,k p_i` for ALL frames. This exactly preserves distances **even if that group wrongly straddles a joint**. It cannot reveal internal incompatibility from its own estimated trajectories, and its independent per-patch states make between-patch consistency entirely optional. It also loses the dense renderer's joint occlusion/silhouette signal and substitutes selected, exclusive, nearest-point RGB-D correspondences.

**Do not claim the two algorithms have the same observability, evidence, or grouping semantics.**

### 2.2 Required architecture for this project (retain exact local rigidity, but only CONDITIONAL on validated membership)

Keep the user's desired **exact local rigidity** on each CURRENT candidate rigid patch, and no per-point orientations. Fix what is missing around it:

1. **Evidence/observation layer independent of imposed rigidity.** Keep per-point raw `(frame,camera,pixel,z,R,rgb,matched/ambiguous/rejected)` and model-free world XYZ backprojections; build time-consistent motion hypotheses only from actual sensor measurements and stated covariance. Grouped trajectories alone must never validate their own group membership.
2. **Reversible canonical surface partition** `g_e(i)` with versioned epochs `e`. A provisional rigid patch is an explicit hypothesis; a held-out-evidence-supported graph cut can split it and refit its *entire* trajectory. A held-out-evidence-supported merge can reduce computational groups ONLY when the combined patch genuinely follows one rigid motion. Original point/sample IDs are immutable.
3. **Observable local tracking**, using physical RGB-D factors and a calibrated/declared prior for a shared group pose and (optionally) twist. For long unobserved spans, report prior-driven predictions and uncertainty rather than declaring motion evidence; test conservative pose-only model alongside 12D. **Do not** merely zero twists or clamp static.
4. **Cross-supernode compatibility inferred from data**, not assumed from spatial proximity. Verify shared metric motion using measured group centroids/held-out raw factors; before confident part membership, keep incompatible or unobservable inter-node links switchable/ambiguous. Local constraints must not force stationarity or weld door/cabinet.
5. **Motion support and physical-model reduction:** after improving local evidence, extract one coherent non-identity moving core versus static support; then **fresh** global rigid RGB-D optimization of the core, then **both** joint models compared on identical independent measurements. No geometry recovery yet.

Implement as controlled outer epochs: `initialize surface graph -> local rigid IEKF -> independent residual and motion-consistency audit -> connected split/merge proposals -> exact re-fit under accepted partition -> repeat until evidence-stable -> coherence -> single rigid IEKF -> joint comparison`. Never silently reuse a prior group's covariance as if it belonged to a new membership. Within any accepted epoch, local rigid transport must remain exactly rigid.

If a patch's current rigid membership is rejected but no connected alternative is identifiable, do **not** call it a valid physical rigid unit. Retain its observations and mark `GROUP_UNRESOLVED`; a local short-window independent XYZ diagnostic or original 6D pointwise baseline can help propose motion separation without making it the accepted output. Neither group predictions nor a mistaken pooled fit may be used to infer part labels.

### 2.3 Measurement likelihood and Jacobian to implement/verify

For canonical sample `p_i`, current group `s`, calibrated world-to-camera `(R_c,t_c)`, and observed raw factor `z_ic,k = (u,v,D)` with covariance `S_z`, define

`x_i,k = R_s,k p_i + t_s,k`, `q_c = R_c x_i,k + t_c`,

`h_c(x_i,k) = [ fx q_x/q_z + cx, fy q_y/q_z + cy, q_z ]ᵀ`.

For **left-world** group pose perturbation `delta_tau=[delta_rho,delta_theta]`:

`G_i = [ I3 | -[x_i,k]_× ]`, `H_ic = J_{pi}(q_c) R_c G_i` (3×6). The full 12D group measurement Jacobian is `[H_ic, zeros(3,6)]` at the appropriate Lie-chart Jacobian/retraction. A centroid-centered chart gives `[I,-[x_i-c_s]_×]` but changes the translation chart; do not mix conventions. Finite-difference these exact Jacobians and verify their adjoints for every camera.

An IEKF group update with fixed `A_s,k` associations minimizes

`E_s(T,xi) = (1/2)|| [Log(T T_minus^{-1});xi-xi_minus] ||^2_(P_minus^{-1}) + (1/2) sum_{(i,c,u) in A_s,k} || z_{c,u} - h_c(T p_i) ||²_(R_{c,u}^{-1})`.

Solve with **one fixed prior** through inner iterations and accepted true nonlinear energy, then carry a properly reset `12×12` posterior (if pose+twist variant). Each raw observation must have unique ownership. RGB may rank candidate associations but is NOT an image residual unless an explicit photometric model is introduced. Preserve calibration/mask/depth semantics; do not import Gaussian rasterizer to point-only code.

**Critical distinction**: selected fixed correspondences give conditional likelihood, NOT evidence that chosen correspondences are correct. Add a **held-out sensor-space predictive score**, evaluated on raw observations not used to fit pose or generate the segmentation/split. Include coverage/visibility diagnostics that separately account for: predicted canonical points unsupported by raw depth and observed surfaces not representable by currently visible endpoint samples. Do not force novel surfaces at other times to match canonical points.

### 2.4 Angular observability: mathematical test, not `rank == 6`

For group-centered points `r_i = x_i - c_s`, the local 3D Jacobian is `[I,-[r_i]_×]`. After removing the translational block (Schur complement), angular Fisher information scales roughly as

`I_theta = sum_i w_i [r_i-rbar]_×ᵀ [r_i-rbar]_×`, with coordinate- and sensor-noise-dependent weights.

Its eigenvalues depend on **spatial spread, direction and independent measured point support**. With a small radius `L`, rotational information scales `~ N L² / sigma_xyz²`. Therefore numerical rank 6 with patches only 1–3 cm wide can still yield unreliable rotation and misleading shared twist. `P_pose` must be interpreted with metres for translations and radians for angles; never rank or trace mixed units without physically normalized scales.

For each group/frame, export: all six normalized singular/eigen values, condition number, translational/rotational posterior uncertainty, centroid/radius, true distinct camera and point support, prior-to-measurement information gain in a declared dimensionless physical scale, last independently informative observation. No `rank6` boolean as the sole acceptance criterion.

### 2.5 Temporal uncertainty: fix the *assumptions* before blaming the integrator

For the current pose+spatial-twist state, `T_{k+1}=Exp(h hat(xi_k)) T_k`, `xi_{k+1}=xi_k`. A legitimate left-error model uses `A=[[ad(xi),I],[0,0]]`, `F(h)=exp(Ah)`, `P^- = F P^+ Fᵀ + Q(h)`, with correctly oriented signed-time noise factors. Existing analytic CPU/CUDA propagation tests reportedly pass. **Do not rewrite correct Lie propagation as a first reaction.**

For short intervals around `xi≈0`, `P_pose_pose(h)≈h² P_xi_xi + (h³/3) Qc`. With `P_xi,0=I6`, the initial 0.04 s fridge step gives ~40 mm 1-sigma per position direction before any sensor factor. This uncertainty scale must be declared and justified, not casually inherited from the Gaussian model; translational and rotational accelerations have different dimensions. Investigate using measured multi-frame displacement/flow and stationary control segments, with robust estimators and explicit assumptions; no GT axis/part labels. Reject a hyperparameter configuration that yields a deceptively wide matching gate and physically unsupported motion while still passing a chi-square gating test.

**Ablation required**: compare unchanged 12D pose+twist with a mathematically correct 6D pose-only random-walk (or properly specified constant-pose) model using the same raw observations and provenance. Record factor retention, held-out raw-depth residuals, prediction-only drift, state conditioning, and downstream rigid coherence. Neither model is automatically best. For 6D, do not create a faux velocity/covariance; document `T_{k+1}^- = T_k^+`, pose covariance evolution, and correct Lie reset. Maintain the 12D baseline until ablation evidence exists.

### 2.6 Identity/reacquisition and correspondence recovery, WITHOUT inventing points

Before changing the optimizer, classify every rejected candidate as `OUTSIDE_CAMERA`, `INVALID_MASK_OR_DEPTH`, `Z_OCCLUDED`, `GEOMETRIC_GATE`, `RGB_AFFINITY`, `RAW_ID_OWNED`, `NUMERIC_TIE`, `NO_LOCAL_CANDIDATE`, etc. Save candidate costs, innovation covariance, gate values, raw source IDs and original UVZ.

The present prediction-gated local nearest-neighbor matcher can self-reinforce a wrong twist: once away from the observed surface, it proposes no useful matches. Implement an auditable **multi-hypothesis rescue** for a group with long missing support:

- Pose prior hypothesis from 12D or 6D motion model.
- Last independently supported pose, properly transformed to the current time with a stated non-informative/conservative dynamics hypothesis; identity relative to canonical may be an explicit **competitor**, NEVER a silently imposed static label.
- Local measured-surface alignment hypothesis (e.g. robust projective RGB-D registration/point-to-plane) constructed from actual independent measurements using mask, calibrated depth and visibility, with covariance.

Evaluate these against the **same held-out raw observations**; choose or leave `AMBIGUOUS`, with covariance and raw-factor attribution. A group with no observations **may still be visible** (matcher failure): distinguish it from truly occluded or out-of-FOV geometry by depth-order tests. Maintain exclusive true factor ownership; do not turn one pixel into many independent updates.

### 2.7 Cross-group compatibility and graph-connected split/merge

Within a provisional rigid group all member distances are exactly preserved by construction; they cannot validate the group. For two groups `s,r` adjacent in the canonical **surface** graph, use their independently measurement-supported trajectories to compute intergroup edge-length strain and/or shared rigid-fit prediction residual:

`d_sr(k) = ||c_s(k)-c_r(k)||` and, if observable, `z_sr(k)=log((d_sr(k)+eps)/(median_{k in measured} d_sr(k)+eps))`.

Propagate centroid covariance through `d_sr` and DO NOT count two group centroids predicted with huge uncertainty as real rigid evidence. The same body preserves all cross-group distances and admits **one** shared rigid `T_k`; different bodies need not. A single near-axis edge can preserve distance across a real revolute boundary, so use a graph of multiple edges and held-out multi-point rigid prediction, not one-edge cuts.

#### Boundary split

Never make unrestricted residual clusters then merely reject them if disconnected. Generate **connected candidate partitions directly on the immutable original surface adjacency graph**, using residual displacement differences from independent actual RAW RGB-D factors across several times. Feasible methods: seeded graph-constrained min-cut over local train residuals, regularized graph labeling with connectedness enforced, or deterministic enumeration of graph-connected cuts near the incoherent interface; do not assume the true part count/labels. Propose multiple graph-valid candidates, compare on independent held-out factors.

For a candidate split `G -> G_A ∪ G_B` (both connected), fit entire time-coupled trajectories `T_A(k),T_B(k)` from their own canonical XYZ and training observations; this can be a filter plus backward fixed-interval smoothing conditioned only on training data (not the backward-from-final independent discovery pass). Marginalize over low-observation frames by propagation; **instantaneous rank 3 is not an unconditional veto** if other frames make the trajectory identifiable. Report total temporal observability and evidence.

Assess using the *same* held-out `(original frame,camera,pixel)` sensor factors for one- and two-body models and predictive covariance. Correct for extra parameters (out-of-sample likelihood / explicit complexity). Preserve cross-camera duplicated physical observations in the same fold. An empty/inconclusive comparison must yield `UNRESOLVED_SPLIT`, not accepting the split and not falsely declaring the original group physically rigid.

#### Optional coarsening

Consecutive well-observed, graph-adjacent groups `s,r` may be **MERGED** only if a single shared free rigid model explains their measurements on held-out frames/cameras at least as well after accounting for fewer parameters; otherwise leave distinct. NEVER merge just to attain 192/512 supernodes or to match RigEuler's Gaussian count. Coarsening is a geometric model-selection operation, not spatial downsampling. Every raw endpoint point stays represented.

### 2.8 Supernode count: model capacity/observability, not a magic 64

Fridge 4,401 groups for 184,274 samples (median 23; 864 singletons), blade 582 groups for 34,369 (median 54.5, 145 with <10). A large number of tiny independent poses allows localized overfitting and many weak rotation estimates. Yet blindly increasing the group size may **weld joint boundaries**. Both risks must be measured.

Keep all raw sample points and make the latent computational resolution adaptive. Run controlled occupancy ablations (`64` original, `128`, `256`, optionally alternative group count imposed only for ablation) and report **actual size quantiles**, graph connectivity, camera support, per-group pose information singular values, normalized held-out RGB-D likelihood, accepted split/merge, computational runtime and cross-node strain. Do NOT select a target occupancy by GT labels, visual appearance alone, or best training residual. If coverage creates isolated singleton/degenerate patches, report them as positional observations but **do not estimate unique unobservable rigid rotations** from them; allow their association to better-observed connected group only upon actual evidence.

### 2.9 Coherence and physical one-moving-part selection after Phase 1

Adapt RigEuler `edge_strain`, `candidate_hierarchy`, `cross_validated_rigidity`, `select_moving_component`, `refine_from_core` to **measurement-supported** group trajectories and true uncertainties. Do not use self-generated within-group rigidity as evidence. Build candidate components from inter-node strain and multiple independent boundary edges. Evaluate on held-out real raw RGB-D: static `T=I`, shared moving free `T_k`, and candidate-relative-to-complement model. A global rigid drift of every group is NOT articulated motion. Use known fixed camera world gauge only after verifying calibration; if complement not coherent, report `REFERENCE_UNRESOLVED`.

Select one confidently moving rigid **core** with per-group and per-point support; expand to additional points via moving versus stationary predictive error on actual independent measurements, not based only on magnitude. Allow `MOVING_RIGID`, `STATIC_COMPATIBLE`, `INCOHERENT_OR_AMBIGUOUS`; near-axis points can remain ambiguous. Freeze support before Phase 2. No assumption on moving fraction; no GT.

### 2.10 Fresh Phase 2 and joint estimation — only after support test passes

On selected `S_mov`, initialize `T_k` with uncertainty-aware Kabsch on independently measurement-supported canonical/current point correspondences. Then run a new shared-rigid `SE(3)` IEKF with **all selected points constrained by the same `T_k`**, directly against the original UVZ factors across all cameras. Require actual held-out improvement over static and unconstrained local alternatives. Carry appropriately correlated group uncertainty; do not treat thousands of positions generated by only a few group poses as independent sensors.

Given this free rigid whole-window trajectory, fit BOTH models with a fixed geometry:

- `P`: `x_i,k^P=p_i+q_k u_P`, `||u_P||=1`; its physical axis origin is NOT identifiable.
- `R`: `x_i,k^R=a+Exp(q_k [u_R]_×)(p_i-a)`, `||u_R||=1`, gauge `u_R·(a-c_ref)=0` (axis point along line unidentifiable).

Initialize with RigEuler `rigid_trajectory_reduction` analogues (PCA translation, finite `(I-R_k) a=t_k`, axis direction from relative rotations; unwrap angles in physical time); refine candidates against same rigid poses, compare models on the **same held-out UVZ factors/covariances** and report axis covariance/conditioning. Do not demand both models identifiable: valid P under pure translation wins even though R finite axis degenerates; valid R under pure rotation wins; when static/ambiguous/unobservable choose `JOINT_TYPE_UNRESOLVED`. Do not introduce an arbitrary residual-winner margin.

## 3. MISSING code or mathematical mechanisms to implement, mapped to local files

| Local file | Surgical responsibility | Do not do |
|---|---|---|
| `group_process.py` | Keep tested signed-time Lie factor; diagnose and calibrate `P_xi0`, `Qc`; pose-only 6D ablation as *separate named* option; physical scale diagnostics | Delete 12D, overwrite old `process6.py`, hide poorly excited motion |
| `group_iekf.py` | Verify group H and manifold reset; save per-factor pre/post residuals, distinct support, angular information; expose held-out factor-prediction and source attribution | Treat accepted factor training error as truth; independent per-point cov duplicate; fake uncertainty |
| `surface_supernodes.py` | Versioned connected canonical graph, real surface edge evidence, actual occupancy/radius/rank report; reversible/mergeable partition | Reduce endpoint XYZ catalogue; force arbitrary target number of physical bodies |
| `group_partition_validation.py` | Connected candidate generation **before** labels; compare one/two rigid models on same untouched held-out raw factors; correctly account for time-coupled low-rank frames | Relax original failing fixture; post-hoc disconnected rejection as the only proposer |
| `phase1_grouped_controller.py` | Alternating **accepted** partition epochs, exact replay/refit, gate/safety, both directions; continue storing chronological physical timestamps | Reuse invalidated group covariances; resume changed membership into prior checkpoint; run recovery |
| `group_pose.py` | Exact rigid `x=T p`; unit tests `SO3`, SE3 Jacobians, camera-frame and centered charts | Per-point rotation; projection with wrong coordinate gauge |
| `pointwise_rgbd/association.py` | **Read-only inspiration**; implement separate group-aware candidates, reason attribution and reacquisition path | Change baseline matcher in place; accept duplicate pixel factors |
| `group_visualization.py`, `assets/group_viewer.html` | Correct debug as specified in §4 below, with actual raw data and selection, not just an RGB/membership cloud | Hide moving motion as outlier using anchor AABB; QA only HTML-load counts |
| New `group_coherence.py` | Observed/uncertain cross-node strain, graph candidate hierarchy, real held-out moving-vs-identity model selection | Fit based on predicted-only or within-group constant distances |
| New `rigid_xyz_phase2.py` | Fresh common rigid IEKF directly against RGB-D | Relabel a posthoc Kabsch fit as Phase-2 optimization |
| New `xyz_joint_reduction.py` | Both candidate P/R fits, same-set held-out score, axes, identifiability and uncertainties | Invent joint if no real support |

If the current uncommitted files differ from this description, document the actual code path and adapt responsibly; do not regenerate everything from zero. Make surgical incremental changes and fail a controlled test if an unresolved issue prevents physical inference.

## 4. Broken debug: required replacement, with testable claims

Inspect the real saved `grouped_blade_103706_test_first02/viewer/index.html`: it creates **three point-cloud layers**—directly observed, prediction-only within a **fixed endpoint-defined bounding box**, prediction-only outside that box—colored by arbitrary computational group ID or canonical RGB. Its `manifest.measured_bounds` describes the **endpoint** geometry, NOT observed surfaces at each frame. Thus "outside measured ROI" is semantically wrong for a moving part at later frames; a truly moving prediction may lie outside the canonical box. The "outlier" toggle can conceal legitimate hypothesis points. It has no raw depth/foreground point cloud or 2D reprojection overlay, no per-group measurement residual/uncertainty, and no per-node supported-motion selection. Existing QA checks buffer counts, not whether scene reconstruction matches measurements. Do not claim screenshot/source of a runtime JS crash without reproducing one.

Replace/extend the viewer (style: supplied `rigid_phase2_debug.html`) with **honest instrumented layers**, no hidden point cap:

1. Load model points, raw measured foreground RGB-D backprojections for the ACTUAL chosen frame and camera(s), and matched measured points as distinct **visually distinguishable** overlays. Actual raw points come from the original dataset or exported measured arrays; never paint model predictions as measured.
2. Render raw XYZ anchor, fitted XYZ current, and actual measured XYZ; support switching point colors: RGB, immutable `supernode_id`, observation status, innovation residual, pose angular uncertainty, proposed/validated physical class, connected components. All class layers absent or labeled `NOT_ESTIMATED` until computed.
3. Show per-frame `frame_id`, physical time in seconds, current camera, number of raw valid foreground pixels, number accepted, rejected per specific reason, prediction-only groups, last direct-support timestamp, pose info singular values, chosen group size/radius, angular and translation standard deviations in correct units, filter iterations/energy. Per-group inspector with click/select and highlight source pixels.
4. 2D original camera RGB, depth/foreground masks, projected point positions, *actual accepted* UVZ rays, depth residual color map, rejected candidate reason overlays, and in/out-of-FOV/occluded masks; include actual 3D camera frusta. Use same original IDs and factor DB as inference. The image viewer must be able to expose a bad factor visually.
5. Show group-to-group original adjacency edges and change in **measured supported** edge lengths (units mm); toggle valid/invalid/ambiguous graph cuts. Show pose axes and trajectory for selected group; later show shared rigid pose and estimated joint only after established. Compare 6D and 12D runs side-by-side at identical timestamps when ablation exists.
6. **Correct ROI semantics**: endpoint/canonical bounding box may be a stable camera focus box, but do not call it a frame's measured ROI. For each frame, calculate raw measured masked XYZ bounds from raw sensor data and a separate full predicted bounding box. Classify prediction-vs-observation discrepancies geometrically with covariance, not simply by leaving the anchor AABB. Keep a stable camera option and `Fit raw measured`, `Fit current model`, `Fit both` controls.
7. Screen controls: fullscreen dark Three.js, right diagnostic sidebar, bottom full-timeline slider/play/pause/speed, orbit/pan/zoom, persistent buffers, deterministic point colors, and explicit ALL/LOD; point count and hidden layers must be displayed. Never pre-render just 512 or 4096 points without an explicit sampling control.
8. Browser QA must check actual loaded image/point buffers and **visually inspect** screenshots for first, mid, last frames. Independently compare projected 3D points with 2D pixel overlays within numerical tolerance; test scene extents/camera orientation at frame transitions; no WebGL errors; no stale double cache/race frames. For no full dataset availability, show an explicit `RAW_MEASUREMENTS_UNAVAILABLE` warning and do not mislabel model predictions.

Debug quality is a **blocking gate**: output has to let a scientist answer for any named group: *Which real camera pixels drove this motion, what was rejected and why, is its rotation observable, and which nearby groups predict the same actual measurements?*

## 5. Implementation sequence and mandatory gates

**Stage A — source and provenance audit.** Inspect all uncommitted modules, verify exact change inventory and 12D math, repeat original 45 baseline tests + existing grouped tests. Audit covariance units, camera transforms, raw factor ownership and actual pixels. Create `CORRECTION_SOURCE_MAP.md` with class/functions/lines, old/new behavior, branch/commit, and a manifest. Do not change unrelated code.

**Stage B — truthful debug + quantitative data diagnosis.** Before altering estimator numerics, reconstruct both eight-frame fridge and blade experiments from their RAW factor stores and viewer. Export actual per-group per-frame support, factor rejection reasons, RGB-D depth residual p50/95/99, distance from predicted point to raw measured foreground surface split observed/predicted, angular information spectra, covariance and supernode size; plot cross-group strain map. Explicitly identify the source for each wrongly predicted group. Fix viewer and rerun browser QA. Do not call a visually improved plot estimator convergence.

**Stage C — noise and correspondence corrective ablations.** Keep 12D as baseline; implement and compare 6D pose-only group model. Diagnose uncalibrated initial twist scale, use documented defensible assumptions/data-noise estimates and report these; no GT tuning. Improve observation reacquisition and group model selection, holding association/factor transaction invariants. Test static/translation/revolute controlled sensor fixtures with variable dt and partial occlusions; exercise multi-camera correlated provenance. Compare held-out RAW factors, not just train residual.

**Stage D — connected split/merge and epoch replay (mandatory).** Implement connected graph cuts from train residual evidence, whole-sequence two-rigid refit with temporal covariance, and same-heldout predictive model comparison. Pass the unchanged original Gate-B fixture; test disconnected candidate, no-motion, crossing joint, group with only one observed point in an intermediate frame, duplicate cameras and outlier depth. Gate must fail CLOSED if physical separation is unobservable. Do not slip a GT split into inference or weaken the test.

**Stage E — group observability/resolution.** Compare target computational occupancies 64, 128, 256 and actual group statistics on **both** real prefixes. Add reversible evidence-based merges where warranted, not a fixed 192-node target. A candidate group too small to identify rotation must be labeled `POSE_ROTATION_UNOBSERVABLE` rather than claiming a fully accurate SE3 pose from a floating-point rank. Both exact within-patch rigidity and surface-graph connectivity must hold in each accepted partition epoch.

**Stage F — independent temporal coherence and moving support.** Implement cross-group coherence with observation support, cross-validated rigid-vs-static-vs-complement, three classes and one moving part only. Synthetic positive/negative case must show valid moving core, static supported groups, and ambiguous regions; single global drift must not yield articulation. For real blade use informative longer interval, not only 0–0.119 s; do not classify a fraction from an arbitrary displacement cutoff.

**Stage G — fresh Phase 2 and two joint candidates.** Only after stages D/F validate, run shared free-rigid RGB-D IEKF, prismatic/revolute candidate fits and same-set held-out UVZ comparison, independent forward and reverse actual endpoint anchors. Report joint uncertainty/degenerate cases, axis sign equivalence, time-base consistency, rescaled rotation+translation comparisons, and motion observability. Compare candidate predictions directly to raw RGB-D rays and render accurate debug. No point recovery.

For every gate, report **tests attempted, tests passed, tests failed, reproducible commands, exact actual datasets/frame IDs and cameras, CPU/GPU peak, timing, file paths, valid numerical plots, unresolved limits.** Full sequence must not be declared complete merely because 8-frame prefix runs. Do not commit or push unless user or project instructions authorize it; keep old baseline untouched.

## 6. Quantitative acceptance criteria (non-arbitrary)

1. **Mathematics:** finite-difference verified `H`, Lie `Exp/Log`, signed-time `F,Q`, covariance reset, nonnegative posterior spectra and CPU/CUDA reference agreement within conditioning-aware FP64 bounds; exact within-group pair distances in every accepted epoch.
2. **Sensor evidence:** held-out negative log predictive density of actual **independent raw** `(u,v,D)` factors reported in physical units, with validity/factor-rejection status and uncertainty. At least two different motion models MUST be evaluated on the same factor IDs. Require no unexplained duplicate/background use. Better training residual alone NEVER constitutes improvement.
3. **Partition:** connected training-proposed split with correctly re-played filter results for both sides, objective out-of-sample improvement after complexity cost, and handling of low instantaneous observation support through time; original wrong-boundary test passes unchanged.
4. **No false positives:** static scene produces `NO_COHERENT_MOVING_COMPONENT` when evidence cannot distinguish motion from noise. Pure P and pure R fixtures fit corresponding observable model; correct joint-specific degeneration; one moving part only. No GT labels used in run.
5. **Observed versus prediction-only:** report drift and covariance separately and compare 12D/6D at same timestamps, including fridge and blade; no unsupported group can become a confidently moving part solely by large predicted displacement.
6. **Supernode count:** report actual occupancy distribution, pose information strength, graph edge strain, held-out residual and runtime across 64/128/256—not cherry-picked count and no coarsening across observed joint boundary.
7. **Real debug:** screenshots at multiple original frames visibly compare actual raw measured geometry and modeled geometry; raw UVZ overlay sampled pixel reprojection quantitatively correct; selected group inspector and layer selectors work. No hidden display caps or misleading fixed ROI.
8. **Outcome:** Phase 1 estimates measurement-supported coherent motion; Phase 2 fits one free rigid trajectory; joint fit returns P/R or explicit unresolved with parameter/observability diagnostics; forward and backward evaluated independently on genuine opposite endpoints. No recovery, no Phase 3.

### What to return

Return a concise implementation report with (a) exact source-level defects confirmed (vs speculation), (b) the mathematical modification with equations and covariance assumptions, (c) detailed modified-file map, (d) failed/passed tests, (e) independent raw RGB-D residual tables/plots, (f) annotated screenshots **from actual measured data**, (g) retained baseline and new run commands, (h) explicit unresolved issues. **No success claims in advance of evidence.**
