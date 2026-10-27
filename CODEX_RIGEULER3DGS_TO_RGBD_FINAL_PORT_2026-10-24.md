# CODEX IMPLEMENTATION CONTRACT — Finish the RigEuler3DGS → RGBDiekf XYZ/RGB-D port

**Date:** 2026-10-24  
**Target:** `SamueleMara/RGBDiekf`, reviewed `main` commit `f913d5628d29fd6c4494aeea5fad83874962dc40`  
**Reference:** `SamueleMara/RigEuler3DGS`, **pinned commit** `481b2cf2d5602932286948cf14114f9caee4de3c`  
**Target scene:** `.../preparation_8cams/partnet_subset_fixed_render_linear/blade_103706`, **all 60 frames, all 8 cameras**; currently authorized `--fps 59`.  
**Goal:** one high-confidence moving rigid part, one measured temporally coherent rigid SE(3) trajectory, a defensible revolute/prismatic joint, independent forward/backward comparison, then classify the existing uncertain endpoint points using **identity vs. estimated joint motion**.

## 0. EXECUTION ORDER — READ THIS FIRST

This is a **port and completion** of the existing working RigEuler method, **not a new research architecture**. The present RGBDiekf main branch already has much of the desired system; change only what is demonstrably different from the reference or broken on measured data.

**DO NOT**:

- Restart the old persistent-physical-supernode experiment, impose `x_i(t)=T_supernode(t)p_i` in Phase 1, implement boundary splitting, or fit multiple moving bodies.
- Introduce background rendering, Gaussian opacity/compositing, black-background losses, alpha carving, free-space appearance supervision, or a new renderer. We have **measured masked XYZ/RGB-D observations**, so use them directly.
- Split cameras into training/model-selection/assessment roles. **ALL available cameras participate in ALL production estimation stages, at EVERY time.** A camera selector in the viewer affects display only.
- Add an alternative 6D-vs-12D Phase-1 filter, redesign the proven pointwise covariance operator, invent thresholds, alter approved noise/association gates, use GT labels/joints or object names for inference, or insert/reconstruct new point geometry.
- Stop implementing downstream code because an unrelated real-data experiment is unresolved. Implement and unit-test the full pipeline on exact synthetic fixtures; keep **real-world acceptance** unresolved until measurements genuinely support it.
- Claim that `FIT`, low in-sample pixel error, covariance SPD, or `RESOLVED` from trajectory-only coherence proves correct articulation.

**DO**:

1. Read the pinned reference functions in §1, then inspect actual call sites in the RGB-D files listed there. Do not infer behavior from filenames or historical documentation.
2. Freeze and reuse compatible complete all-camera Phase-1 histories. First solve the **coherence → rigid → joint** failures from the saved results, not another expensive Phase-1 run.
3. Port only the missing mathematical mechanisms. Preserve all preexisting experiments in their original directories; write new experiments to fresh directories and record precise source commit/hashes.
4. Run a complete synthetic end-to-end regression, then a full measured blade forward and backward pipeline. Report *actual* accepted/unresolved states and quantitative diagnostics.

## 1. Source-to-source mapping: what to reuse, what to change

| Stage | Pinned RigEuler reference | Current RGBDiekf files | Mandatory action |
|---|---|---|---|
| Free Phase-1 motion | `estimator/sparse_iekf.py`, `estimator/supernodes.py`, `tools/kpart_pipeline.py` | `pointwise_rgbd/free_phase1.py`, `grouped_point_iekf.py`, `grouped_covariance.py`, `association.py`, `process6.py` | **Preserve.** Verify semantics: individual point `[x,v]`, computational grouped *increments*, not persistent group poses. All cameras. No unnecessary new filter. |
| Coherence | `estimator/motion_coherence_support.py` (`supernode_trajectories`, `canonical_knn_graph`, `edge_strain`); `estimator/coherence_extraction.py` (`candidate_hierarchy`, `cross_validated_rigidity`, `select_moving_component`, `extract_moving_component`); `tools/run_motion_coherence_support.py` | `pointwise_rgbd/free_coherence.py`, `reference_motion.py`, `reference_coherence.py`, `surface_supernodes.py` | **Fix representative-graph construction and audit observability.** Preserve reference mathematical hierarchy/selection; do not use computational groups as physical parts. |
| Kabsch initialization | `estimator/rigid_part_runtime.py`: `weighted_kabsch`, `kabsch_initialize` | `pointwise_rgbd/shared_rigid_rgbd.py`, `group_partition_validation.py` | Use chronological/processing-order-correct Kabsch poses **and spatial twists** as reference initialization. Export its per-frame unexplained-motion diagnostics. |
| Shared rigid Phase 2 | `estimator/rigid_part_runtime.py`: `RigidState`, `process_matrix`, `process_noise`, `predict`, `run_rigid_window`; `estimator/rigid_part_se3.py` | `pointwise_rgbd/shared_rigid_rgbd.py`, `canonical_schur.py` | **PORT shared pose/twist temporal IEKF** with direct all-camera RGB-D Jacobians. Do not let the existing independent-frame map fit overwrite the temporally estimated trajectory with unconstrained frame poses. |
| Joint after rigid motion | `joint_reduction/rigid_trajectory_reduction.py`: `relative_increments`, `one_dof_diagnostics`, `fit_prismatic`, `fit_revolute_axis`, `fit_axis_point`, `revolute_angles`, `reduce_trajectory` | `pointwise_rgbd/joint_rgbd.py`, `canonical_schur.py` | **PORT complete pose-space reduction and diagnostics FIRST.** Retain direct all-camera RGB-D refinement as a subsequent stage. |
| Two directions | Reference chronological/endpoint-gauge handling in `tools/run_rigid_se3_pipeline.py`, `tools/run_motion_coherence_support.py` | `pointwise_rgbd/joint_bidirectional.py`, runner | Convert each endpoint-anchored trajectory to common world/chronological relative poses; compare physical joint geometry without assuming point IDs correspond. |
| Support completion | RigEuler coherent moving core + shared rigid/joint inference; no rendering code needed | `pointwise_rgbd/joint_support.py` | Use existing endpoint points and **same observed foreground RGB-D**, score `I` versus endpoint-relative joint, preserve ambiguous/occluded cases. |
| Debug | Reference's trajectory/rigidity/reduction/axis diagnostics | `pointwise_rgbd/free_visualization.py`, `assets/free_viewer.html` | Expose measured-vs-predicted evidence for every stage, including **unselected** joint candidates. |

**Source provenance check:** Compare locally loaded reference file hash/revision with `reports/reference_port_provenance.json`. Current RGBDiekf `reference_coherence.py` and `reference_motion.py` already contain copies of reference formulas: prefer reusing them rather than copying the same algorithms again. Ignore historical `group_pose.py`, `group_iekf.py`, `group_coherence.py`, `raw_coherence_validation.py` as production sources unless explicitly used by the current runner. The unused `raw_coherence_validation.py` still refers to camera-partitioned factors: **do not re-enable it**.

## 2. Observation model: why point RGB-D is simpler than Gaussian rendering

For valid foreground observation `m` at frame `k`, camera `c`, the raw observation is

`z_m = [u_m, v_m, d_m]^T`, `R_m = sensor_covariance(d_m)`,

and a world point `x` is predicted by calibrated projection

`h_c(x) = [pi_x(K_c T_cw x), pi_y(K_c T_cw x), (T_cw x)_z]^T`.

The data objective for selected, uniquely assigned raw returns is

`E_sensor = (1/2) Σ_(k,c,m in valid foreground associations) (z_m - h_c(x_m(k)))^T R_m^(-1) (z_m - h_c(x_m(k)))`.

**All cameras `c=0..7` contribute to each estimate.** Keep the existing measured-depth temporal matcher, foreground-only positive validity, exact ownership bookkeeping for Kalman updates, and endpoint source priors counted **once**. RGB can retain its present appearance-based matching role; no photometric rasterizer is needed.

Physical hypotheses are simply:

- stationary world point: `x_i(k) = p_i`;
- moving rigid point: `x_i(k) = T_k p_i` (or endpoint-relative `T_k T_anchor^-1 p_i`);
- moving articulated point after reduction: `x_i(k) = J(q_k) J(q_anchor)^-1 p_i`.

The stationary surface is the **identity motion hypothesis**, not a rendered background. Measured foreground depth, projection and correspondence visibility are sufficient inputs to the existing point-based likelihood. An unobserved/occluded pixel is **not** a zero-residual static measurement or fabricated negative evidence. Do not use background RGB/depth to drive inference. In particular do not add model z-buffer gates that preempt the existing measured-depth matcher.

For left pose correction `T ← Exp(δτ^) T` and current `x=T p`, the point Jacobian is `[I, -[x]_×]`. The UVZ measurement Jacobian is

`H_m = J_(UVZ,c)(x) [ I, -[x]_× ]`.

This is the direct XYZ-only substitute for RigEuler's renderer JVP/VJP; no orientation of an individual point is needed.

## 3. Preserve Phase 1 and reuse its all-camera data

The correct Phase-1 state is **per-point** `s_i(k)=[x_i(k),v_i(k)] in R^6` with per-point marginal `P_i(k) in R^(6×6)` and already implemented cross-point coupling. The computational supernode correction basis is

```
δx_i = δt_g + δω_g × (x_i^canon - c_g)
δv_i = δu_g + δα_g × (x_i^canon - c_g)
δs = G δζ                  (restricted increment, NOT physical state)
```

Preserve current signed-time constant-velocity state prediction, covariance propagation, unique pixel factors, Joseph update and the closed covariance operator **provided the existing tests still pass**. `G` being fixed in canonical coordinates is the currently tested restricted approximation: label it as such, do not quietly reinterpret it as globally exact SE(3) motion.

Keep point trajectories `X_dir[k,i]`, `P_dir[k,i]`, `direct_observed[k,i]`, accepted raw factor IDs, canonical endpoint XYZ and original chronological frame IDs. `X` is stored in chronological order for both passes. Forward and backward have distinct catalogues and need not have matching point IDs.

**Reuse first:**

- `.../all_cameras_20261024/blade_full_64` or `blade_corrected_64` for an all-camera 64-occupancy source;
- `.../all_cameras_20261024/blade_full_256` or `blade_corrected_256` for an all-camera 256-occupancy source.

Read their source manifests, partition digest, camera policy, timestamps, source hashes, covariance semantics and actual completion before accepting caches. The **corrected_*** directories reused the complete Phase-1 histories; the source to pass as `--phase1-from` must contain `first/phase1` and `last/phase1`, compatible `manifest.json`, and `summary.json`. Do not assume mere directory existence means the cache is valid.

Group occupancy is a Phase-1 computational parameter, not an estimated physical part count. Existing eight-frame evidence: occupancy 64 → 582 groups, 128 → 315, 256 → 182; fitted error improves with larger groups but these are **in-sample** UVZ statistics, not a motion-accuracy verdict. Do not auto-change the filter grouping during the port.

**Tests:** preserve 45 baseline tests and current free-pipeline tests; assert no persistent group pose trajectory, all original cameras used, positive source depth, duplicate raw assimilation rejected, forward/backward signed-time, finite positive covariance, and unchanged old source hashes.

## 4. Coherence: exact reference method, independent fixed-count graph

**First downstream correction. Do this entirely from stored Phase-1 trajectories.**

The pinned reference `tools/run_motion_coherence_support.py` explicitly separates `--iekf-supernode-size` (default 64) from `--supernode-count` (default **192**) and builds the second set **by target node COUNT**, independent of Gaussian count. RGBDiekf currently uses `--representative-occupancy 512` via `connected_geodesic_assignment`, which changes the representative graph when endpoint point density/connectivity changes. **Fix this.**

### 4.1 Build independent representatives

In `free_coherence.py`, add a configurable `--coherence-node-count` with default `192`. Build representatives from **canonical endpoint XYZ**, not from per-frame transported positions, with deterministic farthest-point seeds and Lloyd nearest-centroid assignment as in `estimator/supernodes.py:build_supernodes(num_supernodes=...)`. All points belong to some representative; no GT, no semantics, no displacement magnitudes. For the point-only surface, optionally preserve existing measured-surface connectivity *as a documented alternative* where Euclidean assignments bridge disconnected surfaces, but the default port must reproduce/reference-test the pinned fixed-count algorithm first. Never silently swap grouping algorithms within one comparison.

Preserve `computational_membership` and **separately save** `coherence_membership`, representative count, centroid XYZ, number of member points, node trajectories and canonical kNN graph. Use canonical centroids with reference symmetric kNN `k=12`, built ONCE; do not update edges as points move. Reference trajectory for a representative is the mean of its free point positions at each time, with the same fixed member IDs.

### 4.2 Run reference extractor, not a new classifier

From reference `motion_coherence_support.py` and `coherence_extraction.py`, use:

1. `ℓ_ab(k) = ||X_a(k)-X_b(k)||`;
2. `z_ab(k) = log[(ℓ_ab(k)+ε)/(median_t ℓ_ab(t)+ε)]`;
3. `C_ab = sqrt(mean_t z_ab(k)^2)`;
4. graph-persistent component hierarchy with its existing multi-edge merge guard;
5. reference `cross_validated_rigidity`: folds of **representative NODES**, NOT cameras;
6. complement-relative Kabsch reference and world-frame identity comparison;
7. the pinned `select_moving_component` qualification/selection, including its **size-normalized world residual improvement** and exclusion of entire-graph global drift.

The existing reference `refine_from_core` is **not mandatory for initial high-confidence moving core**: keep optional recruitment separate from the frozen Phase-2 core. We deliberately use the joint later to classify initially low-confidence points. Do not turn early reference refinement into an unvalidated whole-scene core.

### 4.3 Audit sensor support without redefining coherence

This adaptation matters for explicit point association: free trajectories can contain prediction-only drift. The reference extractor measures kinematic coherence, not sensor observability. Do **not** change the formula above or insert displacement/point-count heuristics. Instead, publish for each candidate:

- representative IDs, all point IDs, node count and membership, internal/boundary strain, persistence, `eta_cv`, world-frame and complement-relative `RSS_R/RSS_I` and explained displacement **per node-frame**;
- how many frames/points of that component are directly supported by *actual* all-camera raw factors, number of prediction-only segments, covariance spectrum and per-frame coverage;
- complement support, reference-frame pose stability, world-identity residual and reasons for selecting/rejecting the motion core;
- **the measured all-camera RGB-D residual of a Kabsch trajectory fitted only from that candidate**, reported for diagnosis without using the joint type or GT to construct candidates.

A candidate spanning ~all endpoint samples but leaving an unsupported tiny complement is not automatically a trustworthy articulation; demonstrate that the chosen complement actually fixes a measurable world/static reference. **Do not introduce a moving-fraction upper bound or call 99.9% physically impossible by itself.** If the static reference or motion evidence is unidentifiable, return ambiguous with an explicit reason, not `RESOLVED` by default.

**Mandatory real regression:** compare `first`/`last` and Phase-1 occupancies 64/256 using the **SAME coherence-node-count=192**, same k=12 and identical extractor settings; record the resulting core sizes and membership/projection metrics. The previous 64→1130/212 and 256→34328/189 point discrepancy must be investigated from these records, **not covered by a threshold**. A stable physically valid core is the required output before Phase 2 may claim reliable motion.

## 5. Shared rigid Phase 2: port the actual RigEuler temporal SE(3) IEKF

**Most important missing mathematical mechanism.** The current `shared_rigid_rgbd.py` independently fits each frame and then optimizes canonical map + a large set of independent frame poses with `canonical_schur.py`. This can fit all-camera RGB-D yet yield frame-to-frame SE(3) motions that no single fixed joint can explain. RigEuler's working Phase 2 uses a **single rigid part state** with rigid **pose and spatial twist** in a sequential temporal IEKF. Add this, after freezing the selected high-confidence core. Do not touch the Phase-1 point-state model.

### 5.1 Initialization

Port `rigid_part_runtime.kabsch_initialize`:

- source `p_i` is this direction's *actual endpoint canonical XYZ*;
- for each frame fit proper `R_k,t_k` by reflection-safe Kabsch from `p_i` to the Phase-1 free `x_i(k)` of **only the selected moving core**;
- record per-frame Kabsch singular values, residual in meters, displacement in meters and `unexplained_fraction = mean_rigid_residual / mean_point_displacement` (report only, don't hard-threshold);
- obtain initial **spatial twist** from `ΔT_k = T_{k+1} T_k^(-1)` and `ξ_k = Log(ΔT_k)/Δt_k`, using measured timestamps;
- pin the actual endpoint pose exactly to identity `T_anchor=I`; do not assume the backward canonical geometry is `X(0)`.

### 5.2 Temporal state and uncertainty

One **physical** moving part (not per-computational-supernode):

`y_k = (T_k ∈ SE(3), ξ_k ∈ R^6)`, with `P_k ∈ R^(12×12)` and LEFT pose tangent `δτ` (translation-first), world/SPATIAL `ξ`. This 12×12 covariance belongs ONLY to the **one confirmed physical moving body**, in Phase 2.

The exact reference process is

```
E_k = Exp(Δt_k ξ_k^)
T_(k+1)^- = E_k T_k^+
ξ_(k+1)^- = ξ_k^+
F_k = [ Ad_(E_k)       Δt_k J_l(Δt_k ξ_k) ]
      [ 0              I_6              ]
Q_k = [ Δt³/3 Q_a       Δt²/2 Q_a ]
      [ Δt²/2 Q_a       Δt    Q_a ]
P^-_(k+1) = F_k P^+_k F_k^T + Q_k
```

Use RigEuler's `RigidState.process_matrix/process_noise/predict`, `run_rigid_window`, `frame_step` Lie correction and **geometrically correct error-chart covariance reset**. Port formulas and tested functionality without importing a Gaussian renderer. Keep the already authorized physical noise values or document explicitly how existing translational parameters map into the rigid process; do **not** invent new calibrated angular noise or silently pretend point velocity diffusion is an angular acceleration. Any truly missing angular process parameter must be an explicit documented assumption, not a tuned fix.

For backward processing, run in reverse **processing order** with positive elapsed `Δτ = t_previous - t_current` and reverse the final array to ORIGINAL chronological indexing; save processing order, endpoint and original times. The point Phase-1 signed-time semantics remain untouched.

### 5.3 All-camera rigid IEKF update from raw UVZ

At frame `k`, each selected moving point predicts `x_i(k)=T_k p_i` with

`H_m = [ J_(UVZ,c)(x_i) [I,-[x_i]_×], 0_(3×6) ]`.

Build factors from **all cameras** and the existing calibrated raw foreground association mechanism. When a rigid pose changes, recompute associations under that rigid pose (as existing `shared_rigid_rgbd.run` already does). Do not assume associations accepted under the free-point trajectories remain valid for an altered rigid pose. The accepted RGB-D factors share the one rigid state update; never treat N points as N separate rigid-body filters. Maintain proper original-endpoint priors once. Output an actual per-frame prior/update energy, measurement counts per camera, innovation statistics, accepted/rejected iterations, pose/twist covariance and observability.

For each frame the IEKF measurement itself has **zero direct twist columns**. Twist gains information from pose–twist cross-covariance through temporal propagation, just like RigEuler. Assert this in tests.

### 5.4 Integrate the existing canonical map refinement **without another estimator redesign**

`canonical_schur.py` already retains the measured endpoint XYZ priors and their coupling to motion. **Keep it.** However, its existing unconstrained `RIGID` fit must not overwrite the output of the reference-style temporal pose/twist IEKF. Make the temporal filter the authoritative rigid trajectory. For the first working port, refine canonical points **conditionally on those rigid poses** using the current Schur machinery (source priors once), and if the point positions change, run the same temporal IEKF again from the updated map. Do not silently reintroduce free per-frame poses or multiply already-used RGB-D measurements as a second independent information source. Export the original Kabsch poses, temporally updated poses, and fitted canonical XYZ separately. Retain existing covariance qualifications honestly; do not invent independence or artificially reduce the posterior covariance.

## 6. Joint reduction must be a faithful POST-HOC stage

Run the **pinned reference** `joint_reduction/rigid_trajectory_reduction.py:reduce_trajectory` on the completed **temporally optimized** rigid sequence BEFORE joint-constrained RGB-D refinement. This module contains NO renderer and is directly portable.

### 6.1 Normalize pose gauge before reduction

The reference expects chronological `T_0=I`. Forward already has this. For backward endpoint-anchored poses, first calculate

`S_k = T_k T_0^-1`, so `S_0=I`, using **the same physical world convention** for both passes. The underlying backward canonical endpoints remain at frame `K-1` and are NOT replaced with the frame-0 point catalogue.

All joint comparisons use these chronological `S_k` poses, proper left-perturbation covariance chart and world-fixed axis geometry. Do **not** assume forward and backward point IDs or endpoint coordinates match.

### 6.2 Port all reference trajectory diagnostics

Compute `ΔS_k=S_(k+1) S_k^-1` and `ζ_k=Log(ΔS_k)/(t_(k+1)-t_k)` in translation-first SE(3). Define characteristic part scale

`L = sqrt(mean_i ||p_i - mean_j p_j||²)`.

Use `ζ'_k=[v_k/L,ω_k]` and weighted SVD. Export

`γ_1dof = σ_1²/(Σ_j σ_j²)`

plus all singular values and per-frame increments, following the reference; `γ` is a **structural diagnostic, not a stand-alone hard joint classifier**. Use pose uncertainty with its correct LEFT chart as the reference does; do not silently use a right residual with left covariance.

Run the reference complete P/R fits **against the rigid trajectory**:

**Prismatic**: proper uncentered PCA of translations; `R^P_k=I`, `t^P_k=s_k u_P`; no physical axis origin is identifiable or required.

**Revolute**: finite rotation-log axis fitting (with rank/observability), fixed finite axis line via `(I-R_k)a=t_k` in a gauged perpendicular 2D plane, unwrapped angle `θ_k`; `R^R_k=Exp(θ_k [u_R]_×)`, `t^R_k=(I-R^R_k)a`.

Refine P and R in **pose space** with the reference Gauss–Newton routine. Save per-frame position/motion residuals and left/right `SE(3)` residuals, axis radius divided by part scale, axis observability, input vs reconstructed trajectories, `γ_1dof`, and whether the rigid poses were modified (**must be false**). In an endpoint with no excitation, report unobservable, never fabricate an axis.

### 6.3 Then refine P/R against ALL measured RGB-D

Keep the port's `joint_rgbd.py` (proper priors, canonical-point Schur uncertainty and the same all-camera measurement model). Initialize from **pose-reduced P/R**; refine each joint's **one fixed geometry plus time-varying scalar `q_k`**, maintaining the chronological/endpoint transform conventions.

Importantly, the current port reuses the final free-rigid `factors.sqlite` for both joint models without a joint-aware association update. Diagnose this explicitly: after the joint changes predicted positions, **reassociate measured foreground UVZ under the joint hypothesis**, using the existing matcher and ownership constraints. Do not train on physically inconsistent frozen correspondences. For candidate comparison, report the same complete raw foreground observation domain, the set of explained/eligible raw returns and nonexplained counts; do **not** compare two incomparable conditioned factor subsets and call their NLLs directly comparable. Avoid adding a new renderer or global all-pixel combinatorial optimizer. The fixed selected factor set from the original rigid fit remains available as an apples-to-apples *diagnostic* even when refitting associations.

Use measurement residuals to verify the hypotheses, but **do not turn the 1% pointwise association gate into a magical calibrated global chi-square cutoff** on millions of selected, correlated, in-sample residual scalars. The observed reported `χ²/(3N)` values (roughly 13–42) are a real mismatch under the chosen noise model that must be explained, **not silently forgiven**. Expose separately:

1. free rigid trajectory vs P/R in metric XYZ/pose space;
2. raw UVZ residuals for the free rigid, P and R motions over a common measured set;
3. per-camera/per-frame correspondence counts, depth residuals and unmatched/occluded observations;
4. candidate identifiability and predicted uncertainty.

If pose-space residual already fails: investigate core/temporal rigid trajectory/coordinate convention. If pose-space residual fits but raw UVZ fails: investigate association, depth, map and sensor model. **Never relabel a failed joint as successful by removing statistical diagnostics, changing covariances, or introducing GT information.** Joint type should be chosen only when the identifiable model is genuinely better supported; pure translation may select prismatic even when the revolute axis is degenerate, as already agreed.

## 7. Forward/backward comparison, then joint-guided support completion

**This is required; it is not the same as merging the endpoint point clouds.**

- Independently solve from FIRST and LAST endpoint catalogues using **all cameras for the entire sequence**. The catalogs are distinct; do not match their point IDs.
- Compare the physical joint type, direction up to sign, finite revolute **axis line** (not its arbitrary axial point gauge), scalar q trajectory after aligning sign and additive gauge, and reconstructed world XYZ motion.
- Independently retain candidate P/R comparisons even when the current code's final `selected` is null. Current `joint_bidirectional.py:run` stops early if either direction has no accepted selected joint; extend its *diagnostic comparison* to both fitted hypotheses without pretending they are accepted.
- Only combine the two directions when their geometry and actual raw evidence are compatible. A raw pixel `(original_frame, camera, pixel)` is one measurement even if associated in both directions. Combine unique sensor returns and distinct original endpoint priors with proper shared uncertainty; do not multiply two posteriors as independent.
- If only one direction has identifiable, measurement-supported joint and the other is weak due to occlusion, keep that one-direction estimate available for **conditional** support completion with its uncertainty. Do not require artificial symmetric success, but do not claim directional agreement when absent.

For each *already existing* point `p_i` anchored at its actual endpoint `a_i`, predict:

```
STATIC:  x_i^S(k) = p_i
MOVING:  x_i^M(k) = J(q_k) J(q_(a_i))^(-1) p_i
```

The joint transform and axis are fixed/common, but `q_k` varies across all 60 frames. Test both on **all eight cameras** and all frames where the hypotheses are visible and distinguishable. Accumulate original foreground RGB-D residuals using the **same candidate surface evidence** for static/moving, plus the documented source-point covariance and joint uncertainty; avoid double-counting raw returns as new independent Kalman updates. Occluded/unknown rays provide no fake penalty. Produce `STATIC_COMPATIBLE`, `MOVING_JOINT_COMPATIBLE`, or `AMBIGUOUS`, and return explicit evidence counts and residuals. Never insert new points; do not force near-axis or consistently occluded points into a class when indistinguishable.

## 8. Real debugging: must actually show where the pipeline fails

Keep `free_visualization.py`/Three.js, but repair it into an actual **measurement-driven** viewer. In both forward and backward direction with chronological 0..59 timeline display:

- raw RGB images, foreground depth/mask and full original measured XYZ cloud for each of the eight cameras;
- Phase-1 predicted/updated XYZ, direct-observation flags, computational correction groups;
- **separate** coherence representative graph (target 192), candidate node components, selected core, complement and core/complement Kabsch trajectories;
- canonical XYZ, `kabsch_poses`, `temporal_rigid_poses`, moving-core transformed predictions, source endpoint gauge;
- **BOTH** unselected revolute/prismatic joint axes, scalar q plots and predicted motion, even if `JOINT_TYPE_UNRESOLVED`; never hide both hypotheses just because selection failed;
- actual predicted-vs-observed `u/v/depth` maps, per-frame/per-camera error overlays/quantiles, matched and unmatched foreground pixels; observed residual lines, no fabricated blank-background error;
- final support classes and evidence when support completion was actually run. If unresolved, say NOT_EXECUTED.

Fix the current viewer's 250-ms fixed playback tick to follow `dataset.times` (allow speed adjustment), and show camera coordinate frames/world axes. Keep all raw points (no silent LOD), ensure fitted rigid/joint point arrays agree numerically with the saved transform/canonical state. A browser merely loading is **not sufficient**: inspect actual rendered first/middle/last frame screenshots and compare all eight camera projections to numerical source points. Save screenshot checks and computation-to-display error metrics.

## 9. Mandatory tests and quantitative outputs

### 9.1 Synthetic complete pipeline (not separate unrelated stubs)

Run the **full production call graph** using measured synthetic foreground RGB-D generated from two connected rigid surfaces with one static and one moving. Use at least: (i) known finite-axis revolute with speed changes/stops; (ii) pure prismatic; (iii) static/no distinguishable motion; (iv) part initially occluded but exposed at the opposite endpoint; (v) near-axis weakly moving points; (vi) nonuniform times and two endpoint gauges; (vii) exact/degenerate correspondence and multi-camera duplicate-return cases.

For both forward and backward, assert:

- Phase-1 remains individual XYZ/velocity + grouped updates; true positions move without permanent per-supernode rigidity; no invalid-depth/background accepted factors, all cameras contribute;
- fixed 192 coherence nodes independent of computational target occupancy (when enough points); held-out **node** rigidity selects moving surface without GT labels and never calls the stationary whole scene moving;
- reflection-safe Kabsch and rigid IEKF recover **one temporally consistent** shared pose/twist trajectory; pose Jacobians, propagation, covariance reset pass finite differences/dense small reference;
- trajectory reduction reports `γ_1dof`, both P/R hypotheses, stationary unresolved, true prismatic without invented axis point, true revolute with finite fixed axis;
- reproject both joint candidates using original all-camera foreground UVZ and check model agreement quantitatively against the known synthetic truth *only in post-hoc tests*;
- relative endpoint transport gives same predicted XYZ from canonical first/last geometry wherever observed; joint-guided completion correctly classifies seen static and moving endpoint samples and keeps unobservable samples ambiguous;
- viewer source arrays equal backend rigid/joint transforms; debug axes visible even for unresolved fits.

Use the **known synthetic GT only inside assertion/evaluation fixtures**, never to initialize or choose a model during the production run.

### 9.2 Real 60×8 blade, freeze Phase-1 first

1. On **saved** all-camera Phase-1 histories, run the corrected fixed-count coherence graph independently for occupancy 64 and 256 and both anchors. Save candidate hierarchy and observed support metrics. The extreme previous supports `first/last: 1130/212` (64) and `34328/189` (256) must be explained by actual candidate evidence rather than accepted as a physical segmentation.
2. Fit Kabsch and the **new temporal shared-rigid IEKF** to each defensible moving core, all cameras and all frames. Save per-frame pose, twist, covariance, process innovations and raw RGB-D counts/errors. Show the difference vs old independent-frame poses **without rerunning Phase-1**.
3. Run reference full one-DOF reduction, then both direct UVZ-constrained joint fits, preserving the original residual/motion/association reports. Diagnose precisely why the previous raw chi-square counts were enormous. Do not retune noise to force agreement.
4. Compare forward/back physical joint geometry and q. If a joint is genuinely identifiable/supportable, run classification of existing endpoint points; if not, save the full candidate diagnostics and keep classification NOT_EXECUTED on the real scene.
5. Rerun Phase-1 only if a **specific verified source defect** requires changing its observations/grouped correction model. A change to coherence-node count or Stage-C temporal rigid estimator is **not** such a reason.

### 9.3 Required report files

Within the new result root:

```
manifest.json                            # Git SHA + source hashes + all-camera policy
first/phase1/...                        # read-only reusable original cache or new verified run
last/phase1/...
first/coherence/{representatives.npz,candidates.json,selection.json}
last/coherence/{representatives.npz,candidates.json,selection.json}
first/rigid/{kabsch_poses.npy,temporal_poses.npy,twists.npy,pose_covariance.npy,diagnostics.json}
last/rigid/{kabsch_poses.npy,temporal_poses.npy,twists.npy,pose_covariance.npy,diagnostics.json}
first/joint/{trajectory_reduction.json,prismatic.json,revolute.json,comparison.json}
last/joint/{trajectory_reduction.json,prismatic.json,revolute.json,comparison.json}
bidirectional/{comparison.json,selected_joint.json if supported}
first/support/endpoint_support.npz       # ONLY if a genuinely usable joint
last/support/endpoint_support.npz        # ONLY if a genuinely usable joint
viewer/index.html                       # real observations, both candidate axes, overlays
reports/{validation.md,per_frame_metrics.csv,reference_port_diff.md,tests.txt}
result.json                              # complete vs unresolved, exact stage and reason
```

Report **numerical** pose-space P/R discrepancy (meters, radians), `γ_1dof`, per-camera raw UVZ RMSE and counts, missing-return support, core trajectory rigidity, degree of direct observation, temporal innovations, iterations, and forward/back axis line and q mismatch. For all metrics specify whether *in-sample/conditioned* or *synthetic GT*; never call same-set training measurements independent assessment.

## 10. Implementation restrictions and completion criteria

1. Add a clear new coherent runtime in the current `tools/run_free_grouped_rgbd_joint.py`; preserve the old CLI/outputs. Introduce `--coherence-node-count 192` (and migrate/alias the old `--representative-occupancy` explicitly), keep `--target-occupancy` for Phase-1 only. `--fps 59` for the current synthetic blade is the user's existing explicit assumption, not an inferred acquisition clock.
2. Keep the old baseline untouched, and leave legacy persistent-group files as deprecated, not called by production. Do not accidentally import a modified/reference plotting-only `coherence_extraction.py` instead of pinned revision `481b2cf2`.
3. Build exact reference mathematical behavior first, with small unit/dense comparison tests, and then replace only the renderer measurement operators by direct calibrated UVZ operators. This means **no extra rendering model** and **no hand-made segmentation rules**.
4. Do not silently discard a failed hypothesis or an incomplete direction: save the complete candidate outputs, reason and browser state. An unresolved result **does not** mean stop coding the next module; it only blocks asserting successful physical inference on that real run.
5. Do not claim success until the synthetic **full chain** works and the measured blade result shows: sensor-supported moving core, valid shared rigid temporal motion, one identifiable P/R joint consistent with full all-camera foreground evidence, physically comparable direction results, and nontrivial verified endpoint support classification. If blade data cannot identify the joint, explicitly say so and demonstrate with the saved diagnostic arrays; never manufacture a confident result.

### Suggested phased commits (small and independently testable)

- **Commit A:** fixed-count coherence graph + frozen-cache replay + candidate/observability diagnostics; no Phase-1 changes.
- **Commit B:** rigid Kabsch+twist init and 12D **single moving part** temporal SE(3) IEKF with raw all-camera UVZ; preserve canonical map uncertainty.
- **Commit C:** full pinned reference `reduce_trajectory` + corrected 1DOF diagnostics, P/R pose residual comparison and all-camera constrained RGB-D refinement/association verification.
- **Commit D:** forward/back gauge comparison, joint-guided endpoint support, reliable real-data viewer/overlays and full synthetic/real validation.

**Final instruction:** Implement the above in code and test it, not just a new report or speculative design. The reference already contains the needed coherence and rigid/joint mathematics. The XYZ-only port should retain the **same inference logic**, with Gaussian-rendering derivatives replaced by direct calibrated foreground RGB-D point projection.
