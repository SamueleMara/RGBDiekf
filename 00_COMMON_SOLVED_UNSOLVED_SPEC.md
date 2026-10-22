# COMMON MATHEMATICAL CONTRACT — RGB-D Solved/Unsolved SLAM Outer Loop

**MANDATORY FOREGROUND-MASK ADDENDUM:** The binary foreground masks are a hard admissible-measurement and predicted-visible-point feasibility constraint. Read and implement `CODEX_PARALLEL_IEKF_UKF_HARD_FOREGROUND_MASKS.md` before using this document. No background pixel can initialize SOLVED/UNSOLVED, be matched, enter an IEKF/UKF measurement, or be fused into geometry. Occluded solved points remain valid. Identical policy in both pipelines.


**Status:** Normative, shared without modification by the standalone IEKF and UKF implementations. No hybrid filter, no filter selection, no 3DGS, no implicit field, no surface normals, no separate static/dynamic pipeline.

## C0. Inputs and geometric variables

Input: synchronized calibrated multiview RGB-D observations at acquisition times `t[0:K+1]`:

\[
\mathcal Y_k=\{(I_{c,k},D_{c,k},M_{c,k},K_c,T_{cw,c})\}_{c=1}^C.
\]

`M` is a foreground observation-domain mask, NOT a motion or part label. Respect the actual selected scene's timestamps/FPS; `dt_k=t_{k+1}-t_k>0`. Depth is metric and invalid/no-return pixels have a separately documented observation model. Use all frames and all cameras for a production run; subsampling is allowed only in explicitly named diagnostic tests.

Backprojection of a valid depth pixel `u=(u_x,u_y)` into world coordinates:

\[
q_{kcu}=T_{wc,c}\,D_{c,k}(u)K_c^{-1}[u_x,u_y,1]^\top,
\]

with appropriate homogeneous coordinates. Raw observation IDs `(k,c,u)` are immutable.

At any outer iteration maintain ONLY two conceptual sets:

- `SOLVED`: persistent canonical 3D points `p_i`, fixed or revisable RGB `a_i`, originating observation provenance, uncertain rigid-group membership and map covariance/information references. A point is `solved` if assigned a currently accepted geometric/rigid-motion explanation. Initially the fused points observed at `k=0` populate `SOLVED`; their motion group memberships are provisional because frame 0 alone cannot identify motion.
- `UNSOLVED[k]`: raw foreground RGB-D observation IDs that the currently accepted explanation does not claim. These are NOT persistent state variables. All observations remain on disk and may be revisited; only a compact index, flags and local hypotheses are retained.

The groups are indexed `g` with shared poses `T[g,k] in SE(3)`; the number of groups may change under accepted outer-loop model edits. An identity trajectory `T[g,k]=I` describes the stationary case **without any distinguished static class or static-specific codepath**. A group can be parametrized relative to any reference frame `s_g` through `T[g,s_g]=I`; transform its canonical coordinates if its gauge is changed. No ground-truth joint type, joint angle, motion label or part segmentation participates in inference.

The sole predicted position of solved point `i` at `k` is

\[
x_{i,k}=T_{g(i),k}p_i,
\qquad
x_{i,k}=T_{g(i),k}T_{g(i),s}^{-1}q_{i,s}\quad\text{for a point acquired at }s.
\]

This relative transport formula is IDENTICAL forward and backward. Store no independently moving trajectory or velocity per point.

## C1. Direct RGB-D measurement and sparse association

For a solved point `p_i`, given a candidate pose `T[g,k]`, transform to camera coordinates `X_{ick}=T_{cw,c} T[g,k] p_i`, project `u_hat=pi_c(X)`, predicted depth `d_hat=e_3^T X`, and predicted RGB `a_i`.

**Canonical matched-observation model:** associate `i` with an *actual* target raw pixel `u` using a one-to-one or explicitly many-to-one sampling-aware matching model:

\[
z_{kcu}=\begin{bmatrix}u_x&u_y&D_{c,k}(u)&I_{c,k}(u)^\top\end{bmatrix}^\top,
\qquad
h_{ick}(T,p_i)=\begin{bmatrix}\pi_c(X)^\top&e_3^\top X&a_i^\top\end{bmatrix}^\top.
\]

`r=z-h`, with correctly configured covariance `R_{kcu}` for pixel localization / association, depth and RGB. The image coordinates are measurements of the associated target location; they are not arbitrarily interpolated depth. The RGB part informs association and/or the shared point's appearance; at fixed association and frozen appearance it has zero pose derivative. For direct photometric alignment instead, use `I_k(pi(X))-a_i` with an analytical image-gradient Jacobian and **propagate correlations between overlapping bilinear samples** from the original raw pixels. Do not count both forms as independent observations of the same RGB pixels.

Matching is local: project predicted solved points to each camera, use pixel/tile bins and a depth-buffer/order test, query compatible observed pixels in the projected support, and solve deterministic local assignment with an explicit `unassigned` alternative. Use the actual RGB/depth/pixel uncertainty and a declared model prior. No hand-chosen norm-based novelty threshold. Keep raw observation IDs so one measured pixel is not incorporated independently several times. Simple point samples and pixel quantization are adequate for the matching objective; **normals, TSDFs, surfels, Gaussian rendering and differentiable splatting are NOT required**.

A point sample does not reconstruct a continuous image everywhere. Interpret predicted RGB-D images as **partial sparse projections** with valid/unknown masks and z-buffering; do not claim dense-pixel likelihoods for uncovered pixels. Unexplained measured pixels stay in `UNSOLVED`. A full generative image model would require specifying pixel support; do not silently assume one.

## C2. Exact solved/unsolved outer schedule

### Initialization
1. Load cameras, masks, metric depth and source timestamps/FPS. Determine RGB-D and process-noise assumptions from the authorized existing configuration; distinguish assumptions from independent calibration.
2. Backproject and fuse overlapping **frame-0** observations across cameras through local correspondence, constrained by uncertainty and matching evidence. Do not concatenate millions of duplicates into permanent map variables.
3. Set `SOLVED <- frame0_fused_points`; set `UNSOLVED[k>0] <- all valid masked observations`; use frame-0 correspondence provenance to avoid treating the same measurements again as independent prior evidence.
4. Bootstrap rigid-group motion **from observed interframe correspondences**. Start with the fewest supported motion hypotheses, but do not freeze all frame-0 points into one un-splittable body. A persistent residual subgroup can propose a split; accept/reject through the common scoring rule in C5. No preassigned moving/static pattern.

### Forward tracking sweep
For `k=1..K`:
- Predict poses/twists and their uncertainty from the preceding state (IEKF or UKF according to the standalone implementation).
- Move only the points currently in `SOLVED` using common group transformations. Project, depth-order and find correspondences in observed RGB-D point measurements.
- Run the **inner filter for matched, supported group states only**. If correspondence structure changes, update it externally between inner solves, not by storing all pixels in the state.
- Mark matched measured observations as explained; leave others in `UNSOLVED[k]`. A point that disappears because it is behind measured foreground is retained in `SOLVED` but simply receives no surface observation update while occluded.
- Save filtered state, predicted state, associated observations, uncertainty and evidence needed for backward smoothing.

After `k=K`, smooth the SAME tracked motion trajectory backward via the filter-specific smoother. This alone does not discover new geometry.

### Late-to-early discovery sweep
For `s=K..1`, consider coherent local groups of `UNSOLVED[s]`, **not an eager permanent map entry for each pixel**:
- Test transporting candidate measured 3D points backward by each currently solved group trajectory `T[g,k] T[g,s]^{-1}`. Include a new rigid-group hypothesis if existing groups fail to account for joint cross-frame evidence. Identity trajectories are allowed automatically as normal poses.
- Use sparse backward projective matching and visibility with earlier RGB-D images. Behind a nearer depth measurement = possible occlusion, projected into measured free space = contradiction, off-image / invalid reading = uninformative. Evaluate with declared uncertainty, not an arbitrary fixed residual cutoff.
- Where several raw observations describe the SAME physical 3D point under one consistent rigid motion, fuse them into ONE canonical point with source provenance and covariance/cross-information. Missing or weakly constrained observations remain `UNSOLVED`.
- Optimize proposed new group motions using **the same chosen inner filter** operating in reverse observation order or by using its filter-specific temporal smoother; never create a second unrelated trajectory. Proposal construction may use deterministic weighted rigid registration as an initializer, not ground-truth labels.
- Accept a new point, group or re-assignment ONLY when the shared full-sequence score including unmatched likelihood and model complexity improves. Until accepted, keep edits provisional (transactional map edit).

### Progressive chronological frontier
For `s=1..K`:
- Reproject all currently solved geometry at frame `s`; recompute matched/unmatched observation IDs.
- For `UNSOLVED[s]`, search backward through `s-1..0` for consistent relative rigid transport, including existing-group and new-group hypotheses. Add accepted coherent geometry to `SOLVED` and revise the affected group trajectories.
- Regenerate `UNSOLVED` membership from observation ownership; no cumulative single-use 'discarding' of an observation.

Repeat: forward filter -> filter-specific backward smoothing -> late-to-early discovery -> progressive frontier until no accepted edits or meaningful accepted improvement remains under the fixed numerical optimizer precision and unchanged scoring definition. Never promise all observed points become solved: some observations are fundamentally unidentifiable or noisy.

## C3. No ad-hoc status threshold: deterministic decisions via a declared probability model

Every candidate association, unassigned observation, rigid group and fused point must be evaluated with a consistent negative log posterior:

\[
J(\mathcal S,\mathcal G,\mathcal A,\mathbf T,\mathbf p)=
\sum_{\text{owned raw observations}}[-\log p(z_a\mid\text{assigned point, pose})]
+\sum_{a\in\mathcal U}[-\log p_{\mathrm{unassigned}}(z_a)]
+J_{\mathrm{motion\ prior}}+J_{\mathrm{map\ prior}}-\log p(\mathcal G,\mathcal A).
\]

All model probabilities and priors must be explicitly specified/provenanced, normalized where required, and used in both pipelines. If the user supplied no prior or a parameter is genuinely not identifiable, report it. A standard observation likelihood permits a deterministic MAP decision (including `unassigned`) without a hand-coded distance threshold. **The exact combinatorial global association optimum is generally computationally intractable; local sparse assignment and group proposals are algorithmic restrictions, not magically exact global Bayesian inference.** Identify the optimization class and record accepted candidates.

A proposal is accepted if its score with the new model is lower than the same score with the old model; numerical tolerances relate only to solve accuracy. Unfavorable proposals are rolled back. Never count a source-frame observation as both an independent landmark prior and a new measurement.

## C4. Geometry uncertainty and shared motion uncertainty

For `p=T_{g,s}^{-1} q_s`, initialize `Cov(p)` using Jacobians w.r.t. RGB-D observation and pose:

\[
P_{pp}=J_q R_q J_q^\top + J_T P_{TT}J_T^\top
+\text{correlation cross terms (if present)}.
\]

Also store `P_{pT}` or sparse information factors; two new points sharing an uncertain source pose are correlated. Do not count them as independent motion constraints. Only the **conditional inner-motion filter** may operate on a small pose/twist covariance; joint map-motion marginal uncertainties require selected sparse covariance queries or marginalization with cross terms. Gauge fix one reference rigid pose for each canonical group chart; do not invert a singular full pose/map gauge. For new group or first observed point, retain a conditional prior, not falsely precise initial pose or identity-motion certainty.

Use a well-defined initial `P0`, `Q(dt)`, sensor `R`, and output-no-return/association model; read FPS/timestamps from actual input metadata. Approved numerical guesses from previously supplied configuration are assumptions and must be logged with provenance. Do not invent unrelated constants and call them measurements.

## C5. GPU and memory invariants

- Store `N_map x 3` canonical XYZ, RGB and group IDs and small covariance/information metadata; `UNSOLVED` stores indexed raw observations or compact per-frame bitsets.
- Camera rays and observations are **streamed one frame/batch at a time**. No `N_frames*N_cameras*N_pixels*N_groups` persistent tensor and no cloned XYZ per group.
- Project and bin only solved visible points; candidate discovery operates on current unresolved patches and nearby frames, streaming observations.
- Each accepted data association contributes an `O(1)`-width Jacobian of the relevant motion state, not a per-pixel state variable.
- Per frame memory `O(N_map+N_active+N_raw_frame+12^2 B^2)`, plus optional stored `O(K(12B)^2)` smoother covariance. Point/map cross-information is retained sparsely or via selected solves, never dense `N_map x N_map`.
- GPU kernels: batched `SE(3)` point transport, calibrated projection, sparse image-space matching, depth-order/visibility, segmented reductions or unscented low-rank moments, and optional deterministic rigid alignments.
- Respect physical RGB-D mask domain; do not treat entire unmasked environment as object geometry.

## C6. Shared outputs and mandatory comparable experiments

Each implementation outputs identically named *semantic* products under separate directories:

`solved_points.ply`, `solved_points.csv` (point IDs and group IDs), `unresolved_observations.csv` (frame/camera/pixel IDs and reasons/probabilities), `group_poses.npz`, `group_twists.npz`, `group_covariance.npz` (appropriately labeled conditional/marginal), `match_log.csv`, `stage_log.csv`, `objective_history.csv`, `forward_depth/`, `forward_rgb/`, `backward_depth/`, `backward_rgb/`, `run_config_resolved.yaml`, `validation_report.json`.

Both implementations MUST read the exact same dataset and matching/noise config, start with the exact same frame-0 fused points, use identical association proposals and objective definitions, and differ ONLY in their inner filtering and temporal smoothing. Do not favor one by using different sensors, segmentation, initialization, or GPU budgets.

### Acceptance criteria

1. No normals, surfels, TSDF, Gaussian renderer or dense point/observation state in either code path.
2. Full scene (all actual timestamps/cameras) loaded and used; masks, camera conventions and metric depths tested.
3. Frame-0 initialization uses only frame 0; source observations are not counted twice.
4. Successful measured forward tracking; nonidentity motion not artificially suppressed.
5. At least one real backward discovery attempt with actual source and earlier frame indices; new points fused with provenance if evidence permits.
6. Progressive frame-frontier pass revisits remaining unsolved observations.
7. Pose state covariance has correct process and observation updates; back smoother returns smoothed pose AND covariance.
8. Actual reconstructed RGB/depth projections exported with valid/unknown masks; no fabricated dense image coverage.
9. Objective must be evaluated on one unchanged likelihood/prior across outer updates; accepted stage values nonincreasing.
10. No `3**N`, no fixed whole-scene assignment patterns, no dense full-measurement covariance, no arbitrary per-observation permanent map variables.
11. Full run completes cycles; numerical tests alone or a first objective evaluation are not production success.
12. Synthetic occluded/delayed-appearance and motion-group-split tests recover the correct groups without ground-truth inference leakage.

### Explicit mathematical limitations

Matched-point correspondence is a discrete variable and rigid motion alone may be unobservable from one point or symmetric geometry. Neither IEKF nor UKF guarantees the globally optimal association or complete reconstruction of never-observed geometry. Monotonic accepted posterior scores do not prove convergence to the true physical scene.
