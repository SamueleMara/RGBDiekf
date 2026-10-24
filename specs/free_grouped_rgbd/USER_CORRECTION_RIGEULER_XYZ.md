# RGBDiekf — Correct implementation based on RigEuler3DGS

The previous instructions incorrectly turned computational supernodes into permanently rigid physical bodies and introduced an unnecessary boundary-splitting algorithm.

**Discard that architectural interpretation. Implement the actual RigEuler3DGS pipeline adapted to XYZ-only RGB-D points.**

## A. Phase 1: Free per-point IEKF with grouped updates

Use these reference implementations:

- `RigEuler3DGS/estimator/sparse_iekf.py`
- `RigEuler3DGS/estimator/supernodes.py`

Maintain a 6D `[XYZ, velocity]` state and 6×6 covariance for each tracked point.

Reuse the existing working RGBDiekf RGB-D projection, measurement covariance, correspondence handling and per-point process model.

Implement RigEuler-style grouped IEKF corrections:

\[
\delta s=G\delta z,\qquad
A_{\mathrm{group}}=G^\top A_{\mathrm{point}}G,\qquad
b_{\mathrm{group}}=G^\top b_{\mathrm{point}}.
\]

For each supernode, the correction has translation, rotation, linear-velocity and angular-velocity components, expanded into **XYZ and linear-velocity corrections of member points**.

These are grouped optimization coordinates, not persistent rigid poses or rotations belonging to individual points.

After every IEKF update, retain the corrected individual states. Propagate those states to the following frame using the original pointwise process model.

Do not enforce `x_i(k) = T_supernode(k) p_i` across the whole sequence.

Do not introduce physical boundary splitting, persistent group poses, per-supernode rigid trajectory filters, or a new 6D-versus-12D estimator ablation.

Build computational supernodes from canonical XYZ geometry. Keep their identities consistent across the sequence, so they can subsequently be used for coherence analysis.

Use a manageable computational group resolution, test grouping sizes for runtime and measurement performance, and separately construct the smaller representative-supernode graph used by the coherence extractor. Do not confuse either count with the number of physical moving parts.

Run the estimator independently:

- Forward: initialize from actual first-frame RGB-D points.
- Backward: initialize from actual final-frame RGB-D points and process the complete sequence in reverse chronological order.

Use real signed time differences and correct covariance propagation.

Store all individual XYZ trajectories, observed flags, actual accepted raw factors, and uncertainty.

## B. Motion coherence: discover the single moving part

Port the actual method from:

- `RigEuler3DGS/estimator/motion_coherence_support.py`
- `RigEuler3DGS/estimator/coherence_extraction.py`

Use the free point trajectories from Phase 1.

Construct representative-supernode trajectories and a fixed canonical neighborhood graph.

Measure temporal changes in distances between neighboring nodes:

\[
C_{ij}
=
\sqrt{\operatorname{mean}_k
\left[
\log\frac{\ell_{ij}(k)+\epsilon}
{\operatorname{median}_t\ell_{ij}(t)+\epsilon}
\right]^2}.
\]

Use the existing RigEuler graph-component hierarchy and held-out rigid-motion validation to extract a high-confidence moving core.

Compare the selected rigid motion against identity and, where necessary, the complementary reference motion.

Do not classify points as moving simply because their displacement is large.

Do not classify a computational supernode as physically rigid merely because its IEKF corrections were grouped.

Preserve ambiguous and poorly observed points for later model-based classification.

For now, extract one high-confidence moving rigid part, not multiple articulated parts.

## C. Phase 2: one shared rigid motion

Use:

- `RigEuler3DGS/estimator/rigid_part_runtime.py`
- `RigEuler3DGS/estimator/rigid_part_se3.py`

Fit one shared rigid transformation per frame from the highly confident moving trajectories using weighted Kabsch.

Then refine this common rigid trajectory with the original multiview RGB-D measurements.

All selected moving points must satisfy:

\[
x_{i,k}=R_kp_i+t_k.
\]

Unlike Phase 1, **this is where exact physical rigidity of the entire moving part is imposed**.

Do not introduce additional physical partitions or subparts.

Optimize a complete rigid trajectory for both forward and backward runs.

## D. Joint geometry

Use:

- `RigEuler3DGS/joint_reduction/rigid_trajectory_reduction.py`

From the optimized rigid trajectory, estimate:

**Prismatic:**

\[
x_{i,k}=p_i+q_k u_P.
\]

**Revolute:**

\[
x_{i,k}=a_R+\operatorname{Exp}(q_k[u_R]_\times)(p_i-a_R).
\]

Estimate fixed geometry and time-varying displacement.

Fit both hypotheses and compare their residuals against the same independent RGB-D measurements.

Use the previously corrected candidate-specific identifiability rule. A degenerate revolute fit must not prevent selection of a valid prismatic model.

Return unresolved if the measurements cannot distinguish them.

## E. Compare forward and backward joint estimates

The two passes have different point catalogues and different canonical endpoint frames.

Convert their estimated motion and joint geometry to the same calibrated world reference and chronological frame indexing.

Compare:

- Joint type.
- Axis direction, accounting for sign ambiguity.
- Revolute axis line, accounting for its axial gauge.
- Joint displacement trajectories, accounting for their endpoint gauge.
- Measured RGB-D residuals and confidence.

Do not assume point IDs correspond across forward and backward runs.

Use compatible joint evidence from both directions to obtain the best supported joint geometry and trajectory. Do not force agreement when they genuinely disagree.

## F. Complete the moving support using the identified joint

This is the important final step.

Once the joint motion is confidently estimated, revisit ALL initial and final endpoint points, including points excluded from the original high-confidence moving core.

For a point anchored at its actual endpoint frame \(a_i\), compare two predictions at every observable frame:

**Static hypothesis:**

\[
x_{i,k}^{S}=p_i.
\]

**Moving hypothesis:**

\[
x_{i,k}^{M}
=
T_J(k)T_J(a_i)^{-1}p_i.
\]

Evaluate both using the original masked RGB-D observations, camera calibration, depth ordering and valid measurement correspondences.

Accumulate the measurement evidence across the whole sequence.

Assign the point to whichever motion model is better supported by the available independent RGB-D evidence:

- STATIC_COMPATIBLE
- MOVING_JOINT_COMPATIBLE
- AMBIGUOUS

Do not force a decision for points that remain unobservable or whose motion models are indistinguishable.

This stage completes and improves the support of the **existing endpoint point clouds**. It does not yet insert arbitrary new points into the reconstruction.

## G. Debug visualization

Repair the viewer so it correctly displays:

- Actual raw multiview RGB-D points at the selected frame.
- Estimated free Phase-1 positions.
- Computational supernodes.
- High-confidence moving support.
- Optimized shared rigid motion.
- Fitted joint axis and motion.
- Final static/moving/ambiguous support.
- Actual RGB-D residuals and observed-versus-predicted point status.

Do not substitute a canonical bounding box for currently measured geometry.

Use consistent world coordinates, frame indexes, timestamps and colors. No hidden display subsampling.

## H. Required delivery

Implement the complete stages A through G.

Test each stage with independent synthetic inputs, then execute the full pipeline on the real articulated blade sequence.

Report forward and backward results separately and jointly.

Preserve the baseline implementation and existing data.

Do not block implementation of coherence, rigid fitting or joint estimation because of a previous test associated with the discarded physical supernode-splitting architecture.

Do not implement point recovery or K-joint inference yet.

**Success means obtaining credible moving support, a common rigid trajectory, a revolute/prismatic estimate, forward/backward comparison, and final joint-guided support completion—validated against actual RGB-D measurements.**