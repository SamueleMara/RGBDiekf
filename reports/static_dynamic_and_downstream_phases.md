# Static/dynamic extraction and downstream phases: how they work now

Code at commit `29a23a7`, plus the uncommitted debug and `--stage debug` changes, which leave the estimators untouched. Numbers come from the storage run `results_11_10_26/RGBDekf/projective_storage_full_20261028_0706` and the batch `projective_20261028_batch`.

Each direction runs the same chain on its own:

```
free pass F1 / B1 (one 6D IEKF per endpoint point)
  └─ E  coherence: static/dynamic extraction          estimator/projective_coherence.py
       └─ F2 / B2  rigid SE(3)+twist IEKF on the core  estimator/rigid_kalman_cuda.py
            └─ G  joint reduction + decision            joint_reduction/joint_projective.py
                 ├─ (q, q̇) IEKF   (accepted joint only)
                 ├─ support labels (accepted joint only)  joint_reduction/support_projective.py
                 └─ forward vs backward comparison (never fused)
```

Forward and backward never share points, labels or states. They meet only in the final comparison.

---

## 1. What the extraction receives

The free pass gives, for every endpoint point `i`, its filtered position `x_i(k)` at each frame and a status:

| Status | Meaning |
|---|---|
| 0 | the anchor (canonical endpoint) |
| 1 | measured: at least one valid RGB-D row at the final linearisation |
| 2 | prediction only: no valid row, so the state is the constant-velocity prediction |

The status files are `free_<d>/posterior_xyz.npy` and `free_<d>/point_status.npy`. No ground truth, and no image other than through these states, enters the extraction.

On storage, a point is measured in about 50% of frames.

## 2. Static/dynamic extraction (Stage E): coherence

**Entry point:** `run_coherence`, `estimator/projective_coherence.py:26`. It calls the pinned RigEuler extractor `run`, `estimator/motion_coherence_support.py:60`. That extractor reduces to RigEuler3DGS `481b2cf` when every weight is 1; the parity is tested in `tests/test_coherence_parity.py`.

### 2.1 Representative nodes, not points

- **Nodes:** `build_supernodes(canonical_xyz, 192)` (`motion_coherence_support.py:26`) seeds 192 nodes by farthest-point sampling from a fixed random point (seed 0). It then runs at most 25 Lloyd iterations on the canonical endpoint XYZ.
  - This is a fixed-count spatial partition, not a part count.
  - Every point inherits its node's label, so the static/dynamic decision is made per node: about 1,800 points per node on storage.
- **Node trajectories**, from `supported_node_trajectories` (`supported_coherence.py:40`):
  - `ω_s(k)` is the fraction of the node's members observed at frame k. The anchor counts as observed.
  - `x_s(k)` is the node's anchor centroid plus the mean displacement of its observed members.
- **Two policies run on the same free trajectories:**
  - **`pinned_raw`:** every point-frame counts as observed (ω ≡ 1). This is exactly the pinned RigEuler statistic. It drives the rigid pass by default (`--coherence-weighting`).
  - **`measurement_supported`:** a point counts only on frames where it was measured (status ≠ 2). This is the RGB-D adaptation, so prediction-only states are not read as observed motion.

### 2.2 Graph and strain

- **Graph:** `canonical_knn_graph(x_s(anchor), k=12)` (`motion_coherence_support.py:50`) builds a symmetric kNN graph once, at the anchor. That gives about 1,260 edges on 192 nodes.
- **Edge strain**, from `weighted_edge_strain` (`supported_coherence.py:56`):
  - The edge length at frame k is `ℓ_ab(k) = |x_b(k) − x_a(k)|`.
  - The reference length `ℓ̄_ab` is the median over supported frames.
  - The log strain is `z_ab(k) = log(ℓ_ab(k)/ℓ̄_ab)`.
  - The edge's strain is `C_ab = sqrt(Σ_k w z² / Σ_k w)`, with `w = ω_a ω_b`.
  - An edge with no support outside the anchor frame is removed: no evidence is not the same as zero strain.

### 2.3 Candidate components

`candidate_hierarchy` (`rigeuler_ref/coherence_extraction.py:50`) runs graph-constrained single linkage over edges sorted by increasing `C`.
- Every merge records the component it closes, with its birth cost, death cost and persistence.
- A merge needs two independent edges, capped by the size of the smaller side. This stops a single cheap bridge edge from chaining two bodies together.
- Candidates need at least 3 nodes. At most 48 are kept, binned by log₂(size) and ranked by persistence within each bin.

### 2.4 Held-out rigidity

For each candidate, from `supported_coherence.py:94-147` and `:176`:

- **Boundary statistics:** `edge_distributions` (`coherence_extraction.py:149`) reports internal versus boundary strain quantiles and the boundary edge count.
- **World-frame test:** `cross_validated_rigidity` checks the candidate's nodes in world coordinates.
  - The nodes are split into 5 folds.
  - For each fold and frame, a weighted Kabsch fit on the training nodes (at least 3 supported) predicts the held-out nodes.
  - It reports `rss_rigid_cv` against `rss_identity_cv` (the "nothing moved" prediction), and from those `δRSS_cv`, `η_cv = 1 − RSS_rigid/RSS_identity`, and a BIC.
- **Complement-relative test:** the same test, after expressing every frame in the frame of the complement nodes (`to_reference_frame`).
  - The complement's own Kabsch fit is removed per frame.
  - A frame where the complement has fewer than 3 supported nodes carries no evidence.
  - It also reports `relative_motion_ratio = sqrt(RSS_identity / Σω) / (member extent)`.

### 2.5 Selection

`select_moving_component` (`coherence_extraction.py:290`) qualifies a candidate only if all four conditions hold:

| Condition | Why |
|---|---|
| not degenerate | it has enough nodes to fit |
| at least one boundary edge | something exists to move relative to; the whole graph is never "the part" |
| `relative_motion_ratio > sqrt(ε_machine)` | it moves above the numerical floor |
| complement-relative `η_cv > 0` and `δRSS_cv > 0` | a common rigid motion predicts unseen nodes better than identity |

Among the qualified candidates, the moving side is the one with the largest world-frame held-out `δRSS_cv` per node-frame. The world frame breaks the symmetry because the cameras are fixed. If nothing qualifies, the result is `NO_COHERENT_MOVING_COMPONENT`. `δBIC` is reported but does not gate.

### 2.6 Refinement and labels

- **Refinement:** `refine_from_core` (`supported_coherence.py:150`) fits the selected core's per-frame rigid motion. Any other node is recruited if that motion explains its trajectory better than identity: the weighted error with motion is smaller than without, `E^M < E^I`. Seed and recruited nodes are stored separately.
- **Node labels:**
  - **MOVING_RIGID:** the refined core.
  - **STATIC_COMPATIBLE:** the members of the largest candidate whose complement-relative `δBIC ≤ 0`, i.e. a rigid motion does not beat identity.
  - **INCOHERENT_OR_AMBIGUOUS:** everything else.
- **Verdict:** this repository's wrapper (`projective_coherence.py:41-46`) turns the result into one verdict:
  - `RESOLVED` when the pinned status is RESOLVED and the core is non-empty and smaller than the whole graph;
  - `UNRESOLVED_FULL_GRAPH_SELECTED` when the core is the whole graph;
  - `NO_IDENTIFIABLE_MOVING_CORE` otherwise.
- **Outputs:** `coherence_<d>/<policy>/moving_core_point_ids.npy`, `representatives.npz` (membership, node trajectories, ω, edges, C, log strain, labels), `result.json`, and `coherence_summary.json`. The summary includes the agreement between the two policies.

### 2.7 What it gives on storage

| Direction | Policy | Seed + recruited = core nodes | Core points | Precision / recall vs GT (post hoc) | Node labels (moving / static / ambiguous) |
|---|---|---|---|---|---|
| forward | pinned_raw (drives the rigid pass) | 6 + 8 = 14 of 192 | 22,891 | 0.976 / 0.641 | 14 / 43 / 135 |
| forward | measurement_supported | 9 + 8 = 17 | 29,275 | 0.876 / 0.735 | 17 / 44 / 131 |
| backward | pinned_raw (drives the rigid pass) | 5 + 23 = 28 | 55,084 | 1.000 / 0.812 | 28 / 15 / 149 |
| backward | measurement_supported | 22 + 6 = 28 | 53,865 | 0.999 / 0.793 | 28 / 55 / 109 |

On storage, forward, 27 candidates qualified; on backward, 30.

### 2.8 How to read the labels (important)

- **Only the moving core is a positive decision.** "Static" is not a positive decision for most of the scene: 135–149 of 192 nodes stay INCOHERENT_OR_AMBIGUOUS, and STATIC_COMPATIBLE covers only one candidate's members. The extraction answers "which coherent part moves", not "label every point static or dynamic". A complete per-point labelling is the support stage's job (§5), and that only runs after a joint is accepted.
- **The extractor always finds a component.** It resolved a component on the static ray-cast scene too, as documented in `test_pinned_extractor_resolves_a_component_on_static_noise`. With the free pass's drift (33% of static points end more than 10 cm off on storage), some coherent drifting group can always qualify. Rejecting a static scene is left to the joint decision (§4), which currently fails at it.
- **It holds up despite the per-point drift because it works on node means over about 1,800 points.** Static drift is in random directions, so it averages out inside a node, while the drawer's motion is shared. That is why the backward core is clean: precision 1.00.
- **Recall is limited by the node granularity and the hierarchy.** Drawer nodes whose mean motion is polluted by drifting members, or that straddle the drawer boundary, are not recruited.

## 3. Rigid pass (Stage F: F2 / B2)

**Entry point:** `run_rigid`, `estimator/rigid_kalman_cuda.py:154`. It is skipped with `SKIPPED_UNRESOLVED_COHERENCE` unless the driving policy's verdict is RESOLVED.

### 3.1 Inputs

| Input | Role |
|---|---|
| Core points `p_i` | the moving core at their canonical endpoint XYZ and canonical RGB; the only measured points |
| STATIC_COMPATIBLE points outside the core (`:168`) | kept fixed at their canonical XYZ, with their source covariance, as occluders in the per-pixel point table only |
| Ambiguous points | not used at all: neither measured nor occluding |
| Free trajectories | not used, except for the core membership |

### 3.2 State and model

- **State:** `(T_k, ξ_k)` with `ξ = [v, ω]` in m/s and rad/s. The covariance is 12×12 in the left tangent: `T = Exp(δ^) T̂`.
- **Initial state:** at the anchor, `T = I` with zero pose covariance (the gauge). The twist prior comes from the noise config.
- **Prediction** (`rigid_predict`, `:37`), with signed `dt`:
  - `T⁻ = Exp(dt ξ) T` and `F = [[Ad(E), dt J_l(dt ξ)], [0, I]]`;
  - `Q` is the integrated white twist acceleration with density `Qc` (noise config block 6:12).
- **Measurement** (`chain_measurement`, `:47`): the same projective RGB-D terms as the free filter, evaluated at `x_i = R p_i + t`.
  - Same bilinear sampler, the same 4-channel residual `[a − I; Z − D]`, the same R and the same validity rules.
  - The terms are chained through `B_i = [I, −[x_i]×]`, so `H = H_x B_i`.
  - The pixel assignment is frozen per frame at the prior.
- **Update** (`rigid_update`, `:97`): an iterated EKF with a fixed prior and a left-chart prior error.
  - The prior error is `e = [Log(T T⁻⁻¹); ξ − ξ⁻]` with Jacobian `L = blkdiag(J_l(e)⁻¹, I)`.
  - Each step solves `A δ = b`, with `A = Lᵀ P⁻⁻¹ L + Σ Hᵀ R⁻¹ H` and `b = Σ Hᵀ R⁻¹ r − Lᵀ P⁻⁻¹ e`.
  - It stops when `δᵀb < 1e-10` or after 10 iterations.
  - The covariance is computed once, in the final pose's chart (no reset afterwards). A local-chart Joseph form is computed as a cross-check in the traces.
  - A frame with no valid row keeps the prediction.
- **Rigidity check** (`rigidity_check`, `:139`): pairwise distances between core points, as a numerical check (about 1e-12 m).

### 3.3 Storage results

| | Backward (B2) | Forward (F2) |
|---|---|---|
| Core points | 55,084 | 22,891 |
| Valid rows per frame (median) | 46,316 | 16,401 |
| NIS per dof at the prior (median) | 0.62 | 26 |
| Final pose error vs GT | 1.6 mm, 0.04° | 0.48 m, 92° |
| Pose σ | 0.13 mm at most | — |
| Frames not converged | 27 / 59 | 57 / 59 |

- **Backward is accurate but overconfident.** Its pose σ is at most 0.13 mm while the actual error is 1.6 mm: the per-row noise is treated as independent over about 46k points × 4 channels, so the information is overcounted.
- **Forward goes wrong in two stages.** It builds up a false rotation of about 0.11 rad by frame 39, while its translation stays close to the truth (0.39 m at frame 39, true 0.35 m). Then the valid rows collapse from 14,471 at frame 43 to 224 by frame 49, and it diverges.

## 4. Joint reduction and decision (Stage G)

**Entry point:** `run_joint`, `joint_reduction/joint_projective.py:169`. It needs a COMPLETE rigid pass and uses no images.

### 4.1 Reduction

The reduction is the pinned `reduce_trajectory`, `rigeuler_ref/rigid_trajectory_reduction.py:593`, vendored verbatim. It reads the filtered poses `T_k` and their 6×6 pose covariances, fed in processing order with the anchor first.
- **Prismatic fit:** the axis `e_P` is the weighted principal direction of the translations, not mean-centred because `T_anchor = I`. Then `s_k = e_Pᵀ p_k`, followed by a weighted Gauss–Newton refinement.
- **Revolute fit:**
  - the axis `e_R` is the principal eigenvector of `Σ w φ_k φ_kᵀ` with `φ_k = Log(R_k)`, i.e. from the finite rotations;
  - the axis point `a` minimises `Σ w |(I − R_k) a − p_k|²`, constrained to the point on the axis closest to the core centroid;
  - then the angles `θ_k` and a Gauss–Newton refinement.
- **Residuals and diagnostics:**
  - the pose residual is in the left chart: `Log(T̂ T_M⁻¹)`, weighted by the inverse pose covariance;
  - `γ_1DOF` is a rank-1 ratio of the increment twists, reported and not thresholded;
  - the far-axis ratio is the axis distance divided by the part size, also reported and not thresholded.

### 4.2 Decision (`reduce_and_decide`, `:52`)

The decision uses `α = 0.01` from `configs/image_joint_policy.json`. The anchor frame is excluded throughout. With `m = K − 1`:

| Test | Statistic | Distribution under the hypothesis |
|---|---|---|
| static | `S0 = Σ Log(T_k)ᵀ P_k⁻¹ Log(T_k)` | `χ²(6m)` |
| prismatic | `E_P = Σ r_kᵀ P_k⁻¹ r_k` | `χ²(6m − 2 − m)` |
| revolute | `E_R`, the same form | `χ²(6m − 4 − m)` |
| revolute vs prismatic (nested) | `E_P − E_R` | `χ²(2)` |

The resulting status is:
- **STATIC** if the static test is not rejected;
- **PRISMATIC** if prismatic is adequate and revolute is not significantly better;
- **REVOLUTE** if revolute is adequate and prismatic is rejected or significantly worse;
- **UNIDENTIFIABLE** otherwise.

Stated approximation: the pose marginals are treated as independent across frames.

### 4.3 Other outputs

- **Axis constancy** (`axis_constancy`, `:106`): each increment `Log(T_{k+1} T_k⁻¹)/dt` is compared with the fitted line, as a per-increment angle and distance. Reported only.
- **Candidates:** both models' axis, point and `q(t)` (gauge `q(anchor) = 0`) are always written, along with `q̇` by finite differences, whatever the status.

### 4.4 Storage results

| | Backward | Forward |
|---|---|---|
| Prismatic residual (rms) | 2.9 mm, 4.5 mrad | 0.50 m, 0.99 rad |
| `E_P` (dof 293) | 193,587 → about 660 per dof | rigid pass diverged |
| Revolute far-axis ratio | 194, i.e. a translation | — |
| Axis vs GT (post hoc) | 0.11° | 13° |
| q RMSE vs GT (post hoc) | 0.4 mm | — |
| Status | UNIDENTIFIABLE | UNIDENTIFIABLE |

- **Backward:** the prismatic fit is about 660 per dof, so with σ 26 times too small the χ² test rejects every model. The status is UNIDENTIFIABLE even though the geometry is excellent.
- **Forward:** the rigid pass diverged, so UNIDENTIFIABLE is correct there.

### 4.5 Outcomes across the batch

| Scene | Joint decision |
|---|---|
| blade (all subsampling factors), laptop, storage | UNIDENTIFIABLE |
| static ray-cast scene | REVOLUTE forward, PRISMATIC backward (false) |

The static scene's false joints are why `test_static_scene_identifies_no_joint` fails.

## 5. Downstream of an accepted joint

These stages have run end to end only on the synthetic ray-cast tests (`tests/test_four_passes.py::test_coherence_rigid_joint_support`, passing) and on the static scene's false joints. They have never run on a real scene, because no real scene has had a joint accepted.

### 5.1 (q, q̇) IEKF

`q_filter`, `joint_projective.py:130`. It runs with `--joint-q-filter`, on by default, for PRISMATIC or REVOLUTE.
- **State:** `[q, q̇]`, with `F = [[1, dt], [0, 1]]` and white-acceleration `Q`. The density and the initial `q̇` variance come from the noise config's translational or rotational block.
- **Core motion:** core points move as `x_i(q) = T_J(q) p_i`, with:
  - prismatic: `T_J = [I, q e]`;
  - revolute: `T_J = [R_e(q), (I − R_e(q)) a]`.
- **Measurement:** the same projective terms, chained through `∂x/∂q = e` (prismatic) or `e × (x − a)` (revolute).
- **Update:** a fixed-prior iterated update.
- **Stated approximation:** the axis and point stay fixed, and their uncertainty is not propagated.

### 5.2 Support labels

`run_support`, `support_projective.py:90`. It labels every free-pass point outside the core:
1. Each point is placed under two hypotheses at every non-anchor frame: static `p_i`, and moving `T_J(q_k) p_i`, using `q` from the q filter when it ran.
2. Each hypothesis is projected into every camera, and the real frame is sampled (`_costs`, `:70`):
   - the cost is `rᵀ R⁻¹ r` over the 4 channels;
   - a camera is unusable when its sample is invalid, or when the measured surface is in front by more than nσ (the point could be hidden or wrong).
3. The per-frame score `d_k` is the mean of `c_static − c_moving` over the cameras usable under both hypotheses.
4. One-sided tests run on the mean of `d_k` (`hac_mean_test`, `:34`): a trend-detrended long-run variance `max(Newey–West, AR(1))`, the HLN correction and `t(n − 1)`. Benjamini–Hochberg is applied per family at α (`:62`).
5. The labels are:

| Label | Rule |
|---|---|
| MOVING_CORE | the rigid core |
| MOVING_JOINT_COMPATIBLE | moving significantly better than static |
| STATIC_COMPATIBLE | static significantly better |
| AMBIGUOUS | neither |
| OCCLUDED_OR_UNOBSERVED | fewer than 3 frames with a common usable camera |

This is the only stage that gives a per-point static/dynamic labelling of the whole cloud.

### 5.3 Forward vs backward comparison

`compare_directions`, `joint_projective.py:206`. For each model type it reports:
- the axis angle (up to sign), and for revolute the distance between the two axis lines;
- the RMSE between the forward `q` and the backward `q` mapped to chronological order.

It runs whenever both directions produced candidates, even if both are UNIDENTIFIABLE. Nothing is fused. On storage: prismatic axes 13° apart, `q` RMSE 0.41 m. That gap comes from the forward divergence.

## 6. Where the chain breaks, in order

1. **The free pass drifts.** About 33% of static points end more than 10 cm off, and the drawer's direction is off by 16–24° per point. Coherence absorbs this by averaging over nodes, so the moving core is usable (backward precision 1.00). But the extractor never says "nothing moves", so static rejection rests entirely on the joint test.
2. **The rigid covariance is overconfident by about 26× in σ.** The per-row noise is treated as independent over tens of thousands of rows. So the χ² decision rejects every model even when the pose is mm-accurate. This is the main reason no real joint is accepted.
3. **The same overconfidence makes a static scene look like it moves.** Small, coherent spurious poses on the static scene are "significant" against tiny σ, so a joint is accepted there: the false positive in the failing test.
4. **The rigid pass can diverge** (storage forward, laptop backward). A false rotation builds up while the translation is still roughly right, then the core loses its valid rows.
5. **The q filter, support labels and anything after them have no real-data evidence,** because steps 2–4 block them.
6. **The noise values are uncalibrated placeholders** (`configs/example_uncalibrated.json`). Every χ² and HAC threshold assumes calibrated noise.

## 7. Where to look in a run

| Question | File |
|---|---|
| which nodes and points are the moving core | `coherence_<d>/coherence_summary.json`, `coherence_<d>/<policy>/moving_core_point_ids.npy` |
| why a candidate was or wasn't selected | `coherence_<d>/<policy>/result.json` (`candidates`, `selection`, `refinement`) |
| rigid pose per frame, convergence, rows, NIS | `rigid_<d>/numeric/timeline.csv`, `frame_XXXX.json` |
| joint tests and both candidates | `joint/<d>/result.json`; per increment: `reduction_increments.json` |
| forward vs backward | `bidirectional/comparison.json` |
| GT scores (post hoc only) | `evaluation_gt/evaluation.md` |
| visual | `debug/coherence_<d>.html`, `rigid_<d>_3d.html`, `joint_<d>.html`, `bidirectional.html` |
