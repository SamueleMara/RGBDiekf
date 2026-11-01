# CODEX — FINAL TARGETED TEST: JOINT PIXEL RESPONSIBILITY AND STATIC-DRIFT CAUSALITY

**Date:** 2026-10-30  
**Repository:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Primary case:** `storage_45135`, canonical point IDs **8444** and **45588**, source cameras **0** and **1**, respectively.  
**Mode:** **READ-ONLY PRODUCTION; ADD DIAGNOSTIC TOOLS/TESTS ONLY.** No changes to production IEKF, default configuration, prior/noise parameters, owner selection, masks, outer IEKS, or existing artifacts. No commit. No full-scene F1 refiltering.

## 0. The exact causal question

We now know that in the standard frame-1 F1 update:

- Point **8444** was reconstructed from source camera **0**, source pixel **(282,237)**, with canonical `x0 = [0.5880764625962563, -0.22182831062353028, 0.37366995913302475]` m and source-canonical RGB `[0.5764705882352941, 0.40784313725490196, 0.3254901960784314]`.
- Point **45588**, sourced from camera **1** pixel **(231,232)**, wins 8444's camera-0 pixel at frame 1. Their *canonical* 3D separation is approximately **1.694 mm**; the *frame-1 projected optical-depth difference at the collision* is approximately **0.103 mm**. Do **not** equate canonical separation to the two points' current separation unless the required frame-1 states are actually available.
- `projective_visibility.visibility` assigns one owner per rounded pixel (primarily ordered by optical depth, with deterministic point-ID tie-breaking). It suppresses 8444 in camera 0 as `DUPLICATE_PIXEL_SUPPRESSED`. The `per-frame` assignment remains frozen through its IEKF iterations, even if the frontmost identity changes later in the inner iteration.
- Camera **1 alone** supplies 8444's accepted frame-1 information. Its RGB discrepancy is already present at frame 0, temporally constant, and produces a **24.156 mm** false position correction, converted to approximately **1.41 m/s** velocity.
- Fifth-pass conditional replays through frame 10: `A_ALL_RAW` ~**229.7 mm** displacement; `B_SOURCE_RAW` ~**0** but loses usable observations; `SRC_PRIORITY_COUNTERFACTUAL` ~**176.8 mm**; `D_ALL_REFERENCED` **0** while retaining camera-1 rows. Thus forcing source ownership does **not** resolve the camera-1 likelihood mismatch.
- Critically, all fifth-pass interventions hold *other point trajectories* at their archived production trajectories. Transferring a pixel slot to 8444 while leaving 45588 unchanged is **not a coupled model** and is not sufficient to decide whether joint responsibility helps.

**Precise goal:** When **both** competing points' states and the measurements they claim are solved consistently, is the initial false displacement of 8444 caused by (H1) exclusive pixel allocation alone, (H2) a persistent camera-dependent observation discrepancy under a valid association, (H3) the actual physical association of a camera pixel with the wrong point/material patch, (H4) numerical nonconvergence, or an identifiable interaction? Establish the counterfactual effects on **both 8444 and 45588**, not just 8444, and on the **same observed pixels**, without using ground truth to choose assignments.

**Possible valid final result:** *material correspondence cannot be uniquely identified from this RGB-D data*. Prove such non-identifiability if applicable; do not force a winner. Calling static drift “fixed” merely because a point received zero rows is a **failed test**.

## 1. Preflight: audit what is actually available

Read local source and prior outputs, not an assumed public GitHub snapshot:

- `scene/endpoint.py`, `estimator/free_projective_run.py`: original point ID, `original_source_ids`, canonical RGB-D/provenance, `x0`, `P0`, frame manifest and timing.
- `estimator/projective_camera.py`, `projective_sensor.py`, `projective_terms.py`, `projective_visibility.py`, `point_kalman_cuda.py`: world/camera convention, bilinear and 2x2-mask support, same-surface/occlusion rules, exact residual/Jacobian/covariance, assignment freeze, 6D update and stopping/acceptance.
- `tools/root_cause_oracle.py`, `tools/f1_second_pass.py`, `tools/f1_sequential_causal_replay.py`, `tools/f1_source_responsibility_audit.py`, `tools/f1_source_responsibility_population.py`, `tools/f1_diagnostic_units.py`: reuse the float64 oracle and no-GT replay code. No copy/paste duplicate estimator.
- Relevant reports: `second_pass_20261030/SECOND_PASS_ROOT_CAUSE.md`, fourth-pass causal report, fifth-pass `SOURCE_RESPONSIBILITY_CAUSAL_REPORT.md` and the relevant CSVs.
- `RigEuler3DGS/estimator/renderer_bridge.py` and `estimator/sparse_iekf.py` only to confirm that it freezes contributors during **one linear solve**, then re-renders/re-sorts following accepted updates. **Do not** implement a 3DGS/Gaussian renderer in RGBDiekf.

Locations:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
FIFTH="$FOR/fifth_pass_source_responsibility_20261030"
OUT="$FOR/final_joint_responsibility_8444_45588_20261030"
```

Before implementing, establish **exact** availability for each frame 0–10:

1. Both point IDs' full forward F1 predicted/posterior 6D states and 6×6 covariances (for 45588, **do not assume** it was previously traced); source/point indices after subsampling; acceptance decisions; per-camera row flags.
2. Predicted positions, depths and pixel-owner candidates of *every other point* affecting the local camera-0/camera-1 pixels (e.g. fifth-pass records already show competitors 8298, 312769, 8595, 8301 in some 8444 inner iterations). These are not a fixed two-point-only scene.
3. Whether archived arrays actually store the required frame-1 **global** point positions and covariances, rather than only every 10th checkpoint or a compact per-ID trace. If exact covariances are unavailable, **do not fabricate them or report parity**. At frame 1, reconstruction from canonical frame-0 `P0` plus the exact production prediction can be valid; verify numerically.
4. Original multi-camera RGB, depth and binary masks; intrinsics/extrinsics, row-gate policy, calibrated `R`, `P0`, `Q`, `dt`, nonlinear iterate policy. Save SHA256 manifest, first original production array hashes, code git status and module hashes.
5. An explicit *reconstruction tier*:
   - **TIER_EXACT_ARCHIVE**: full states and all external competitor positions at a frame can be reconstructed/verified against actual production;
   - **TIER_CONTROLLED_LOCAL**: pair states exact, but external competitors frozen/approximated from an existing saved snapshot, hence claims conditional;
   - **TIER_UNAVAILABLE**: necessary pair/collision information missing; do not run a misleading “joint” test.

The primary **frame-1** test must be achieved from frame-0 inputs and full canonical point geometry if possible. Do not perform a prohibited all-scene trajectory rerun to fake missing frame-1 histories. A single frame's **projection/ownership** of the complete canonical point cloud is permitted, as is local CPU updating of a small selected cohort. Do not update the entire point cloud.

## 2. Phase A — independently reconstruct the exact baseline conflict

Create diagnostic `tools/f1_joint_responsibility_audit.py`; no call or import from production runner into a new training/inference route. The first diagnostic arm **J0** must reproduce the original frame-1 prior and update of point 8444 with source camera-0 suppressed and camera-1 admitted:

- `s_minus`, `P_minus`, exact `F,Q`, full 6×6 MAP Hessian and RHS, point/camera row contributions, NIS and per-inner-iteration support; final `s_plus`, `P_plus`, accepted rows, stopping flag, reason codes.
- Reconstruct **45588** from its own source pixel; independently evaluate its first-frame state and all its admissible measurement rows. Report whether its exact production frame-1 posterior is verifiable. Never claim production parity for it without archived evidence.
- At the collision, log *both points*' projected subpixels, rounded pixels, depths and their difference, depth uncertainties, original per-pixel winner, mask/depth-cell validity, row-gate status, per-row information and source-camera provenance.
- For pixel (0,282,237), report all occupants (including other points), not just 8444 and 45588. Likewise at camera-1 projected/source pixels (around (231,232)) and any other collision pixels affecting the pair.
- Compute the *original exact* camera-1 RGB/depth residuals on 8444, the corresponding frame-0 differences, and all four component force terms under the actual `H,R`. Distinguish what is a genuine sampled foreground pixel versus the **same material point** (usually unknowable).
- Prove algebraically and numerically that initial Δv comes from `P_vx P_xx^{-1} Δx` with separately labeled units.

**Mandatory Gate A0:** 8444's original first-frame state, covariance, rows and acceptance agree with saved production, using previously validated unit-safe tolerances; for 45588 declare either `PARITY_PASS` or `PARITY_NOT_VERIFIABLE` with precise missing arrays. If 8444 parity fails, stop intervention claims and diagnose parity; do not substitute a plausible-looking answer.

## 3. Phase B — construct a genuinely coupled local observation problem

### 3.1 Variables and world state

For the main two points maintain **both** states and full covariances:

\[
 s_{a,t}=\begin{bmatrix}x_{8444,t}\\v_{8444,t}\end{bmatrix},\qquad
 s_{b,t}=\begin{bmatrix}x_{45588,t}\\v_{45588,t}\end{bmatrix},\qquad
 S_t=(s_{a,t},s_{b,t})\in\mathbb R^{12}.
\]

The *joint* prior is the block diagonal of the actual two production 6×6 priors **only if those priors are initially independent**; prove this from the initialization, do not assume independence after a coupled update. Once a shared pixel has been assimilated in a genuinely coupled way, preserve the induced **cross-point covariance** through subsequent frames. A block-marginal approximation must be clearly labeled as such and never described as exact joint Bayesian propagation.

Other points form an *external environment* `E_t` used for measured visibility and pixel competition. They must be obtained from verified original states or exactly predicted canonical states and must be explicitly **frozen** in a local diagnostic. They are not silently updated. If the selected pair is not sufficient to explain the collision because another candidate wins, extend to the **minimal locally relevant cohort** while keeping a finite, explicitly listed set; do not suppress external competitors or invent full-scene consistency.

### 3.2 One measured pixel, one likelihood (no independent double use)

Treat each real camera pixel (and, separately, its bilinear 2×2 support when sampling subpixels) as a **unique observation**. A camera-0 pixel used by 8444 cannot simultaneously be assimilated independently by 45588. The same restriction holds across any camera/point pair in the cohort. Record every pixel ID and every point whose footprint/sampling kernel uses it. **Do not mislabel bilinear sample overlap as exact same-pixel ownership:** detect/report overlap and its shared sensor noise covariance separately.

Define an explicit local observed-pixel set \(\mathcal U_{c,t}\), chosen from the union of fixed, known source pixels, the original predicted collision pixels and their necessary interpolation neighborhoods. Choose this set **before** comparing ownership hypotheses; keep it identical between arms. All pixels must remain in the hard foreground masks. Never pull background data into the likelihood.

For each pixel \(p\in\mathcal U\), an exclusive latent assignment \(z_{c,p,t}\) can take a physically possible point identity (8444, 45588, explicitly identified external frontmost point) or an explicit **unexplained/other** category if supported by a principled nuisance model. Source provenance is observable and persistent; future ownership is **not** known merely because the point originated in that camera.

For a fixed assignment \(z\), the factor must evaluate the **same observed RGB-D pixel values** under one candidate point's predicted geometry and appearance, or a well-specified null/other model. Do not switch to a different target pixel without recording a correspondingly different observation event, otherwise comparing costs across assignments is invalid. Carry all pixel-ownership, eligibility and mask/visibility constraints into the candidate model.

A possible proper mixture model, for a fixed observed-pixel universe, is

\[
  p(Y_{\mathcal U}\mid S_t)
  =\prod_{(c,p)\in\mathcal U}
    \sum_{i\in\mathcal C_{c,p}}
      \pi_{i,c,p}\,p(y_{c,p}\mid s_{i,t},i,c,p),
\]

with **normalized** assignment priors and per-component likelihood normalizers. This is a diagnostic candidate, **not permission to treat two assignments as independent observations**. A shared/interpolated pixel, uncertain reference RGB and jointly estimated ownership invalidate naive product-independence; either use proper shared-noise modeling or label this a conditional approximation.

**Do not compare MAP costs with different row counts by simply dropping rejected rows.** If an observation is assigned to `OTHER` or censored, use a *specified* likelihood for that outcome, consistent across arms. If that likelihood cannot be calibrated from existing measurements, perform only **matched-support likelihood comparisons** and say that absolute model evidence is **not identifiable**. Arbitrary null penalties, a hand-chosen source bonus or omitted-row-zero-cost are prohibited as deciding evidence.

### 3.3 Real image residuals versus candidate point identity

Implement **two explicitly separate diagnostic representations**:

- `LEGACY_PROJECTION`: faithfully reproduces current RGBDiekf candidate row sampling at the *candidate point's changing projected subpixel* and the current bilinear/depth Jacobians. Use this only for conditional replays and parity; rows under different projected supports are not assumed directly comparable.
- `FIXED_OBSERVATION_ASSOCIATION`: holds the observed RGB-D pixel(s) fixed and gives each candidate point an explicit predicted measurement and pixel-location/footprint term, with correctly transformed noise and mask support. Work out whether a *point with no size/footprint* supplies an identifiable prediction at a particular observed pixel. If not, report that a consistent fixed-pixel likelihood would require an additional correspondence model; do **not** smuggle a Gaussian footprint, arbitrary 2D tolerance or zero-cost assignment into the test.

For any newly introduced image-coordinate residual, state \(h\), pixel noise \(R\), Jacobian \(H\), coordinate conventions and source of the uncertainty. Validate finite differences. No invented confidence parameter is allowed to decide the main hypothesis.

### 3.4 Coupled information and covariance

If a differentiable shared-pixel factor is supportable, the 12D normal equations must contain cross-point blocks:

\[
 A=\Lambda^-+\nabla^2_{S}E_{\mathrm{pixel}}(S),\qquad
 b=-\nabla_{S}E(S),\qquad A\Delta S=b.
\]

Where Gauss–Newton/Fisher is used rather than the exact Hessian, name the approximation and show its construction. For exact discrete assignment-conditioned fits, the two state blocks may be independent *conditional on assignment* but assignment uncertainty couples them after marginalization—do not invent nonzero off-diagonal Hessian terms. For a soft mixture, verify genuine mixed derivatives numerically. The posterior covariance is **not automatically** `inv(A)` under a non-Gaussian mixture; if using a Laplace approximation, label it, test symmetry/PSD and report multimodality.

Enforce consistent one-frame nonlinear optimization: fixed prior for all inner iterations; no reassimilation of the same pixel per iteration; step accepted only on a comparable full-data objective; deterministic ordering; no truth-based choices; record convergence and all local minima explored. Do not quietly change the original IEKF acceptance rule in baseline J0.

## 4. Phase C — targeted factorial interventions, not a source-only shortcut

Implement the following arms **on the exact same frame-1 initialization and measured pixel universe**. Run with camera-0/camera-1 current masks and visibility. If a proposed arm is not statistically well-defined from available data, label `UNDEFINED_MODEL` with proof rather than returning zero displacement.

| Arm | Ownership / correspondence | Observation model | Why |
|---|---|---|---|
| **J0 archived F1** | Production nearest-pixel winner, both points evaluated with frozen production eligibility | Absolute raw RGB-D | Required baseline; 8444 parity and 45588 parity/provenance |
| **J1 atomic source transfer** | Claim camera-0 collided pixel for 8444 **and remove it from 45588** atomically | Absolute raw | Source responsibility alone, without double counting; compare both outcomes |
| **J2 joint hard assignments** | Enumerate physically eligible assignments for the ambiguous pixels; optimize **both states** for each | Raw RGB-D | Is there a jointly consistent assignment, and is it unique? |
| **J3 joint hard assignments + frame-0 correction** | Same fixed assignments as J2 | Camera/point-specific frame-0 discrepancy, only where valid | Separate association error from persistent bias |
| **J4 current ownership + reference correction** | Unchanged archived production owner eligibility | Valid camera/point reference, with stated uncertainty | Bias-only control, compare to J3 |
| **J5 source-prior sensitivity** | Soft assignment/mixture, if well-defined; show *flat/equal* versus data-supported provenance prior and uncertainty | Raw and referenced separately | Can data determine ownership without imposed source favoritism? |
| **J6 no-row / source-only negative control** | Only source-valid information | Raw | Detect apparently stationary but unobserved trajectories; not a success |

For J2/J3 enumerate assignments for a fixed set of **actual conflicting camera pixels** and include external frontmost candidates only with legitimate likelihoods. The number of tested assignments and eliminated physically impossible ones must be explicit. Do not trivially force 8444 to win because it originated from that pixel.

**Fixed-prior one-frame contrast:** when evaluating frame 1, `s^-`, `P^-`, `R`, `F`, `Q`, `dt`, masks and camera calibration must be identical across J0–J6. Only assignment and explicitly specified reference model can differ. Report \(\Delta x\) **and** \(\Delta v\) for both points, separately, plus accepted measurements, joint objective under common support, evidence/identifiability, covariance and real observation reuse.

**Exactly attributable camera test:** within each candidate assignment, remove camera-1's *specific* implicated factor for 8444 while retaining all other pixel likelihoods on a consistently defined common observation model (e.g. reassign to an explained external/other component, rather than silently delete it); separately remove the persistent bias while keeping that camera factor and its eligibility. Report their interaction rather than attributing the entire effect to the factor removed first.

**Reference uncertainty:** Source-canonical RGB and `x0` originate from frame-0 data and are *not independent* of frame-0 reference observations. A frame-0 differenced residual must not be treated as a perfectly known independent bias. Provide two variants:

- `ALGEBRAIC_DIAGNOSTIC`: reproduce fifth-pass static counterfactual exactly, but do not call resulting covariance a valid posterior;
- `LATENT_REFERENCE`: a **small** joint model with camera/point nuisance offsets constrained by available frame-0 observations, with shared-reference cross-time covariance/marginalization if doing sequential analysis; no arbitrary independent new measurement at `t=0` for source camera. If the covariance cannot be uniquely calibrated, return sensitivity ranges and `NOT_IDENTIFIED`, not a manufactured winner.

No per-camera global bias: bias is tied to **point and camera** (and potentially local surface geometry), not a single correction shared indiscriminately across all scene points.

## 5. Phase D — continuous joint replay, only after frame-1 gates

If and only if Phase A baseline parity is established and at least one Phase C likelihood is mathematically meaningful, continue selected arms through **frames 0–10** with the following conditions:

1. Both target states evolve together. At each frame predict both 6D states and the complete **12×12** joint covariance, including cross blocks introduced by prior coupled assignments.
2. Recompute current candidate pixel competition from both *updated* point positions plus the archived/frozen environment points. Preserve true masks, all original visibility and depth validity constraints. A pair changing ownership must not cause the same camera pixel to update two points independently.
3. After every accepted nonlinear update, allow new pixel locations/responsibilities within the defined correspondence model; **do not freeze source pixel coordinates** or permanently ban cameras not seeing the point at frame 0.
4. Track any new colliders and decide whether the **minimal local cohort** requires another actual point ID. If the cohort expands materially, disclose and stop rather than silently pretending the 2-point system represents the whole cloud.
5. Record actual observation support, source-camera ownership and reference validity at each frame; if no valid observation exists, propagate the prior and uncertainty—**do not report stationary as a tracking success**.
6. All competitor trajectory/environment dependence is explicitly conditional unless all are actually jointly updated. A local cohort is not a validated full-scene replacement.
7. For uncertain bias, preserve its correlation across time; a repeated frame-0 reference is not ten independent observations. Do not double count priors, canonical colors or masks.
8. Perform matching-support objective and acceptance checks at each inner iteration; if objective is undefined because a pixel becomes physically invalid, record `UNDEFINED_SUPPORT`, not zero, and do not claim MAP descent.

If full 0–10 sequence cannot meet the data/identifiability requirements, still complete the definitive **frame-1 joint analysis**, and explicitly explain the missing prerequisite. This is better than a false success.

## 6. Phase E — falsification controls, kept very small

These tests prevent overfitting to 8444:

- **Storage stable control**: point 60019 or 100564, preserving genuine observed rows and consistent 0–10 behavior. The joint intervention must not create artificial motion or double count shared measurements.
- **USB 67827**: frame-1 camera 5(source)/camera 6(crease), and its later camera-4 admission (no valid frame-0 reference). Show whether the same responsibility logic prevents an invalid *late* observation without freezing camera identity. No assumption that all crease pixels are identical material points.
- **USB moving 87290**: inspect *only* first support-collapse frames (2,3, and optionally 8) and see whether any proposed policy provides a usable observation without GT-based recentering. True motion and GT depth **evaluation only**.
- **USB moving 85696**: inspect duplicate suppression at frames 2–5; state whether the local two-point ownership mechanism can even address its many-competitor problem. Do not claim a successful recovery if it cannot.
- **Storage knife-edge 8416**: repeat separate x (m) and v (m/s) perturbation metrics. Because its historical baseline replay fails parity, mark intervention scores `INDETERMINATE` unless parity is independently established. Do not let it invalidate the 8444 main result by unit confusion.

Do **not** run all six/factorial arms on all populations or run a new full-scene filter. These are out-of-sample **local** falsification controls, not a global performance benchmark.

## 7. Numeric and engineering gates (do not skip)

Implement deterministic, CPU float64 unit tests; GPU not required. Gate failures block *claims*, not data collection. Use source-derived tolerances and explain all tolerances by numerical precision/covariance scale; do not invent speed or spatial gates to improve results.

- **G0 — provenance:** 8444 and 45588 source IDs/pixels/RGB and all collision owners correctly decoded (not case-sensitive/scene-name assumptions); masks and calibration hash verified; exact original point ID mapping after subsampling.
- **G1 — parity:** J0 reproduces production 8444 first frame with correct row ownership, `x+`, `v+`, covariance and acceptance; 45588 parity verified independently or explicitly `NOT_VERIFIABLE`. All 6D states must report physical units separately.
- **G2 — atomic ownership:** transferring a pixel changes its sole active owner: `(owner=8444)` means 45588 cannot also use that pixel. True foreground mask and depth/occlusion invalidity never bypassed.
- **G3 — evidence preservation:** every competing assignment evaluated on identical fixed observed-pixel IDs or reported incomparable; no advantage from silently dropped data or arbitrary null cost. If identity is not identifiable, show a numerical ambiguity/evidence interval and report `UNDETERMINED_MATERIAL_ID`.
- **G4 — joint correctness:** when a shared probabilistic factor is truly used, finite-difference gradients/Hessians and 12×12 cross-blocks agree with independent CPU oracles; when hard conditioning makes blocks zero, they are **correctly zero**. SPD/PSD checks and cross-time covariance reset are explicit.
- **G5 — robust acceptance:** comparable fixed-support objective descends for every accepted *new diagnostic* optimization step, or report nonconvergence/undefined. Keep original J0 acceptance faithful even when it fails this ideal check.
- **G6 — genuine observation:** 0 displacement with 0 accepted rows is explicitly classified `UNOBSERVED_COAST`, not successful static tracking.
- **G7 — frame persistence:** compare both points 0–10 with properly propagated state/covariance; no stale competitor snapshots mislabeled jointly consistent.
- **G8 — no truth leakage:** all policy inputs depend exclusively on RGB, depth, masks, calibration, provenance and filter priors; ground-truth part labels/poses/trajectories are used only for final evaluation in a separate module/file.
- **G9 — safety and reproducibility:** original production files/previous outputs unchanged, deterministic output under repeated run, all random initializations seeded and logged, and zero writes outside the new output directory except new `tools/` and `tests/` source files.

Required synthetic unit tests include: identical vs different point RGB with same depth, nonidentifiable textureless neighborhood, two points exchanging frontmost depth order, actual source occlusion, disjoint pixel assignments, bilinear samples with overlapping four-pixel supports (correlated noise), invalid masks, RGB bias with fixed same material, wrong-surface crease, a well-posed two-point mixture with demonstrably nonzero mixed Hessian, and a source-only unobserved moving point.

## 8. Outputs (new directory only)

Write concise machine-readable evidence **and** a precise human-readable causal report:

```text
OUT/
  README.md
  provenance.json
  inputs_sha256.json
  archive_completeness.json
  tests.log
  tests.json
  baseline_joint_parity.csv
  source_collision_candidates.csv
  fixed_observation_pixels.csv
  candidate_likelihoods.csv
  joint_assignment_enumeration.csv
  per_point_per_camera_factors.csv
  joint_iteration_history.csv
  joint_covariance_12x12.npz
  reference_bias_uncertainty.csv
  joint_sequence_0_to_10.csv                 # only if full replay is valid
  ownership_changes.csv
  heldout_controls.csv
  model_comparison_identifiability.csv
  figures/
    storage_8444_45588_camera0_camera1_ownership.png
    storage_8444_45588_cost_landscape.png
    storage_8444_45588_pair_trajectories.png  # if sequence available
  FINAL_JOINT_RESPONSIBILITY_CAUSAL_REPORT.md
```

Every row needs `scene`, `point_id`, `other_point_id`, camera, frame, source pixel/source camera, projected subpixel/current pixel, `arm`, mask/depth validity, actual observation IDs, assignment, owner, reference validity, original / candidate residual, `R`, `H`, joint prior/covariance where applicable, likelihood comparability flag, convergence, accepted rows and acceptance reason, and data-quality tier. Use JSON arrays for matrices and accompany them with shape/unit metadata. Avoid dumping full-scene point clouds or per-frame image copies.

The report **must** answer the following, with explicit numbers, even if the answer is `UNIDENTIFIABLE`:

1. Is 45588 a valid material correspondent for camera-0 pixel (282,237), or merely a depth-nearby competing sample? What can the images prove and what can they not prove?
2. What does **45588 itself** do under the original versus the atomic-transfer and joint-assignment models? Does solving 8444 merely push the error into 45588?
3. On a **common observed-pixel likelihood**, is either physical assignment more probable after accounting for canonical appearance, depth, initial priors and source-provenance uncertainty? Are differences robust to the nuisance/null model?
4. Does correcting camera-1's persistent discrepancy while preserving its measurement explain 8444's stability more convincingly than altering ownership? Does this remain true when both points are updated together?
5. Are the two errors causally independent, additive, or interacting? Give first-frame \(\Delta x\) (mm), \(\Delta v\) (m/s), matched-support likelihood and observation counts **for both points**.
6. Can both points be tracked 0–10 while still receiving real foreground observations, with legitimate occlusion and no double assimilation? If not, first exact failure event and mathematical reason.
7. Does this evidence support a **minimal production design change**, and what additional assumptions would that change require? If identification is impossible, identify exactly what new information/measurement model would be needed, without writing that model to production.

Use clear verdicts per mechanism: `CONFIRMED_CAUSAL`, `REFUTED_AS_SOLE_CAUSE`, `SUPPORTED_BUT_NOT_ISOLATED`, `UNIDENTIFIABLE`, `NUMERICALLY_INDETERMINATE`. **No generic conclusion** such as “camera bias causes drift” without tracing the exact admitted physical factor and its impact on both points.

## 9. Implementation sequence and CLI

Create one investigation tool, one test file, and a small report generator if justified:

```text
tools/f1_joint_responsibility_audit.py
tests/test_f1_joint_responsibility_audit.py
```

Implement in this strict order:

1. Read fifth-pass evidence, map IDs and artifact availability, write `archive_completeness.json`.
2. Frame-1 baseline parity (8444, then 45588), competitor enumeration, and the exact camera0 collision table.
3. Unit tests for pixel reuse, hard masks, source/point identity, likelihood comparability, finite differences, covariance and no GT leakage.
4. Common-observation likelihood, all well-defined J0–J6 first-frame arms, explicit `UNDEFINED_MODEL` for any nonidentifiable arm.
5. Short coupled sequential replay only if the above gates pass.
6. Minimal falsification controls, report, and deterministic rerun/hash check.

Suggested interface (implement these flags exactly; no unsupported magic arguments):

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
OUT="$FOR/final_joint_responsibility_8444_45588_20261030"

CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_joint_mpl \
python -u -B tools/f1_joint_responsibility_audit.py \
  --forensics-root "$FOR" \
  --fifth-pass-root "$FOR/fifth_pass_source_responsibility_20261030" \
  --out "$OUT" --experiment tests

CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_joint_mpl \
python -u -B tools/f1_joint_responsibility_audit.py \
  --forensics-root "$FOR" \
  --fifth-pass-root "$FOR/fifth_pass_source_responsibility_20261030" \
  --out "$OUT" --experiment all --frame-start 0 --frame-end 10

# Inspect, do not claim success just from a process exit code:
cat "$OUT/archive_completeness.json"
cat "$OUT/FINAL_JOINT_RESPONSIBILITY_CAUSAL_REPORT.md"
```

If the run requires an original production file that does not exist, return a **specific blocker with file path, array name, and earliest frame**. Do not use GT, invent a cached posterior, modify production, or trigger a full-scene rerun to conceal the blocker. A rigorous `UNIDENTIFIABLE` result with a validated frame-1 reconstruction is more valuable than a misleading “fixed” ten-frame trace.

## 10. Explicit prohibitions

- Do **not** treat depth proximity of 0.103 mm or canonical separation of 1.694 mm as proof of same-material identity.
- Do **not** count the camera-0 pixel twice by admitting both 8444 and 45588 as independent observations.
- Do **not** treat source identity as permanent visibility or freeze a moving point's pixel coordinate.
- Do **not** infer correctness from source-only 0 drift without accepted measurements.
- Do **not** report cross-point covariance as exactly zero after marginalizing shared ownership if the posterior has coupled assignments.
- Do **not** compare incomparable models by subtracting unlike row-set objective values.
- Do **not** quietly discard camera-1's measured pixel to make the candidate cost decrease.
- Do **not** use real static/moving GT labels, GT joint trajectories or rendered GT depth to choose correspondence or optimize either point; use them only in clearly separated evaluation.
- Do **not** change the existing `estimator/` tree, production configs or previous reports, train any 3DGS, run a full-scene F1 filter, or commit changes.

**Success criterion:** A verified quantitative answer to whether *joint, measurement-consistent source/cross-camera responsibility* changes the false static correction **without suppressing evidence or moving the error into 45588**. If RGB-D data do not identify material correspondence, demonstrate the degeneracy and name the missing information precisely rather than proposing another generic gate or arbitrary weight.
