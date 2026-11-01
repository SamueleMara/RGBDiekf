# CODEX — DEFINITIVE, FALSIFIABLE ROOT-CAUSE FORENSICS OF STATIC-POINT DRIFT IN THE STANDARD F1 IEKF

**Date:** 2026-10-30  
**Repository:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Task type:** **investigate, reproduce, quantify, and falsify; NOT implement a new estimator**  
**Estimator under investigation:** **standard directional forward F1 IEKF** (`[x,v]`, original per-point 6×6 filter). **Exclude RTS, IEKS, EM, group/ridig-part inference, and temporal outer loops.**  
**Priority:** obtain an exact, independently audited causal explanation of *why the specific physically static points acquire their first nonzero position/velocity corrections, why those corrections are accepted, and why the resulting errors persist.* A population-level slogan such as “depth,” “color,” “large Pvv,” “lack of observability,” “pixel ownership,” or “numerical instability” is **not** an acceptable final result.

> **Operating rule:** Do not start from the hypothesis that a camera-bias variable, frame-0 subtraction, a smaller velocity prior, a different depth interpolation, or another solver is correct. All are competing **offline** diagnostic models, not authorized production fixes. Demonstrate causal effects while holding the other mechanisms fixed. Report negative evidence and contradictions as prominently as confirming evidence.

> **Execution rule:** No inference-source modification, no change to config/defaults/prior/R/noise, no GT fed to estimator, no outer-loop runs, no complete scene re-filtering. Read-only use of existing outputs and raw RGB-D; if necessary, create narrowly scoped CPU point/frame replays using copied state and covariance. All new files go under a fresh `fifth_pass_*` analysis directory and `tools/`/`tests/` investigation-only files. Do not commit without instruction. Do not rewrite existing reports; produce a new report with explicit errata cross-references.

---

## 0. Starting evidence: treat as hypotheses, not conclusions

Use these exact user-reviewed inputs. **Actually open the files and parse numerical arrays; do not rely on the conversation summary or earlier high-level conclusions.**

```bash
REPO=/home/samuelemara/Joint3DGS/RGBDekf
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
S2="$FOR/second_pass_20261030"
S3="$FOR/third_pass_temporal_reference_20261030"
S4="$FOR/fourth_pass_sequential_replay_20261030"
OUT="$FOR/fifth_pass_definitive_static_drift_20261030"
```

**Required files:**

- `$S2/SECOND_PASS_ROOT_CAUSE.md`, `first_update_iterations.jsonl`, `first_update_decomposition.csv`, `frame0_frame1_raw_residuals.csv`, `iteration_support_objectives.csv`, `local_counterfactuals.csv`.
- `$S3/` third-pass `README.md`, traces, reference validity, depth surface fits, controlled arms, associated tests; discover and record actual names.
- `$S4/FOURTH_PASS_CAUSAL_REPORT.md`, `baseline_parity.csv`, `sequence_state.csv`, `sequence_camera_rows.csv`, `sequence_accepted_steps.csv`, `sequence_objective.csv`, `sequence_failure_events.csv`, `reference_covariance_tests.csv`, `ownership_mode_C_parity.csv`, `pixel_shift_sequence_8416.csv`, `diagnostic_units_corrections.csv`, `tests.log`, `sequence_arrays.npz`.
- Original per-scene `baseline`, `traced`, canonical point catalogues, original RGB-D/masks/calibrations, manifests and existing numerical traces for USB, Storage, Blade.
- Inspection-only code: `estimator/free_projective_run.py`, `point_kalman_cuda.py`, `projective_sensor.py`, `projective_terms.py`, `projective_visibility.py`, `projective_camera.py`, `scene/endpoint.py`, `scene/noise_calibration.py`, `tools/root_cause_oracle.py`, `tools/f1_second_pass.py`, `tools/f1_temporal_reference_diagnostics.py`, `tools/f1_sequential_causal_replay.py`, `tools/f1_diagnostic_units.py`.

**Known established facts to verify, not assume:**

1. At the frame-1 update of Storage point **8444**, only **camera 1** is admitted. The camera-1 RGB discrepancy at fixed `x_ref` is the same across frames 0, 1, 2. A0 takes about **24.16 mm** and injects ~**1.41 m/s**; the model's comparable-row objective falls from ~**7.27** to ~**2.93**; A3 (both referenced) stays near exactly zero through frame 10 while still being observed. Depth-only referencing does not eliminate that correction. **A3 zero residual at an unchanged static position is algebraically expected, not independent proof of a correct generative model.**
2. USB point **67827** starts on an admitted **camera-6** crease-depth row (~**−41 depth-noise σ**), **not camera 4**. Earlier reports erroneously listed frozen-out cameras. A2 depth reference keeps drift ≤~0.6 mm through frame 10. The A3/A4 trajectory later jumps when **camera 4** becomes eligible despite having **NO_VALID_REFERENCE** at frame 0; its ~**−80σ** crease residual is not new temporal information. A2's later avoidance can hinge on **NIS 16.23 versus cutoff ~16.25**, so treat support discontinuity as an alternative explanation.
3. Storage **8416** baseline offline parity **fails at frame 1**; report its position and velocity errors separately (11.06 mm and 0.646 m/s at frame 1 for one comparison). Never call a comparison involving this point a production-validated intervention until parity is established. Likewise do not relabel 0.65/0.79 as meters: those historical values mixed units and were velocity-dominated.
4. USB moving **87290** has real motion information in some views, but A0 loses row admission by frame 3 and reference arms eventually lose support; USB **85696** is suppressed in all cameras at frames 3–5 under pixel ownership. These are *negative controls* for an overfitted static fix, not this task's main targets.
5. On seven of eight fourth-pass points, A0 sequential Mode P matches saved production to roughly 1.2e−12 m position and 7e−11 m/s velocity with matching decisions; **do not weaken this reproduction standard without explanation**. Fourth-pass 17 tests pass.
6. Source frame 0 initializes the catalogued 3D point from a source depth pixel and its color, but F1 does not run a full multiview filter correction at frame 0. A previous frame-0 fusion experiment did *not* remove the large frame-1 false update. This omission is a hypothesis component, not a proven primary cause.
7. Per-camera listings must use **production-admitted rows**, not all candidate depth samples. Ground-truth labels in these synthetic scenes are based on closest labelled point-cloud samples with ~centimeter spacing; never portray this as exact physical material-point identity.

**Make an evidence manifest first:** hashes and relative paths; git commit/dirty diff; branch; timestamps; code/manifest policy values; provenance of RGB, depth, masks, `K`, `T_cw`, timestamps, calibration, `R`, `P0`, `Pvv0`, `q_c`, all row gates, assignment, convergence and acceptance rules. Store `evidence_inventory.json`. If datasets/configs/trace are missing, explicitly mark unavailable; do not synthesize numbers.

---

## 1. Precise target and non-negotiable scientific distinctions

For each anchored point `i`, frame `k`, and camera `c`, reconstruct:

1. **What did the sensor actually measure?** Original 2×2 pixel samples, masks/validity, interpolation, effective depth/color noise, camera coordinates.
2. **Why was that particular camera row admitted?** Reconstruct the full chain: projection, nearest-pixel owner, frozen owner, source/reference visibility, same-surface check, occlusion, full RGB-D NIS, final acceptance and recheck. Distinguish *raw sample valid*, *candidate*, *admitted*, *used in a solver iteration*, *used at accepted state*.
3. **What exact measurement drives the correction?** Identify per-camera/channel residual, Jacobian, RHS, Hessian, posterior eigenvectors and actual nonlinear path, not a generic verbal attribution.
4. **What does the correction physically represent?** Is it new temporal evidence, a pre-existing cross-view appearance/geometric inconsistency, a view-identity or pixel-ownership mistake, a local interpolation model mismatch, an unobservable tangential direction, an actual numerical solver error, or some combination? Which are distinguishable with available evidence?
5. **Where and why is that position correction converted into velocity?** Reproduce all six state components, 6×6 covariance cross-blocks, Kalman/IEKF solve and propagation, at the *actual accepted iterate*, including any fallback.
6. **Why is it not corrected later?** For every frame until drift becomes irreversible: accepted RGB-D information, rejection/camera ownership, prediction-only intervals, loss of observability; find exact first loss, not only the endpoint error.
7. **Are any diagnoses observationally indistinguishable?** Say so with a nullspace/likelihood argument. Do not claim unique cause from equivalent observations.

### Required categories of certainty

Every claim must carry one of:

- `EXACT_CODE_AND_ARITHMETIC`: checked against production trace and independent oracle.
- `COUNTERFACTUAL_CAUSAL_WITH_FIXED_CONTEXT`: a one-variable intervention changes the result while everything else named is held fixed; state explicitly that it is a local causal statement.
- `OBSERVATIONALLY_SUPPORTED`: images/geometry favor the mechanism but alternative explanations remain.
- `NOT_IDENTIFIABLE`: available evidence cannot separate mechanisms (give concrete mathematical reason).
- `FALSIFIED`: a designed counterexample rejects the hypothesis.
- `INDETERMINATE`: missing evidence, lost baseline parity, invalid camera sample, or ambiguous comparisons.

Never use the word **“proved”** without identifying the *exact* mathematical statement and assumptions that were proved.

---

## 2. Mandatory cases, controls and selection-bias check

### Fixed forensic point IDs (must all be covered)

| Scene | IDs | Role / competing hypotheses |
|---|---|---|
| `storage_45135` | **8444** | Main static; persistent admitted camera-1 RGB offset; very large false first step |
| `storage_45135` | **49199** | Static; small residuals, nonconverged iteration; whether RGB effect is robust or arbitrary path |
| `storage_45135` | **262894** | Static; prior false claim about occluder rows must remain retracted; actual cameras 5/7 drive update |
| `storage_45135` | **67801**, **102354** (frames 1 and 3 as relevant) | Accepted updates whose original comparable-row objective may increase, and nonconvergence |
| `storage_45135` | **8416**, **8417**, **8418** | Floating-pixel-boundary diagnostics, **not reliable parity-qualified causal estimates** |
| `storage_45135` | **60019**, **100564** | Static controls, including small benign updates |
| `USB_100109` | **67827** | Main static; admitted camera-6 depth crease, later previously unreferenceable camera 4 |
| `USB_100109` | **30502** | Late first measurement, large prior covariance, crease plus appearance ambiguity |
| `USB_100109` | **51975** vs **51974** | Adjacent static point comparison: large difference in cameras/photometric evidence |
| `USB_100109` | **30503** | Stable neighboring control |
| `blade_103706` | **10251**, **14104** | Depth-only tangential gradient/nonconvergence and amended actual admitted crease cameras |
| `USB_100109` | **87290**, **85696** | Moving negative controls to falsify any static-only interpretation of general measurement behavior |

Add other existing strongly drifting points from `selected_point_ids.csv` only if their saved traces establish admissible production rows. Do not omit a difficult case because it contradicts the leading hypothesis.

### Population generality without outcome-selection bias

The fixed cases were chosen because they fail. They can establish mechanisms but **not their prevalence**. Using **saved baseline F1 arrays only**, construct a predeclared reproducible stratification for each of the three scenes:

- GT-grounded static points; group by first accepted-update frame, initial admitted-camera count, source camera and final drift bands; mark points with incomplete GT material attribution separately.
- Stable controls that have **actual accepted non-source rows**, not just no observations; use comparable source camera/depth/image gradient/support when possible.
- A reproducible uniform random holdout from the full GT-grounded static population, selected by a fixed, reported PRNG seed derived from the hashed baseline manifest **before** examining individual residuals.
- Explicitly include both high- and low-error groups; do not rank only by residual magnitude (collider/selection bias). Report group denominators, unmeasured fraction and source-camera ownership imbalance.

Start with at most 12 random holdouts plus 12 stratified representatives per scene. The exact number is an **analysis budget**, not an estimator threshold: if sampled evidence remains contradictory or intervals are too uncertain, say so instead of pretending “definitive generalization.” Do not launch a full-scene F1 run; aggregate population statistics from saved arrays. Report empirical proportions descriptively, not as causal rates; clustered points in one surface are not IID.

---

## 3. Hard gate G0 — truthful baseline reproduction and unit safety

Before any hypothesis experiments:

1. Re-run the previously saved **A0 single-point baseline** replays with the same frame 0–10 initial states, covariance, `F(dt)`, `Q(dt)`, per-frame ownership Mode P, and sensor frames. Compare every frame **componentwise** against production and existing fourth-pass arrays.
2. For every frame, separately report `||Δx||₂` (m), `||Δv||₂` (m/s), `||ΔPxx||`, `||ΔPvx||`, `||ΔPvv||` with units and scale; exact row IDs and reasons; original/accepted NIS, step/iteration/acceptance. Recompute the prediction closure.
3. Check Mode C ownership reproduction against the exact original *competing cloud states*, distinguish `owner_changed` from `owner_near_tie`; Mode C is a counterfactual under a partially frozen cloud, not an updated fully coupled estimation.
4. Reproduce historical errata: never include frozen-out USB 67827 camera 4 in its frame-1 update; never include Storage 262894 camera-3/4 occluders or Blade 14104 camera-0 occluder as admitted rows.
5. **8416/8417/8418:** retain full one-sided floating precision differences; where parity fails, do not label counterfactual change as causal relative to production. If needed, compare under identical chosen float64 branch with fixed `nextafter` direction. Report branches and pixel support.
6. Verify every legacy mixed-six-state metric is split: no max across position and velocity is ever labelled meters. Existing fourth-pass unit errata remain valid.
7. Compute SHA-256 for input images, baseline arrays and config; point-ID mapping (`subsampled_to_full_index`), canonical source (frame/camera/u/v), world coordinates; no silent point-ID remap.

**Gate:** Exclude `PARITY_FAILED` cases from quantitative experimental benefit claims. Still examine them as numerical counterexamples. If any previously parity-qualified point fails, stop that point's interventions, locate first diverging operation, and report it. Do not continue from an untrusted state.

---

## 4. Hard gate G1 — exact anatomy of the FIRST accepted false update

For each static point, find the **earliest accepted update** causing non-negligible canonical displacement or velocity by reading all saved continuous states; do not assume frame 1. Record raw continuous values (no threshold-dependent omission). Track first `||x+−x-||`, first `||v+−v-||`, and first accumulated `||x_k−x_0||`. Report exact frame, time and camera(s).

### A. Complete raw evidence for every camera, including excluded rows

At prediction, at every inner iterate, at accepted iterate, and at the fixed physical `x_ref` on frames 0 and onset:

- actual `T_cw`, `K`, `X_c`, `Z`, `(u,v)`, exact float64 binary representation/ULPs near integer boundaries;
- camera owner and all conflicting point IDs, owner depth, nearest pixel and frozen assignment; whether exclusion is from ownership, mask, occlusion, depth discontinuity, camera footprint or gate;
- 4 original depths and RGB triplets, masks and original validity flags; `(du,dv)` and interpolation weights; *cell coordinates* and 2×2/3×3/15×15 context;
- interpolated RGB/depth and gradients; exact analytic `J_uv_x`, `J_Z_x`, `H_RGB`, `H_D`; independent finite-difference/one-sided checks when differentiable;
- raw signed 4D residual, `R` (measurement), `H P^- Hᵀ`, `S`, `ν`, joint NIS, threshold and all intermediate decision codes;
- whether row was actually in the production solve for that iteration, separately whether in final covariance and accepted state.

**Create a `CAMERA_ROW_LIFECYCLE` diagram/table for each point.** The inability to distinguish candidate from admitted rows caused a material error in the first report and must not recur.

### B. Complete exact 6×6 IEKF step

At each iteration `j`, reconstruct (same sign conventions as production):

\[
A_j = (P^-)^{-1}+\sum_{c\in\mathcal A_j}H_{c,j}^{\mathsf T}R_{c,j}^{-1}H_{c,j},
\]
\[
b_j = -(P^-)^{-1}(s_j-s^-)+\sum_{c\in\mathcal A_j}H_{c,j}^{\mathsf T}R_{c,j}^{-1}r_{c,j},
\qquad
\delta s_j=A_j^{-1}b_j.
\]

Save `A` 6×6, `b` 6×1, `δs` 6×1, condition number, smallest eigenpairs, Cholesky residual, prior cost, total measurement cost by camera/channel, valid row set, convergence statistic, accepted decision and fallback source. Independently reproduce in NumPy float64; for borderline cases use higher precision at **unchanged** cell membership and explain any one-sided derivative differences.

Compute the position/velocity update relationship with the *actual* prior covariance:

\[
\Delta v=(P^-_{vx})(P^-_{xx})^{-1}\Delta x
\]

for any position-only measurement that satisfies the model assumptions; check its residual numerically in the accepted solution, or derive the Schur counterpart if the iterative relinearization/fallback introduces distinctions. Do not substitute the approximation `Δv≈Δx/Δt` for the exact measured matrix.

### C. Rigorous attribution and its limits

At a **fixed** linearization and fixed admitted row set, compute each camera×channel information-force vector `g_{c,q}=HᵀR⁻¹r`, and displacement contribution `δs_{c,q}=A⁻¹g_{c,q}` **using the same total Hessian A**. Include the prior-gradient contribution. Verify exact algebraic closure `Σ δs_component = δs_j` in all 6 state elements.

Then, **separately** run nonlinear leave-one-camera-out and RGB-only/depth-only re-solves with the same prior and same other conditions; report changes in accepted rows, Hessian, pixels, convergence and covariance. These are *not additive* attribution and must not be presented as such. For key points, compute camera-order-invariant subset attribution across ≤8 cameras (exact subset enumeration feasible at small point scale) **if** row support is held fixed and objective comparability remains valid; otherwise explain why no invariant assignment is possible.

Plot the accepted correction in a local depth-surface frame (surface normal `n`, tangential basis `t1,t2`) obtained independently and validated for local surface planarity. If surface normal cannot be reliably estimated at a crease, report `NORMAL_NOT_IDENTIFIABLE`; do not invent a normal or classify drift as tangential there.

---

## 5. Hard gate G2 — determine whether an observation contains ANY new temporal evidence

This is the strongest missing experiment. Distinguish *same observed pixels*, *same camera-specific residual at fixed x_ref*, *same surface identity*, and *same accepted measurement*. These are NOT interchangeable.

### 5.1 Test camera/time geometry before differencing

For each scene and selected point:

- verify per-frame camera intrinsics/extrinsics, depth units, image transformations, timestamps, object poses and actual camera motion from metadata; do not assume cameras fixed;
- for stationary camera and static point, project `x_ref` into each frame and compare the **same physical location**; for moving cameras, use the appropriate time-specific projection and do not subtract different ray observations without considering geometry;
- compare pixel neighborhoods frame 0 versus onset **pixelwise**: raw RGB channel values, raw metric depth, mask, visibility, ownership and interpolation weight, recording exact equality, differences and quantization units;
- also compare a *matched valid surface neighborhood* independently of the chosen point's current trajectory to detect occlusion/silhouette changes; if correspondence cannot be established, mark that hypothesis inconclusive.

### 5.2 Absolute residual versus temporal change

Let `x_ref` be immutable frame-0 canonical geometry. Write exactly what production measures:

\[
r_{i,c,t}(x)=\begin{bmatrix}a_i-I_{c,t}(\pi_{c,t}(x))\\Z_{c,t}(x)-D_{c,t}(\pi_{c,t}(x))\end{bmatrix}.
\]

Where both frame-0 and frame-t observations are **physically valid at x_ref**, evaluate:

\[
b_{i,c,0}=r_{i,c,0}(x_{ref}),\qquad
\Delta r_{i,c,t}(x_{ref})=r_{i,c,t}(x_{ref})-b_{i,c,0}.
\]

For each of four channels separately, give raw `r0`, `rt`, `Δr`, normalized by **both** declared sensor R and an empirically estimated temporal-difference covariance when supportable. Report reference validity and physical occlusion separately from frozen duplicate ownership. If invalid, **do not set reference to zero**.

**Falsifiable discriminator:** if full raw images/patches and camera geometry are unchanged between frames, a solver producing nonzero inferred *temporal motion* from that same spatial mismatch is fitting an absolute appearance/geometry discrepancy, not a changed observation. This is a valid local causal statement under fixed camera/visibility assumptions. It does **not** imply that every frame-0 mismatch should be permanently subtracted under actual motion.

### 5.3 Frozen-image experiment (essential independent confirmation)

For point+frame replay only, construct paired observation inputs:

- **T0:** actual frame-t raw RGB-D and masks, original prediction/covariance/owners.
- **T1:** *all cameras' raw frame-0 RGB-D/masks* used as a synthetic, temporally unchanged observation at that same frame-t prediction/covariance; keep the time-step and prior unchanged. Mark this as a **null intervention**, not a physically consistent articulated scene; recompute or hold owner table in explicitly separate arms. Track any artifact due to moving competitors.
- **T2:** identical frame-t observations but replace only **the specific admitted offending camera's** RGB or depth field with its frame-0 field, retaining all other channels and camera geometry; confirm same visibility/mask or mark support incomparable.
- **T3:** actual raw images and **fixed position x_ref** (one-step score only); compare time-0 and time-t residual and gradient, no optimization.

Report the exact change in `Δx`, `Δv`, objective, camera rows, and accepted iterate. If T0 ≈ T1 at onset despite a moving scene, this directly demonstrates that **no new time-varying signal is required** for that correction. If T0≠T1, identify the first differing image/camera/pixel/owner and do not attribute all motion to a persistent bias.

### 5.4 Why zero-residual referencing is not independent proof

Perform the previously validated offline RGB/depth/RGB-D `r_t-b_0` arms, but show:

- `r_delta(x_ref) = 0` for unchanged images is an **identity**, not experimental ground truth;
- change of objective landscape `J_original(x)` versus `J_delta(x)` on a common valid sample support;
- whether reference subtraction changes the *local Jacobian*, *row gate*, *accepted camera set*, *acceptance threshold result*, or simply RHS;
- whether the reference itself is built from a source pixel also defining `x_ref`, `Pxx0`, and `a_i` (noise correlations and data reuse).

Never present A3 stabilization as evidence that a production bias model has been statistically identified.

---

## 6. Hard gate G3 — independently diagnose RGB and depth model errors, without forcing a preferred explanation

### 6.1 RGB: which specific physical/model discrepancy?

For **8444**, **49199**, **262894**, **51975**, and controls:

1. Report original source camera/canonical `a_i`, source-pixel RGB, every admitted camera's 2×2 RGB, the per-camera RGB residual at `x_ref` and image gradient per channel; evaluate temporal stationarity at frames 0–5.
2. Determine if the error is attributable to camera-specific *global* gain/offset, per-point/per-camera surface appearance, view-dependent illumination/BRDF, 8-bit rounding/interpolation, occlusion, or sampling the wrong surface. Use data-driven cross-camera difference maps across multiple static points, including unselected random controls.
3. Test models **offline**, one at a time: constant camera-wide RGB offset/gain (estimated from held-out static pixels only for diagnosis); point×camera constant offset; source-color re-estimation from independent valid cameras at frame 0; camera-specific temporal difference; robust no-color/depth-only fit. Use held-out point/camera scoring to detect overfitting.
4. Avoid assigning a unique physical cause (lighting vs geometry) if identical RGB observations can be generated by both. Report image/camera evidence needed to identify it.
5. In consistent residual cases, diagnose whether the solution moves by *one interpolant pixel*, how long it stays in a given cell, and whether the apparent RGB gradient is a quantization edge. Record exact grad vectors and the amount of displacement needed to cancel each per-channel residual (linear prediction versus actual resampled cost).

**8444 stopping question:** Why, with *only admitted camera 1* and a temporally constant RGB discrepancy, does the correct original likelihood produce the accepted ~24.16 mm step? Give actual `H`, `R`, prior/Hessian eigenvectors, full Newton iterates and original/differenced costs; show which specific image pixel changes lead to cost 7.27→2.93. Determine whether the same correction also occurs under frozen frame-0 images and whether the data supply ANY evidence for temporal motion.

### 6.2 Depth: distinguish crease, true surface, interpolation, calibration and occlusion

For **67827 cameras 6 AND 4**, **30502 camera 1**, **Blade 10251**, **Blade 14104**, etc.:

- Extract exact raw metric depths and masks of every 2×2 and nearby valid 3×3/5×5 neighborhood; show finite differences, mixed second differences and local inverse-depth plane fits with residuals and conditioning; separately compare arithmetic metric-depth versus harmonic inverse-depth interpolation and Jacobians. Never equate a large second difference with a proven inter-object occluder.
- For each candidate plane/face, require sufficient single-surface support; a candidate fit requiring a different face, hidden depth or image pixels outside mask is **not a valid replacement**. Label `FACE_IDENTITY_UNPROVEN` unless a trusted face-specific renderer/mesh or dense material association exists.
- Compare `D(u,v)` with `Z` at `x_ref`, and with predicted/proposed shifted point; compute `√R_D`, `√S_DD`, joint RGB-D NIS, and the *exact* same-surface allowance actually calculated from `z`, focal length, 80° policy and sensor parameters.
- Split cases: genuine depth discontinuity within 2×2, valid sloped plane but wrong interpolation gradient, wrong front/back surface, inconsistent extrinsics/intrinsics, depth definition `Z` versus range, floating pixel cell boundary, and no adequate surface evidence. Explicitly attempt to **falsify** each with an independent check.
- Report the **same raw depth cell at time 0** and at the decisive frame. If the crease is already there, do not call it dynamic motion.
- Run exact one-off counterfactuals (held same prediction and row assignment): remove offending row; replace its sample by a geometry-consistent oracle *only when valid*; re-evaluate with inverse depth; change only depth versus RGB. If replacement is not physically validated, do not report the result as a correction—mark `HYPOTHETICAL_SYNTHETIC_SAMPLE`.

**67827 stopping question:** For frame 1, show camera **6**'s exact ~−41σ depth row and why the full gate admits it; for frames 4 and 7, independently verify when camera **4** first becomes admissible, why frame-0 reference does not exist, whether the raw −80σ crease is identical to frame 0, and why one arm passes/fails NIS around **16.23/16.25**. Identify whether late false motion is caused by a *newly exposed pre-existing discrepancy*, by actual new camera information, or by borderline numerical support selection.

---

## 7. Hard gate G4 — identifiability and information geometry, not generic “unobservable” labels

For each decisive update compute the **actual** position-information matrix and its eigenstructure:

\[
\Lambda_x^{\mathrm{meas}}=\sum H_{x,c,q}^\top R_{c,q}^{-1}H_{x,c,q};\quad
\Lambda_x^{\mathrm{post}}=\Lambda_x^{\mathrm{prior}}+\Lambda_x^{\mathrm{meas}}.
\]

- Eigenvalues/eigenvectors **in consistent units** (normalization if needed) and overlap with actual displacement `Δx`; report whether the displacement aligns with weak observed directions or is actively driven by a strong but inconsistent measurement.
- For a locally planar surface, compute the true/inferred surface normal and local tangential basis; compare depth versus RGB singular values in normal and tangent directions, and sensitivity to camera baseline/number of admitted cameras. Do not compute normals/surfels/meshes in the *production pipeline*: this is **offline geometric analysis only**.
- Build the stacked temporal linearized observability matrix for frames with **physically valid, production-admitted** rows,

\[
\mathcal O=\begin{bmatrix}H_1F_{1,0}\\H_2F_{2,0}\\\vdots\end{bmatrix},
\]

and examine singular values/nullspace of the `[δx_0, δv_0]` trajectory. Distinguish time-invariant bias directions from genuine velocity directions. If association support changes, report separate segment-local observability; do not extrapolate a single linearization over discontinuous image cells.
- Evaluate how much the prior `Pvv0=I m²/s²`, `Pxx0`, `Pvx0`, and `Q` each contribute to the actual gain. Use exact algebra, not just correlations. **Do not tune priors.** A theoretical sensitivity curve with frozen data may vary one model parameter to test the mechanism, labelled `PRIOR_SENSITIVITY_ONLY`.
- Explain what cannot be inferred from a single camera/single time: for a position observation `x_0+t v`, the decomposition between initial position correction and velocity is ambiguous without additional structure. **Demonstrate this nullspace numerically** for the failing point's observation Jacobians.
- Separate **measurement uncertainty** (sensor R), **prediction uncertainty** (P), and **association uncertainty** (not equivalent to either). Do not interpret `√S_DD ≫ √R_D` alone as proof a row is wrong.

---

## 8. Hard gate G5 — is a latent camera/point bias identified, or just an unfalsifiable explanation?

This is a diagnostic model-comparison study, **not** permission to add a bias state to production.

Use a local linearization of the raw time series on a *fixed, physically valid observation support*:

\[
y_{c,t} = H_{c,t}\delta x_t + B_{c,t}\beta_{i,c} + e_{c,t},
\]

where `β_{i,c}` denotes a **point×camera×channel** persistent discrepancy, not an assumed universal camera-wide calibration bias. Explicitly distinguish:

- `M0`: current absolute RGB-D residual, no extra bias;
- `M1`: frozen per-camera frame-0 difference (the previous algebraic control);
- `M2`: uncertain constant point×camera offset estimated jointly from allowed frames (shared across time);
- `M3`: constant camera-wide calibration/appearance offset shared across points, estimated on independent controls;
- `M4`: time-varying nuisance (only if independent data support it); this is an overflexible **negative model** that can absorb motion, not an endorsed solution;
- `M5`: purely stationary trajectory with no bias, and `M6`: moving trajectory with no bias, **only for diagnostic fit/identifiability; do not choose states or labels in production**.

**Analytical requirements:**

1. For Gaussian models with valid correspondences, fixed linearization and proper priors, implement the exact joint normal equations and independently verify the Schur complement after eliminating nuisance parameters:

\[
\begin{bmatrix}A&B\\B^\top&C\end{bmatrix}
\begin{bmatrix}\delta x\\\delta\beta\end{bmatrix}
=\begin{bmatrix}g_x\\g_\beta\end{bmatrix}
\Rightarrow
(A-BC^{-1}B^\top)\delta x=g_x-BC^{-1}g_\beta.
\]

2. Independently compare with dense GLS with shared frame-0 covariance, including off-diagonal correlations. For independent frame errors `e_t`, differenced residuals `d_t=e_t-e_0` satisfy

\[
\operatorname{Cov}(d_t,d_s)=\delta_{ts}R_t+R_0\quad (t,s>0),
\]

**only** under stated independence and fixed geometry; source backprojection and canonical appearance can introduce extra cross-correlation that must be derived or flagged unknown.
3. Report model rank, condition number, prior sensitivity and parameter identifiability. If missing frame-0 camera 4 allows arbitrary `β` to explain either static geometry discrepancy or motion, explicitly demonstrate the gauge ambiguity. Do not infer a unique bias from a single late observation.
4. Do not compare normalized likelihood/evidence across different camera sets, invalid/occluded pixels, or unnormalized selection rules. If the original model's selected-row objective is all that exists, label comparisons **conditional on that support**.
5. Fit nuisance parameters on one temporal/camera subset and score on **held-out valid observations**. Avoid evaluating a model only on the same observation used to define its bias. Where no valid independent holdout exists, report `NO_INDEPENDENT_VALIDATION`.
6. Test whether a stationary model plus fixed bias reproduces the same RGB-D time series as a moving model plus no bias on the observed rows. If the data cannot tell them apart, report `NOT_IDENTIFIABLE`, not `STATIC_BIAS_PROVED`.

**Important:** The existing A4 `R_t+R_0` per-frame run **ignores across-frame correlations** and is only a sensitivity probe. The prior fourth-pass `reference_covariance_tests.csv` already demonstrates differences between naive and exact marginalization; reproduce those oracles first.

---

## 9. Hard gate G6 — numerical instability and acceptance, isolated from physics

For source-pixel-boundary/oscillatory cases **8416, 8417, 8418, 8444 frame 7, 49199, 102354, Blade 10251/14104**:

1. Reconstruct the exact floating-point projection path for source pixel at the frame of the failing update: save `u,v` at binary64 precision, nearest integer, ULP distance, `floor`, all four candidate neighboring cells, one-sided analytic Jacobians, RGB-D values (continuous but generally nondifferentiable at boundaries).
2. Separate changes caused by **residual itself**, **Jacobian only**, **cell membership**, **camera ownership**, **row gate**, **nonlinear acceptance**, and **finite-precision solve**. One factor per controlled replay; name the held-fixed values.
3. Run one-sided `nextafter`, ±1e−12, ±1e−10, ±1e−8 pixels with actual numeric representation; measure distinct `||Δx||` (m) and `||Δv||` (m/s). Do not sum or take a max over mixed-unit state blocks.
4. Plot per-inner-iteration `J_actual`, prior cost, per-camera/channel residual cost, iteration step, gradient/normal-equation residual, row set, and pixel cell. Calculate *fixed-support* cost at the prediction **and** at accepted state when samples remain physically valid. If samples become invalid, return `UNDEFINED_SUPPORT`, not zero cost.
5. Confirm whether the accepted step is monotone on the **same** valid physical measurement support, whether `last_consistent_iterate` is at an iteration cap, and whether a chi-square adequate update increases an actual comparable objective. Produce exact reproducible examples, not general assertions.
6. Offline solver controls: only **for analysis**, run the *same* frozen-support objective with a safeguarded line search/trust-region or dense reference minimize. Compare its local stationary point to the production IEKF candidate. This is an **oracle** to determine whether the production solve itself is unstable; do not substitute it into inference.
7. If two one-sided branches produce different legitimate minima because of non-smooth image interpolation, identify this as **non-unique/non-smooth optimization** and distinguish it from floating-point arithmetic errors. A high-precision solve inside the same cell cannot prove uniqueness across cells.

**Stop rule:** do not report a “bias correction prevents drift” for a point if the baseline or controlled trajectory does not reproduce across its pixel-boundary branch. Use `NUMERICAL_CAUSAL_INTERVENTION_INDETERMINATE`.

---

## 10. Hard gate G7 — explain the entire drift trajectory, not only frame 1

For each static case, analyze **frame 0 through at least frame 10**, then continue by reading all saved baseline frames through frame 59 to identify any later reattachment and accumulation. Recompute full CPU sequential replays only for decisive short windows; no need to replay all points for all 60 frames.

Exact temporal decomposition:

\[
x^+_T-x^+_0=
\sum_{k=1}^{T}\underbrace{(x^-_k-x^+_{k-1})}_{\Delta t_k v^+_{k-1}}
+\sum_{k=1}^{T}\underbrace{(x^+_k-x^-_k)}_{\mathrm{accepted\ position\ correction}}.
\]

Also decompose velocity increments, integrated early velocity versus later corrections, first predictive departure, first loss of physical source surface, first all-camera nonmeasurement, first correct-row rejection, first newly admitted biased camera, and effect of competition/occlusion. Reproduce vector closure, not only magnitudes.

For each of 8444, 67827 and 51975, create a **mechanism-specific timeline** that links:

`source observation -> frame0 multi-camera inconsistency -> actual first admitted row -> exact first Δx, Δv -> frame-2 prediction -> rows at every update -> first support/visibility change -> final accumulated drift`.

For 67827, explicitly include the alternate A2/A3/A4 timeline and the **late camera-4** event; depth referencing only suppresses this point in particular arms because other arms re-admit an unreferenced camera. Do not claim universal suppression.

Distinguish observation support reduction as a *cause of continuing drift* from a *cause of the first false update*. The population's previously reported 89–97% “drift before row loss” must be revalidated using production-admitted rows and correct timestamp ordering.

---

## 11. Unbiased controls: what would falsify the favored explanation?

Create a preregistered **intervention matrix**, holding solver/step/covariance/owners fixed wherever physically possible:

| Intervention | What it isolates | What finding rejects the hypothesis |
|---|---|---|
| Hold actual images, remove one **admitted** row | Particular camera's causal influence | Same onset Δx/Δv despite removal, or support becomes incomparable |
| Replace frame-t image with frame-0 image (all cameras) | Need for temporal image change | Correction disappears only because owners/validity change; cannot attribute to time signal |
| Replace only offender's RGB with its frame-0 RGB | Persistent RGB discrepancy | Same correction persists with identical accepted rows and objective model |
| Replace only offender's depth with frame-0 depth | Persistent depth discrepancy | Same correction persists or same-surface validity changes |
| Camera-specific residual difference, covariance **unchanged** | Algebraic constant-offset subtraction | Zero residual is tautological; nonzero movement from numerical/gating issues remains |
| Joint latent-bias marginalization, exact covariance | Statistical nuisance hypothesis | Posterior still assigns same motion, or bias unidentifiable on holdout |
| Global camera vs point×camera bias | Which spatial structure is supported | Global model fails on held-out points; local model only memorizes training data |
| Fixed physical row set / freeze ownership | Support switching | Movement persists on identical rows, so row dropping not necessary |
| Reference stable source pixel branch left/right | Nondifferentiable gradient | Different accepted motion even with identical images/physics |
| Enforce depth-only, RGB-only, full RGB-D | Channel causality | Wrong channel still creates correction after fixed-support, covariance-matched replay |
| Fix velocity exactly at zero **offline only** | Position/velocity separation | Explains suppression by construction, **not** cause/valid estimator |
| Same original residual with varied prior Pvv (offline only) | Prior sensitivity | Claim that prior alone caused false residual must be rejected |
| True GT motion position as **evaluation oracle only** for moving controls | Evidence accessible vs gate rejection | Physical view unavailable, so gate is not necessarily root cause |

Every intervention must record its **valid domain**, effective input changes, early `Δx,Δv`, full row set and 10-frame trajectory. When several variables change, mark **confounded**, even if error is reduced. Do not set a row's residual to zero and describe it as a physically realizable sensor intervention unless it is actually realizable.

### Additional cross-scene falsification

- Test the same proposed explanation on an actually measured stable control with *nonzero* cross-camera residual; if stable anyway, determine which factor (row omission, Hessian nullspace, convergence/velocity transfer, prior) blocks drift.
- Test a failure with **only a source camera** row (source pixel exact zero at frame 0); does it drift because the frame image changes, interpolation gradient at an integer boundary, erroneous ownership, or actual physical motion?
- Repeat on different surface orientations and cameras: do not infer universal cross-view RGB “bias” from Storage 8444 alone. Use USB crease and Blade depth-only cases as deliberate adversarial examples.

---

## 12. Required outputs (small, auditable, machine-checkable)

Create `OUT` if it does not exist; refuse to overwrite existing results. If third/fourth-pass source artifacts are incomplete, note failures in `limitations.md` instead of inventing outputs.

### Source and proof artifacts

1. `evidence_inventory.json` — SHA-256, source paths, git state, exact resolved configs and point-ID mappings.
2. `baseline_parity_detailed.csv` — every point/frame, position/velocity/covariance deltas, accepted rows/decisions, original/branch parity.
3. `point_catalogue_and_selection.csv` — fixed IDs plus predeclared holdout selection and source-point metadata; never GT inference.
4. `camera_row_lifecycle.csv` — every camera, each decisive frame/inner iterate: raw sample validity, ownership, frozen suppression, gate, accepted use, exact reason codes and owner IDs.
5. `frame0_vs_onset_raw_images.csv` — raw 2×2/patch differences, camera pose changes, raw and normalized RGB-D residuals, temporal differences, validity, physical identity caveats.
6. `inner_information_exact.npz` + index CSV — `A`, `b`, full `P`, `H`, `R`, `S`, `ν`, per-channel forces, eigenvectors, trace/provenance for decisive iterations only.
7. `causal_interventions.csv` — controlled variable, held-fixed variables, validity, counterfactual prior/position/velocity, accepted rows, objective, convergence, covariance and uncertainty of comparison.
8. `depth_surface_evidence.csv` — each 2×2/3×3/5×5 test, arithmetic-vs-inverse depth, planar fit residuals, visibility, uncertainty, `FACE_IDENTITY_UNPROVEN` statuses.
9. `rgb_appearance_evidence.csv` — per-channel source/readout appearance, temporal constancy, camera-wide vs local fitted residual and held-out scores.
10. `observability_identifiability.csv` — singular values, nullspace/gauge directions, latent-bias identifiability, existence of independent held-out observations.
11. `numerical_branch_stability.csv` — one-sided cells, exact px perturbations, position/velocity errors with units, accepted solve, row/cost changes.
12. `static_full_timeline.csv` — 0–10 replay, and baseline through frame 59; drift decomposition, first event chronology, sample persistence and support changes.
13. `negative_controls_moving.csv` — 87290/85696 changes in true-motion evidence, gating and ownership, **evaluation-only GT** columns.
14. `tests.log` — tests themselves, executable commands, final `OK`, hashes of inputs.
15. A **small curated** `figures/` folder with at least: each key point's exact RGB-D 2×2 + adjacent patch at frame0/onset; 1-D/2-D original/differenced objective along actual correction; camera/row history; singular-vector view; drift and velocity over time; floating point left/right branch for 8416. No massive per-frame screenshot dump.

### Main deliverable: `DEFINITIVE_F1_STATIC_DRIFT_ROOT_CAUSE.md`

Open with a compact **evidence table** whose claims are immediately traceable to CSV rows, traces and plots. Include one full dossier per mandatory failing point:

```
SCENE / POINT ID / STATUS: PARITY_PASS | PARITY_FAIL
GT ground/static confidence & caveats (evaluation only)
Original source frame, camera, pixel; original x_ref, R, Pxx0, Pvv0
FIRST false update: exact k, t, Δx vector & magnitude, Δv vector & magnitude
CAUSAL CAMERA(S): production-ADMITTED only; owner, pixel cell, raw 2×2, r, H, R, S, NIS, gate
TEMPORAL CHANGE: r(frame0 at x_ref), r(onset at x_ref), raw image delta, pose/visibility validity
FORCE DECOMPOSITION: 6×6 A, b, exact per-camera/channel contributions, eigenpairs, closure
NONLINEAR PATH: each IEKF iteration, changing cells/rows, accepted fallback, objective and convergence
CONTROLLED ABLATIONS: one mechanism varied at a time; fixed-context causal outcome + confounders
PROGRESSION: how first false Δv propagates, when/why additional information stops it (or fails)
BIAS MODEL TEST: which statistical models are identifiable, held-out score, covariance caveats
ALTERNATIVES TESTED & FALSIFIED: explicit results and numbers
UNRESOLVED/NOT IDENTIFIABLE facts: exactly which missing physical observation would decide
VERDICT: initiating source; optimizer/association involvement; velocity amplifier; persistence mechanism
EVIDENCE LEVEL: exact arithmetic / controlled causal / support / not identifiable / indeterminate
```

End with **three independent conclusions**, not one slogan:

1. **Mechanistic within-case result:** for each representative point, exact observation / update that causes the *first false displacement*, and its subsequent dynamics.
2. **Across-case scope:** what is common across different failures, what differs (RGB vs depth vs solver vs ownership), and the distribution on the unbiased static holdout; do not project the hand-picked sample proportions onto the full scene.
3. **Physical identifiability boundary:** what inference is impossible from the measurements without an assumption, which missing data would resolve it, and whether a putative latent bias or numerical fix is justified *in principle* but **not yet authorized for production**.

Add `OPEN_QUESTIONS_AND_DISCONFIRMING_EVIDENCE.md`: rank contradictions, negative controls, all invalid comparisons, all cases in which the favored hypothesis fails. The document must be critical of its own explanation.

---

## 13. Tests required (no test success, no accepted root-cause verdict)

Extend CPU-only independent tests under `tests/test_f1_definitive_drift_forensics.py` (split into modules if useful). At minimum:

**Production agreement / ownership**

- `G0a`: A0 exact accepted state, covariance, reasons for 8444, 67827, 51975, 51974 and controls across consecutive frames; known 8416 nonparity flagged.
- `G0b`: frozen duplicate ownership is excluded from every row/Hessian/cause attribution; regression test for 67827/262894/14104 errata.
- `G0c`: manifest hash/point ID/source camera/pixel parity, frame order/timestamp/extrinsics.

**Measurement / camera / physics**

- `G1a`: source pixel backprojection and 2×2 sampling reproduce RGB-D exactly at pixel center, with one-sided derivatives handled explicitly.
- `G1b`: finite-difference `H_RGB,H_D` within a differentiable cell; test *both* sides of integer boundaries.
- `G1c`: same-surface allowance and all RGB-D NIS gate decisions independently match production.
- `G1d`: frame0/frame1 exact equality for unchanged static image, including raw pixels and masks; distinguish 'raw image same' from 'reference valid'.
- `G1e`: source camera data correlated with `x_ref,a_i`; test no accidental independent resampling claims.

**Kalman/IEKF math and causal diagnosis**

- `G2a`: 6×6 normal equation independent dense solve, per-channel RHS/Hessian closure to machine precision.
- `G2b`: exact Δv–Δx cross-covariance relation for valid position-only row update; explicit process-noise propagation.
- `G2c`: off-support invalid camera must be `UNDEFINED`, never silently zero cost.
- `G2d`: frozen-image null intervention can reproduce false static update without actual temporal change when assumptions hold.
- `G2e`: finite synthetic stationary trajectory with persistent bias vs genuine moving trajectory; verify identifiability/nullspace claims.
- `G2f`: exact latent-bias marginalization equals dense correlated GLS under same likelihood and correctly handles missing references; naive R+R0 is **not** falsely declared equivalent.
- `G2g`: one-variable counterfactual restoration and deterministic repeatability; counterfactuals never modify production arrays.

**Numerical and negative controls**

- `G3a`: ±1e−12 px results keep position and velocity units separate, and flag parity failure independently of sensitivity.
- `G3b`: accepted nonconverged iterate and higher-comparable-cost counterexamples reproduced with full row-set evidence.
- `G3c`: `NO_VALID_REFERENCE` never imputed with zero; first valid later observation does not automatically infer a free bias.
- `G3d`: moving 87290/85696 diagnose support loss but GT cannot modify prediction, gate, bias or ownership.
- `G3e`: no data leakage: GT variable flows solely into evaluator; every hypothesis test is conducted with identical raw files and immutable baseline unless explicitly labelled intervention.

Run existing 17 fourth-pass tests plus earlier applicable unit tests in the same test report; failures must be discussed, not hidden. Check no new diagnostic code path is imported from production inference by default.

---

## 14. Execution order and stop criteria

1. **Preflight**: print evidence inventory/hashes, code paths, policy, original production trace and all earlier errata. Do not launch computations before identifying real input paths.
2. **G0 (parity)**: reproduce required cases and owners. On failure, isolate first divergence, record `INDETERMINATE`, do not substitute a nonmatching replay.
3. **G1 + G2 (first update)**: 8444, 67827, 51975/51974, 49199/controls. Produce exact admitted camera rows and six-state step, including a fixed-context force decomposition.
4. **G2 temporal null**: compare raw frames, camera-specific residuals and frozen-frame-0 interventions. Only then test bias subtraction and model variants.
5. **G3 + G4**: RGB/depth geometry and identifiability oracles. Reproduce both confirming and disconfirming cases.
6. **G5**: compare latent nuisance and GLS mathematically, with missing-reference cases and valid held-out data. This is offline and does not alter inference.
7. **G6**: numerical branch and objective/cell-row switching cases. Do not conflate mixed-unit sensitivity with displacement.
8. **G7**: timeline from first false correction to first lost row and late reattachment (including 67827 camera4), and moving negative controls.
9. **Population sample**: aggregate baseline arrays; analyze the predeclared holdout; rank mechanisms by support only where demonstrable.
10. **Report and tests**: write definitive report and disconfirmation appendix. Output the final path and top-level summary only after validations.

**Mandatory stop/qualification rules:**

- If camera support differs, do not compare raw selected-row objectives as though they were the same likelihood. Record `SUPPORT_NOT_COMPARABLE` and the exact lost/new rows.
- If a camera has no valid reference, do not subtract a manufactured offset or assume it is zero; classify `NO_VALID_REFERENCE` and explore identifiability.
- If a face/material identity cannot be independently verified, report that limitation rather than inventing mesh identity.
- If source camera owns integer-pixel discontinuity and A0 parity fails, do not claim counterfactual benefit for that point.
- If raw frame0/frame-t data differ in a truly moving or newly occluded region, do not assert static-pixel invariance.
- If model choice is based on GT to identify a static point, GT may be used only to **select/evaluate** diagnostic points; never enter a production or simulated online estimator update.
- Do not optimize a new filter, tune covariance to reduce endpoint GT error, introduce velocity caps, impose fixed static labels in inference, or alter the current first-pass baseline.
- If no experiment separates two causal explanations, present both with the **exact equivalence/nonidentifiability**. A justified `NOT_IDENTIFIABLE` is a valid scientific conclusion; a fabricated unique origin is not.

---

## 15. Final acceptance standard

I will consider this investigation complete only if a reader can take **8444** and **67827**, follow the raw measured pixels from frame 0 into a single *production-admitted* measurement row, reconstruct its residual/Jacobian/R/S/NIS, obtain exactly the accepted six-state update from an independent solver, explain why the original objective prefers the resulting position, determine whether any temporal evidence was new, and follow the injected velocity until the later loss or re-admission of cameras. The same method must work or transparently fail on the adversarial numerical and depth-only cases.

**Do not end the report with only** “persistent camera bias + velocity amplification.” That is the leading *description*, not the requested deep explanation. Identify **which physical/model mismatch** caused the residual, **which specific pixel/gradient or crease** translated it into a 3D spatial correction, **why** the nonlinear objective accepted that correction, and **which information was absent or confounded** so that physical motion could not be distinguished from a spatial discrepancy.

**Deliver the forensic evidence, not a new estimator.** If the result genuinely supports a principled model change, place proposed alternatives in a short final section as *unimplemented hypotheses* with predicted falsification tests—not as code changes, defaults or promises of guaranteed improvement.
