# CODEX IMPLEMENTATION CONTRACT — F1 FOURTH-PASS CAUSAL REPLAY AND UNIT-SAFE DIAGNOSTIC FIX

**Date:** 2026-10-30  
**Repository:** `/home/samuelemara/Joint3DGS/RGBDekf`  
**Task type:** implement **diagnostic code, tests and reports only**. Do **not** modify the production estimator, inferencer, noise configuration, camera geometry, defaults, or datasets. Do not commit unless subsequently instructed.  
**Primary question:** Does removing an already-existing, camera-specific frame-0 RGB/depth discrepancy prevent false velocity **over consecutive frames**, without preventing actual moving points from being tracked? At the first failure, exactly which observation, gate, numerical step or missing information is responsible?

This is **not** authorization to deploy temporal-reference subtraction as a new production model. The result is an auditable, **conditional single-point CPU replay**, plus correction of a concrete diagnostic **units bug**. No RTS, IEKS, EM, physical part constraints, GT-driven state corrections, covariance tuning, or full-scene reruns.

## 0. Mandatory inputs; read before coding

The actual third-pass evidence was supplied, not just its summary. Read:

```
<FOR>/second_pass_20261030/SECOND_PASS_ROOT_CAUSE.md
<FOR>/third_pass_temporal_reference_20261030/THIRD_PASS_CAUSAL_RESULT.md
<FOR>/third_pass_temporal_reference_20261030/cases/*.md
<FOR>/third_pass_temporal_reference_20261030/csv/*.csv
<FOR>/third_pass_temporal_reference_20261030/traces/{case_info,input_hashes}.json
```

Where:

```bash
REPO=/home/samuelemara/Joint3DGS/RGBDekf
FOR='/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030'
THIRD="$FOR/third_pass_temporal_reference_20261030"
OUT="$FOR/fourth_pass_sequential_replay_20261030"
```

Discover current filenames before coding; do not assume the source files match an older GitHub commit. Inspect these existing modules and reuse rather than duplicating their math:

```
tools/f1_temporal_reference_diagnostics.py
tools/f1_second_pass.py
tools/root_cause_oracle.py
tools/root_cause_trace.py
tools/forensic_f1_drift.py
estimator/point_kalman_cuda.py
estimator/projective_sensor.py
estimator/projective_terms.py
estimator/projective_visibility.py
estimator/free_projective_run.py
tests/test_f1_temporal_reference_diagnostics.py
```

**Do not change any code under `estimator/`, `scene/`, or `train.py`.** Use original source images, calibration, masks, canonical point catalogues and saved production F1 trajectories. Abort clearly if inputs are absent; do not fabricate replacements.

### Verified facts that must not be misrepresented

1. **Storage 8444, frame 1:** production uses **camera 1 only**; its RGB residual at fixed `x_ref` is identical at frames 0–2. Original single-frame correction = **24.156 mm**, velocity change norm about **1.411 m/s**. RGB-referenced replay gives approximately **0.006 mm**, RGBD-referenced gives ~0. This does **not** establish a correct ten-frame trajectory.
2. **USB 67827, frame 1:** production uses **cameras 5 and 6**, **not camera 4**. Camera-6 depth-crease discrepancy is persistent from frame 0; removing/referencing its depth term reduces the false step from **5.155 mm** to **0.163 mm** in the single-frame test. The depth-crease physical face identity is **not proven**.
3. **USB 87290:** its last successful update is frame 2; frame-3 predictions fail the chi-square gate at cameras 0/1/7 (reported NIS about **202/263/200**), although an offline GT-position evaluation shows potentially discriminating observations. GT recentering **has not** been a successful causal tracking test.
4. **USB 85696:** loses the last usable row predominantly to **frozen duplicate-pixel ownership**. A single camera with a reference supplies weak/ambiguous motion evidence. This is a **required negative control**, not an error to hide.
5. **Storage 8416:** source-pixel knife-edge and nonconvergent iterates; previous summaries describe a mixed six-state difference as if it were a position difference. This is a **diagnostic units bug**. Do not reuse the previously reported `0.65 m` as a verified position error.
6. An earlier report listed camera rows frozen out by per-pixel ownership. Use **production-admitted** rows; separately report potential/raw samples, but never attribute a correction to frozen-out camera rows.
7. The source-camera frame-0 observation *defines* canonical depth/colour/position. It is not an independent additional measurement. Correlation between `x_ref`, canonical colour `a`, and the reference source pixels must be acknowledged.
8. Original point IDs are retained; moving points can be misplaced or prediction-only. Never say they were deleted.

## 1. Deliverables

Implement or extend investigation-only tooling:

- `tools/f1_diagnostic_units.py` (or equivalent small reusable helper), with strictly typed/explicit state comparison utilities.
- **Correct** affected fields, charts and report labels in `tools/f1_temporal_reference_diagnostics.py` and, after a grep-based audit, in the earlier *diagnostic-only* tools if the same bug appears.
- `tools/f1_sequential_causal_replay.py`: deterministic CPU single-point **sequential** filter replay, with controls and counterfactual arms; no full-scene filtering.
- `tests/test_f1_diagnostic_units.py` and `tests/test_f1_sequential_causal_replay.py` (names may differ, but these cases must exist).
- New output directory (never overwrite third-pass results) containing:
  - `FOURTH_PASS_CAUSAL_REPORT.md` and a short `README.md` with the exact command;
  - `baseline_parity.csv`, `sequence_state.csv`, `sequence_camera_rows.csv`, `sequence_accepted_steps.csv`, `sequence_failure_events.csv`;
  - `sequence_objective.csv`, `reference_covariance_tests.csv`, `diagnostic_units_corrections.csv`;
  - compact `.npz` arrays of states/covariances and fixed-reference data;
  - figures for static drift, moving-point GT error **evaluation-only**, position and velocity, per-camera NIS, and observation support;
  - `tests.log` and `input_hashes.json` containing SHA256, Git commit, dirty status, command, resolved options and source versions.

All old result directories, original raw data, original traced arrays, and production outputs **remain byte-identical**. New files must be written in a **new** output directory; fail if it is nonempty unless a distinct explicit suffix is supplied.

## 2. FIX FIRST — dimensional error in six-state diagnostics

State:

\[
\mathbf s = [x,y,z,v_x,v_y,v_z]^\top,
\quad \mathbf x\ [\mathrm m],\quad \mathbf v\ [\mathrm{m/s}].
\]

A scalar computed from all six components, such as

```python
np.abs(s_a - s_b).max()
```

**has no physical unit.** It must never be labeled `m`, and must not determine a physical position-parity or positional-stability verdict.

### 2.1 Required exact code audit

In `tools/f1_temporal_reference_diagnostics.py`, inspect and correct at least the source positions corresponding to:

- `production_s_plus_max_abs_m = abs(s_plus - production_state).max()`;
- `accepted_state_diff_vs_unshifted_m = abs(s_plus_shifted - s_plus_unshifted).max()`;
- `iterate_diff_vs_unshifted_m` computed by subtracting full `[x,v]` iterates;
- `first_iteration_diverging_over_1e-9_m` computed from the mixed six-state scalar;
- axis labels and Markdown tables consuming those fields, including the `storage_8416_one_sided_sensitivity` figure and previous “baseline parity ... m” lines.

Also grep other `tools/*root*cause*`, `tools/f1_second_pass.py`, `tools/forensic_f1_drift.py`, and reports for this **same mixed-units pattern**. Do not opportunistically refactor anything else.

Use **separate** metrics:

```python
from numpy.linalg import norm

def compare_state6(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape == (6,)
    dx = a[:3] - b[:3]
    dv = a[3:] - b[3:]
    return {
        "position_l2_m": float(norm(dx)),
        "position_linf_m": float(np.abs(dx).max()),
        "velocity_l2_m_s": float(norm(dv)),
        "velocity_linf_m_s": float(np.abs(dv).max()),
    }
```

For arrays of several iterates, apply it per iterate. Use **two independent** first-divergence indices:

- `first_iteration_position_diverging_over_1e-9_m`;
- `first_iteration_velocity_diverging_over_1e-9_m_s`.

The numerical cutoffs are **diagnostic resolution**, not physics-based acceptance thresholds. Record actual differences instead of implying that values below the cutoff are identical.

Rename report/CSV fields to explicit units. Either **remove** the old mixed fields from new outputs or retain them only as `legacy_mixed_state_max_UNITLESS_INVALID` with a loud warning; no user-facing claim of meters may derive from them. Don't quietly reinterpret old numbers.

Do **not** turn a 6×6 state covariance into a single `m²` covariance. Split covariance comparisons into `P_xx` (m²), `P_xv` (m²/s), and `P_vv` (m²/s²), and report appropriate blockwise norms when comparing replays.

**Mandatory correction report:** from the stored original and perturbed six-vectors, recalculate position and velocity differences for Storage 8416, USB 67827 and the oracle-parity entries 8416/8417/8418. Explain whether the prior reported large scalar was position- or velocity-dominated. Do not invent the corrected values before running the code.

### 2.2 Unit tests

- Construct two states with **identical positions**, velocity differing by 2 m/s. Position difference must be exactly 0 m, velocity difference 2 m/s.
- Construct two states with 4 mm positional difference and equal velocities; report 0.004 m, 0 m/s.
- Mixed nonzero changes: both components retain distinct units; no scalar is treated as a distance.
- Reference every historical field producing the prior invalid labels; fail a grep/AST-style diagnostic audit when old fields still feed charts or conclusions.
- Verify the original per-point production-vs-oracle parity with **position and velocity separately**. Do not claim the old 0.65/0.79 is a distance.

## 3. EXACT replay semantics — do not reset at each frame

The existing third-pass counterfactuals independently solve individual frames using production's pre-saved `s^-_k, P^-_k`. Those tests **cannot** answer whether a corrected frame-1 update changes frame-2 prediction and prevents frame-3 failure.

For each arm and each selected point, construct one continuous sequence from canonical frame 0 through frames **1–10 inclusive**, optionally extending to frame 15 **only if** a first loss occurs near frame 10. Do not launch another 60-frame full scene.

### 3.1 Initial state and prediction

Use exactly the existing F1 canonical point and covariance, not GT:

\[
\mathbf s_0^+=[\mathbf x_{\mathrm{ref}},\mathbf0],\qquad
P_0^+=\begin{bmatrix}P_{xx,0}&0\\0&P_{vv,0}\end{bmatrix}.
\]

Use **the actual archived** `P_{xx,0}`, `P_{vv,0}`, timestamps, `q_c`, and the production constant-velocity transition:

\[
F_k=\begin{bmatrix}I&\Delta t_k I\\0&I\end{bmatrix},\qquad
\mathbf s_k^-=F_k\mathbf s_{k-1}^+,
\qquad P_k^-=F_kP_{k-1}^+F_k^\top+Q_k.
\]

The exact `Q_k` (including cross-block signs) must match `estimator/point_kalman_cuda.py::process_noise`, **not** be reconstructed from memory or an alternate convention.

After a valid measurement update, persist both the accepted **full six-state** posterior and the **full 6×6** covariance. On a prediction-only frame, `s^+=s^-`, `P^+=P^-`. Preserve the original F1 inner iterations, gate and acceptance rule **unchanged** in every replay arm.

Never restart an arm at each frame from the saved baseline predicted state. Do not overwrite estimated velocity with a finite-difference GT velocity. Do not freeze a point classified static by GT.

### 3.2 Other points and pixel ownership: explicitly conditional replay

The per-point F1 update depends on the positions of **other points** through pixel ownership and self-occlusion. A single-point replay cannot fully reproduce how those other points would move under a globally changed measurement model.

Implement **two separately labeled modes**:

**Mode P: PRODUCTION_OWNERSHIP_REPLAY (primary causal control).** At each frame, use the original production frame's frozen eligibility/ownership information for the selected point, as taken from its trace or recomputed exactly from the original full-cloud state. Then evaluate the replayed point's own geometric/sensor validity, RGB-D row gate, and final recheck as applicable. Label the fact that other points are not re-estimated. If a replayed point changes pixel, the borrowed original ownership is only a conditional diagnostic; log `OWNERSHIP_NOT_PHYSICALLY_RECOMPUTED` instead of claiming physical validity.

**Mode C: CONDITIONAL_RECOMPUTED_OWNERSHIP (sensitivity).** Construct other points' predicted positions from the **saved original F1 state** at each frame, replace only the selected point's position with this arm's prediction, and recompute per-pixel front-most assignment and self-occlusion exactly by production tie-breaking. Freeze each frame's ownership during its inner iterations, then apply the same final self-occlusion recheck. Do not use GT. Report the fact that this mixes the replayed point with baseline trajectories of all competitors.

Where full-cloud saved positions or float32 quantization prevent exact reconstruction, flag `OWNER_PARITY_UNRESOLVED` with point/pixel/frame; **never quietly substitute an approximate owner** and then call a result identical to production. A per-frame original trace of owner IDs is preferred to approximating with rounded positions.

**Keep both modes' results distinct.** Mode P has strongest apples-to-apples control of original row selection; Mode C tests whether ownership changes would dominate once the point's motion changes. Neither equals a production full-scene run.

### 3.3 Mandatory baseline parity before interpreting counterfactuals

For each selected point, arm `A0_ORIGINAL_F1`, production ownership Mode P, original measurement covariance and policy, starting at frame 0, must reproduce production F1 **frame by frame**, not just the first frame. Check:

- `s^-`, `s^+` position, velocity (separate units), `P^-`, `P^+` blockwise;
- admitted camera IDs, per-camera reason, ownership, row count;
- accepted iteration/`converged`/acceptance code and predicted-only transitions;
- source images, calibration/intrinsics, timestamps, process covariance, and image interpolation precision.

No single universal numerical threshold may be used to conceal the known pixel-boundary instability. For stable cases **8444/67827**, require at least `position_linf <= 1e-9 m` and `velocity_linf <= 1e-7 m/s` where production trace precision supports it; otherwise show the exact divergence, first frame/iteration and cause. For 8416 explicitly allow a **reported parity failure** rather than an invented tolerance. If the original baseline fails parity on a point, all causality claims based on that replay are **INDETERMINATE**, while the separately valid first-frame tests remain usable.

Do not assume saved float32 velocities are ground-truth float64 states; use full-precision traces where available, and explain precision limitations.

## 4. Measurement counterfactual arms

All arms use **the same original state and prior** at frame 0, original camera calibration, measured images, hard foreground masks, observation-validity rules, and original IEKF transition. The only intended difference within an arm is the indicated measurement residual/covariance; do not silently change process noise, gates, acceptance, or maximum iterations.

Use the already implemented third-pass reference constructor. For camera `c`, cache the immutable valid frame-0 reference:

\[
\mathbf b_{c,0}=h_{c,0}(\mathbf x_{\mathrm{ref}}),\quad
h_{c,t}(\mathbf x)=\begin{bmatrix}\mathbf a-I_{c,t}(\pi_c(\mathbf x))\\ Z_c(\mathbf x)-D_{c,t}(\pi_c(\mathbf x))\end{bmatrix}.
\]

The referenced residual is:

\[
\mathbf r^\Delta_{c,t}(\mathbf x)=h_{c,t}(\mathbf x)-M\mathbf b_{c,0},
\]

where `M = diag(1,1,1,0)` for RGB, `diag(0,0,0,1)` for depth, `I4` for RGBD. The reference is **fixed at `x_ref`**; it is never resampled at the new moving position. Jacobian w.r.t. current x is still the derivative of `h_{c,t}(x)`; no reference derivative w.r.t. x is added when the reference is treated as fixed in this conditional diagnostic.

Arms, minimum:

| ID | Temporal residual | Measurement variance treatment | Purpose |
|---|---|---|---|
| `A0_ORIGINAL_F1` | absolute original RGBD | original `R_t` | baseline control |
| `A1_REFERENCE_RGB` | subtract frame-0 RGB bias | original R (`ALGEBRA_ONLY`) | isolate appearance offset |
| `A2_REFERENCE_DEPTH` | subtract frame-0 depth bias | original R (`ALGEBRA_ONLY`) | isolate depth offset |
| `A3_REFERENCE_RGBD` | subtract all valid frame-0 channels | original R (`ALGEBRA_ONLY`) | combined causal test |
| `A4_REFERENCE_RGBD_RPLUSR0` | as A3 | `R_t + R_0` for referenced channels only | one-step independent-samples **sensitivity approximation**, not calibrated likelihood |

If using additional arms (e.g. source-camera-excluded reference, fixed-row gate diagnostic), label them **additional diagnostic**; do not promote to main results.

For a camera without a valid frame-0 reference, **do not invent b=0**, do not fetch a substitute pixel, do not use GT. Either retain the original residual with an explicit `NO_VALID_REFERENCE_ORIGINAL_USED` flag (primary policy matching the third pass), or exclude it in a separately labeled sensitivity arm. Record the choice per camera and frame.

The source-camera frame-0 `b=0` arises from the construction of the initial point and canonical color. It is **not** an extra independent observation. Include a sensitivity analysis that flags this correlation; do not make unjustified information-gain or posterior-calibration claims.

Recompute the innovation gate from the counterfactual residual and declared R **at each frame**, without substituting the original baseline gate decision. Freeze the per-frame assignment only as specified in §3.2. Record reasons (including `MASK_EXCLUDED`, `DEPTH_DISCONTINUITY`, `SELF_OCCLUDED`, `DUPLICATE_PIXEL_SUPPRESSED`, `INNOVATION_INCONSISTENT`, `MEASURED_SURFACE_IN_FRONT`) separately, in correct production priority.

### 4.1 Covariance integrity: non-negotiable qualification

The same frame-0 observation enters **every** later difference. In general, the temporal differences are correlated. If observations have noise e_t and e_0, for arbitrary covariance blocks `C_ts = Cov(e_t,e_s)`,

\[
\operatorname{Cov}(e_t-e_0,e_s-e_0)
=C_{ts}-C_{t0}-C_{0s}+C_{00}.
\]

Under the *additional* assumption of independent sensor noises across different times, for t,s > 0:

\[
\operatorname{Cov}(e_t-e_0,e_s-e_0)=\delta_{ts}R_t+R_0.
\]

`A1–A3` (original R) are **algebraic-causality probes only**, and `A4` treats each reference difference's **marginal** variance more appropriately under independence but still neglects cross-time correlation in its recursive Kalman update. Neither yields a guaranteed calibrated posterior over 10 frames. Furthermore, deterministic render mismatch / camera appearance bias is **not** equivalent to independent pixel noise, and frame-0 source noise is already correlated with `x_ref` and `a`.

Implement a **separate independent offline covariance oracle** over a 3–5-time synthetic linearized fixture:

1. Build the full block covariance above and verify SPD (or PSD with exact conditioning when singular).
2. Compare dense generalized least squares using full covariance vs a latent shared reference-noise variable marginalized analytically. Their means/covariances/log-likelihoods must agree numerically.
3. Compare to naïve repeated independent `R_t+R_0` to illustrate its different uncertainty. Do **not** assert it is exact.
4. Save a `reference_covariance_tests.csv` with assumptions, dimensions, eigenspectra and relative errors.

**Do not implement a new correlated-noise Kalman estimator in production.** If a proper sequential correlated-reference estimator would require shared latent b states and canonical-source cross-covariance, report that as an outstanding requirement, not a silently completed task.

## 5. Mandatory point cases and time horizon

**Primary cases:**

1. `storage_45135:8444` — static, RGB-driven false velocity, start at frame 0 and trace at least through frame 10.
2. `USB_100109:67827` — static, camera-6 crease depth mismatch, frames 0–10.
3. `USB_100109:87290` — moving, last rows frame 2, first all-camera rejection frame 3, frames 0–10.
4. `USB_100109:85696` — moving negative control, duplicate-pixel ownership, frames 0–10.
5. `storage_45135:8416` — moving, source-pixel knife edge, frames 0–5; numerical divergence and unit correction required even if parity fails.

**Additional cheap controls:** `USB_100109:51974` (stable), `USB_100109:51975` (static failing), and `storage_45135:60019` (stable). Use the same arms/frames; at least 0–5 for controls.

For every point/arm/frame save both state (position, velocity), covariance, geometric projection, row eligibility, actual sampled 2×2 pixel cells, per-camera values, residuals, gradients, NIS, gate threshold, accepted vs rejected update, and whether inner iteration converged. **Do not infer from only the final endpoint.**

### 5.1 Static point acceptance questions (8444, 67827)

For each arm, calculate:

- `drift_from_xref_m(k) = ||x^+_k - x_ref||` and `speed_m_s(k) = ||v^+_k||`;
- vector decomposition: `x^+_k-x_ref = sum_{j=1}^k (dt_j v^+_{j-1} + (x^+_j-x^-_j))` (verify closure);
- first frame with measurable false update, first nonconverged accepted iterate, first row loss, first row readmission;
- residuals at **fixed x_ref** from all valid cameras and their temporal differences, and residuals at the replayed position;
- per-step camera ownership and admitted row IDs; gate vs sensor noise variance, with `sqrt(S)` and `sqrt(R)` distinguished;
- full time series of actual motion and uncertainty, not just the initial zero correction.

**Decisive test:** if A3 produces near-zero frame-1 position/velocity but later develops significant motion at frame 2–10, trace the **first later** nonzero update by exact camera, pixel, depth/color channel, and temporal change. Do not summarize as “reference solves static drift.” If A1 succeeds for 8444 but A2 does not, explicitly report that distinction; analogously for 67827.

A static control held at x_ref with no valid observations is **not evidence of successful tracking**. Report observed-frame counts. Also distinguish x_ref transport drift from any actual GT geometry error.

### 5.2 Moving point 87290: test genuine non-GT recovery

This case is successful **only** if the sequential estimate starting at frame 0 (never recentered using GT) retains reliable rows through the original frame-3 failure or demonstrably reduces its GT-evaluated tracking error while keeping physical observation support.

At frames 1,2,3,4,5 and 10, record for cameras **0,1,7**:

- projection at each arm's own predicted state, depth and RGB residuals, covariance-propagated NIS, and chi-square thresholds;
- accepted/rejected row status with **exact reason**;
- deviation of the predicted and posterior states from GT **evaluation-only**, separate from residual scoring and update;
- counterfactual **at GT** (tag `GT_ORACLE_ONLY`) to determine whether evidence exists but is unreachable from the predicted state;
- whether the frame-3 first zero-row event moves later, disappears, or remains.

A `GT_ORACLE_ONLY` state **must not** become a prediction, initialization, gate center, motion prior, or fallback iterate in a replay arm. If a model only works after GT recentering, classify it as `MOTION_EVIDENCE_EXISTS_BUT_INFERENCE_FAILED`, **not** recovered.

### 5.3 Moving point 85696: do not fake observability

At frame 3 and adjacent frames, log the exact **last camera** and why each other camera was suppressed. Disaggregate:

- frozen duplicate ownership;
- self-occlusion;
- mask or invalid depth;
- depth discontinuity;
- innovation rejection.

Compare both Mode P and Mode C; distinguish a merely available image from an eligible **independent point observation**. If all cameras remain suppressed, the outcome is `NO_OBSERVABILITY_UNDER_CURRENT_PIXEL_OWNERSHIP`, not failure of reference subtraction alone.

At fixed x_ref and GT x (GT only for scoring), report whether the referenced rows distinguish actual displacement; do not claim motion recovery if GT and stationary costs are similar.

### 5.4 Numerical knife-edge 8416

Keep production solver unchanged. Run original and referenced single-point local sequences with **explicit one-sided** `+/-1e-12, +/-1e-10, +/-1e-8` pixel perturbations for the original source projection in the affected step. Perturb only the requested diagnostic projection input, not camera calibration or GT.

For each arm save:

- `position_l2_m`, `position_linf_m`, `velocity_l2_m_s`, `velocity_linf_m_s` compared to unperturbed;
- separate per-iteration position/velocity divergence and first-divergence indices;
- side-selected cell/one-sided Jacobian, per-iteration accepted rows, objective values, stopping and fallback;
- comparison to production full-precision trace, separately for position and velocity.

Report whether reference subtraction reduces **physical positional sensitivity** or only velocity sensitivity. Do not call a trajectory numerically stable simply because the first position step is smaller.

## 6. Objective and gate reporting — no false monotonicity

For a fixed arm and a **named fixed set of camera identities**, compute the original 6D prior cost and the nonlinear resampled residual cost:

\[
J_{\mathcal C}(\mathbf s)=
(\mathbf s-\mathbf s^-)^\top(P^-)^{-1}(\mathbf s-\mathbf s^-)
+\sum_{c\in\mathcal C} r_c(\mathbf x)^\top R_c^{-1}r_c(\mathbf x).
\]

Here r_c and R_c must belong to the **same declared arm**, and projection/resampling happens at the tested state. Note that the camera **identity set** may be held constant while underlying pixels change; log pixel changes separately. If a factor is physically invalid at the candidate, mark that comparison `UNDEFINED_SUPPORT_CHANGED`, never substitute zero residual or simply drop its cost.

Also report the original production's **own-valid-row adequacy statistic** separately, including differing row counts. Never compare objective values of A0 (absolute residual) against A3 (reference residual) and declare one model “monotonically better”: the residual models differ.

No replay arm is authorized to change the production IEKF acceptance rule. A proposed future monotone solver is **not part of this diagnostic**. Flag accepting a higher fixed-factor cost and accepting nonconvergence as outcomes, not as exceptions to silently fix.

## 7. Exact tests (CPU)

Implement deterministic tests without the full dataset where possible:

**U1 — unit regression.** Three state comparison fixtures from §2 plus full-iteration traces and exported CSV headers; fail if a mixed six-state scalar is labeled meters.

**U2 — baseline continuous parity.** For 8444 and 67827 at least frames 0–3, A0/Mode P matches production per-frame states and row reasons at the precision warranted by saved traces. If not, test must FAIL and report first divergence; do not weaken to acceptance-only parity. For 8416 record a known numerical parity limitation distinctly.

**U3 — prediction-only identity.** A point with no admissible row preserves the mean propagation, full covariance and all cross-blocks exactly to numerical tolerance; no artificial contraction or reset.

**U4 — frame-0 reference invariance.** On a static synthetic image, all valid fixed-position temporal differences vanish; Jacobians for current x remain unchanged; the source camera is not counted a second time.

**U5 — motion remains detectable where observable.** Synthetic moving textured point produces nonzero temporal evidence; a textureless plane tangential-motion fixture yields ambiguity. Do not assert every motion must be recoverable.

**U6 — shared-reference covariance.** Dense GLS and latent shared-noise representation agree on a 3–5-frame linear Gaussian oracle; naïve independent treatment demonstrably gives a different posterior/covariance for generic nonzero R0.

**U7 — invalid reference.** If frame-0 sample is masked, discontinuous, occluded or missing: it remains `NO_VALID_REFERENCE`; no zero fill, no substituted source camera, no GT.

**U8 — ownership priority.** Frozen-out cameras do not contribute to the information equation; Mode P and Mode C log their explicit ownership differences. Reproduce 8444 camera 1 only and 67827 cameras 5/6 at frame 1.

**U9 — GT separation.** Monkey-patch/instrument all inference inputs and assert no GT arrays, GT transformations, GT joint state or GT recentered state are used in `predict`, `gate`, `rows`, `solve` or acceptance; only `evaluate(...)` may read GT.

**U10 — reproducibility and nonmutation.** Rerun selected small cases; hashes of original source artifacts/production outputs remain unchanged and new numeric outputs are deterministic within clearly stated precision; no GPU used.

**U11 — expected negative control.** With all camera rows frozen as duplicate for 85696, the point obtains no measurement information and prediction-only transport occurs. The test must NOT invent a successful reacquisition.

**U12 — honest loss reporting.** If a fixed-factor camera becomes invalid after movement, objective comparison returns `UNDEFINED_SUPPORT_CHANGED`. Separate position and velocity plots show correct units and labels.

Capture *full test output* in `tests.log`, including number of tests, failures/skips, actual assertions and source revision. Do not claim success based on a truncated `tail -4` that hides failures.

## 8. Reporting template — completion requires all rows

`FOURTH_PASS_CAUSAL_REPORT.md` must contain these sections:

1. **Executive status:** `COMPLETE`, `PARTIAL` or `BLOCKED`, with exact skipped/failed items. Never claim complete if A0 baseline parity or any required point is missing.
2. **Diagnostic bug correction:** original incorrect field names, replacement names/units, exact corrected values for 8416/67827 and oracle parity, test outcomes, and which historical statements must be retracted.
3. **Reproduction/parity:** per-point first divergence (position, velocity, covariance, rows, acceptance) relative to production; note conditional ownership and source precision.
4. **Static 8444:** per-frame table for all A0–A4 (and P/C where available): drift, speed, variance, admitted cameras, first accepted false update, first loss.
5. **Static 67827:** same plus camera-6 depth-crease row and explicit `FACE_IDENTITY_UNPROVEN` where needed.
6. **Moving 87290:** comparison of before/after frame-3 gate, camera 0/1/7 NIS, accepted measurements and actual GT error over time (GT for evaluation only).
7. **Moving 85696:** per-camera last-support cause, negative observability verdict, P vs C sensitivity.
8. **8416 numerical boundary:** pixel perturbation **position and velocity** separately, iteration-map divergence, and honest baseline parity.
9. **Covariance status:** independent-samples approximations vs full shared-reference covariance oracle; clear note that current causal replay's reference posterior is **not a calibrated probabilistic claim**.
10. **Failure analysis:** which update/event in which frame starts a new false velocity, with camera/pixel/channels/Jacobians/NIS; no vague “still drifts.”
11. **Next research decision:** do the results justify formalizing an observation-bias latent state / correct correlated reference model, or does failure arise earlier from invalid ownership, nonconvergence or missing information? **No code implementing that next research decision.**

For each case/arm provide one of:

- `STATIC_FALSE_DRIFT_SUPPRESSED_THROUGH_FRAME10` (and actual drift/speed evidence);
- `STATIC_FALSE_DRIFT_DELAYED` (name first later bad frame);
- `STATIC_FALSE_DRIFT_PERSISTS` (name first bad update);
- `MOTION_TRACKED_WITHOUT_ORACLE` (GT evaluation supports, no GT used in inference);
- `MOTION_EVIDENCE_EXISTS_BUT_INFERENCE_FAILED`;
- `NO_OBSERVABILITY_UNDER_CURRENT_PIXEL_OWNERSHIP`;
- `BASELINE_PARITY_FAILED`;
- `INDETERMINATE` (specific missing information).

No purely invented velocity/speed threshold should determine success. Use physical displacements and error curves; report any thresholds as **evaluation-only**, not a hidden model assumption. The qualitative verdict must be supported by exact numbers.

### Suggested command shape

Use supported CLI options of the actual implementation; do not add misleading switches. Example:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
FOR='/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030'
OUT="$FOR/fourth_pass_sequential_replay_20261030"
CUDA_VISIBLE_DEVICES='' ~/miniconda3/envs/rigeuler3dgs/bin/python \
  tools/f1_sequential_causal_replay.py \
  --forensics-root "$FOR" \
  --third-pass-root "$FOR/third_pass_temporal_reference_20261030" \
  --out "$OUT" --frames 0:10
CUDA_VISIBLE_DEVICES='' ~/miniconda3/envs/rigeuler3dgs/bin/python \
  -m unittest tests.test_f1_diagnostic_units tests.test_f1_sequential_causal_replay -v \
  > "$OUT/tests.log" 2>&1
```

Check that `$OUT` is newly created/empty **before** running. If the CLI differs, record the **actual exact invocation** in README; don't paste an untested example as the executed command. Use CPU-only, selected points, no entire 60-frame re-filtering.

## 9. Non-negotiable stop conditions

- **If original A0 multi-frame parity fails:** stop interpreting differences for that point and first isolate the replay discrepancy (ownership, float32 archive precision, nonlinear iterates, row selection). A counterfactual built on an irreproducible baseline cannot establish a causal improvement.
- **If a referenced static point remains stationary only because it has no accepted measurements:** report `UNOBSERVED`, not success.
- **If a moving point only tracks after GT recentering:** report failure without oracle.
- **If changed pixel support makes an objective undefined:** do not compare by treating disappearing rows as zero loss.
- **If covariance assumptions are not valid:** report descriptive state trajectories, not calibrated posterior or Bayes-factor claims.
- **If the diagnostic state-unit bug is not corrected and retested:** the implementation is incomplete even if the replay figures look good.

**Final Codex response must include:** exact files changed (diagnostic/tools/tests only), executed commands, test totals, output paths, corrected 8416 position-versus-velocity numbers, baseline parity status, 8444/67827 trajectories through frame 10, whether 87290 keeps frame-3 measurements without GT, why 85696 loses its last camera, and explicit unresolved limitations. **Do not implement or suggest changing the outer IEKS during this task.**
