# Why static points drift and moving points get lost in the standard F1 IEKF

> **ERRATUM (2026-10-30, found by the second pass, `tools/f1_second_pass.py`).**
>
> **What was wrong.** The per-camera row listings of the section 4 stop-rule blocks, and of the onset information table built on them, listed every camera whose sample was valid at the prediction. They included cameras that production had already **frozen out by pixel ownership**: DUPLICATE_PIXEL_SUPPRESSED, i.e. the camera's pixel is owned by another point.
>
> **The rows actually admitted at s⁻:**
> - USB: 45971 cameras 3, 4 (not 5); 67827 cameras 5, 6 (not 4: the −80σ crease row was **not** used; the used crease row is camera 6 at −41σ); 30502 camera 1 only.
> - Storage: 8444 camera 1 only (not 0, 7); 49199 cameras 2, 3; 205921 cameras 4, 5; **262894 cameras 5, 7 only. The occluder rows of cameras 3/4 (67/31 mm in front) never entered the update, so "they steer the iterations" is wrong.**
> - Blade: 10251 cameras 0, 1; 27016 camera 5; 11634 cameras 1, 4; **14104 cameras 3, 4. The camera-0 occluder (30 mm in front) was frozen out; the admitted wrong-surface row is the camera-4 crease (−47σ).**
>
> **What is unaffected:**
> - the population statistics of section 3, which use production's own camera_reason;
> - the same-A splits and whole-Hessian counterfactuals, which use the oracle's frozen candidacy;
> - the amplifier, the oracle reproduction and the H4/H5/H6/H7 evidence.
>
> **Retracted.** Occluder admission by the row gate is withdrawn as a demonstrated F1 initiator for 262894 and 14104. What remains true is that the row-level gate would pass such rows if ownership did not exclude them.
>
> See `forensics_F1_inner_drift_20261030/second_pass_20261030/SECOND_PASS_ROOT_CAUSE.md` once that pass is run.

> **ERRATUM 2 (2026-10-30, fourth pass, `tools/f1_diagnostic_units.py`): units bug.**
>
> Every "accepted state differs by … m" / "reproduces to … m" number in this report and in the second- and third-pass outputs was a max over the MIXED 6-state difference [x (m), v (m/s)]. That number has no unit.
>
> **Corrected values:**
> - Section 7, moving points 8416–8418: "accepted states 0.03–0.79 m apart" is wrong. Recomputed from the stored vectors, 8416 differs by 11.1 mm in position and 0.646 m/s in velocity under a +10⁻¹² px shift; 8417 differs by 13.6 mm and 0.793 m/s. The large values were velocity.
> - "400 of 413 decisive updates reproduce to better than 1e-9 m" used the same mixed scalar.
>
> Separate position and velocity values are in `fourth_pass_sequential_replay_20261030/diagnostic_units_corrections.csv` (or the smoke-run equivalent).

Forensic investigation following `RGBDiekf_Static_Drift_INNER_IEKF_Only_Codex_2026-10-30.md`. Scenes: USB_100109, storage_45135 and blade_103706 (the spec named USB only; storage and blade were added on request). Estimator: the **standard directional forward F1 IEKF** only. No RTS, IEKS, temporal solver or EM was run or read. Nothing in the estimator was changed. Ground truth was used only to pick and score points after estimation.

## 1. Answer

**On all three scenes, a static point drifts through the same three-link chain. The code does exactly what its equations say:** an independent float64 oracle reproduces 400 of the 413 decisive frame updates to better than 1e-9 m. The 13 that do not match are explained in section 7.

1. **Initiator.** At the point's first measured frame k, the position prior is very weak along the surface: σ ≈ k·dt·σ_v0. That is 17 mm at frame 1, because P_vv0 = 1 m²/s², the documented uncalibrated initial guess. The point's admitted rows then pull it along the surface by millimetres. This happens in one of two ways.
   - **(a) Noise-level rows (the majority).** The rows agree with the static position to within noise: |z_D| ≤ 3, colour residuals 1–4σ. They pull the point along directions that the data barely constrain:
     - depth constrains only the surface normal;
     - colour gradients along the surface are weak, or are 8-bit quantization steps;
     - the bilinear interpolant adds spurious gradients along the surface.

     The accepted correction is 0.01–0.3 prior-σ, i.e. 1–30 mm.
   - **(b) Structurally wrong rows (a minority).** The 2×2 depth cell straddles a crease, an edge or an occluder, so the row is 40–240 000 σ_R off at the static position. Such rows are admitted for two reasons:
     - the same-surface test allows up to about 5 cm of depth spread (the 80° maximum-incidence term);
     - the row gate's innovation covariance is dominated by the prior (√S_DD = 17–760 mm, against √R_D = 5.7–56 µm).
2. **Amplifier.** For position-only rows the velocity correction is Δv = P_vx P_xx⁻¹ Δx, and at the first measured frame this gain is 1/(k·dt). Measured |Δv|/|Δx| at onset is 58.4–58.9 s⁻¹ at frame 1, 29.4–29.5 at frame 2 and 19.6–19.7 at frame 3, in every traced onset of every scene. Each millimetre of correction becomes about 1 mm per frame of permanent motion.
3. **Persistence.** The point now slides along its own surface, so its depth rows stay consistent and nothing in the measurement contradicts the motion. Integrated velocity is 94–98% (median) of the final departure. Over 89–97% of drifters reach 1 mm of drift *before* they ever lose their rows. Later, sliding carries points into creases, occluders and pixels owned by other points, so they lose rows and coast. Coasting points never stop.

**Moving-part points** are lost through support collapse, not deletion. Pixel ownership (DUPLICATE_PIXEL_SUPPRESSED) and self-occlusion remove their rows from frames 2–4, and they coast with whatever velocity the first one or two updates gave them. Some re-attach to wrong surfaces: about 20% of measured moving points are more than 5 cm from ground truth. All of them stay in the output arrays.

None of this is an arithmetic bug. The causes are:
- **Model and prior:** an uncalibrated velocity prior combined with a state that is unobservable along the surface.
- **Two admission rules** that accept non-single-surface rows.
- **Two numerical sensitivities:** non-converged, expanding inner iterations, and knife-edge Jacobians at source pixels.

## 2. Scope, provenance and invariance

| | USB_100109 | storage_45135 | blade_103706 |
|---|---|---|---|
| standard F1 baseline | `static_equilibrium_audit_20261029_1718/USB_100109/baseline` (e1e663b) | `…/storage_45135/baseline` (e1e663b) | new, same CLI: `forensics_F1_inner_drift_20261030/blade_103706/baseline` |
| traced re-run (same resolved arguments + `--debug-point-ids` / all `--debug-frames`) | `forensics_F1_inner_drift_20261030/USB_100109/traced` | `…/storage_45135/traced` | `…/blade_103706/traced` |
| tracing invariance (every published array) | IDENTICAL | IDENTICAL | IDENTICAL |
| argument differences | none (`free_solver` / `temporal_config` exist only after e1e663b, at their standard default) | none | none |
| points / frames / cameras | 95 967 / 60 / 8 | 346 437 / 60 / 8 | 34 369 / 60 / 8 |
| √R_D (measured, sum_w2 + model) | 55 µm | 5.7 µm | 11.5 µm |
| prior | P_vv0 = I m²/s², q_c = 0.01 I m²/s³ (documented-initial-guess), P_xv0 = 0; dt = 1/59 s | same | same |
| policy | `last_consistent_iterate`, `rgbd_chi2`, `recheck_self_occlusion`, `per-frame` pixel ownership, float64, 10 inner iterations | same | same |

- **Selection:** 30 point IDs per scene, chosen deterministically from the standard F1 output (`selected_point_ids.csv`):
  - 9 static points that drifted >10 cm and 6 that drifted 1–10 cm;
  - 6 stable controls (<1 mm), matched by source camera, depth, slope, texture and rows;
  - 7 moving-part points;
  - 2 never-measured controls.
- **Traces:** every frame of every selected ID.
- **Ground-truth caveat:** "static" means the nearest labelled point in the ground-truth part cloud is static, with cloud spacing h = 1.8–2.0 cm. Transport drift ‖x⁺ₖ − x⁺₀‖ is reported as canonical deviation, not as exact material-point error.

Outputs are in `/media/…/RGBDekf/forensics_F1_inner_drift_20261030/<scene>/forensics/`, 36–92 MB per scene. Each folder contains:
- `README.md` with the exact commands;
- `selected_point_ids.csv`, `population_summary.json`, `population_where_when.json`;
- `f1_state_timeline.csv` (every frame of every ID) and `f1_state_full_covariance.npz` (full 6×6 P⁻/P⁺);
- `f1_inner_iterations.jsonl`, `depth_samples_first_events.jsonl` (raw 2×2 rows at prediction, accepted state and physical reference, with ground-truth surface identity), `f1_row_admission.csv`;
- `f1_update_oracle.csv`, `f1_channel_counterfactuals.csv`, `c4_pure_prediction.csv`, `h1_depth_at_physical_reference.csv`, `f1_moving_point_survival.csv`;
- `hypotheses.json`, `point_events.json`, `provenance.json`, `root_cause_report.md` (generated tables), `oracle_tests.txt`.

Code (new, investigation only):
- `tools/forensic_f1_drift.py` (`select`, `trace-run`, `analyze`, `where`);
- `tests/test_forensic_f1_oracles.py` (5 oracle tests, passing);
- `tools/root_cause_oracle.py`: added an independent inverse-depth branch. The default (metric) is unchanged and its tests pass.

## 3. Where and when: every ground-truth static point (saved F1 arrays)

| | USB | storage | blade |
|---|---|---|---|
| static points | 34 796 | 202 439 | 18 976 |
| final drift p50 / p90 / p99 | 13.6 / 224 / 1167 mm | 13.9 / 686 / 1940 mm | 0 / 48 / 684 mm |
| > 10 cm / > 1 cm / < 1 mm whole sequence | 17.8% / 53.4% / 34.9% | 31.4% / 54.2% / 10.3% | 7.8% / 13.8% / 84.2% |
| onset frame (drift > 1 mm) of the > 10 cm drifters, p10 / p50 / p90 | 1 / 2 / 7 | 1 / 1 / 3 | 1 / 1 / 5 |
| > 10 cm drifters whose onset precedes any row loss | 5494 / 6202 (89%) | 61 805 / 63 520 (97%) | 1379 / 1477 (93%) |
| integrated-velocity share of the departure (p50) | 0.945 | 0.977 | 0.953 |
| first accepted correction of > 10 cm drifters, p50 | 2.5 mm → 0.12 m/s | 8.2 mm → 0.41 m/s | 3.5 mm → 0.16 m/s |
| Δv/Δx at that update, p50 | 58.4 s⁻¹ | 58.4 s⁻¹ | 58.2 s⁻¹ (1/dt = 59) |
| share of final departure that is the pure prediction of the first measured velocity, p50 | 0.17 | 0.87 | 0.64 |

**When.** Drift starts at the first measured frame, frame 1 for most points, in all three scenes.

**Where, by distance to the moving part.** Drift does not depend on proximity to the moving part.

| Distance to moving part | 2h | 4h | 8h | 16h | 32h |
|---|---|---|---|---|---|
| USB, >10 cm | 22% | 14% | 16% | 23% | – |
| storage, >10 cm | 34% | 31% | 33% | 34% | 30% |

**Where, by cameras measuring at frame 1.** The more cameras measure a static point at frame 1, the more it drifts:

| Rows at frame 1 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| USB, >10 cm | 12% | 16% | 28% | 49% | 100% |
| storage, >10 cm | 20% | 33% | 42% | 47% | – |
| blade, >10 cm | 3.5% | 8% | 55% | 78% | – |

Measurement is the cause. **Stable points are mostly unmeasured, or measured only by their own source camera.** That camera sees the identical pixel in a static, noiseless render, so its residual is exactly 0. At their onset frame, drifting points have VALID rows in 19–22% of camera slots, against 5–6% for stable points at the same frames.

**The rows actually used at frame 1, evaluated at the static position x0** (`rows_used_at_frame1_evaluated_at_x0`):

| | USB | storage | blade |
|---|---|---|---|
| drifters > 10 cm with a used frame-1 row at \|z_D\| > 3 | 36.5% | 4.6% | 50.8% |
| median max \|z_D\| of their used rows | 1.19 | 0.70 | 4.06 |
| stable points with such a row | 0.18% | 0.27% | 0.85% |
| P(> 10 cm) when max \|z_D\| ≤ 3 / 3–10 / 10–100 / > 100 | 15 / 68 / 86 / 96% | 36 / 55 / 42 / 69–82% | 7.5 / 41 / 58 / 81–100% |

So there are **two initiator classes**:
- **(a) Consistent rows.** These are the majority: 63% of USB drifters, 95% of storage, 49% of blade.
- **(b) Grossly inconsistent rows.** Points with such rows drift almost surely, at 55–100% depending on the scene.

## 4. The causal chain, proven on individual points (spec section 11)

Each block follows the spec's stop rule: frame k, the decisive inner iteration j, camera c, cell (u, v), the four depth samples, r_D, H_D, R_D, S_DD, NIS and gate, the accepted Δx and Δv, oracle agreement, the next prediction integrating v⁺, and the later observation trace. Full records are in `depth_samples_first_events.jsonl` / `f1_inner_iterations.jsonl`, and the generator is `stop_rule` on the forensics outputs.

### 4.1 Class (a): consistent rows, motion along the surface

**storage 8444** (drifts 1.31 m)
- **Frame 1, prior:** x⁻ = x0, v⁻ = 0, √diag P⁻_xx = 17.0 mm, P⁻_xv = 0.01695 I.
- **Rows (all VALID):**
  - camera 0 is the source camera: (282.000, 237.000), r = 0 exactly;
  - camera 1, cell (231, 232): depths 3.75579 / 3.74813 / 3.75656 / 3.74889 m, r_D = +6.0 µm = 1.05σ, √S_DD = 28.4 mm, NIS 0.44, colour r/σ = (1.63, 1.21, 1.43), |∇I| = 0.033/px;
  - camera 7, cell (363, 236): r_D = −4.2 µm = −0.74σ, colour (1.31, 1.18, 0.94).
- **Iterations:** 10 inner iterations, crossing 7–8 pixel cells per iteration; largest step at j = 5 (17.3 mm). Accepted final iterate with J = 2.93 ≤ 16.25.
- **Accepted update:** Δx = (0.003, 23.617, 5.073) mm, Δv = (0.0004, 1.3795, 0.2963) m/s, |Δv|/|Δx| = 58.4, P_vx P_xx⁻¹ ≈ 58.4 I.
- **Channels:** depth contributes 0.015 mm and colour 24.14 mm (same-A split). A colour-only re-solve gives 15.2 mm; a depth-only re-solve gives 0.004 mm.
- **Oracle:** |s⁺ − production| = 1.7e-14 m.
- **Next prediction:** x⁻₂ − x⁺₁ = (0.008, 23.381, 5.023) mm = dt·v⁺₁.
- **Afterwards:**
  - Drift is 50 mm at frame 2 (camera 1 only), then 70, 93 and 116 mm through frames 3–5 (prediction-only at 4–5).
  - It reaches 230 mm by frame 10 with occasional single-camera rows that no longer correct the velocity, and 1.31 m at the end.
- **Verdict:** colour residuals of 1–1.6σ (view-dependent shading, inside the declared colour noise) are explained by a 1-pixel slide along the surface, which a 17 mm prior permits. This is class (a), driven by colour.

**storage 49199**
- **Frame 1:** rows from camera 1 (source, r = 0) and cameras 2 and 3, all with |z_D| ≤ 0.86 and colour residuals ≤ 0.8σ.
- **Accepted update:** 10 inner iterations, non-converged; Δx = 7.9 mm, Δv = 0.46 m/s (ratio 58.7). Colour contributes 8.1 mm (same-A) and 8.3 mm (colour-only re-solve).
- **Oracle:** 1.4e-11 m.
- **Afterwards:** drift reaches 69 mm by frame 13 and 124 mm at the end.

**blade 10251**
- **Frame 2, prior:** x is measured at frame 1, so P⁻_xx = (1.3, 33.9, 34.0) mm.
- **Rows:** depth only, with no colour texture (|∇I| = 0) and r_D = 0.0 to 0.81σ.
- **Accepted update:** 10 non-converged inner iterations crossing 3–6 cells each. The last iterate is accepted with Δx = 12.6 mm along the surface (y), Δv = 0.37 m/s (ratio 29.5 = 1/(2dt)).
- **Counterfactuals:** a depth-only re-solve gives 12.6 mm; with inverse depth on the same rows it is 1.4 mm. The tangential "information" here is the bilinear interpolant's cell-gradient artefact, not geometry.
- **Oracle:** 6.5e-9 m (non-converged path, section 7).

**USB 51975**
- **Frame 2, rows (cameras 3, 4, 5):** |z_D| ≤ 0.13; colour residuals of −1.9σ on G and B in cameras 3 and 5. USB's "texture" is 8-bit quantization: |∇I| = 0 or 0.0039 = 1/255 per px.
- **Accepted update:** 10 non-converged iterations, crossing 7–8 cells each. Δx = 1.6 mm, Δv = 0.046 m/s.
- **Afterwards:** 9 mm at frame 3, then 33 mm at frame 5 (a rejected frame-4 update in between), 0.37 m at the end.
- **Its pixel neighbour 51974:** measured only by its own source camera 4, so r = 0, and it never moves.

### 4.2 Class (b): crease, edge or occluder rows admitted by the gate

**USB 67827**
- **Frame 1, camera 4:** (358.567, 280.511), cell (358, 280).
  - The depth column at u = 358 changes by 39, 38, 22 and 7 mm per pixel: a **crease** (second difference 15 mm). The cell depths are 3.65247 / 3.65247 / 3.63029 / 3.63029 m, spread 22.2 mm, which is under the 50.5 mm allowance.
  - D = 3.641133 m against Z = 3.636679 m: r_D = −4.45 mm = −80σ. Inverse depth gives 3.641099 m, essentially the same.
  - √S_DD = 61.6 mm, so NIS = 0.76 ≤ 16.25 and the row is VALID.
- **Camera 6:** a diagonal crease or silhouette (spread 51.1 mm, under the 52.5 mm allowance), r_D = −2.28 mm = −41σ, √S_DD = 120 mm.
- **Accepted update:** Δx = 5.15 mm (depth 4.97 mm same-A), Δv = 0.302 m/s (ratio 58.7). Next prediction moves by dt·v⁺ = 5.1 mm.
- **Oracle:** 4e-13 m.
- **Afterwards:** 31 mm at frame 2 and 104 mm at frame 6, then 0.90 m at the end.

**USB 30502**
- **First measured at frame 3** (frames 1–2 had no rows: all pixel-suppressed or masked), so the prior σ is 51 mm.
- **Camera 1:** cell (446, 275) on a crease (39 → 38 → 19 → 9 mm/px), r_D = −6.41 mm = −116σ, √S_DD = 233 mm, NIS 4.87, VALID.
- **Camera 3:** crease, −3.31 mm = −59σ.
- **Accepted update:** Δx = 14.6 mm, Δv = 0.287 m/s = Δx/(3dt). At the accepted state camera 1 samples the far face with r_D = 0.
- **Its neighbour 30503:** measured only by its source camera 2, so r = 0, and it stays within 10 nm.

**storage 262894**
- **Frame 1:** cameras 3 and 4 see a surface **67 mm and 31 mm in front** of the point, an occluder: r_D = +67.2 mm = +11 733σ and +30.9 mm = +5 398σ.
  - Both still pass, with NIS 11.8 and 6.1 under 16.25, because √S_DD = 28.9 and 17.0 mm.
  - The "measured surface in front" test needs ν_D > 3√S_DD ≈ 87 / 51 mm.
- **What happens:** they steer the iterations and are then rejected at the accepted state. The accepted 27.2 mm step (Δv = 1.59 m/s) is driven by colour residuals of 3.7–4σ in cameras 5 and 7.
- **Afterwards:** prediction-only from frame 6, 0.37 m by frame 13, 1.7 m at the end.

**blade 14104**
- **Frame 1, camera 0:** a surface 30 mm in front (r_D = +30.2 mm = 2624σ, √S_DD = 73 mm, NIS 9.0); camera 4 is on a crease (−47σ).
- **Accepted update:** 10 non-converged iterations crossing 7 cells each. An earlier consistent iterate is accepted (J = 13.15 ≤ 16.25): Δx = 30.6 mm, Δv = 1.79 m/s (ratio 58.6).
- **Afterwards:** **no row ever again.** The point coasts at 30 mm per frame and ends 1.72 m away.

**Never measured until late (USB 34391, storage 317931).** The first rows arrive at frames 34 and 31. By then the prior σ along the correction is 526–577 mm, and the IEKF jumps 1.14–1.35 m onto a surface 1.15 m away.

### 4.3 The amplifier, exactly

At the first measured frame, P⁻ = F^k P0 F^kᵀ + ΣQ, so P⁻_xx ≈ (k·dt)²·P_vv0 + P0 and P⁻_vx ≈ k·dt·P_vv0. For position-only rows the Kalman velocity correction is exactly P_vx P_xx⁻¹ Δx (oracle test 3, identity to 1e-15). With P0 ≪ (k dt)² P_vv0 this is Δx/(k dt). Measured values:
- the gain matrix is 58.4–59.0 I at k = 1 and 29.4 I at k = 2 (blade 10251's first measured axis gives 59.3, because it was measured at frame 1);
- |Δv|/|Δx| at every onset in the three scenes is 58.2–58.9, 29.4–29.5 and 19.6–19.7.

The drift identity x⁺_T − x⁺_0 = Σ dt v⁺_{k−1} + Σ (x⁺_k − x⁻_k) closes to 2e-15 m on every traced point.

## 5. Hypotheses

| | verdict | evidence |
|---|---|---|
| **H1** interpolation bias at the stationary position | **DISPROVED as a bias**; **PARTIAL as an interpolant-gradient artefact** | **Bias.** Exact slanted plane: metric bilinear bias ≤ 0.66 µm (oracle 1), < 0.06 √R_D on USB and < 0.12 on storage. At x0, metric and inverse residuals are equal to 3 digits on all class-(b) rows, i.e. creases, which no single-plane interpolant fixes. **Gradient artefact.** In consistent-row onsets with long gaps (USB 21772, 88465, 44724; blade 10251), the inverse-depth re-solve on the same rows shrinks the correction 2–10×: 7.85 → 0.76, 18.5 → 3.3, 40 → 6.6, 12.6 → 1.4 mm. The bilinear cell gradients are what supply the spurious tangential information. Storage onsets are unchanged by inverse depth. |
| **H2** wrong-surface / tangentially ambiguous association | **CONFIRMED for class (b); tangential ambiguity CONFIRMED for class (a)** | **Class (b).** Creases (second difference 15–41 mm in the 2×2 neighbourhood), silhouettes and occluders (+30 to +67 mm in front) are admitted at frame 1. **Identity.** At part level the ground-truth label of all 4 pixels is STATIC_PART. Face- or material-level identity is `identity_unproven`: the labelled cloud has 2 cm spacing; the geometric kink is the proof used here. **Tangential ambiguity.** Depth information along the normal is 4e4–4e5 m⁻¹; along the surface it is 0–300 m⁻¹ (depth) and 0–1500 m⁻¹ (colour). Δx is mostly tangential. |
| **H3** prediction covariance widens admission; a precise depth row then drives the update | **CONFIRMED for class (b)** | √S_DD of admitted crease and occluder rows is 17–255 mm at the onset frames (1–3) and up to 665 mm late, against √R_D = 5.7–56 µm. Rows at 40–11 733σ_R pass NIS ≤ 16.25 because H P⁻ Hᵀ ≫ R. |
| **H4** position–velocity coupling turns a correction into persistent motion | **CONFIRMED (amplifier)** | Section 4.3. Velocity-integration share is 0.945 / 0.977 / 0.953. The pure prediction of the first velocity explains 17% / 87% / 64% of the final departure (p50). Drift grows linearly through prediction-only frames (14104: 30 mm/frame for 58 frames). |
| **H5** standard F1 acceptance / iteration | **PARTIAL: behaviour, not a contract violation** | The rule is a χ² adequacy test of the iterate, documented in `point_kalman_cuda.py` lines 18–30, not a decrease test. Among decisive events, 22/95 (USB) and 46/96 (storage) accepted states are non-converged 10-iteration iterates. 3 genuine storage cases accept a state with higher fixed-support cost than the prediction (67801 frame 2: 8.3 → 19.4; 102354 frame 3: 8.2 → 14.2; 8418). Recheck of self-occlusion is applied; no accepted row is invalid at the accepted state; P⁺ uses the accepted rows; no covariance shrinkage without accepted rows (oracle 4). Oracle 5 (ray-cast fixture): 0 of 61 accepted updates raised the cost, so the behaviour appears on the hard scenes only. |
| **H6** moving-point loss | **CONFIRMED: support collapse, not deletion** | Section 6. |
| **H7** numerical bug / data / provenance | **DISPROVED as an arithmetic bug**; **two numerical sensitivities found** (section 7) | Prediction identity 4e-16; Joseph vs information 1.8e-9; drift identity 2e-15; dt = 1/59 s constant (timing.json fps). Depth is Z, not range: corr(r_D, range − Z) = −0.16 / −0.30. Every acceptance decision is identical between production and oracle. |

### Counterfactuals (offline algebra on saved frames; never published states)

**C0, reproduce the frame.** 132/137 (USB), 133/140 (storage) and 135/136 (blade) decisive frame updates reproduce to ≤ 1e-9 m. Every acceptance decision matches. The 13 others are explained in section 7: non-converged expanding iterations (up to 3e-4 m on static points) and source-pixel knife edges (up to 0.79 m on moving points 8416–8418).

**C1, metric vs inverse depth (same rows).**
- Removes most of the correction only for consistent-row, long-gap depth onsets (H1, gradient artefact).
- Unchanged for class (b) and for the colour-driven storage onsets.
- Sometimes larger: storage 262894 goes 27 → 33 mm.

**C2, re-solve with one channel only.**
- *Storage:* colour alone reproduces most onsets (4–17 mm); depth alone gives ≤ 2 mm.
- *USB and blade:* mixed. Class (b) is depth-driven; quantized-colour cases are colour-driven (USB 51975: colour-only 9.9 mm, depth-only 0.7 mm).

**C3, remove one row.**
- No row was ground-truth-proven to image the other part (0/137, 1/140, 0 events), so no oracle-audit exclusion was possible at part level.
- Removing the largest-|z_D| row: USB 45971 goes 5.2 → 0 mm; 67827 goes 5.2 → 0.9 mm; others unchanged, or lose all rows.

**C4, pure prediction.** See the "pure prediction" row of the section 3 table and `c4_pure_prediction.csv`.

**C5, rows at prediction vs accepted state.** Pixel cells change for most cameras (`f1_row_admission.csv: pixel_support_changed`), and a crease row at the prediction often samples the far face at the accepted state (USB 30502 camera 1: −6.41 mm → 0). The accepted state is therefore consistent with a different surface than the one the gate tested.

## 6. Moving-part points

| | USB | storage | blade |
|---|---|---|---|
| moving points (grounded) | 22 037 | 21 689 | 1 396 |
| final GT error p50 / within 1 cm / within 5 cm | 0.74 m / 0.14% / 4.0% | 0.30 m / 10.6% / 24.0% | 0.84 m / 1.7% / 4.2% |
| first row loss p10 / p50 / p90 (frame) | 2 / 3 / 8 | 2 / 4 / 10 | 2 / 4 / 11 |
| never lose rows | 5.0% | 1.4% | 40.8% |

USB, frame by frame (`f1_moving_point_survival.csv`):

| Frame | With rows | Prediction-only | Rows but >5 cm from GT |
|---|---|---|---|
| 1 | 60% | 40% | 24 |
| 6 | 33% | 67% | 4292 |
| 50 | 9% | 91% | 1747 |

- **Why rows are lost.** In nearly every point (98%), at least one camera is suppressed as a duplicate pixel (`per-frame` front-most ownership); 86% have a self-occluded camera.
- **Not offscreen, not deleted.** At most 7 of 22 037 points are fully offscreen, and none are deleted.
- **What a point does without rows.** It coasts with the velocity its first one or two updates gave it:
  - USB 87290: 1.87 m/s from frames 1–2, then 0 rows → 0.32 m error by frame 8;
  - USB 85696: v ≈ 0 while the part moves 10 mm/frame → error grows by 10 mm/frame;
  - USB 26556: picks up 0.066 m/s at frame 1, then gets no further accepted rows (rejected or none) while the part moves away; error reaches 0.11 m by frame 8.
- **"Tracked" is mostly coincidental.** The selected "tracked" points are within 1 cm only at the last frame; mid-sequence they are 40–100 mm off.

The moving part fails through the same two weaknesses as the static part (unobservable tangential motion and a 17 mm prior), plus a much larger true velocity. Ownership and self-occlusion then remove the rows that could correct it.

## 7. Numerical sensitivities (not arithmetic bugs)

1. **Expanding, non-converged inner iterations.** Where production and the oracle disagree by more than 1e-9 m, they start identical at iteration 0 and the gap grows by about 10–40× per iteration:
   - USB 44724, frame 31: 2e-16 → 2e-12 → 1e-9 → 1e-5 → 1.5e-4 m, on the same cells and rows;
   - USB 45971, frame 4: up to 2.7e-7 m.

   The iteration map is locally expanding: the IEKF oscillates between bilinear cells and never meets its tolerance. The accepted non-converged state is therefore determined by rounding. This applies to 22–46% of decisive accepted updates.
2. **Knife-edge Jacobian at the source pixel.**
   - x0 is the back-projection of an integer pixel centre, so in its own source camera every point projects *exactly* onto integer (u, v), on the border between four bilinear cells.
   - Which one-sided gradient F1 uses there depends on the last rounding bit. Production gets v = 237.0 and the NumPy oracle gets v = 236.99999999999997 (storage 8416).
   - For static points the residual there is exactly 0, so the row adds (arbitrary) information but no push.
   - For moving points 8416–8418 (frame 1), the arbitrary gradient plus the expanding iteration gives accepted states 0.03–0.79 m apart between two correct float64 evaluations.
   - The spec's G4.1 warns against exactly this ("no arbitrary selection of the gradient at a nondifferentiable point").

## 8. Initiator / amplifier / association / consequence

- **Initiator:**
  - Class (a): noise-level residuals (view-dependent colour within the declared model variance, 8-bit colour quantization steps, bilinear depth cell gradients) along directions with almost no data information. Combined with a frame-k position prior σ ≈ k·dt·√P_vv0 (17 mm at frame 1), this produces accepted 1–30 mm moves along the surface (0.01–0.3 prior-σ).
  - Class (b): 2×2 support across creases, edges and occluders. It is admitted by the same-surface allowance (n·√2·σ_D + z·tan 80°·√2/f ≈ 44–62 mm) and by a prior-dominated row gate.
- **Amplifier:** Δv = P_vx P_xx⁻¹ Δx = Δx/(k dt) (P_vv0 = 1 m²/s², uncalibrated documented guess).
- **Association failure:** sliding along the surface keeps depth consistent, so the motion is never contradicted. Pixel ownership and self-occlusion later remove rows. Crease, edge and occluder rows pass the gate.
- **Consequence:** static points depart at about 1 mm/frame per mm of initial correction (13.6 / 13.9 / 0 mm median, 7.8–31% beyond 10 cm). Moving points lose support by frame 3–4 and coast or re-attach to wrong surfaces. Points are never deleted.

## 9. Two candidate corrections (evidence-backed, NOT implemented, for your review)

1. **Calibrate the velocity prior instead of P_vv0 = 1 m²/s² (and q_c).**
   - **Why:** this one number sets both the permission to slide (σ_x at frame k = k·dt·σ_v) and the gain (1/(k dt)) that converts every accepted correction into permanent motion. Every traced onset in three scenes shows exactly that gain, and onsets are only 0.01–0.3 prior-σ long.
   - **Rules:** per the earlier decisions, the value must come from you (`--velocity-prior-speed99`) or from identifiable data (spec P1.7), never be picked silently.
   - **Limits:** it does not remove class (b), and it trades directly against tracking the real moving part. Storage's drawer needs about 0.5 m/s.
2. **Refuse rows whose 2×2 support is not one surface, and test occlusion against the measured depth.**
   - **What exists now:**
     - the same-surface rule only checks spread ≤ allowance, and the 80° term makes that allowance about 5 cm, so it admits creases and edges;
     - the row gate tests the innovation against S, which is dominated by the prior, so a surface 30–67 mm in front passes at frame 1;
     - point-cloud visibility knows nothing about the measured occluder.
   - **Proposed checks:**
     - test the planarity of the 2×2 or 3×3 depth support (second differences);
     - compare the measured depth with the predicted Z using √R_D-scale tolerances for occlusion, instead of the prior-inflated S.
   - **Evidence:** class (b) explains 37% (USB), 5% (storage) and 51% (blade) of >10 cm drifters. Their drift probability is 55–100%, against 7.5–36% for consistent rows.

Not proposed as fixes, but to keep in view: the non-converged expanding iterations and the knife-edge source-pixel Jacobian (section 7). Both make accepted states depend on rounding.

## 10. What remains UNPROVEN

- **Face- or material-level identity of crease rows:** the ground-truth labelled cloud (2 cm spacing) cannot separate faces of one part. The evidence is geometric: the depth kink.
- **The causal weight of class (a) vs (b) across the whole population:** this is measured at frame 1 only (`rows_used_at_frame1_evaluated_at_x0`); later onsets are covered only on the traced IDs.
- **Real scenes:** they were not part of this investigation, per the spec. Real depth noise (σ_D in mm) and real texture change the balance between the two classes.
- **Whether the candidate corrections fix the drift:** not tested. This was an investigation only; no estimator was changed.
