# SVGD Stage 1 — decisive falsification and causal root cause (2026-11-03)

Spec: `CODEX_RGBDiekf_SVGD_Falsification_And_Root_Cause_2026-11-03.md`. Tool: `tools/audit_svgd_falsification.py` (post hoc, separate process; GT never in
inference), synthetic runner `tools/svgd_synthetic_run.py`, contract tests `tests/test_svgd_falsification.py`. Nothing in inference, configs, priors or
covariances was changed; DEV-0012/0013 and A1–A7 are NOT implemented or authorized.

## Answer

**The motion is not recovered because three components of the current model fail, in a case-dependent order; no single optimizer fix can cure it.**

| case (full masked clouds, all cameras) | E_obs prefers GT? | E_total* (V minimized) prefers GT? | along I(α) | proposal | classification |
|---|---|---|---|---|---|
| Blade, 2 frames (8 cams, N = 34 369) | yes, weakly: −84 | **no: +1038** (per-point velocity prior +1122) | E_total* minimum near α ≈ 0.05–0.1 | A/B rejected; D accepted (0.65 mm, cos GT 0.999) | `MOTION_PRIOR_OR_RELATIVE_CALIBRATION_FAILURE` |
| Blade, 10 frames | yes, every frame: −84 … −970, Σ −5763 | **yes: −4641** | monotone to α = 1, no barrier | A/B rejected; C/D only 0.2 µm steps | `KDE_OR_FEATURE_DIRECTION_MISMATCH` |
| synthetic textured plate translation, 10 f (3 cams) | yes at α = 1: −1684 | yes: −1682 | **barrier +3.6e7 at α = 0.5** | Φ points to GT (cos 0.998) but is an ASCENT direction (+1.6e7) | `KDE_OR_FEATURE_DIRECTION_MISMATCH` |
| synthetic hinge rotation, 10 f (0.9 rad) | **no: +2.96e8** (frames ≥ 5) | no | monotone increase | accepted µm steps | `OBSERVATION_DISCRIMINATION_FAILURE` |

The components, in causal order:

1. **Observation model E_obs** (finite-RFF feature mean, V_t = all with weights 1/N, frozen i.i.d. Γ_t = C_Q/M_t + C_P/N; DEV-0001/0002/0005).
   (a) Under rotation the cameras sample the turning plate less and less (observed plate samples 242 → 214 → … → 36 → 22 at frame 9) while h_t keeps all 242
   canonical plate points at weight 1/N: the GT trajectory is charged for mass that is no longer observed (+2.94e8 at frame 9). This is the missing
   visibility / sampling-density law (PDF 8.2–8.4), not an optimizer issue.
   (b) For textured tangential motion the likelihood has a narrow global minimum at GT behind an enormous barrier (+3.6e7 at α = 0.5): Γ ≈ C/M is orders of
   magnitude smaller than the systematic feature mismatch of every intermediate configuration, so acceptance against E_obs blocks any continuous path.
   (c) Where it is well behaved (uniform-colour Blade) its per-frame preference is weak (−84 nats at 14 mm): the blade slides along its own 1.46 m length,
   95 % of the frame-1 samples coincide with the frame-0 cloud (aperture problem), and only the ends carry motion information.
2. **Temporal prior E_time** (repository noise file: σ_v = 1 m/s per axis, q_c = 0.01 m²/s³; measured source covariance).
   The prior is per point and independent: the coherent 0.835 m/s blade motion costs ½|v|²/σ_v² ≈ 0.35 nat for each of 3218 points = **+1122 nats** (all in the
   frame-0 velocity block; process +0.06). Over 2 frames this exceeds the observation gain 13×; over 10 frames the data win. A shared rigid motion would pay
   this once (≈ 1 nat) — the missing coupled-rigidity factor (audit F1) is quantitatively the difference.
   The frame-0 position anchor uses the measured depth σ = q/√12 = 68.8 nm (float32 quantum 2.384e-7 m; verified equal), so any proposal that moves frame 0
   costs 2.8e9 nats: this alone rejects every original step (arms A/B).
3. **Proposal (restricted physical-particle Stein field)**: it is not a descent direction of the objective it is accepted against.
   cos(Φ, −∇E_obs) on moving points ≤ 0.27 (Blade) and down to −0.19 (synthetic translation), ⟨∇E,P⟩ > 0 on the synthetic translation; on the Blade the KDE
   responsibility of every stationary blade point peaks on the observed sample at its OWN old position (peak-sample distance to the true position = the full
   GT displacement: 14.16, 28.31, …, 127.40 mm), so the per-frame field does not grow with displacement (component along the true axis 8.75, 2.9, 1.1, 3.8,
   −0.5, −0.2, 2.8, 1.3, 1.0 vs the required ∝ t), and its temporal shape is inconsistent with the white-acceleration prior (E_time curvature along D 3.4e13):
   even with frame 0 held and exact velocities (arm D) only a 0.2 µm step is accepted on 10 frames.

Gradients are numerically right (FD closure 1.6e-9 … 7.6e-9 on Blade, ≤ 1.6e-8 synthetic): the failure is the model and the proposal, not arithmetic.

## TF report (spec section 13)

```text
CHECKOUT_COMMIT / UNCOMMITTED_CHANGES / FILES_EDITED
  b62c7ceeb300c852a58f34dd748d5a1ce26d987a; working tree = the earlier audit-response changes (svgd_* modules, tests, reports) + 16 root spec .md files
  deleted by the user (untouched). NEW this pass: tools/audit_svgd_falsification.py, tools/svgd_synthetic_run.py, tests/test_svgd_falsification.py,
  reports/svgd/falsification_root_cause_2026-11-03.md (+ display-only viewer files, see VISUAL_QA). No inference/config/prior/covariance file edited.
ORACLE_VERIFICATION
  Model rebuild VALID on all 4 runs (10 checks: config SHA-256, noise re-estimate, cloud hashes, canonical, Lambda_X, H, RFF hash, Gamma log det,
  E(stationary) == timeline init bit for bit, inference sources unchanged). Logged line search reproduced exactly (A/B max rel. error 0.0).
  GT: GT_TRAJECTORY_VERIFIED (Blade 3218 moving / 31151 static; synthetic 242 / 1086); X_GT[0] == X0 bitwise. Criterion: on observed samples unexplained by
  X0 in joint XYZ-RGB (d^2 = |dx|^2/s^2 + dc^T Lambda_C^-1 dc <= chi2_6(.99)), the GT scale lam = 1 explains as many as the best lam in [-1.5, 1.5]
  (within 2 binomial SE) and more than lam = 0. (Plain XYZ NN distances cannot verify tangential sliding: the stationary blade is 0.08 mm from the frame-1
  cloud, the true one 2.1 mm, because pixel-identical re-sampling of the slid surface favours "stationary" — itself a finding.)
  Blade label association (labelled_endpoints): median 18 mm (moving) / 17 mm (static) — sparse label cloud, reported as a limitation.
MODEL_PARAMETERS_AND_UNCERTAINTY_PROVENANCE (covariance_provenance.json per run)
  Blade: depth variance 4.737e-15 m^2 = q^2/12 (measured, float32 floor of a noiseless render), source sigma eigen median [6.9e-8, 1.97e-3, 1.97e-3] m,
  RGB measured, q_c = 0.01 I m^2/s^3 and P_vv = 1 I m^2/s^2 (repository noise file, documented-initial-guess), Lambda_X spacing 2.6 mm, H_X^1/2 = 118 mm,
  B = H, D = 1024, seed 20261103, Gamma i.i.d. (DEV-0002..0005 PROPOSED_NOT_AUTHORIZED). Source uncertainty enters once (P_0). Frame-0 E_obs is in-sample.
E_obs(GT)-E_obs(STATIONARY)  Blade-2f: [0, -84.35]; Blade-10f: [0, -84.35, -278.8, -467.6, -603.7, -722.5, -852.2, -853.0, -931.7, -969.6] sum -5763.4;
  synthetic translation: sum -1684 (frames 4, 7 positive: +151.8, +10.5); synthetic rotation: [0, -306, -294, -115, -176, +1030, +2880, +6979, +1.72e6, +2.94e8].
E_time^*(GT)-E_time^*(STATIONARY)  Blade 2f/10f: +1122.3 (anchor vv +1122.3, anchor xx 0, process +0.06 / +0.05); synthetic translation +2.72;
  rotation +154.5. KKT residual of the exact velocity minimizer <= 5.5e-15.
E_total^*(GT)-E_total^*(STATIONARY)  Blade-2f +1038.0; Blade-10f -4641.1; synthetic translation -1681.6; rotation +2.96e8.
E_obs and E_total^* versus alpha  (energy_oracle.csv; plots on the viewer numerical page)
  Blade-10f E_total*: 0 / -109 / -517 / -971 / -2082 / -3513 / -4593 / -4641 / -762 for alpha 0 .01 .05 .1 .25 .5 .75 1 1.25 (no barrier).
  Blade-2f E_total*: minimum -5.55 near alpha 0.05, +1038 at alpha 1. Synthetic translation E_obs: +3.8e3 / +1.2e5 / +8.0e5 / +1.1e7 / +3.6e7 / +1.2e7 / -1.7e3 / +5.9e7 (barrier).
  Diagnostic observation models (DIFFERENT models, same positions, Blade-10f): B^1/2 = 17.6 mm -4840; 2.6 mm D=1024 -265, D=4096 -1021 (finite-D error
  large at small B); production seed+1 -5830; production D=4096 NUMERICAL_FAILURE (Gamma not SPD). No scale changes the Blade-2f verdict (best -141 vs +1122).
ORIGINAL_STEIN_GRADIENT_FD_CLOSURE  central differences on a 13-step h ladder: <grad E_obs, Phi> 5.9e-11 (2f) / 1.6e-9 (10f); <grad E_total, P> 7.5e-10 / 7.6e-9.
COUNTERFACTUAL_A/B/C/D (line_search_counterfactual.csv; offline only)
  Blade-2f: A, B never accepted (E_time curvature 3.1e16, anchor); C accepted at eta 2.3e-9 (0 mm); D accepted at eta 7.4e-5: dE_obs -5.91, anchor +2.91
  (velocity block), process +1.6e-4, moving 0.65 mm, static 0.012 mm, cos(step, GT) 0.999 — physically correct but the objective's minimum along the path is
  at ~7 % of the true motion.
  Blade-10f: A, B never accepted; C at 1.8e-8 (0.1 µm), D at 3.7e-8 (0.2 µm; curvature 3.4e13) — not physically meaningful (< 1e-3 of 127 mm).
  Synthetic translation: no arm has a descent direction (<grad E, P> = +1.6e7). Rotation: all arms accept 1.5 µm steps along GT (cos 0.997).
KDE_BASIN_AND_SOURCE_CROSS_SUPPORT_AUDIT (stein_field_audit.{csv,json})
  Blade-10f: cos(Phi, GT) moving 0.999, 0.876, 0.720, 0.786, -0.182, -0.099, 0.597, 0.426, 0.325; KDE log p of moving points at stationary 19.8 -> 7.6 vs at
  GT ~21 every frame; peak-responsibility sample = the point's own old sample (aperture lock); static points dominated by moving sources 15.6–17.5 %;
  moving receivers take 21.8 % of kernel mass from static sources (static receivers 0.01 %); the colour factor scales the median kernel mass by 0.47
  (colour reduces, but does not remove, the cross-support coupling).
ROOT_CAUSE_CLASSIFICATION  see the table above: observation model (visibility/sampling law, Gamma scale) for rotation and textured sliding; per-point velocity
  prior + 68.8 nm frame-0 anchor for short horizons; Stein KDE proposal misaligned with E (aperture lock, saturating per-frame field) for the Blade.
SYNTHETIC_NOISE_AND_REAL_NOISE_SEMANTICS  same equations and the same measured-noise routine (temporal_sensor_only) for both; synthetic fixture declared
  depth noise 0.5 mm (measured by the same routine), Blade a noiseless render (measured = quantization floor). No noise-free branch, no floors, no inflation.
TESTS_AND_INVARIANCES  tests/test_svgd_falsification.py: 8/8 PASS (1170 s, CPU): rebuild + archived line-search parity, frame-0 identity, E_time
  decomposition closure incl. xv cross terms + KKT, FD closure, C/D leave X0 bitwise and zero the position anchor, camera-order invariance (energy-scaled
  tolerance), GT-leak AST sentinel + artifact immutability (no p_dynamic.npy), injected bugs (Gamma normalizer dropped, seed changed, lift sign flipped) caught.
VISUAL_QA / PERFORMANCE  display-only viewer debug/svgd_falsification_debug.py (+ assets svgd_falsification_{3d,numerical}.html), QA
  tools/qa_svgd_falsification_debug.py: real Blade-10f audit PASS 68/68 checks (viewer at RUN/posthoc/svgd_falsification_1/viewer/, N_display 20000 of
  34369, 600 Stein vectors, screenshots + sha256 in viewer/qa/); synthetic PASS 71/71; tests/test_svgd_falsification_visual.py 6/6, test_svgd_visual_debug.py
  9/9 still pass. The viewer writes only inside viewer/ (hash-checked). Audit wall time: Blade-2f 510 s, Blade-10f 771 s (with diagnostic feature models).
DEVIATIONS_PROPOSED_NOT_AUTHORIZED  none new. Counterfactual arms C/D and diagnostic feature models evaluated offline only (theory_deviation_audit.md per run).
EXACT_NEXT_DECISION_REQUIRED  the observation model must be revised on paper before any further optimizer work: (1) a visibility / sampling-density law
  for h_t (camera-stratified, footprint-weighted) and a calibrated Gamma (A2/A5) — rotation fails without it; (2) a shared rigid-support motion prior (A4) —
  the per-point velocity prior charges coherent motion N times; (3) only then a proposal consistent with that objective (the KDE Stein field is not).
```

Artifacts per run: `RUN/posthoc/svgd_falsification_1/` (`preflight.json`, `energy_oracle.csv`, `line_search_counterfactual.csv`, `stein_field_audit.{csv,json}`,
`covariance_provenance.json`, `diagnosis.md`, `theory_deviation_audit.md`, `falsification_summary.json`, `classification_final.json`, `oracle_trajectories.npz`)
for `svgd_blade_stein_audit_none_20261103_{1808,1814}` and `svgd_falsification_synthetic_{translate,rotate}_10f_20261103_2019`. The folders
`RUN/posthoc/svgd_falsification/` next to them are SUPERSEDED intermediate outputs (earlier verification/acceptance criteria); a removal was blocked by a
safety check and is left to the user.
