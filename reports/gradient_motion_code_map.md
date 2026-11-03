# Gradient-motion code map (Gate 0) — local working tree, 2026-11-02

**Baseline.**
- HEAD `9465cb1`. The local-correlated and material-following work is uncommitted (`git status`: 7 modified, 30 untracked files).
- The focused suite passes **129/129** before any change in this task: 0 failures, 0 skips, no sources changed during the run. Saved at `…/scratchpad/gate0_baseline_tests.json`.
- Source hashes are SHA-256 prefixes at Gate 0.

## Existing functions the spec's equations map onto

| Spec | Local code (file:line, hash) | What it already computes |
|---|---|---|
| Projection u = π_c(X), J_π (2×3, world), Z_c(X), J_Z = e₃ᵀR_cw | `estimator/projective_camera.py:79 CameraRig.project_rows` (`00d42232…`) | uv, Z, Jπ = Jπ_cam·R_cw, J_Z = R_cw[2,:] for fixed cameras |
| Back-projection X_c(u) = D·K⁻¹[u,v,1] | `projective_camera.py:97 backproject` | measured depth → world point |
| First-depth ownership | `estimator/temporal_zbuffer.py:27 first_hit_pixel_owner`, `:98 owners_at_gpu` (`763c3d1f…`) | nearest-integer, all points compete, no background |
| Camera-local references | `estimator/temporal_reference.py ReferenceTable`; material commit `estimator/temporal_material_terms.py:236` (`c674b66c…`) | latest accepted same-camera (frame, measured pixel p_ref, state); lag-window expiry |
| **Finite-displacement material residual** (spec §2.4 anchor convention u_t = p_ref + π(X_t) − π(X_r)) | `temporal_material_terms.py:185` (q_t), `:103 material_prediction` | f = Σw_a(y_a−η_a) − (y_ref−η_ref) − e₄ΔZ |
| H_X,t = ∇I_t(q)J_π(X_t); depth row ∇D_t(q)J_π(X_t) − J_Z(X_t) | `temporal_material_terms.py:110,112` | **identical to the spec's G_{i,c}** (§3.3) at the material anchor |
| H_X,r = −∇I_t(q)J_π(X_r); +J_Z(X_r) | `temporal_material_terms.py:111,113` | reference-state derivatives (spec §2.4); FD-tested in `tests/test_temporal_material_warp.py` |
| Velocity columns 0 (instantaneous residual, spec §3.2) | `temporal_material_terms.py:110-113` write only `[:, :, :3]` | `[H_X, 0]`; velocity enters through P⁻_vX |
| Constant-velocity F, Q(dt), signed dt | `estimator/point_kalman_cuda.py:63 transition`, `:68 process_noise` (`2f7382ca…`) | reused unchanged |
| Within-point joint Gaussian (state, reference states / source chart, raw keys) | `estimator/temporal_local_covariance.py`, `temporal_local_iekf.py:104 LocalFrameUpdate` (`c301cfb5…`), `temporal_material_iekf.py:53 MaterialFrameUpdate` (`95165544…`) | shared raw keys reused, never duplicated; Joseph posterior `temporal_local_iekf.py:464` |
| Per-camera stacking per point | `temporal_material_iekf.py:112 _system` | one stacked 4C-row system per point with Σ (all local variables) and R_warp |
| Current-sample noise | `temporal_material_iekf.py:167 identity_costs` (marginal NLL with log det) and `_system` (fresh current keys, −w_a I) | Σ w_a²R_a integrated out exactly |
| SE(3) | `estimator/se3.py` (twist [v, ω], world-left Exp, `left_jacobian`, `point_jacobian` B = [I, −[x]×]) | **uncentred** world-left; spec §4.2 needs the pivot-centred chart → new code |
| Non-GT part membership | `estimator/projective_coherence.py:26 run_coherence` (legacy free tracks only) | **none exists for temporal-rgbd runs** (all used `--stop-after free`) → Arm C `PART_MEMBERSHIP_UNAVAILABLE` |

## Double-use analysis (spec rule 6)

- The spec's per-camera matrix G_{i,c} = [∇I J_π ; ∇D J_π − J_Z] is **exactly** the material-following `H_t[:, :, :3]`. The current image gradient is that of the clean bilinear interpolant at q_t, with the same raw pixels and the same covariance.
- The differential equations (2.1–2.2) are therefore a **linearization of the same measurement**. Adding them as rows next to material-following would count each current and reference raw pixel twice.
- Consequently:
  - `--temporal-gradient-guide diagnostic` only reads the final linearization; the state is bitwise unchanged.
  - `replace` re-assembles the same residual and Jacobian through the new gradient/geometry modules, to be proven equal by a parity test.
  - Neither may be described as new sensor information.
- The only new information in the spec is **shared part motion (§4)**. It pools different points' gradients, so it needs membership, which is GT-only (oracle, offline) here.

## Current material-following assumptions (unchanged by this task)

- Fixed measured reference pixel; moving current bilinear sample; R_warp = 0 under `ideal`.
- Declared sensor noise is quantization-level.
- The measured same-instant cross-camera depth discrepancy is 1.1e-4 m (USB) and 6.9e-5 m (Blade); see `reports/temporal_measurement_information_study.md`.
- Acceptance is on stable material identities; first-hit visibility is recomputed at every candidate state.
