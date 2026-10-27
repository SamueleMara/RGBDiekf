# Theory-to-code audit: RigEuler3DGS 481b2cf2 vs. the RGBDekf rendered-image pipeline

Written **before** the corrective implementation required by `CLAUDE_RIGEULER_RGBD_RIGOROUS_IMPLEMENTATION_CONTRACT.md` (§1).

* **Reference.** `RigEuler3DGS` at `481b2cf2d5602932286948cf14114f9caee4de3c`. Every listed reference file is unchanged between that commit and the reference HEAD (`git diff --stat`).
* **Caution on the reference working tree.** It holds an *uncommitted* 1511-line rewrite of `estimator/coherence_extraction.py`. Only the pinned blob is used.
* **Local code audited.** The uncommitted RGBDekf work of 24 October 2026 (Claude Code): `pointwise_rgbd/{rgbd_point_renderer, image_observation, image_iekf, image_phase1, image_coherence, rigid_image_iekf, joint_image, image_bidirectional, image_support, rigeuler_debug}.py`, `rigeuler_ref/*`, `tools/run_rgbd_image_iekf.py`, `tests/test_image_*`.

**Verdicts.** Each verdict compares mathematics, not outputs.

| Verdict | Meaning |
|---|---|
| **EQ** | Algebraically equivalent; the mapping is stated |
| **ADAPT** | Required adaptation from Gaussians to coloured points; the derivation is stated |
| **DEV** | Deviation from the reference that the contract does not allow, or does not justify |
| **WRONG** | Mathematically incorrect |

Each row names the test that does or will establish the claim.

## 1. Phase-1 state, prediction and process noise

| # | Reference equation / function | Local function | Dims | Verdict | Test / action |
|---|---|---|---|---|---|
| 1.1 | `process.process_predict`: `mu+=v dt`, `v` carried (12D: plus `R`, `omega`) | `process6.predict` | 6 per point (no `R`, `omega`) | **ADAPT**. Rotation and angular velocity are dropped because a point has no orientation (contract §2). | FD of `F`; dense prediction test (to add) |
| 1.2 | `error_state_transition`: `F=[[I,dt I],[0,I]]` (translational rows) | `process6.transition(dt, reverse)` | 6×6 | **EQ** forward. **Reverse:** the reference uses positive elapsed *processing* time with a reverse-time velocity. Locally, `reverse=True` uses `F=[[I,-dt I],[0,I]]` with the physical (chronological) velocity. The two are equivalent under `v_p=-v`, `dt_p=dt`. | Analytic forward/reverse test + FD (to add) |
| 1.3 | `covariance.make_process_noise`: `Q_xx=σ_pos² dt + q dt³/3`, `Q_xv=q dt²/2`, `Q_vv=q dt` (densities) | `process6.process_covariance` with `qc` (acceleration density) and `qx` (position random-walk density) | 6×6 | **EQ** in form. The reverse cross term is `-q dt²/2`, the correct sign for physical velocity. Values: approved config `qc=0.01 m²/s³` (equal to the reference `σ_vel=0.1`), `qx=1e-4 m²/s` (reference `σ_pos=1e-3` ⇒ `1e-6`). This is a documented user-approved value. | PSD + sign test (exists: `test_reverse_process_signs_and_psd`) |
| 1.4 | Prediction once per frame; fixed prior | `image_phase1.run` + `image_iekf.frame_step` | – | **EQ** (one `predict` call before `frame_step`) | One-assimilation invariant test (to add) |

## 2. Computational grouping (`supernodes.py`)

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 2.1 | `build_supernodes`: farthest-point + Lloyd k-means on **canonical** means, `target_size=64` (Phase 1 `--iekf-supernode-size 64`) | `surface_supernodes.build_partition`: geodesic surface-graph partition. Its graph uses the association chi-square gates (`config.gate`). | **DEV**. A different grouping, and it depends on association-policy parameters. | Use the reference `build_supernodes(target_size=64)` on the endpoint catalogue |
| 2.2 | `GroupedObservation.expand`: `dmu_i = dt_c + dθ_c × (mu_i⁰ − c_c⁰)`; the velocity block uses the same map (`sparse_iekf._G`) | `grouped_point_iekf.basis(normalize=True)` | 6×12 per point | **EQ in range, DEV in coordinates.** The normalised basis `G' = G T_g` spans `range(G)`, so `K_G` is invariant. The trust-region metric, preconditioner and radius, however, act in the transformed coordinates, which differ from the reference. | Use the raw reference basis `ζ=[dt,dθ,du,dω]`. FD tests of `G`, `Gᵀ` (adjoint) |

## 3. Measurement function

| # | Reference | Local | Verdict | Action / test |
|---|---|---|---|---|
| 3.1 | 3DGS alpha-compositing renderer; `FrameLinearization.predicted/apply_H/apply_HT/build_block_fisher` | `rgbd_point_renderer.render/Linearization` (normalised compact-footprint splats, front band `z ≤ z_front+3σ_D+ρ`, coverage `A=1−exp(−W)`) | **ADAPT** (contract §3: a point rasteriser, no alpha). The Jacobian is exact for frozen contributors. | Exist: loop oracle, FD-JVP, adjoint, CPU=CUDA, Fisher blocks. To add: all-8-camera adjoint, ordering-transition test, overlap test |
| 3.2 | `observation_terms.camera_terms`: per-element **Student-t ν=5** on RGB and depth; depth gated by predicted alpha `>0.05`; σ_rgb, σ_D(z) fixed | `image_observation.FrameObservation.terms`: Gaussian + uniform-outlier mixture weighted by coverage, `-log[εU + c(1−ε)N]` | **DEV**. The likelihood is invented here, not the reference's robust likelihood. ε and U were declared without a reference. | Replace with the reference Student-t (ν=5, IRLS weight `(ν+1)/(ν+z²)`, objective `(ν+1)/2 log1p(z²/ν)`) on observed-foreground ∩ coverage-gated pixels (gate 0.05, the reference `min_alpha_for_depth`) |
| 3.3 | Masked mode (default): `½κ(M−A)²`, `κ=tr(W)/12` (the expectation over an unknown uniform background) | coverage entered only through the mixture | **DEV** | **Explicit coverage observation model** (contract §3): mask `M_p = A_p(s)+n`, `n~N(0,1/κ)`, with the reference κ. It is applied on observed-foreground pixels (`M=1`). Negative mask pixels count only under a declared object-only premise; it is not declared for the blade, so they don't count. No RGB/depth background value is ever used. The mismatch counts are reported separately. |
| 3.4 | Model-error map `R_model` (Phase 2/3 only) | not used in estimation; image-space or point-attached map used in support only | **DEV, measured** (synthetic hinge rotation error 0.37° → 0.64–0.70°) | Contract §3/§8: the *same* observation model as Phase 1 in all estimation stages. Model mismatch is reported as a diagnostic, not used in estimation. Ablation retained. |

## 4. Phase-1 iterated update (`sparse_iekf.iekf_frame_step`)

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 4.1 | `b = Gᵀ(Hᵀ W r − Λ⁻ e)`, `A = Gᵀ(Λ⁻ + HᵀWH + λI)G`, `λ=lm_initial=1e-3` (solver term) | `image_iekf._iterate` (same b, A; LM damping `λ·diag(D)` adapted) | **EQ** in b and the undamped A. **DEV** in the damping form. | Port: fixed `λ I` (1e-3), as in the reference `A_mv` |
| 4.2 | Steihaug-CG trust region, `ScalingMetric.from_physical_scales(σ_mu=.05, σ_phi=.05, σ_v=.5, σ_ω=.5, rms)`, `adapt_radius(η=.1,.25,.75)`, `forcing_eta(c=.5, η∈[1e-6,1e-2])`, `predicted_reduction`, significance floor 1e-12, `group_precond` (block Jacobi of `GᵀMG` + 1e-6 trace jitter) | custom LM + displacement radius + backtracking + PCG | **DEV** | Port vendored `trust_region.steihaug_cg/adapt_radius/ScalingMetric/forcing_eta` and `pcg.block_jacobi_preconditioner` |
| 4.3 | Convergence `rel_decrease<1e-6 AND scaled step<1e-7`; trust-radius floor; `max_iters=10` | `rel<1e-5`, step `<1e-5 m`, budget 30 | **DEV** | Reference criteria and budget |
| 4.4 | — | coarse-to-fine ×2 pyramid (coarse steps accepted on the *coarse* objective) | **DEV** (contract §4.5: no changing goals) | Removed from production; kept as an ablation |

## 5. Phase-1 posterior (`covariance.py`)

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 5.1 | `approximate_posterior_information`: `Λ⁺_i = Λ⁻_i + F_i` (per-Gaussian Fisher) → `invert_blocks` → `inflate_weak_covariance` → `project_psd`. **The reference documents this as an approximation** (cross-Gaussian terms dropped; optimistic). | `Λ⁺_i = Λ⁻_i + blockdiag(F_i,0)`, block inverse; no inflation | **EQ to the reference's approximation; WRONG as the covariance of the implemented restricted estimator.** It ignores the restriction `δ=Gζ` (contract §5). Measured: 20% too low a position variance on a dense case. | Named interface `posterior_blocks(method)`. Production: **exact marginal blocks of the restricted-gain Joseph covariance** `P⁺=(I−K_G H)P⁻(I−K_G H)ᵀ+K_G R K_Gᵀ`, `K_G=G B⁻¹GᵀHᵀW`, `B=Gᵀ(Λ⁻+HᵀWH)G`, under a block-diagonal (assumed-density) prior. Error vs exact dense 6N recursion quantified over ≥5 frames on growing fixtures. The RigEuler method is kept as a comparison. |
| 5.2 | `project_psd` / jitter logged | `inverse_spd` raises on non-SPD | **EQ** in spirit | Log any PSD repair |

## 6. Coherence (`motion_coherence_support.py`, `coherence_extraction.py`, `run_motion_coherence_support.py`)

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 6.1 | `build_supernodes(num_supernodes=192, seed=0)`; `canonical_knn_graph(k=12)` on `traj[0]`; `edge_strain` (median rest length, `C = RMS z`); `extract_moving_component(folds=5)` | `image_coherence.run` (ported functions verbatim except the progress bar; the extractor diffs only in its progress bar) | **EQ** for the functions | Use the vendored pinned `coherence_extraction` |
| 6.2 | Uniform member weights; raw trajectories (the reference's theory §5 asks for visibility/uncertainty weights) | point weights = visibility × precision; point positions zero-order-held at unobserved frames | **DEV without derivation** (contract §7) | Derive and test: node-level information-supported trajectories, per-frame edge weights `w_ab(k)`, simulated occlusion and drift tests against the reference metric and zero-order hold |
| 6.3 | Canonical frame = `traj[0]` (chronological frame 0 even for backward; the reference's own audit lists this as a deviation) | the direction's actual endpoint | **ADAPT (correct)** | Keep; test reverse |

## 7. Phase 2 (`rigid_part_runtime.py`, `joint_iekf.py`)

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 7.1 | `RigidState.process_matrix`: `F=[[Ad_E, dt J_l(dt ξ)],[0,I]]`, `E=Exp(dt ξ)`; `process_noise`: `s²[dt³/3, dt²/2, dt]`; positive processing dt | `rigid_image_iekf.process` (signed physical dt; `|dt|³/3`, `dt|dt|/2`, `|dt|`) | **EQ** under `ξ_p=-ξ`, `dt_p=-dt`, which leaves `E`, `F` and `Q` unchanged in the tangent | FD of `F` (exists, both signs). Add the explicit equivalence test against the reference formula |
| 7.2 | `prior_error`: `[Log(T T⁻⁻¹); ξ−ξ⁻]`; Jacobian through the retraction | analytic `J_e = blockdiag(J_l(e_τ)⁻¹, I)` | **EQ** (FD-tested) | – |
| 7.3 | `frame_step`: Steihaug-CG, `_metric` = prior-information diagonal / n, dense `A` probe + inverse as preconditioner, `max_iters=8`, `radius_init=1`, accept when `adapt_radius` accepts AND `E_new<E` | `small_state_step` (LM + displacement radius + backtracking + pyramid) | **DEV** | Port exactly |
| 7.4 | Posterior `inv(J_eᵀΛ⁻J_e + H_info)` at the accepted state, PSD floor | same | **EQ** | – |
| 7.5 | `kabsch_initialize` (uniform weights; only the initial twist is used) | same | **EQ** | – |
| 7.6 | Stored in chronological order | same | **EQ** | – |

## 8. Joint reduction and refinement

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 8.1 | `reduce_trajectory(R,p,mu0,members,P_tau,times)` with increasing times (positive dt is clamped) | `joint_image.run`: chronological poses and times | **EQ** | – |
| 8.2 | — | `rigeuler_debug.write`: backward runs pass **decreasing processing-order times**, so `relative_increments` clamps `dt` to 1e-300 | **WRONG** (debug pages only) | Pass positive elapsed processing time, as the reference does |
| 8.3 | Phase 3: `(q,qdot)` process `[[1,h],[0,1]]`, `σ_q̈²[h³/3,h²/2,h]`; same small-state `frame_step` | `joint_image.refine` | **EQ** process; **DEV** solver (see 7.3) | Port solver |
| 8.4 | Reference makes **no** type decision | `joint_image`: chi-square LR tests treating pixels as independent + `far_axis_ratio=10` + `3×median footprint` excitation gate | **WRONG** (contract §9: independent-pixel p-values; a threshold retrofitted after a failed static test) | Replace with the **prequential predictive** comparison: per-frame predicted (pre-update) image NLLs of each model, compared with a Diebold–Mariano test using a Newey–West HAC variance across frames (no pixel independence); `NO_IDENTIFIABLE_JOINT` when static is not predictively worse, or when neither 1-DOF model is adequate. The far-axis tell is reported, not thresholded. |

## 9. Forward/backward and support

| # | Reference | Local | Verdict | Action |
|---|---|---|---|---|
| 9.1 | Directions never merged | `image_bidirectional`: compatible geometry is **averaged** and used for support | **DEV** (contract §10: no ad-hoc fusion) | Comparison only; each direction's support uses its own joint |
| 9.2 | — (no RigEuler counterpart) | `image_support`: contradiction rule + block ICM + point model-error map | **DEV** (contract §11: needs a global latent-label model checked by exact enumeration) | Global labels `z`, `p(z)` = Bernoulli(½), `p(Y∣z)` = complete-scene image likelihood. Block ICM to a local MAP. Per-point frame-replicate HAC test of `E_k(moving)−E_k(static)` at the converged labelling. Validated against exact enumeration on small coupled scenes. |

## 10. Summary of corrections (implementation order)

1. Measurement: Student-t (ν=5) + explicit coverage observation term; no mixture; no model error in estimation.
2. Grouping: reference `build_supernodes` with the raw `G`.
3. Phase-1 solver: the reference trust region and Steihaug-CG, verbatim; no pyramid.
4. Phase-1 posterior: restricted-gain Joseph marginals, with dense multi-frame oracles; the RigEuler block-information method as a comparison.
5. Coherence: information-supported node trajectories with a derivation and occlusion tests.
6. Phase-2 and Phase-3 solver: the reference small-state `frame_step`.
7. Joint decision: predictive Diebold–Mariano/HAC model comparison; remove the excitation and far-axis thresholds.
8. Bidirectional: diagnostics only.
9. Support: global-label model with an exact-enumeration check.
10. Fix the backward times in the debug-page adapter.
