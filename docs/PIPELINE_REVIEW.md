# Pipeline review: theory and implementation (27 October 2026)

This review covers the code as it stands after the restructure. Every claim cites a file and line or a measured report. "Measured" means this repository's own audits on `blade_103706` (34,369 points, 8 cameras, 60 frames) or on its test fixtures. Ground truth was used only after inference.

**Bottom line.** The operators are correct, and the solver is a faithful port of the pinned one. The method on top of them is not yet sound. Phase 1 is numerically chaotic, its carried covariance is overconfident by construction, and its correction space cannot separate a moving part from a static one inside a group. Everything downstream inherits that. No end-to-end result of this pipeline should be reported as validated.

---

## 1. What the pipeline computes

For each anchor (first frame forward, last frame backward):

1. **Catalogue** (`scene/endpoint.py:66`): every valid masked pixel of every camera at the anchor, back-projected to a coloured 3D point with a first-order covariance. No fusion and no subsampling: the blade gives 34k points, `laptop_10211` gives 146k.
2. **Phase 1** (`estimator/phase1.py:71`): every point carries `s_i = [x_i, v_i]`. Each frame runs a constant-velocity prediction (`estimator/process.py:18`), then a MAP solve of `E(s) = ½ (s−s⁻)ᵀ Λ⁻ (s−s⁻) + E_img(render(s))` restricted to `s = s⁻ + G ζ`, with ~64-point rigid groups (`estimator/sparse_iekf.py:89,103`). Then one posterior covariance update (`:178`).
3. **Coherence** (`estimator/motion_coherence_support.py:60`): 192 representative nodes, the pinned held-out rigid-coherence extractor, weighted by how much image support each node had per frame.
4. **Rigid** (`estimator/rigid_part_iekf.py:195`): one SE(3)+twist IEKF for the selected core, rendered against the whole static catalogue.
5. **Joint** (`estimator/joint_iekf.py:105`): pinned pose-space reduction into prismatic and revolute candidates, a causal `(q, q̇)` image IEKF per candidate, and Diebold–Mariano tests on one-step predictive image NLLs.
6. **Support** (`joint_reduction/support.py:218`): static/moving labels for every endpoint point under the chosen joint, from a MAP plus frame-level contradiction tests.

---

## 2. Theory: what is sound

* **Rendered-image measurement, no correspondences.** The innovation is `y − h(s)` on images from all cameras (`estimator/observation_terms.py:93`). This removes the association failure mode of the earlier pipelines in `attic/`.
* **Continuous rasterizer.** The footprint weight `(1−q)²`, the footprint-weighted occlusion depth `ζ = z + b(1/w − 1)`, and the C¹ visibility taper inside the band make the objective continuous in the point positions (`estimator/point_renderer.py:162-221`). The docstring (`:19-22`) records the measured failures of the hard-band alternatives that motivated it.
* **Exact operators.** `H`, `Hᵀ`, `G`, `Gᵀ` and `A_g = Gᵀ(Λ⁻ + HᵀWH + λI)G` agree with dense oracles to about 1e-16. `b_g` agrees with finite differences of the true objective to 1e-7 (`reports/phase1_step2a_audit.md` A1/A2). Reductions are deterministic: sort plus segment sums, no atomics.
* **Fixed prior per frame.** The prior is not relinearized inside the frame, and the step is accepted on the true objective of a fresh render, as in the pinned `sparse_iekf`.
* **Coherence parity.** With unit weights the local extractor equals the pinned blobs node for node (`reports/coherence_reference_parity.md`). The support weighting (`estimator/supported_coherence.py:1-36`) is a principled fix for prediction-only trajectories that look coherent.
* **Joint decision.** One loss per frame (pixels are never treated as independent samples), a persistence null, a causal-prefix geometry, a conservative long-run variance, and one declared α (`configs/image_joint_policy.json`). There are no retrofitted displacement thresholds.
* **Ground-truth isolation.** The evaluation runs in a separate process (`train.py`, `tool('evaluate', …)`).

## 3. Theory: what is not sound, or is unproven

### 3.1 The likelihood is a pseudo-likelihood, and the "posterior" is not a Bayesian posterior

* **RGB and depth.** These are Student-t per element, multiplied by the predicted coverage: `A_p·ρ(r/σ)` (`observation_terms.py:103-110`). Raising a density to the power `A_p` is a tempering, not a likelihood of any sensor model. The docstring says so (`:16-25`).
* **Consequence: the cost can be lowered by uncovering pixels.** The model can lower cost by reducing coverage on badly explained pixels: the gradient in `A` is `κ(1−A) − ρ_rgb − ρ_d` (`:105`). Only `½κ(1−A)²` (`:111`) resists this, with `κ = tr(W_rgb)/12` borrowed from the reference masked mode (`:67`). The balance between "move to explain" and "move away to hide" is set by that κ, which is not derived for this renderer.
* **Pixels are independent in `E`.** Each pixel's sum has 16–26 contributing points on average and up to 130 (`reports/phase1_gpu_profile.md`, counters), and neighbouring pixels share contributors. The Fisher information `HᵀWH` therefore treats correlated residuals as independent. The joint and support stages correct for this at the test level (frame-level losses), but Phase 1 and its covariance do not.
* **Noise values are illustrative.** RGB σ = 0.05, depth σ = 1 cm, `qc = 0.01 m²/s³`, `qx = 1e-4 m²/s`, `σ_v0 = 1 m/s` come from `configs/example_uncalibrated.json`, whose own provenance field reads "Illustrative values only. Not estimated from the dataset." All covariances, gains and test statistics inherit these values.

### 3.2 The Phase-1 correction space cannot represent a part boundary

* **Group-rigid corrections only.** Every correction is `δs = G ζ`: each ~64-point group moves rigidly (`estimator/supernodes.py:95`, the docstring of `estimator/covariance.py:6-23`). A point's deviation from its group's rigid motion is never observed. It evolves only through its own predicted velocity, which is itself corrected only group-rigidly.
* **Groups are geometric.** They are built from the anchor catalogue with no knowledge of the joint. A group that straddles the moving/static boundary must either move as one or not at all. Overlap purity against the 192 coherence nodes is a median of 0.39 (`reports/coherence_reference_parity.md`, stage A). The non-rigid, fragmented moving-part trajectories that broke coherence (Step 1: 3–4-node fragments below the 5-fold CV minimum) are exactly what this restriction produces near the boundary.
* **Why it was kept.** RigEuler uses the same grouped tangent, but its groups are Gaussians optimized with photometric training. Here the groups are fixed, unrefined, and never split.

### 3.3 The carried covariance (Phase 1)

There are three variants, all behind `estimator/covariance.py:198`.

* **Production: `rigeuler_block_information`** (`estimator/restricted_posterior.py:177-184`): `Λ⁺_i = Λ⁻_i + blockdiag(F_i, 0)`, where `F_i` is the point's own 3×3 diagonal block of `HᵀWH`.
  * **Inconsistent with the update actually applied.** The gain is group-restricted (§3.2), but this information assumes each point is observed independently.
  * **Ignores cross-point coupling.** It drops the off-diagonal Fisher blocks. A diagonal block of the inverse is at least the inverse of the diagonal block, so the result is always optimistic: 20–1000× below the exact one-frame covariance on dense fixtures (`covariance.py:30-32`).
  * **Measured effect.** The prior velocity σ shrinks to 0.097 m/s by frame 10 and 0.054 m/s by frame 20, while the true part speed is 0.84 m/s. At frame 10, moving the part to its true position lowers the image energy by 3.4k but raises the prior energy by 684k (`reports/phase1_step2a_audit.md`, mechanism 2). **This is the main cause of the lag of the moving part.**
* **`restricted_joseph_structured`** (`covariance.py:68`): the exact error covariance of the implemented estimator in the `d + Gξ` form. It matches a dense recursion to 1e-6 for 15 frames on a 1,000-point crop. At frame 16 it breaks once cond(B) ≈ 1e14, with no algebraic error found. The true covariance also grows without bound (max variance 21,000 by frame 40) in the directions the grouped gain cannot observe (§3.2). The exact answer is therefore "these directions are unobservable", and no PSD repair can fix that.
* **`restricted_joseph_marginals`**: exact for one frame, then an assumed-density approximation that diverges exponentially at blade scale (`covariance.py:24-29`).
* **Conclusion.** No current variant is both consistent and stable. The correction space (§3.2) and the covariance must be fixed together.

### 3.4 Phase-1 optimization: local, non-smooth and chaotic

* **Stall at the first frames** (frames 1–2). The local solve stalls at 2–6 mm of slide while the image alone prefers 10–23 mm. Steps at the scale of a footprint get a low gain ratio and the trust radius collapses to 6e-5. A kink probe shows a one-sided curvature change at sub-micron scale at the stalled iterate (`reports/phase1_step2a_audit.md`). The front selection is piecewise by construction (`point_renderer.py:29`, "piecewise at ordering ties").
* **Badly conditioned system.** The reduced system has eigenvalues from 0.16 to 3e8. Production stops PCG at 30 iterations and a forcing tolerance of 1e-2. The resulting steps differ from accurate solves by 68–90%. Neither more budget nor accurate solves move the part (A3).
* **Chaos.** A 1e-15 relative perturbation of the prior changes the forward trajectory by 2.4 mm median and 188 mm max within 9 frames (`reports/diagnostics/phase1_roundoff_sensitivity.json`). Over 60 frames, a round-off-level change turned coherence from 192/192 moving into 13 moving nodes (`reports/phase1_b1_small_contractions.md`). Both runs also contain spurious points displaced more than 1 m (16 points and 21 points), against a true slide of about 0.84 m.

### 3.5 Downstream stages

* **Coherence** is a correct port, but it receives unreliable trajectories (§3.2, §3.4). Its outcome is not reproducible: 20, 192 and 13 nodes on three Phase-1 runs of the same scene.
* **Rigid** keeps the canonical geometry fixed and renders non-core points as static (`rigid_part_iekf.py:15-18`). That is a correct hypothesis test only if the core is right. On fixtures it misses its own accuracy gates (§5).
* **Joint**: the test design is sound (§2). With 60 frames, a 3-frame burn-in and a conservative long-run variance, its power is low. The persistence null absorbs constant offsets by design (`joint_iekf.py:21-25`).
* **Support** uses `exp(−E)` as a Gibbs pseudo-likelihood (`support.py:10-13`, stated in the code). The contradiction-plus-support rule is reasonable, but the textured-fixture recall gate fails (§5).
* **Bidirectional**: forward and backward are compared, never fused. This is correct (they reuse the same pixels), but the pipeline then makes no use of the second direction for its result.

---

## 4. Implementation review

### Good

* **One entry point.** `train.py` handles stages, partial runs and dataset roots. Reuse requires identical inputs, settings and stage-source hashes. Each run records a manifest, a source snapshot and per-stage seconds.
* **Deterministic.** Runs reproduce bitwise run to run (`reports/diagnostics/b1_frame1_sensitivity.json`, off vs off).
* **Isolation.** The pinned code is vendored under `rigeuler_ref/` and checked against the git blobs by hash (`tools/audit_coherence_parity.py`). Ground truth is used only in `tools/evaluate_image_iekf_gt.py`, run as a subprocess, and in display layers of `debug/rigeuler_pages.py:73`, which run after inference has written its results.

### Problems (ordered by impact)

1. **Speed.** 5–9 s per frame on the blade, 6–8 min per anchor (`reports/phase1_gpu_profile.md`).
   * 72% of GPU time goes to tiny float64 batched GEMMs. The opt-in `--small-contractions` gives about 2× per frame.
   * The GPU is idle about 40% of the time because of host syncs (`segment_reduce` with host lengths, `nonzero`, `float()` reads in the trust-region loop).
2. **Memory.**
   * The 146k-point scene runs out of memory under a 3.6 GB cap. The peak comes from per-point `[N,12,12]` group blocks while two linearizations are alive.
   * `covariance.npy` is `K×N×6×6` float64: 594 MB per anchor on the blade, and about 2.5 GB at 146k points.
3. **Configuration debt.**
   * `scene/config.py:42-66` still loads the association gate of the old pipeline (`configs/pointwise_association.json`) and calls `config.gate()`.
   * `scene/noise.py` parses a 12-D multi-body noise format of which the pipeline uses six numbers.
   * Neither affects results, but both hide what the model actually assumes.
4. **Stale documentation in code.**
   * The `estimator/covariance.py` docstring says the structured method loses positive definiteness at frame 11. That predates Jacobi equilibration; the current measurement is frame 16 on the crop, as a conditioning limit.
   * The `train.py --posterior` help defers to this review for that reason.
5. **Duplicate code.** `build_supernodes` exists twice: the Phase-1 grouping in `estimator/supernodes.py:49`, and the pinned 192-node version in `estimator/motion_coherence_support.py:26`. They are different functions with the same name.
6. **Per-frame overhead.** Every frame runs an extra `H`/`Hᵀ` adjoint check (`sparse_iekf.py:172-177`). It is cheap but redundant with the tests.
7. **Partial reuse requires a `train.py` run.** Runs made before the restructure are refused for `--reuse`, because their manifests use the old source paths. This is intentional: the sources changed (imports).

## 5. Test status (after the restructure)

`python -m unittest discover -s tests -t .`: 63 tests, **58 pass, 5 fail**.

| Failing test | Measured | Gate |
|---|---|---|
| `test_image_rigid_joint.PipelineTest.test_prismatic_forward` | rotation error 0.81° | < 0.5° |
| `test_image_rigid_joint.PipelineTest.test_revolute_forward` | rotation error 1.99° | < 0.6° |
| `test_image_rigid_joint.RigidFromZeroInitTest.test_image_iekf_recovers_hinge_without_phase1_motion` | translation 11 cm | < 5 mm |
| `test_image_support.SupportLabelsTest.test_labels_on_textured_scene` | recall of uncertain movers 6% | > 30% |
| `test_image_rigid_joint.PipelineTest.test_prismatic_backward_chronological` | translation 2.16 mm | < 2 mm |

All five failures **predate the restructure**. The pre-restructure tree, run from the backup, fails the same five tests with bitwise-identical measured values (`reports/restructure_test_comparison.md`). Four are rigid/joint accuracy gates on synthetic fixtures. One is the support recall gate.

## 6. What to fix, in order

1. **Phase-1 correction space and covariance together.** Either let groups split or refine, as RigEuler's Gaussian training effectively does, or add a per-point residual correction with its own prior. Then carry the covariance that matches the gain actually applied. With an observable state, the exact structured form becomes usable; it should be implemented as square-root or information form to avoid the cond-1e14 break.
2. **Make the per-frame solve robust to the piecewise objective.** Use a better-conditioned reduced system (a group preconditioner that includes the cross-group coupling), and decide whether the coverage tempering is wanted at all, given that it allows "explain by hiding".
3. **Calibrate the noise.** Estimate RGB and depth σ and the process densities from the data, before interpreting any covariance or test.
4. **Only then** run the 40-scene batch, B2 point reduction, and further performance work (host syncs, memory).
