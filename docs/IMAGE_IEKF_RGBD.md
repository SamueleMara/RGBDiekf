# Rendered-image RGB-D IEKF, v2 (RigEuler3DGS 481b2cf2 methodology on coloured XYZ points)

This is the implementation of `CLAUDE_RIGEULER_RGBD_RIGOROUS_IMPLEMENTATION_CONTRACT.md`.

* **Audit before implementation:** [reports/theory_to_code_audit.md](reports/theory_to_code_audit.md).
* **Validation and measured outcomes:** [reports/rigeuler_rgbd_theory_validation.md](reports/rigeuler_rgbd_theory_validation.md).
* **Gate results:** [reports/rigeuler_rgbd_test_matrix.csv](reports/rigeuler_rgbd_test_matrix.csv).

The measurement is the RGB-D image rendered from the transported coloured point cloud into every camera. The observations are the next frame's real images. There is no association, background, alpha or opacity.

## 1. State, process, measurement

* **Phase-1 state.** Each point `i` has state `s_i = [x_i, v_i]` with a physical-time velocity. The process is `F = [[I, h I], [0, I]]` with signed step `h`, and noise `qc [dt³/3, dt²/2; dt²/2, dt] + qx dt`: white acceleration plus the documented position random walk. Backward runs use a positive elapsed duration and `h = -dt` on the physical velocity. This is equivalent to the reference processing-time velocity `v_p = -v` (`test_reverse_time_equals_reference_processing_time_convention`).
* **Renderer** (`pointwise_rgbd/rgbd_point_renderer.py`). Each point has a compact footprint `w = (1-q)²` with radius `s = f ρ / z`.
  * Visibility is continuous. Each contributor has its own band `b_i = 3σ_D(z_i) + ρ_i` and occlusion depth `ζ_i = z_i + b_i (1/w_i - 1)`.
  * The pixel front is `min ζ`. A contributor's visibility is a C¹ taper over `[b, 2b]` of its gap to the front.
  * The pixel prediction is a normalized splat, with coverage `A = 1 - exp(-W)`.
  * The Jacobian is exact for the frozen front contributor. It includes the front-depth derivative, which is added to the front contributor's own row.
  * Why continuous: a hard band, or a front defined by an edge sample, made the fixed objective jump (+50 for a 1e-6 m step). That collapsed the reference trust region with the moving plate half-way.
* **Observation** (`image_observation.py`). The domain is observed foreground pixels with valid depth.
  * RGB and depth use the reference per-element Student-t (ν = 5), tempered by the rendered support: `A_p ρ(z)`.
  * Coverage uses an explicit mask observation `½ κ (1 - A)²`, with `κ = tr(W_rgb)/12` (the reference masked-mode κ).
  * Background values are never read, and nothing is synthesized where no prediction exists.
  * The Joseph noise is the Student-t variance `ν/(ν-2) σ²` and `1/κ`.

## 2. Phase 1: fixed-prior iterated update (`image_iekf.py`, `image_phase1.py`)

* **Correction groups.** The pinned `build_supernodes(target_size=64)` on the endpoint catalogue, with the raw grouped tangent `[dt, dθ, du, dω]` (`supernodes.py`). They are computational only; there is no persistent group pose.
* **Solver.** The pinned trust-region path of `sparse_iekf.iekf_frame_step`:
  * `b = Gᵀ(HᵀWr - Λ⁻e)` and `A = Gᵀ(Λ⁻ + HᵀWH + 10⁻³ I)G`, applied matrix-free over all cameras;
  * Steihaug-CG in the RMS physical metric (σ = .05/.05/.5/.5) with the block-Jacobi group preconditioner;
  * forcing η and `adapt_radius` (.1/.25/.75); acceptance on the true objective after a fresh render;
  * stop on the reference criteria or after 10 iterations. There is no pyramid (`pyramid=(2,1)` is an ablation).
* **Covariance** (`phase1_covariance.py`, named interface). The complete runs carry the reference's `rigeuler_block_information`: `Λ⁺_i = Λ⁻_i + blockdiag(F_i, 0)`. It is bounded but optimistic: 20–1000× below the exact one-frame covariance on the dense fixtures.
  * **Why not Joseph.** Both restricted-Joseph recursions are implemented, and both are exact against dense 6N oracles over 5 fixture frames: the structured exact recursion (≤ 2·10⁻¹⁰) and the assumed-density marginals (exact for one frame). Both DIVERGE on the blade. The restricted gain moves only whole groups, so the image-measured intra-group deviations that the independent-point prior allows are pushed into the group corrections, which in turn inflate every group member.
  * **Measured divergence, assumed-density, blade forward.** The maximum position variance went from 1.3e-2 m² at frame 1 to 6.5e2 at frame 4 and 1e18 at frame 14, ending in a Cholesky failure at frame 20. The exact structured recursion lost positive definiteness at frame 11.
  * This is the earliest failing gate of the implementation (`reports/rigeuler_rgbd_theory_validation.md` §4).

## 3. Coherence (`image_coherence.py`, `supported_coherence.py`)

* **Pinned pipeline.** 192 fixed-count nodes, symmetric kNN with k = 12, log strain, and the held-out extractor.
* **RGB-D adaptation.** Node displacements are averaged over image-supported members only. The node-frame support ω weights every statistic. An edge supported only at the gauge frame is removed.
* **Reduction.** With ω ≡ 1 the result is bitwise the pinned extractor.
* **Documented limitations of the pinned selection.** It resolves a component on static noise and on a pure global drift, so static scenes are rejected downstream.

## 4. Phase 2 and the joint stage (`rigid_image_iekf.py`, `joint_image.py`, `predictive_tests.py`)

* **Phase 2.** One shared SE(3)+twist IEKF: `F = [[Ad_E, h J_l(hξ)], [0, I]]`, left prior error with `J_e`, and a whole-scene render.
  * It uses the pinned small-state `frame_step`: prior-diagonal metric, dense-probe preconditioner, 8 iterations.
  * The posterior is the Joseph form `A⁻¹(N + JᵀWRWJ)A⁻¹`.
* **Joint stage.**
  * The pinned `reduce_trajectory` gives the reported geometry.
  * For each type, a `(q, qdot)` image IEKF runs with causal-prefix geometry, so a model is scored on images its geometry has not seen.
  * The decision is a Diebold–Mariano test on one-step predictive image NLLs, with the conservative `max(Newey–West, AR(1))` long-run variance, the HLN correction and `t(n-1)`.
  * Outcomes: `NO_IDENTIFIABLE_JOINT`, `PRISMATIC`, `REVOLUTE` or `UNRESOLVED_INDISTINGUISHABLE`. There is no excitation or far-axis threshold.

## 5. Forward/backward and support (`image_bidirectional.py`, `image_support.py`)

* **Forward/backward.** World-frame diagnostics only: no averaging, pooling or posterior product.
* **Support model.** Global labels `z`, with `p(Y|z) ∝ exp(-E(z))` on the complete scene and a uniform prior.
* **Inference.**
  * The MAP comes from a monotone block ICM. It reproduces exact enumeration on small coupled scenes.
  * Labels come from per-point, frame-level tests of the single-flip differences at the MAP, with Benjamini–Hochberg control.
  * The fast path is exact against full re-renders.

## 6. Reproduce

```bash
python train.py --scene <scene | dataset root> --output <fresh dir> --fps 59     # everything, debug + GT evaluation + QA
python -m unittest discover -s tests -t .
```

See the [README](../README.md) for partial runs, and the [code map](PIPELINE_CODE_MAP.md) for the files. The pre-restructure commands `tools/test_matrix.py`, `tools/run_ablations.py` and `tools/run_rgbd_image_iekf.py` are in `attic/tools/`.
