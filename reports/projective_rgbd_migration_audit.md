# Projective RGB-D migration audit (step 1 of `CODEX_RGBD_FOUR_PASS_PROJECTIVE_IEKF_CUDA_SUBSAMPLING_2026-10-27.md`)

Date: 2026-10-28.

## Base checkout

| Item | Value |
|---|---|
| `git rev-parse HEAD` | `34131bcdcedbbd02f6118f1a766fd8394bed3b3e`, the commit the contract was written against |
| Worktree | The uncommitted RigEuler-style restructure of 2026-10-27: legacy code moved to `attic/` (308 tracked paths show as deleted, 24 modified, 13 untracked). The worktree is the source of truth, as the contract requires. |
| Snapshot before editing | `/home/samuelemara/Joint3DGS/RGBDekf_snapshot_before_projective_20261028.tar.gz`, sha256 `86677e45…0fa164` (excludes `attic/`) |
| Earlier backup | `/home/samuelemara/Joint3DGS/RGBDekf_backup_before_restructure_20261027.tar.gz` |
| CUDA hardware | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB, driver 580.178.04. torch 2.11.0+cu128, CUDA 12.8. Shared with the user's own jobs: check `nvidia-smi` before GPU work. |
| Archived results | Nothing under `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/` is modified. |

## Contract table (§10) mapped to the current worktree

The contract's file names are those of commit `34131bcd`. After the restructure they live at the paths below. These rendered-splat modules **stay in the repository, labelled as the legacy rendered-splat baseline**, and are reachable only through `tools/legacy_rendered_splat_train.py`. **No new production module imports them.**

| Contract name (at `34131bcd`) | Current path | Current behaviour | New production replacement |
|---|---|---|---|
| `pointwise_rgbd/rgbd_point_renderer.py` (`_candidates`, `render_view`, `Linearization`) | `estimator/point_renderer.py` | Footprint splats, normalized contributor RGB-D, visibility taper, coverage, synthetic-image Jacobian | `estimator/projective_camera.py`, `estimator/projective_sensor.py`, `estimator/projective_visibility.py`: projection of points into the actual cameras and bilinear sampling of the **measured** RGB-D |
| `pointwise_rgbd/image_observation.py` (`student_t`, `FrameObservation.terms`, `.pixel_energy`, `.apply_W`) | `estimator/observation_terms.py` | Student-t, coverage-tempered energy, IRLS | `estimator/projective_terms.py`: 4-channel `r = −h`, analytic `H`, diagonal sensor `R`, reason codes |
| `pointwise_rgbd/image_iekf.py` (`linear_system`, `trust_region_solve`, `_frame_step`) | `estimator/sparse_iekf.py` | `Gᵀ(HᵀWH+Λ)G`, Steihaug-PCG trust region on a rendered-image objective | `estimator/point_kalman_cuda.py`: independent per-point 6D IEKF with fixed prior, batched 6×6 information solve, final covariance |
| `pointwise_rgbd/phase1_covariance.py`, `restricted_posterior.py` | `estimator/covariance.py`, `estimator/restricted_posterior.py` | Grouped block-information and restricted-Joseph models | Per-point 6×6 covariance in `estimator/point_kalman_cuda.py` (pointwise-independence approximation, declared) |
| `pointwise_rgbd/supernodes.py` (`GroupBasis`) | `estimator/supernodes.py` | 12D grouped correction tangent `G ζ` | **Not used** in any free update. Point chunks are memory batches only (`--point-chunk-size`). |
| `pointwise_rgbd/image_phase1.py` (`endpoint_catalogue`, `run`) | `estimator/phase1.py` | Endpoint samples, then footprint radius, `render()` and `frame_step()` | Endpoint extraction reused (`scene/endpoint.py`, unchanged). `scene/canonical_subsample.py` then `estimator/free_projective_run.py` for F1/B1. |
| `pointwise_rgbd/image_coherence.py`, `supported_coherence.py`, `rigeuler_ref/coherence_extraction.py` | `estimator/motion_coherence_support.py`, `estimator/supported_coherence.py`, `rigeuler_ref/coherence_extraction.py` | Pinned coherence, image-support weighting | **Kept unchanged** (parity-tested). `estimator/projective_coherence.py` feeds them the free trajectories and the measurement validity of the new filter, and runs both the pinned-raw and measurement-supported weightings. |
| `pointwise_rgbd/rigid_image_iekf.py` (`small_state_step`, `run`) | `estimator/rigid_part_iekf.py` | Shared pose/twist via a rendered-image trust-region energy | `estimator/rigid_kalman_cuda.py`: 12D Lie IEKF on the same projective measurement, with `estimator/se3.py` (Exp/Log/Ad/J_l, [v,ω], finite-difference tested) |
| `pointwise_rgbd/joint_image.py` (`reduce`, `refine`, `run`) | `estimator/joint_iekf.py` | Joint reduction plus rendered-image refinement and DM tests | `joint_reduction/joint_projective.py`: pinned `reduce_trajectory` on F2/B2 poses, χ² adequacy decision, optional `(q, q̇)` projective IEKF |
| `pointwise_rgbd/image_support.py` | `joint_reduction/support.py` | Splat flip/remove energies, block ICM, Gibbs pseudo-likelihood | `joint_reduction/support_projective.py`: measured projective consistency of static vs joint transport, frame-level tests |
| `pointwise_rgbd/image_visualization.py`, `rigeuler_debug.py` | `debug/viewer.py`, `debug/rigeuler_pages.py` | Pages built on rendered images | `debug/projective_debug.py` plus `debug/assets/projective_*.html`: measured RGB-D and point-only pages |
| `tools/run_rgbd_image_iekf.py` | the old `train.py`, moved to `tools/legacy_rendered_splat_train.py` | Rendered-splat pipeline | `train.py` (four passes; `tools/run_rgbd_projective_pipeline.py` is the contract's alias name) |

## Reused without change

- `scene/` loading: `metadata/cameras.json` is converted once to world→camera `T_cw` by `scene.dataloader.camera_from_metadata`; masks are strictly binary; masks are required; `depth_npy` must be float metres; unrectified distortion is refused by `scene/dataset.py`.
- `scene/endpoint.py`: measured endpoint back-projection with source provenance `(frame, camera, pixel)` and first-order covariance.
- Coherence: `estimator/motion_coherence_support.py`, `estimator/supported_coherence.py`, `rigeuler_ref/coherence_extraction.py` (`tests/test_coherence_parity.py`).
- Joint reduction: `rigeuler_ref/rigid_trajectory_reduction.reduce_trajectory` (pinned, vendored verbatim).
