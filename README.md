> **24 October correction:** production uses ALL available cameras jointly at every frame and stage. Coherence cross-validation splits supernodes, not cameras. See [current execution contract](FREE_GROUPED_RGBD.md) and [all-camera results](reports/free_grouped_all_cameras_2026-10-24_validation.md). Historical camera-partitioned results below are superseded.

> **23 October correction:** the physical-supernode and boundary-splitting architecture below is superseded. The current runner implements free individual 6D point states with grouped increments, reference coherence, shared rigid RGB-D fitting, joint inference and endpoint support completion. See [current architecture and commands](FREE_GROUPED_RGBD.md). Real-data acceptance is determined by the new measured validation report, not the historical gates below.

> **Grouped contract (22 October 2026):** isolated grouped Phase 1 is implemented, but the required joint-boundary gate fails. Recovery/full reconstruction remain paused. See [the exact audit and measured status](reports/grouped_2026-10-22_validation.md). This is not a completed grouped/coherence/rigid/joint pipeline.

> **Correspondence correction (22 October):** geometric surface coverage is independent of exclusive IEKF ownership. See [the audit and matching equations](AUDIT_CORRESPONDENCE_2026-10-22.md) and [real-data comparison, incremental validation and commands](reports/correspondence_2026-10-22_results.md). Unlimited discovery has not been validated or launched by this correction.

> **22 October 2026:** the current request is the isolated **6D pointwise** algorithm. Use [POINTWISE_RGBD.md](POINTWISE_RGBD.md), [its audit](AUDIT_POINTWISE_2026-10-22.md), and [current validation](reports/pointwise_2026-10-22_validation.md). The commands and rigid/group/UKF model-selection blockers below concern superseded pipelines; they are historical, not blockers for the new pointwise estimator.

> Historical status: the earlier rigid reconstruction was **NOT complete/accepted**. See [LATEST_REQUEST_VALIDATION.md](LATEST_REQUEST_VALIDATION.md) and [RUNNING.md](RUNNING.md).

> Workspace update (20 October): the masked point-reprojection v2 theory is now
> authoritative. No visibility-policy configuration is required. Current commands
> and measured validation status are in [RUNNING.md](RUNNING.md) and
> [VALIDATION.md](VALIDATION.md); prior visibility-guard notes are superseded.

# Masked RGB-D point SLAM — two independent filter pipelines (20 October 2026)

## Files

- `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.pdf`: complete 26-page theoretical specification, including source-to-point model correspondence (M1--M8), common solved/unsolved algorithm (C), analytical IEKF+RTS (A), Lie-group UKF+unscented RTS (B), sparse CUDA implementation (D), and numerical/real-data acceptance requirements.
- `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md`: editable complete theory source.
- `CODEX_IMPLEMENT_PARALLEL_MASKED_RGBD_IEKF_UKF_v2.md`: actionable mandatory Codex development, correctness, GPU, mask, and end-to-end validation contract.
- `reference_3dgs_masked.pdf`: user-supplied reference note containing the 3DGS measurement model, especially Section 13. This is a scientific source, **not** a point-renderer implementation target.

## Correct interpretation

Only mask-positive, genuinely valid metric-depth observations can initialize points, enter correspondences, become UNSOLVED candidates, or provide Kalman measurement updates. RGB is considered at their corresponding measured foreground pixels. A point-only model has no Gaussian alpha, so randomized background compositing and transparency-carving gradients do not transfer. The two pipelines share data/matching/outer reconstruction logic, but their filter and smoother equations are independent.

## Running Codex after copying these files into a local repository

From the RGBDekf repo, place these four files under `specs/masked_point_v2/` and invoke:

```bash
codex exec -C /home/samuelemara/Joint3DGS/RGBDekf --sandbox workspace-write - <<'PROMPT'
Implement the complete TWO-INDEPENDENT-PIPELINE masked RGB-D point SLAM specification.
Read specs/masked_point_v2/CODEX_IMPLEMENT_PARALLEL_MASKED_RGBD_IEKF_UKF_v2.md and specs/masked_point_v2/Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.pdf completely before editing code. Study specs/masked_point_v2/reference_3dgs_masked.pdf Section 13 only for the measured RGB/depth/mask preparation.
Use one common solved/unsolved outer loop, direct point reprojection and foreground-only observations. Do not transfer 3DGS alpha or artificial backgrounds. Implement analytical IEKF plus RTS and separately manifold UKF plus unscented RTS, both with sparse CUDA kernels. Remove all visibility-policy guards. Run ALL required tests, then both actual complete full-dataset forward/backward/discovery/frontier/second-forward pipelines. Report actual results and failures, never preflight/unit-tests as completed reconstruction.
PROMPT
```

This provides instructions for implementation; it is not itself implemented workstation source code.
