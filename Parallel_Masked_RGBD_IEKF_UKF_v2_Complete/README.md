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
