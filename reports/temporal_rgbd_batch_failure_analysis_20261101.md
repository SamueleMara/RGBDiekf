# Actual temporal batch failure analysis — 2026-11-01

**No estimated motion was produced.** This batch is a failed execution, not a reconstruction.

Inspected batch: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_rgbd_20261031_185807_opzr4l9k`. Analysis is read-only with respect to estimation code and original results. No inference was launched and no files were deleted.

The previous all-scene command was inappropriate given the already failed full-cloud pilot. Exposing `--stop-after support` only requested existing downstream stages; it did not make the full temporal solver production-capable. The command-generation and array-shape checks were not end-to-end validation.

| Scene | Points | Reported filtering seconds | Saved updates | Failure | GPU peak GiB |
|---|---:|---:|---:|---|---:|
| USB_100109 | 95967 | 902.7 | 0 | FULL_RUNTIME_LIMIT | 2.46 |
| blade_103706 | 34369 | 2414.2 | 0 | FULL_RUNTIME_LIMIT | 1.67 |
| foldchair_102255 | 103960 | 903.0 | 0 | FULL_RUNTIME_LIMIT | 2.35 |
| fridge_10905 | 187239 | 903.5 | 0 | FULL_RUNTIME_LIMIT | 2.43 |
| laptop_10211 | 145973 | 903.7 | 0 | FULL_RUNTIME_LIMIT | 2.19 |
| oven_101917 | 343176 | 904.3 | 0 | FULL_RUNTIME_LIMIT | 3.91 |
| scissor_11100 | 47033 | 902.4 | 0 | FULL_RUNTIME_LIMIT | 2.19 |
| stapler_103111 | 137186 | — | 0 | ENOSPC during array allocation | — |

Seven recorded filtering failures consumed **130.6 minutes** in aggregate. This excludes time outside the per-pass filtering timer. `completed_frames: 1` means only the frame-0 anchor; zero of the 59 filtering updates were saved.

## What the files actually contain

- Seven `posterior_xyz.npy` files have nominal 60-frame shapes, but **every scalar in frames 1–59 is zero**. They were preallocated and zero-filled before filtering. These are unwritten output slots, not trajectories collapsing to the world origin.
- The seven saved `update_acceptance.npy` arrays contain zero accepted updates. No per-frame temporal diagnostics, innovation summaries, accepted posterior covariances, smoothing results or joint estimates exist.
- Every saved RGB-D panel is frame 0. USB and Blade panels were visually inspected: input RGB/depth/mask and initial ownership are populated; temporal-warp and residual panels are empty. The HTML pages therefore show initialization only.
- Stapler posterior array is truncated: 45,088,768 actual bytes versus 197,547,968 required by its NPY header. It must not be loaded as a completed trajectory.
- No backward, coherence, rigid, joint or support directories exist in these runs. No simplified-mode run was reached.
- `commands.jsonl` records eight child failures. `storage_45135_full.log` is empty; there is no Storage run directory, no Washer run and no final batch summary. An empty log is not evidence that estimation started.

## Code responsibility and causal chain

1. `tools/run_temporal_rgbd_critical.py:71` loops over all full-mode scenes before any simplified-mode scene. Lines 94–101 continue after nonzero child exits. The same systematic solver failure was therefore repeated seven times instead of stopping the workload.
2. `estimator/temporal_run.py:143–150` allocates and zero-fills all 60-frame result arrays before the first update. There is no disk-capacity reservation. This explains valid-looking, full-shaped, zero-tail files after failure.
3. `estimator/temporal_run.py:229,266,269` assembles anchors, process and measurement factors through Python loops. `estimator/temporal_full_graph.py:282–305` linearizes factors individually, performs repeated tiny GPU algebra and scalar synchronizations, then constructs a sparse COO operator. Connected components use bounded dense QR/SVD rather than a scalable batched sparse factorization. Calling this a production CUDA sparse solver was not supported by the measured runtime.
4. USB and Foldchair died while linearizing the prior-predictive innovation calculation. Fridge and Laptop died during measurement-factor assembly; Oven during process-factor assembly; Scissor during the first nonlinear linearization. Blade reached component backtracking but saved no accepted update.
5. The deadline is cooperative: `check_budget()` is called at selected boundaries. QR/SVD, objective evaluation and global line-search loops do not enforce a hard wall-clock timeout. Blade ran **2414.2 seconds** despite a nominal **900-second** budget; the traceback reaches the later check at `solve_sparse_map:399`. The evidence does not isolate which uninterruptible section consumed the excess.
6. Stapler hit `OSError: [Errno 28] No space left on device` in `numpy.open_memmap`. Writing the failure report then also failed. The batch writer has no protected finalization path for disk-full failures, which explains missing status and batch-summary files.

## Resource and statistical interpretation

- The run directories currently occupy **5.876 GiB allocated**. The SSD currently reports **13.78 GiB available** and 100% used after rounding. The ENOSPC log establishes a past allocation failure; this audit does not attribute all existing SSD contents to this batch.
- Recorded GPU peaks range from 1.67 to 3.91 GiB. These failures are explicit runtime-limit exceptions and one disk-full exception, not recorded CUDA out-of-memory errors.
- Preflight reports only a lower bound on component workspace (roughly 7–27 MB for the largest current-state-only component in these runs). Passing it does not establish runtime or full latent-variable QR feasibility.
- Seven intact manifests retain 60 frames, eight cameras, times 0–1 s and the original canonical point-ID order. Correct loading and IDs do not establish successful estimation.
- The noise label is IDEAL_WARP_ASSUMPTION_UNCALIBRATED. Several conditional sensor estimates have a depth constant standard deviation about 6.9e-8 m, essentially quantization scale. These are not independently calibrated physical errors. No innovation or accuracy evidence from this failed batch can validate them.

## Required correction before another batch

First make one real full-cloud three-frame case complete a measured update with bounded assembly/solve cost, valid posterior covariance and an actual frame-1/2 viewer. The factor assembly, synchronization and component solve paths require profiling and replacement by scalable exact operations. Also enforce meaningful runtime checks, disk-capacity handling, explicit completed-frame validity and batch fail-fast behavior. None of those are solved by simply increasing the budget, removing correlations or switching to simplified mode.

This analysis changes no estimator, launches no new optimization and deletes no output. Motion accuracy and covariance consistency remain unmeasurable for this latest batch.
