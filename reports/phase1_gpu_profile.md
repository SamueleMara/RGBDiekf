# Step 2A-P: Phase-1 GPU profile (baseline, before any optimization)

| Item | Value |
|---|---|
| Tool | `tools/profile_phase1_rgbd.py --run <v2 run> --scene blade_103706 --frames 1 10 30 --kernel-table --large-scene laptop_10211 --memory-fraction 0.45` (read-only; writes only `reports/phase1_gpu_profile.{csv,json}` and `reports/phase1_gpu_kernels.txt`) |
| Method | Each instrumented call is timed with `torch.cuda.Event` and a synchronize. One warm-up frame. Fixed inputs: saved prior = `process6.predict` of the saved posterior of frame k−1, production `IEKFConfig()` |
| Kernel table | `torch.profiler` (CPU + CUDA) over the frame-30 update |
| Environment | torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, TF32 off, 16 CPU threads. RTX 4070 Laptop 8 GB, driver 580.178.04 |
| Contention | A user process (PID 1599382) holds 3.8 GB at 84–93% GPU utilization throughout, so absolute times are contended. The per-stage split is the result. This process was capped at 45% of VRAM (3.6 GB) |
| Note on percentages | Timers nest. "Steihaug-CG" includes the `H`, `Hᵀ`, `W`, `G`, `Gᵀ` calls it makes, and "linear system assembly" includes `block_fisher`. Percentages of the same nesting level add up; the `H:`/`G:` rows are a sub-breakdown |

## Counters (blade, 34,369 points, 537 groups, 8 cameras, 640×480)

| Frame | Frame update (ms) | Newton iters (acc/rej) | PCG iterations per Newton iteration | Point–pixel contributions | Occlusion candidates (occluded) | Contributors per covered pixel (mean/max) | Peak allocated / reserved (MB) |
|---|---|---|---|---|---|---|---|
| 1 | 8,741 | 10 (8/2) | 9, 11, 30, 30, 30, 30, 8, 10, 2, 2 | 961,032 | 1,873,188 (912,156) | 25.6 / 94 | 2,052 / 2,214 |
| 10 | 6,196 | 10 (8/2) | 9, 12, 14, 12, 12, 9, 2, 4, 1, 1 | 885,985 | 1,845,905 (959,920) | 22.4 / 113 | 1,920 / 2,440 |
| 30 | 6,515 | 10 (9/1) | 9, 8, 11, 11, 11, 22, 11, 2, 1, 1 | 825,036 | 1,846,644 (1,021,608) | 16.7 / 130 | 1,870 / 2,440 |

Image decode and GPU upload of the 8 RGB-D frames (`levels`) take 10–39 ms per frame, which is negligible. Per camera, about 21k points project, and about 49–55% of the footprint candidates are rejected as occluded.

## Per-stage breakdown (CUDA-event time)

| Stage | Frame 1 ms (%) | Frame 10 ms (%) | Frame 30 ms (%) | Calls (f10) | Median / p95 per call (f10) |
|---|---|---|---|---|---|
| **Steihaug-CG solve** (incl. operator applications) | 3,942 (45.1) | 1,659 (26.8) | 2,427 (37.3) | 10 | 194 / 269 ms |
| **Linear system assembly** (b, Fisher, preconditioner) | 1,988 (22.7) | 1,832 (29.6) | 1,831 (28.1) | 10 | 183 / 184 ms |
| ↳ `H: block_fisher` (per-point 3×3 HᵀWH) | 2,093 | 1,904 | 1,740 | 11 | 160 / 231 ms |
| ↳ group preconditioner (GᵀMG + block Cholesky) | 142 | 144 | 159 | 10 | 14.4 ms |
| **Render, all 8 cameras** (11 = 1 + 10 trial renders) | 1,465 (16.8) | 1,504 (24.3) | 1,259 (19.3) | 11 | 106 / 275 ms |
| ↳ projection + footprint candidates | 329 | 356 | 306 | 88 | 3.5 ms per camera |
| **Robust terms + IRLS weights** | 358 (4.1) | 286 (4.6) | 281 (4.3) | 11 | 26 ms |
| **Posterior** (`rigeuler_block_information`) | 424 (4.9) | 430 (6.9) | 236 (3.6) | 1 | — |
| Sub-breakdown: `H v` (`apply_H`) | 1,457 | 644 | 917 | 92 | 6.9 ms |
| Sub-breakdown: `Hᵀ w` (`apply_HT`) | 1,560 | 721 | 981 | 102 | 6.8 ms |
| Sub-breakdown: `W` (`apply_W`) | 729 | 347 | 411 | 91 | 3.8 ms |
| Sub-breakdown: `G z` / `Gᵀ f` | 66 / 150 | 34 / 72 | 38 / 112 | 101 / 101 | 0.31 / 0.71 ms |
| **Frame update total** | **8,741** | **6,196** | **6,515** | | |

The `H v` and `Hᵀ w` counts are about one per PCG iteration plus one per Newton iteration (the right-hand side and the predicted reduction). Frame 1 is the most expensive because four Newton iterations hit the 30-iteration PCG cap.

## Kernel level (frame 30, `torch.profiler`, [phase1_gpu_kernels.txt](phase1_gpu_kernels.txt))

| Kernel group | CUDA time | Share of CUDA time | Calls | Source |
|---|---|---|---|---|
| `aten::bmm`, float64 batched GEMM (CUTLASS `d884gemm` 1.72 s at 11.2 ms per call, plus `gemv` 0.86 s) | 2.70 s | **71.6%** | 2,485 | Tiny per-contribution and per-point products: `Jᵀ P J` in `block_fisher`, `J v` / `Jᵀ a` in `apply_H`/`apply_HT`, `gᵀ M g` in `GroupBasis.blocks`, `Λ e` |
| Elementwise `mul`/`copy`/`add`/`div` | ≈ 0.55 s | ≈ 15% | ~25k | everywhere |
| Indexing (`index`, `index_put_`) | 0.15 s | 4% | ~15k | gathers into contributions |
| `segment_reduce` (`_segsum`) | 0.11 s CUDA (3.47 s CPU total, mostly waits) | 1.5% | 2,449 | per-pixel / per-point sums |
| Sort / argsort | 0.035 s | 0.9% | 352 | front selection, segments |

Total CUDA kernel time was 3.78 s against 6.5 s of event time for the same update. The GPU is therefore idle about 40% of the update. The gaps come from host synchronizations: 2,449 `segment_reduce` calls (the `lengths` argument), 2,520 `nonzero` calls, the `float(...)` scalars of the trust-region loop, and this profiler's own per-call synchronize.

**Microbenchmark** of the same contractions, same shapes, float64, this GPU (scratch script; the results are equal to round-off):

| Contraction | cuBLAS batched GEMM | Broadcast multiply + sum | Speedup | Relative difference |
|---|---|---|---|---|
| `Jᵀ P J`, 130k × (3×3) | 24.5 ms | 0.78 ms | 31× | 1.1e-16 |
| `gᵀ M g`, 34k × (6×12) | 12.2 ms | 4.6 ms | 2.7× | 1.9e-16 |
| `J v`, 130k × (5×3) | 0.53 ms | 0.22 ms | 2.4× | 8.5e-17 |
| `Jᵀ a` | 0.58 ms | 0.17 ms | 3.5× | 1.1e-16 |
| `Λ e`, 34k × (6×6) | 0.14 ms | 0.08 ms | 1.6× | 1.2e-16 |

## Larger scene: `laptop_10211` (145,973 points, 2,280 groups, frame 1 from the anchor)

From `tools/diagnostics/large_scene_memory.py`, written to `reports/diagnostics/large_scene_memory_laptop_10211.json`:

| Item | Value |
|---|---|
| Point–pixel contributions (8 cameras) | 5,568,268, which is 6.3× the blade for 4.2× the points |
| Render only, peak allocated | 1,611 MB. One linearization's Jacobians take 637 MB |
| Frame update under the 3.6 GB cap | **Out of memory** (162 MB request at a 3,318 MB peak) in `GroupBasis.blocks` → `g.transpose(1,2)@M@g` → `_segsum`. It materializes per-point `[N,6,12]` and `[N,12,12]` float64 temporaries while both the accepted and the trial linearizations are alive |
| Timing | Not measured. It cannot be run without exceeding the memory left by the other user's job |

## Ranked bottlenecks (measured)

1. **Tiny float64 batched GEMMs: 71.6% of CUDA kernel time.** The largest single item is `block_fisher` at 27–31% of the frame update. cuBLAS batched GEMM is about 30× slower than an equivalent elementwise contraction at these shapes.
2. **Host-synchronization gaps: about 40% of the update with the GPU idle.** Causes are `segment_reduce` with host `lengths`, `nonzero`, and scalar reads in the loop.
3. **Steihaug-CG operator applications: 27–45% of the update.** About 7 ms per `H v` and per `Hᵀ w` (8 cameras), which is mostly bottleneck 1 inside them.
4. **Rendering: 17–24%.** 11 renders per frame (1 + 10 trials); candidate generation is about 3.5 ms per camera.
5. **Memory at scale.** The per-point `[N,12,12]` group-block temporaries cause the 146k-point OOM.

Robust terms (4–5%), the block-information posterior (4–7%), `G`/`Gᵀ` (≤ 2.5%) and image I/O (< 1%) are not worth optimizing first.

**B1 choice, by measured priority:** replace the tiny batched GEMMs (item 1) with exact broadcast contractions, opt-in behind a flag. Group blocks are computed from the structured `[N,3,6]` lever block, which also removes the `[N,6,12]` temporary from item 5. Items 2–4 are left for later, separate changes.
