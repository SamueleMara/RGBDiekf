# Step 2B-B1: tiny batched GEMMs as broadcast contractions (opt-in)

| Item | Value |
|---|---|
| Change | `pointwise_rgbd/small_contractions.py`, with `matvec`, `rmatvec`, `congruence` and `gram`. These are used by `Linearization.apply_H`/`apply_HT`/`block_fisher`, `GroupBasis.expand`/`reduce`/`blocks` and the two `Λ e` products of `image_iekf.linear_system` |
| Flag | `IEKFConfig.small_contractions` (default **False**); CLI `tools/run_rgbd_image_iekf.py --small-contractions`. Enabled per frame update by `frame_step`, and the flag is restored afterwards |
| Default path | With the flag off, each helper evaluates the original expression. A fresh flag-off Phase 1 reproduces the saved v2 production `state.npy` **bitwise** for frames 0–9. All 35 existing tests (`test_image_iekf`, `test_image_renderer`, `test_coherence_parity`) pass unchanged |
| With the flag on | The same sums are evaluated in a different order: equal to float64 round-off, not bitwise. `GroupBasis.blocks` uses `G_i = blockdiag(pos_i, pos_i)`, so four `[N,6,6]` congruences replace the `[N,6,12]` temporary |
| Tests | `tests/test_small_contractions.py` (3 tests, all pass): flag off is bitwise equal to the original expressions and flag on is within 1e-14 (CPU and CUDA); G/Gᵀ/blocks/H/Hᵀ/Fisher on vs off within 1e-13, adjoint preserved; one fixture frame update on vs off within 1e-9 with the same accept sequence |
| Pre-change sources | `rgbd_point_renderer.py` `6b8ed71e…`, `supernodes.py` `10e6e2d2…`, `image_iekf.py` `48a4ff25…` |

## Equal-input benchmark (blade, same saved prior and images, 3 alternating repeats)

Command: `tools/benchmark_small_contractions.py --frames 1 10 30 --repeats 3 --large-scene laptop_10211 --memory-fraction 0.45`. Output: [phase1_b1_small_contractions_benchmark.json](phase1_b1_small_contractions_benchmark.json).

| Frame | Wall time, off (s) | Wall time, on (s) | Median speedup | Peak allocated, off / on (MB) | Same accept + PCG sequence | State correction difference (relative) | Posterior blocks (relative) |
|---|---|---|---|---|---|---|---|
| 1 | 10.4, 11.0, 8.5 | 6.3, 8.4, 8.0 | **1.31×** | 2,003 / 1,998 | yes | 2.3e-2 (see below) | 1.6e-2 |
| 10 | 8.9, 6.2, 12.7 | 3.2, 4.1, 10.8 | **2.19×** | 1,867 / 1,863 | yes | 1.7e-11 | 7.8e-12 |
| 30 | 13.4, 7.4, 8.9 | 4.8, 3.4, 4.3 | **2.06×** | 1,819 / 1,825 | yes | 8.7e-12 | 9.0e-13 |

Times are contended: a user job held the GPU at 85–93% throughout, which explains the spread.

Operator discrepancy at the flag-off final linearization, maximum over the three frames (relative): `H v` 9e-17, `Hᵀ a` 1.8e-16, `block_fisher` 4e-17, `GᵀMG` blocks 1.3e-18, `b_g` 4.4e-16, `A_g z` 1e-16, preconditioner 1.4e-14.

Peak memory is unchanged. The large scene (`laptop_10211`, 146k points) still runs **out of memory with both variants** under the 3.6 GB cap, with requests of 56 MB (on) and 32 MB (off). Removing the `[N,6,12]` temporary does not lower the peak enough, so the large scene is still not profiled.

## Frame 1 and the full sequence: Phase 1 amplifies round-off

At frame 1 both variants are bitwise deterministic run to run (`reports/diagnostics/b1_frame1_sensitivity.json`). They first differ at Newton iteration 2, the first iteration whose PCG stops at the 30-iteration cap: relative residual 3e-2, with eigenvalues of the reduced system spanning 0.16 to 3e8. A 1e-16 operator difference becomes a 2.3% difference in the correction (maximum 2.0 mm position, 0.048 m/s velocity). The moving slide differs by 0.006 mm.

Over 10 frames of the full forward Phase 1 (`reports/diagnostics/phase1_roundoff_sensitivity.json`):

| Pair | Frame 1: median / p90 / max (mm) | Frame 9: median / p90 / max (mm) |
|---|---|---|
| off vs on | 0.006 / 0.03 / 2.0 | 1.5 / 9.9 / 1,861 |
| off vs off with the prior × (1 + 1e-15·N(0,1)) | 0.46 / 1.6 / 11.9 | 2.4 / 12.5 / 188 |
| off-perturbed vs on | 0.46 / 1.6 / 11.9 | 2.2 / 12.1 / 1,709 |

A round-off-sized perturbation of the **unchanged** production path diverges as much as B1 does, and from frame 1 more. The sequence sensitivity is a property of the current Phase 1, not an error introduced by B1.

**Full 60-frame forward run with the flag on.** Output is in `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/b1_small_contractions_blade_first_20261025`, with `--stop-after coherence`; comparison in `reports/diagnostics/b1_full_run_vs_v2.json`.

| | v2 production (off) | B1 (on) |
|---|---|---|
| Phase-1 wall time (not equal-input: different trajectories and contention) | 378 s (6.09 s per frame, median) | 306 s (4.95 s per frame, median) |
| Position difference from v2 at frame 59 | — | median 7.4 mm, p90 67 mm, 14,901 points over 1 cm |
| Points displaced > 1 m (the true slide is about 0.84 m) | 16 (max 1.47 m) | 21 (max 14.5 m) |
| Coherence seed → final | 183 → 192/192 moving | 5 → 13 moving, 13 static, 166 ambiguous |
| Coherence partition (192 nodes) | identical | identical |

**The 192-node coherence result does not agree.** The difference cannot be attributed to B1. The baseline's own 192/192 outcome is not stable under a 1e-15 perturbation of its input, so neither outcome is a reproducible property of the scene. This also qualifies the Step-1 finding: the v2 trajectories that the pinned extractor turns into 192/192 are one round-off realization of an unstable filter.

## Status

* **Operator verified** (Step 2A). B1 operators agree to ≤ 1.4e-14.
* **Speedup verified for the same completed frame update**: 1.31× / 2.19× / 2.06× median at frames 1 / 10 / 30, with no extra memory. **Not verified for full-sequence equivalence or coherence agreement.** The 60-frame trajectory and the coherence labels differ, and the experiment above shows that any round-off-level change to the current Phase 1 does the same. B1 stays opt-in (default off). It should not become the default until Phase 1 is numerically stable over a sequence.
* **Moving-motion lag diagnosed** (Step 2A).

## Not done, and why

* B2 (point reduction) and B3 (active set) were not started. The spec allows one optimization at a time, and the round-off finding has to be resolved first: no reduced representation can be judged against an unstable baseline.
* Host-synchronization gaps (about 40% GPU idle) and the large-scene memory peak are the next measured bottlenecks. They were not touched.
* The 40-scene batch was not run. Its gate (operators and benchmark correct on small cases) is met for single frame updates, but the sequence-level instability makes batch coherence outcomes not reproducible at round-off level.
