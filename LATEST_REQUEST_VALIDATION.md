# Latest request: implementation and measured failures — 20 October 2026

**NOT COMPLETE. Neither real reconstruction is accepted.** This report supersedes earlier running/completion statements in the repository. Existing experiment files were preserved; no existing worker was interrupted. No reconstruction workers were running when checked.

Changes: UKF now retains nominal valid foreground raw associations throughout conditional nonlinear quadrature. Raster-bin changes do not delete factors. All 25 sigma outputs use M4 and full RGB-D covariance in M9/M10; nonpositive/nonfinite sigma depth raises an explicit domain error instead of clipping or dropping data. Nominal hard-mask feasibility and outer foreground occlusion tests remain separate. This follows the latest execution request Section 2; it changes the earlier M5/B3 sigma-visibility deferral behavior. It is a conditional measurement approximation, not marginalization of uncertain visibility.

Defaults now stream one original decoded observation in RAM, store sparse score/ownership working arrays in SQLite, keep one atomically replaced checkpoint payload, and export compact maps/trajectories/covariance summaries/metrics. No copied raw frames, per-frame image/factor/step dumps, repeated source ZIPs or checkpoint generations. Detailed historical outputs require explicit `--debug-artifacts`. Resume refuses to terminate a live worker. Source provenance is rechecked against real valid foreground before final export.

Removed production inventions: `p(B)=1/[B(B+1)]`, `N log B`, and `p(N)=1/[(N+1)(N+2)]`. Those old numerical rules exist only in the explicitly labeled regression fixture, never as production defaults. The previous nine-group result was a fitted initializer result, not a literal hardcoded nine, but its model-selection score was unjustified. Production no longer uses it.

The exact mathematical gap is theory C3's symbolic `-log p(G,A)` plus M6/A7/D7 and implementation Sections 5/10: the normalized model-selection law and reference measure across differing point/group/trajectory dimensions are not supplied. Approved P0/Q/R do not define this operation. For the removed score, a physically identical split from 10 to 11 groups decreases J by **917.3332** on the actual 328-time clock and 160,943 points. Replacing it with another hand-selected penalty, group cap or hidden evidence approximation would violate the request. `rgbd_common/model_contract.py` reports `UNSPECIFIED_MODEL_SELECTION`; no new visibility/noise/timing parameter is requested.

| Measured real result | IEKF | UKF |
|---|---:|---:|
| Validated times / cameras | 328 / 3 | 328 / 3 |
| Timing | 25 Hz, scene timestamp CSV | same |
| Production attempt runtime (seconds) | 52.988 | 51.254 |
| Status | mathematical specification gap | same |
| Actual forward / smoother / reverse / frontier / second-forward passes | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 / 0 |
| Final SOLVED / UNSOLVED, groups, motion | not computed | not computed |

The separate CUDA frame-0 geometry component fuses **160,943 points**, 56,194,820 host map bytes, in 1.144s. CUDA peak allocated: 112,009,216 bytes. This is initialization only, with no inferred physical grouping; it is not either production reconstruction or a full-cycle memory benchmark. Input has 54,970,499 valid foreground observations. Calibration metadata says `CANDIDATE_NOT_VALIDATED`; timestamps use recorded indices/FPS, with hardware synchronization not independently validated. No GT motion/part labels entered estimation. The object-only/no-external-occluder premise is absent, so M3 negative-mask visibility remains undetermined, not artificially certified.

Numerical suite: **126/127 passed; 1 failure, 0 errors, 0 skips**, actual RTX 4070 Laptop CUDA, 96.605s. Runner: installed standard-library unittest. No pytest installation or claimed pytest run. Source SHA256: `d859ae16df3751d6ba50782c21cf34a011227c607afb67d1c9bd467b105cc7d3`. The unchanged failing assertion is UKF progressive motion greater than 0.05 m.

| Synthetic diagnostic (not real reconstruction) | IEKF | UKF |
|---|---:|---:|
| Conditional measured-correspondence 6-DoF translation error (m) | 0.000918835 | 0.002236213 |
| Corresponding rotation error (rad) | 0.000614429 | 0.000343296 |
| Progressive accepted maximum translation (m) | 0.128210338 | 0.000166077 |
| SOLVED initial -> final | 1024 -> 1085 | 1024 -> 1024 |
| UNSOLVED raw observations | 253 | 448 |
| Groups (fixture truth: 2) | 7 — FAIL | 1 — FAIL |
| Runtime (seconds) | 4.006 | 8.655 |

The progressive diagnostics intentionally use the historical TEST-ONLY structural model, to preserve the original failing assertion and expose the remaining defect. They do not supply the missing production mathematics. UKF retains all 960 frame-1 matched factors after the fix, yet associates incorrectly/collapses to one almost stationary group. Its conditional filter math test passes; progressive motion recovery and geometry growth do not. IEKF also over-fragments; scalar motion magnitude alone is not correct physical reconstruction.

| Latest request validation item | Evidence / acceptance |
|---|---|
| 1. Rigid motion recovery | Conditional 6-DoF test passes both; progressive UKF FAIL; real not established |
| 2. Common motion grouping without redundant fragments | FAIL both progressive fixtures; undefined production model-selection term |
| 3. Foreground-only observations | CPU/CUDA M1 invariance, raw provenance, background/invalid perturbation tests PASS; no real estimated map accepted |
| 4. Forward/reverse transport | Relative SE(3) and bounded GPU/reference tests PASS; no real reverse sweep |
| 5. Both covariance methods | Analytical, Joseph, fixed-prior, own RTS; sigma mean/covariance, Woodbury/dense S, own unscented cross-smoother and shared-source tests PASS. Conditional, not joint-map marginals |
| 6. Real SOLVED growth through backward discovery | FAIL/not completed; synthetic IEKF +61, UKF +0 |
| 7. Full second forward with expanded geometry | FAIL/not completed on real data; synthetic IEKF only expands |
| 8. All frames/cameras under one defined objective | Input integrity PASS; reconstruction FAIL because model-selection objective is unspecified |
| 9. Sparse CUDA and practical memory | Component parity PASS; measured frame-0 allocation above. Full reconstruction runtime/memory NOT validated |

Equation mapping: M1/M2 -> dataloader, foreground, geometry; M3/M5 -> foreground and matching_cuda; M4/M6--M8 and A2--A5 -> iekf dynamics/measurement/filter/smoother; M9/M10 and B2--B4 -> ukf sigma_points/dynamics/measurement/filter/smoother; M11/D4 -> source_factors and map_uncertainty; D1--D6 -> shared Torch CUDA/projector/matcher/backward kernels; C3/D7 -> explicit unresolved model-selection contract. Foreground assertion records are in `tests_final/MASK_PORT_VALIDATION.json`; all pass/fail acceptance gates are in `acceptance/comparison.json`.

Results: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/parallel_request_20261020_compact`. Reproduction commands and expected current failure are in RUNNING.md. Source hashes are saved without duplicating source archives. Actual real-attempt source hashes are recorded separately from the later final-test hash; output-only/checkpoint validation changed afterward. No fair full-scene IEKF/UKF accuracy or performance comparison is available.
