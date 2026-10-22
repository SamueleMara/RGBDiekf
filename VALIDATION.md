> Latest user-request results: [LATEST_REQUEST_VALIDATION.md](LATEST_REQUEST_VALIDATION.md). Production reconstruction is **not accepted**: invented model-count priors were removed; the required model-selection measure is undefined. UKF progressive motion recovery still fails. Earlier experiments and running-status statements below are historical.

# Current v2 result — acceptance NOT passed

The final complete GPU suite is `parallel_masked_v2_20261020/tests_06/`:
**123 tests, 122 passed, one UKF motion-recovery failure, no errors/skips**,
120.22s. Exact snapshot SHA256:
`8d174e2a4a92a0bb53b66705ab30a9fc8f63601a0290854efcb6aa1bf7d261fb`.
All source-to-mask invariance, analytical/sigma/covariance references and the
new tiny coupled-prior test pass. The physical UKF reconstruction assertion
still fails; the separate six-DoF experiment also over-fragments both bodies.

Both real runs have completed their first full 328-frame, three-camera forward
pass and their OWN backward smoother. IEKF has accepted a new-group proposal
with two new points after actual backward discovery; other geometry edits were
rejected by the full objective. The full reverse sweep, chronological frontier,
final expanded-map forward pass and convergence remain incomplete/running.
Their current accepted counts come from checkpoints, not initializer proposal
fields. No complete reconstruction is claimed.

All persistent results are on the requested SSD under
`parallel_masked_v2_20261020/`. Current reports update each minute in `reports/`.
The exact group-count scoring counterexample is a remaining mathematical issue:
at N=160,945, copying identical trajectories from ten groups into eleven improves
the current MAP score by 917.14 without changing RGB-D predictions. No invented
regularizer or replacement objective has been inserted to conceal that issue.
The required physical recovery, controlled performance comparison and full
real-cycle acceptance therefore have not passed. Commands are in RUNNING.md.

The sections below retain historical attempts and their source/test snapshots;
the v2 updates at the end describe the actual corrections and failed gates.

> Current authority: `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md` (20 October).
> Earlier visibility-policy requirements and background-depth ordering below
> are historical, superseded behavior. See the v2 update at the end of this file.

# Historical validation through 19 October — superseded by v2

Status: **INCOMPLETE; mandatory full real cycles and convergence have not passed.**
All paths below are under:
`/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/parallel_validation_20261019`.

The historical 19 October suite is `tests_v13_hard_masks/test_results.json`: **99 passed,
zero failures/errors/skips**, actual CUDA available, 83.87 seconds. `junit.xml`,
`tests.log` and `source_snapshot.zip` retain cases and the exact tested source.
Source SHA256: `59e1f35030934b13692c14dfc5c796be67d507d02fee41b9ff8b6c5a6d61f3f6`.
The existing environment has no pytest;
these are executed standard-library unittest tests, not an assertion that
`pytest -q` ran. All 328 times / three cameras were scanned in the input test.

| Requirement | Executed evidence | Interpretation |
|---|---|---|
| Camera conventions, directory/metadata/ZIP loading, actual timing | test_camera, test_data_loading, test_real_dataset_smoke | All input images scanned; export/reprojection consistency, not independent physical calibration |
| Lie Exp/Log/Ad and composition | test_lie | Numerical references |
| IEKF F/G/H and fixed-prior Gaussian/Joseph update | test_iekf_math | Actual estimator vs finite differences / linear Gaussian reference; damping excluded from covariance |
| UKF sigma construction, augmented process and Woodbury | test_ukf_math | Own moments; coupled priors use joint sigma dimensions; batched/scalar operator parity |
| Each own backward smoother | test_smoothing | Means, diagonal and every selected cross-time block vs dense tiny Gaussian posterior |
| Late-point, shared-source and raw-noise coupling | test_covariance, test_fusion_and_visibility | Joint pushforward / Monte Carlo verification; source-factor representation, not full-scene joint marginals |
| Fusion, ownership, occlusion, missing depth, inverse transport | test_fusion_and_visibility, test_data_association | Point-only local likelihood and immutable raw IDs |
| Forward / own smoother / reverse / frontier / second forward | test_outer_loop | Both independent synthetic point-image integrations; **not real acceptance** |
| GPU/CPU math, deterministic reductions, subset-invariant projection | test_gpu_equivalence | Actual GPU; bitwise discrete projection invariant to point batch size |
| Immutable streaming caches and portable visual exports | test_frame_cache, test_visual_debug, test_resume_cli | Sparse sample normalization, mask bits, unchanged background score, true-path resume rejection |
| Original stochastic priors after resume; explicit numerical migration | test_outer_loop | Old posterior never reused as a prior; old score epoch archived, not silently accepted |
| Historical identical frame-0 input | tests_v7/current_operator_initialization.json | Pre-amendment only: both 160,943 points; not certification of the later hard-mask/GPU-backprojection implementation |
| Real-map UKF scalar / batched moments | tests_v4/ukf_sigma_benchmark.json | Same A/M/b in recorded case; concurrent operator timing, not whole-scene benchmark |
| Full real cycles and final products, both estimators | iekf/, ukf/ stage logs | **Pending**; previous partial loops/initialization are not completion |
| Real convergence and physical body accuracy | Full reports required | **Unverified**; no global or physical convergence claim |

Real input: 328 times, three cameras, 640x480, 25 Hz from
`metadata/frame_timestamps.csv:timestamp_s`; 62,210,415 foreground observations,
54,970,499 valid foreground depths. The camera metadata explicitly says
`CANDIDATE_NOT_VALIDATED`; timestamps derive from source indices/recording FPS,
not independent hardware synchronization. Noise values remain approved guesses.

`rotating_scene_diagnostic_v2/accuracy_diagnostic.json` preserves an additional
known rotating/translating point-scene experiment. Both filters completed the
small temporal cycle but produced seven groups and inaccurate moving-body
poses. The fixture also contains a stationary rear body; its per-group errors
are against the moving transform and must be interpreted accordingly. This is
an accuracy limitation, not a successful physical reconstruction test.

The scalar-score comparison across the old float32 BLAS projection and corrected
fixed-operation projection is **invalid**. Old checkpoints/logs/source snapshots
are preserved. Explicit migration logs both baselines, archives the old score
history, and starts score epoch 2. Current-epoch accepted changes must all be
checked against that one corrected operator. No tolerance was widened.

Additional unresolved model-selection issue: `reports/group_score_counterexample.json`
and AUDIT.md give an exact redundant-group counterexample for the current
explicit structural prior. Proper count priors do not themselves resolve
comparison of MAP densities across continuous trajectory dimensions. No new
prior strength, group-count cap or assumed calibration was introduced to hide it.
Full joint map-motion posterior marginals remain uncomputed; saved covariance is
conditional and the sparse source-pose dependencies are explicitly retained.

Reproduce the suite in the existing environment:

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B scripts/run_parallel_tests.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/tests_new'
```

Fresh independent entry commands and their current unresolved-model outcome are
in RUNNING.md. The actual resumed
experiment commands and exact source ZIPs/hashes are saved in the SSD experiment
root. `--max-outer 1` is a full-cycle experiment ceiling, **not convergence**.

## Mandatory hard foreground amendment

The amendment was read completely. The user's visibility decision remains
unset, as explicitly requested. The 0.99 confidence used by numerical fixtures
is a test input only, not an adopted production assumption. No mask morphology,
soft silhouette weight or guessed visibility probability is used.

| Amendment Section 7 criterion | Executed evidence | Current result |
|---|---|---|
| 1. Exact binary parsing | `test_exact_binary_values_and_invalid_greys` | PASS |
| 2. Required aligned masks | `test_missing_misaligned_and_nonbinary_dataset_masks_fail`, complete real input scan | PASS |
| 3. Foreground-only initialization/provenance | `test_frame_zero_backprojection_and_unsolved_source_domain`, GPU backprojection/reference | PASS on fixtures |
| 4. Foreground-only Kalman targets | Forged-background-target rejection, actual IEKF/UKF moment parity | PASS on fixtures |
| 5. Foreground ownership/count identity | Raw ownership guards and matched/unassigned identity | PASS on fixtures |
| 6. Visible exclusion; hidden/no-return preservation | `test_visible_background_is_infeasible_hidden_and_missing_are_retained` | PASS under explicit fixture decision |
| 7. Actual late backward point discovery | `test_real_backward_discovery_keeps_hidden_late_point_and_rejects_visible_background` | PASS on synthetic RGB-D; not real-scene completion |
| 8. Separate camera/time masks | Per-camera boundary test and six-view backward CPU/CUDA parity | PASS on fixtures |
| 9. Exact nearest mask sampling | Subpixel boundary regression; binary parser | PASS |
| 10. Each independent filter rejects infeasible nominal | `test_both_filter_candidates_respect_the_same_feasible_set`, own covariance tests | PASS on fixtures; no sigma clipping |
| 11. Background RGB cannot change inference | Background RGB perturbation, foreground-only initializer image, source/target/ownership guards | PASS on fixtures |
| 12. CUDA/CPU parity including reverse discovery | GPU backprojection/full covariance, matching, both reductions, six-view backward, background exclusion | PASS on actual GPU |
| 13. Both complete real cycles | Actual entry attempts stop before estimation | **NOT PASSED** |
| 14. Required mask/violation products | Actual export pixel-set regression and both synthetic full-cycle exports | Implemented/tested; **final real products pending** |

The complete real attempts are under the sibling directory
`parallel_hard_mask_preflight_20261019/`:

| Current real execution | IEKF | UKF |
|---|---|---|
| Images read and validated | 328 times x 3 cameras | 328 times x 3 cameras |
| Resolved acquisition timing | 25 Hz, scene timestamp CSV | Same |
| Approved P0/Q/R loaded | Yes | Yes |
| Status | INPUT_MODEL_UNRESOLVED | INPUT_MODEL_UNRESOLVED |
| Recorded preflight runtime | 49.42 s | 48.16 s |
| Completed constrained filter/discovery/frontier/new-map cycles | None | None |
| Final constrained reconstruction accepted | No | No |

Both fail explicitly with `MISSING_VISIBILITY_DECISION_MODEL`. These attempts
precede the final backprojection/cache refinements and are not experimental
validation of those later stages. The current component suite exercises their
actual GPU operators and scans the complete real input; a production cycle still
requires the user's rule. The old unconstrained workers were interrupted and all
checkpoint/source/log files preserved. No old reconstruction is relabeled as a
hard-mask result.

Additional unresolved requirements remain explicit: full real convergence and
physical accuracy; actual full-scene output coverage and accepted-mean covariance;
the documented redundant-group MAP-density issue; and consistency of candidate
extrinsics/masks at frame zero. Full joint map-motion marginals are not computed
and conditional Gaussian uncertainty does not represent hard-set truncation.

## Current V2 validation — 20 October 2026

All paths in this section are under the SSD result root
`parallel_masked_v2_20261020/`. V2 supersedes the old missing-policy outcome and
older background-depth ordering tests. Current status: **NOT ACCEPTED**.

Complete actual-GPU suite `tests_02/test_results.json`: **120 tests, 119 passed,
one failure, zero errors and zero skips**, 131.57 seconds. Source snapshot/hash:
`3ea61873080ba1dad7f5cb4a0b22a5fe6704f6574b462a2d585ef04ad54721ed`.
JUnit and `MASK_PORT_VALIDATION.json` retain actual assertion cases, branch,
device and raw source IDs. The full input test reread every frame and camera.
The sole failure is the UKF branch of
`test_both_independent_full_forward_reverse_frontier_forward_cycles`: accepted
maximum translation is 0 m, below the unchanged >0.05 m assertion. IEKF passes
that synthetic motion/cycle test. No test was skipped or weakened to hide this.

`ukf_fixture_failure/motion_recovery_diagnostic.json` preserves the failing
fixture, actual UKF traces, map, own covariance, rendered point images and HTML.
At frames 1,2,3 it receives 960,864,800 raw matched factors respectively, with
zero active factors after full-map sigma visibility deferral. This is a synthetic
diagnostic, not a real-scene reconstruction, even though its temporal stages run.

`kernel_benchmark.json` evaluates the actual 160,943-point frame-0 map (nine
bootstrap groups), real frame1/camera0, with the authorized propagated P0/Q.
It finds 54,456 raw matches; group0 UKF retains 238 factors and defers 33,985.
Measured Torch peak: 133,715,968 bytes allocated, 153,092,096 reserved;
usable device 8,444,575,744 bytes, RSS peak 1,525,600,256 bytes. It explicitly
uses the genuine prior, rather than a tiny covariance chosen to make sigma
support pass. Timings include startup/concurrent activity and are operator
measurements, not a fair whole-pipeline performance comparison.

| V2 M8 / implementation Section15 gate | Evidence | Result |
|---|---|---|
| Exact P=MI and used validity | Branch/device-specific preparation tests; explicit raw-valid directory/ZIP/cache test | PASS |
| Binary masks / dimensions / cameras / automatic timing | hard-foreground, loader and complete-scene integrity tests | PASS |
| Arbitrary background RGB and depth invariance | Both filters, CPU/CUDA; bit-identical targets, outputs, score, initialization and discovery | PASS |
| Invalid foreground depth invariance with fixed Vraw | Explicit validity fixture, zero/NaN/Inf/large/negative replacements | PASS |
| No observation leakage / unique raw IDs | Foreground source/target guards, CSR ownership identity and fusion tests | PASS on tested fixtures; full real final-map audit pending |
| Empty masks | No initialized point, target factor, update or discovery in both branches/devices | PASS |
| Conditional M4/M7 output and analytic derivatives | IEKF FD / full-noise math tests on CPU/CUDA; fixed RGB zero pose rows | PASS |
| UKF moments / augmented process / own smoother | Linear Kalman, full-noise dense Woodbury, scalar/batch sigma and smoother references | PASS numerically |
| Stable z-buffer / foreground occlusion / off-camera | Explicit point-ID ties, hidden FG and missing-depth regressions | PASS |
| Mask-negative trial rejection without policy knob | Declared-premise feasibility, each filter rejection and transaction rollback | PASS |
| Late inverse transport / once-only unresolved | Reverse matching, source dedup and branch/device once-visible tests | PASS on fixtures |
| Optional photometric interpolation | Baseline disabled; no additional warped-RGB factor | OFF |
| Cache pose/map trials and rollback | Full/incremental J parity; fast/full feasibility; cached/fresh reverse projector comparisons | PASS |
| Nonzero motion and expanded-map second pass | Existing synthetic complete-cycle assertions | IEKF PASS; UKF FAIL |
| Full real F -> own smoother -> discovery -> frontier -> second F | `full_scene_02/` actual worker logs | PENDING, not acceptance |
| No prohibited renderer / normals / hybrid / catalogue | Production import graph plus point-only source audit | PASS |

The first real v2 trial `full_scene_01/` was interrupted only by verified worker
PID, preserving every file, before replacing repeated feasibility/projection work
with mathematically equivalent cache queries. The reason, source, command and
exit signal are saved. It is not a completed reconstruction.

The fresh `full_scene_02/` experiments request one complete outer cycle each on
all 328 times and three cameras (25Hz); `--max-outer 1` is a cycle ceiling, not
convergence. Both input audits and 160,943-point initial maps are recorded.
Their workers are independent processes sharing GPU/SSD, so their elapsed times
must not be advertised as a controlled speed ranking. Real acceptance will be
reported from their actual stage traces and products, never from initialization.

Equation traceability: M1/M2 -> dataloader, foreground compaction, geometry/CSR;
M3/M5 -> shared projection/foreground trial check; M4/M6/M7 -> observation_model,
iekf.measurement; M8 -> iekf.filter; M9/M10 -> ukf.measurement/filter;
M11 -> projection_cuda.inverse_transport_unmatched and backward_cuda/discovery;
A1--A8 -> iekf dynamics/filter/smoother plus shared outer_loop;
B1--B7 -> ukf dynamics/sigma_points/measurement/filter/smoother;
D1--D4 -> common geometry, raw cache, projector/matcher/source factors;
D5/D6 -> independent CUDA information/sigma reductions;
D7 -> common score/score_cache/transactional controller;
D8--D10 -> config, timing, diagnostics, outputs and acceptance reporting.

Latest complete suite after exact UKF reuse: `tests_03/test_results.json`,
**119/120 passed, one unchanged UKF motion-recovery failure, zero errors/skips**,
103.97s, source hash
`f1ee896914de6e27a106566a5b25bd1b8b95b3c547bf968058b32094caf68a93`.
`kernel_benchmark_exact_reuse.json` compares the final operator against the full
reference on the same actual map/frame/prior: A, M and b differences are exactly
zero. The optimized sigma stage measures 1.319s, full reference 2.817s in that
check; neither is a controlled full-pipeline comparison.

IEKF continues under `full_scene_02/iekf`. Only the slow UKF worker from that
trial was preserved/interrupted; its verified-PID record states the reason.
The latest optimized independent UKF runs under `full_scene_03/ukf`, from the
original priors and identical common code/data. Production IEKF/common modules
have not changed since its snapshot. Both runs remain subject to the real-cycle
and physical convergence gates. Current machine-readable status and the
comparison table are in `reports/v2_current_acceptance.{json,md}`. Those reports
use actual stage files and verified live command lines, not an initialization
flag as evidence of reconstruction.

Final fixed-target/GPU foreground suite: `tests_04/test_results.json`, **121 tests,
120 passed, one failing UKF motion-recovery subtest, no errors/skips**, 99.96s.
Snapshot SHA256:
`64269aeeca12f99c6c0e69b258443ac9fe118fc28e6b811a06980136d80ea70a`.
The new fixed-target depth-noise/dense-UKF regression passes. The unchanged
translation assertion still fails: accepted maximum translation is approximately
0.000182 m versus the required >0.05 m. The earlier all-zero diagnostic is
historical; the final factor/trajectory replay is in `ukf_fixture_failure_final/`.
It would be incorrect to describe this remaining failure as a missing visibility
configuration or to substitute a tiny production covariance to make it pass.

Final real-frame kernel check `kernel_benchmark_device_fg.json` reports zero
A/M/b difference and equal active counts against the corrected full reference.
The sigma stage measures 0.1263s in that check; full reference 0.5518s. These are
individual operator timings under recorded activity, not a claim that one full
filter dominates the other. Final full real UKF is `full_scene_04/ukf`; IEKF
continues at `full_scene_02/iekf` with unchanged IEKF/common runtime modules.

The actual IEKF first full forward trace contains all 328 times and three
cameras; its own RTS file has 327 analytical transitions, no UKF sigma cross
array, and finite 328x108x108 conditional smoothed covariance. Frame327's filter
used 66,895 raw matched factors, maximum estimated group translation about
0.2341m, minimum filtered-covariance eigenvalue 3.86e-12. The smoothed trajectory
passed the common full objective, reducing 679,629,355.76 to 596,955,906.99.
This is real filtering/smoothing evidence, **not** completed discovery/frontier/
second-forward acceptance or an independently calibrated accuracy estimate.

`rotating_six_dof_diagnostic/accuracy_diagnostic.json` records the required
additional nonzero-six-DoF fixture. IEKF ends with 1,092 points/eight groups;
UKF with 1,024 points/seven groups, for a scene containing two physical bodies.
Their minimum translation errors against the moving transform are about 0.0586m
and 0.1045m respectively. The stationary body must not be expected to follow that
moving transform; these minima are descriptive, not a correspondence-verified
whole-scene accuracy metric. Extra groups and poor moving-body recovery fail
physical reconstruction acceptance. GT was read only for evaluation after each
inference completed.

The cache point-ID row-reordering correction is included in
`tests_05/test_results.json`: **122 tests, 121 passed, one unchanged UKF motion
failure, no errors/skips**, 105.24s; source snapshot SHA256
`c732430a47250b86fd4017a95ab2365aa616ac6a64f598219d8c9d07d5163083`.
A final covariance audit removed tolerance-based independence classification:
any nonzero cross-body covariance selects joint sigma points, even when its
absolute scale is below machine epsilon in unrelated units. The tiny-coupled
prior test and all UKF math tests pass; its complete suite is recorded in
`tests_06/` when finished. Existing real workers' conditional block priors are
exactly separable, so this correction does not change their operator on those
inputs; their original source snapshots are preserved.

`reports/first_real_smoother_audit.json` verifies both actual first 328-frame
smoothed proposals. Each has finite symmetric PSD 328x108x108 covariance, with
zero minimum eigenvalue from fixed reference gauges. IEKF saves 327 analytical
transitions and no sigma cross array; UKF saves 327 sigma cross arrays and no
analytical transition array. These are their own conditional proposals, not an
assertion that every proposal is the final accepted trajectory or calibrated
physical uncertainty.

`reports/model_selection_counterexample.json` is an exact algebraic unresolved
model-selection check using real timing/noise and N=160,945. With physically
identical RGB-D predictions and trajectories, splitting ten groups into eleven
changes the current score by **-917.14**; twelve into thirteen by **-3374.39**.
The per-group zero-residual motion NLL is -16,257.02. Thus the declared proper
count/membership priors do not prevent a continuous complete-data MAP density
from rewarding redundant motion labels. This is not solved by the v2 mask
amendment. No group cap, invented prior strength, dropped normalization constant,
or replacement evidence objective is hidden in the implementation. A justified
model-selection measure/structural model remains unresolved. Together with the
known physical recovery failures and incomplete real temporal cycles, it blocks
an honest claim that every acceptance criterion is satisfied.

Both real branches have now finished their full first forward and own backward
smoother. IEKF has performed actual backward matching and a full chosen-filter
replay of a new-group hypothesis; rejected geometry proposals are preserved.
Do not confuse a proposal's `groups` field with the accepted map group count:
only checkpoints/accepted transactions define the latter. Real discovery,
chronological frontier, final expanded-map pass and convergence are still
running/pending. The verified-worker monitor updates the current report each
minute and saves a final failed-or-passed assessment only after workers exit.
The equal-budget controlled performance comparison is also pending; concurrent
elapsed times cannot establish dominance.

Final full-suite result after retaining all nonzero cross-body terms:
`tests_06/test_results.json`: **123 tests, 122 passed, one UKF physical motion
failure, no errors/skips**, 120.22s. Snapshot SHA256:
`8d174e2a4a92a0bb53b66705ab30a9fc8f63601a0290854efcb6aa1bf7d261fb`.
`reports/shared_input_code_comparison.json` verifies byte-identical 32 shared
modules in the actual experiment snapshots, identical scene/calibration/timing/
mask/raw fingerprints, identical noise/prior provenance and identical initial
XYZ/RGB/group/source digest. It explicitly does not claim that downstream
hypotheses or concurrent performance constitute a controlled comparison.

Current remaining failure evidence: the corrected UKF drawer run retains useful
factors for one group (25, 184, 520 at frames1..3), but all factors for the other
three provisional groups defer at these frames. Their motions are not recovered;
the final accepted translation remains about 0.000182m. This is an unresolved
reconstruction failure in the current conditional hypothesis/visibility family,
not a missing acquisition input, an unspecified visibility knob, or a proof
that every possible implementation must fail. No failed assertion is removed.
