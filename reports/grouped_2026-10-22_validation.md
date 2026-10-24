# Grouped RGB-D execution audit — 22 October 2026

**Incomplete. Stopped at mandatory Gate B.** The isolated grouped Phase 1 exists
and has numerical and bounded real-data evidence. The joint-boundary regression
fails. Coherence, shared-rigid Phase 2, joint reduction, the complete 328×3 run,
and full-scene forward/reverse acceptance have not been implemented/validated.
Point recovery remains paused. No axis or physical moving support is invented.

The controlling instruction in Section 12 is: **“If a gate fails, stop at that
gate and report the root cause with a narrow patch proposal.”** The full-scene
and downstream CLI paths fail closed before filtering while Gate B is failing.
This report does not substitute unit tests or local rigidity for reconstruction.

## 1. Repository, source preservation and changes

Committed HEAD remains **`dd4426edcbac772b45a6f634b7c6c665b3651b8c`**. The grouped
changes are **uncommitted**; there is no new committed SHA or GitHub push.

Existing `pointwise_rgbd/{process6,filter_iekf,controller,discovery,dataset,
observations,projection,association}.py` are unchanged from HEAD. The static
RGB-D loader, previous masked pipelines, reference repository and old result
histories were not edited. Fresh SSD output directories hold all experiments. Tracked changes are the README/AUDIT status links and the import-audit test (16 added lines / 1 removed line); new grouped source, assets, tools, tests and documentation remain untracked. `grouped_20261022_delivery/git_status.txt`, `tracked_changes.patch` and `delivery_manifest.json` retain the exact status, diff and hashes.
Every experiment has source hashes and baseline code hashes in its manifest.

The reference checkout is at `b194e1691e5eb5a7ec62b830b1cdac89bcaa0a3d`, but all
six inspected reference source files have no diff from the requested
`481b2cf2d5602932286948cf14114f9caee4de3c`. Reference algorithms were inspected;
no Gaussian renderer/runtime or GT motion source was imported.

New source:

- `surface_supernodes.py`, `group_pose.py`, `group_process.py`, `group_iekf.py`.
- `phase1_grouped_controller.py`, `group_partition_validation.py` (boundary prototype and stopped-gate admission report), `group_visualization.py`.
- `assets/group_viewer.html`, vendored Three.js 0.160.0 and MIT license.
- `tools/{run_grouped_rgbd_joint,test_grouped_rgbd,audit_grouped_gate_b,report_grouped_gate_a,qa_grouped_viewer}.py`.
- Four grouped test modules and independent RGB-D test fixtures.

The one existing test changed is the baseline import audit: it now follows the
baseline entrypoint's reachable imports and rejects any dependency on grouped
modules/Lie estimators. Its former package-wide ban wrongly rejected independent
new modules explicitly required by this contract. No estimator assertion was
removed, and the 44 other baseline tests remain unchanged.

The user-supplied contract file and reference ZIP were already untracked on
entry. Only Section 8 and test 11.4 text in that contract were updated, as the
user explicitly authorized: candidate-specific degeneracy does not disqualify
an independently identifiable, measurement-consistent competitor. A prismatic
physical origin is not identifiable and is not required. Runnable joint
selection tests/code are not reached at this failed boundary gate.

## 2. Contract-to-source/test map

| Contract section | Actual implementation and evidence | Status |
|---|---|---|
| 0: XYZ-only, exact local transport, no recovery | `group_pose.transport`; `run_phase1`; exact-rigidity, gauges and forbidden-recovery controller tests. All original endpoint observations retained. | Implemented for Phase 1; physical labels/joint output unreached |
| 1: source audit and isolation | Reference functions inspected; baseline SHA checks; reachable-import audit; independent CLI. | Implemented |
| 2: endpoint/direction preparation | `surface_supernodes.endpoint_samples`, `run_phase1` processing order, real chronological times/IDs. Both directions and resume tested. | Raw RGB-D supported; optional final-PLY explicitly rejected; full-scene reverse not run |
| 3: computational supernodes | `surface_graph`, `connected_geodesic_assignment`, `build_partition`, `Partition.save/digest`; discontinuity, connectivity, isolation and linked-camera provenance tests. | Implemented initial partition; approved split epoch not achieved |
| 4: grouped Lie IEKF | `group_pose`, `group_process`, `group_iekf.associate_groups/correct`; Jacobian/process/reset/SPD/CPU-CUDA/unique-factor tests; actual pair audit. | Numerical core implemented; real residual/iteration limits remain |
| 5: physical boundary validation | `fit_measured_rigid`, `predictive_score`, `propose_split`, `heldout_comparison`, `validate_group`, `split_partition`; independent RGB-D boundary regression. | **FAILED Gate B**; zero approved splits, no two-group refit accepted |
| 6: inter-node coherence / moving/static/ambiguous labels | No implemented extractor, hierarchy, core or classification results. | Not implemented/reached because B fails |
| 7: fresh common rigid RGB-D refinement | No Phase-2 estimator/results. | Not implemented/reached |
| 8: P/R fitting and selection | Contract policy corrected per user; no fits, selection implementation or fabricated axes. | Not implemented/reached |
| 9: outputs and real visualization | Phase-1 arrays, raw refs, group ranks, manifests, residual diagnostics, streamed Three.js viewer and overlays; browser QA. | Partial: eight actual prefix frames, no downstream support/axis layers |
| 10: modules / CLI | Seven isolated modules, standalone CLI, tests and diagnostics tools. `group_coherence.py`, `rigid_xyz_phase2.py`, `xyz_joint_reduction.py` absent. | Partial, explicitly stopped at B |
| 11: mandatory validation | 20 grouped tests, 19 pass / 1 fail; baseline 45 pass. Detailed mandatory-test matrix below. | Not all acceptance tests implemented/run |
| 12: gates | A invariants/math and bounded data exercised; B fails; C/D/E/F not reached. Full source/default downstream execution blocked. | Correctly stopped; overall not accepted |
| 13: deliverables | This source map, real results/logs, rejected proposal, actual viewer and commands. No claims for missing physical outputs. | Partial delivery with explicit failure |

### Mathematical model used by the mapped source

Each group has one SE(3) pose and spatial world twist. Every point position is
**exactly** `x_i=R_group p_i+t_group`; no independently optimized point positions
or per-point orientations/velocities exist in this estimator. Canonical XYZ is
conditioned on the measured endpoint. Its backprojected measurement covariance
is retained as source provenance, not silently treated as a marginalized map
uncertainty. Calibration is conditioned fixed; source calibration remains marked
`CANDIDATE_NOT_VALIDATED`.

Translation-first spatial left error `[delta_pose,delta_twist]`, 12×12 covariance:

```
T_next = Exp(h xi) T,  xi_next=xi,  h=t_next-t_current
A = [[ad(xi), I], [0, 0]]
F(h) = exp(A h)
Q(h>0) = integral_0^h exp(A s) B Qc B^T exp(A^T s) ds
Q(-d) = F(-d) Q(d) F(-d)^T, d>0
```

The Van Loan 24×24 block exponential computes the chosen linearized factor;
blocks are batched/chunked, not one global dense `(12S)²` matrix. Reverse time
uses signed h, including negative pose/twist noise cross blocks. Physical twist
is not sign-flipped. This is a reoriented process factor, **not** Bayesian RTS.

The new continuous spatial-acceleration model is declared explicitly.
`configs/example_uncalibrated.json:process_covariance_rate[6:12,6:12]` supplies
numeric `Qc=0.01 I6`, units outer([m/s,rad/s])/s, and
`initial_twist_covariance=I6`, units outer([m/s,rad/s]). These are documented
initial magnitudes under a newly declared group-process assumption, not sensor
calibration or the copied old point Q. Direct pose diffusion is not copied.
The endpoint pose gauge is conditioned/eliminated; initial twist is uncertain,
not a static label. Approved existing sensor R and the 1% association policy
are retained. Every manifest records values, source keys, chart and units.

For correction, one predicted prior is held fixed throughout nonlinear inner
iterations. Optimize coordinates `d` relative to that prior, use analytic
`J_projection [I,-[x]_cross]`, compose with `Jl(d_pose)`, accept only decreasing
fixed-association energies, and reset final-information covariance by
`diag(Jl(d_pose),I)`. Measurement RGB only ranks correspondence candidates;
it is not an image-rendering measurement residual.

Temporary derived point-position marginals adapt the existing candidate matcher;
they are not independent point states/priors. The update aggregates shared
12D group information. `G_i P_pose G_j^T` cross-point covariance is retained
analytically; the held-out joint prediction calculation uses a rank-six
Woodbury reduction and matches a dense small-system reference. Conditional
independence between groups and fixed association/calibration are explicit
approximations; global Bayesian association/calibration uncertainty is not solved.

## 3. True tests and mandatory-test coverage

Final grouped report: `grouped_20261022_gate_tests_delivery_final/tests.json` and
`tests.log`: **20 run, 19 passed, 1 failed, 0 errors, 0 skipped**, CUDA true,
3.796500 s. The failing test is
`GroupBoundaryTests.test_wrong_initial_group_rejected_from_heldout_rgbd_and_both_groups_refit`.
It has not been weakened, skipped, or converted to an expected failure.

Baseline: `grouped_20261022_baseline_tests_final/`: **45 passed**, no errors or
skips, CUDA true, 21.087975 s. Earlier failed attempts remain preserved.

CPU/CUDA maximum covariance difference: **3.326506e-12**. Forward error is checked
against the actual condition-number/FP64 error budget (posterior condition
1.776323e7), not an arbitrary physical tolerance. The independent covariance
reference has normalized backward error **2.322248e-16**, below the 12D FP64
roundoff budget 2.664535e-15. Initial attempts using unexplained tiny absolute
inversion tolerances failed and remain in earlier reports; no estimator noise,
association gate or covariance repair was changed to pass them.

| Mandatory test §11 | Actual evidence / remaining work |
|---|---|
| 1 exact rigidity / update / prediction / resume / directions | Passed math/controller tests plus all-pair bounded data audits |
| 2 one/two/non-collinear observability | Passed: pose ranks 3 / 5 / 6; information spectra exported |
| 3 static | Identity grouped poses tested both directions; moving-classifier false-positive test not reached |
| 4 translation | Constant translation Phase 1 tested; variable speed, shared Phase 2 and prismatic selection not reached |
| 5 rotation / reversals / wraps | Arbitrary rigid rotations/Jacobians tested; full joint-identification fixture not implemented |
| 6 wrong joint-boundary group | **Failed**, rejected disconnected proposal; accepted split/refit not demonstrated |
| 7 occlusion | Passed partial occlusion and 20 fully unobserved frames; covariance grows, exact geometry retained |
| 8 no false coherence | Not implemented; extractor not reached; no within-group distance used as motion evidence |
| 9 ownership vs coverage | Group duplicate-factor rejection and actual raw-factor audit passed; existing coverage/baseline suite passed; coverage is not a group update |
| 10 Lie math / signed covariance / reset / CUDA | Passed finite differences, reverse-factor identity, information-covariance reference and CPU/CUDA checks |
| 11 held-out leakage | Passed training-proposal independence from altered held-out values and linked multiview provenance; common held-out predictive covariance checked against dense reference; full P/R same-set comparison not reached |
| 12 directions / original frames / gauge | Passed synthetic directions/resume and real prefix directions; whole-scene endpoint-327 experiment not run |
| 13 impossible / tiny motion | Pose-rank rejection present; final coherence/joint ambiguity logic not implemented/reached |
| 14 bounded real performance / viewer then full scene | Both bounded real prefix runs and browser QA performed; full scene blocked |
| 15 baseline preservation | Eight estimator/loader source files identical to HEAD; 45 baseline tests passed; old results not overwritten |

### Bounded actual real-data results

Actual scene: `/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314`.
The loader reads actual 25 Hz `metadata/frame_timestamps.csv:timestamp_s`.
Both experiments use every available camera and **all raw endpoint samples**
from an explicitly selected eight-frame prefix, chronological original IDs
**0–7**, times **0–0.28 s**. The reverse experiment is anchored at prefix frame
7, **not full-scene frame 327**. No matching point identity across the two
independent endpoint catalogues is assumed.

| Measured quantity | Forward anchor 0 | Reverse prefix anchor 7 |
|---|---:|---:|
| Endpoint raw samples retained | 184,274 | 184,640 |
| Computational groups | 4,401 | 4,431 |
| Distinct accepted filter factors, excluding endpoint conditioning | 761,977 | 782,530 |
| Duplicate / background / invalid factors | 0 / 0 / 0 | 0 / 0 / 0 |
| Observed group-frames, excluding anchor | 24,239 | 24,549 |
| Prediction-only group-frames, excluding anchor | 6,568 | 6,468 |
| Rank-6 measured group-frames | 17,352 | 17,499 |
| Inner-iteration-limit group-frames | **777** | **784** |
| Maximum within-group pair-distance error | 4.614365e-16 m | 4.250073e-16 m |
| End-to-end bounded runtime | 92.115082 s | 93.351411 s |
| Peak CUDA allocation | 2,596,884,992 bytes | 2,613,721,600 bytes |
| Peak RSS | 2,323,763,200 bytes | 2,304,405,504 bytes |

The forward rigidity audit checked **78,811,136** actual within-group pairs over
all stored prefix frames. Exact-transport error was at most 2.22e-16 m and
rotation orthogonality error 6.00e-15. Anchor poses are exactly identity, no
anchor observation is reassimilated, and outputs are chronological in both runs.

| Final-association residual distribution | Forward median / p99 / max | Reverse median / p99 / max |
|---|---|---|
| Pixel-ray norm | 0.265595 / 2.968084 / 67.285772 px | 0.214525 / 2.521920 / 71.781387 px |
| Absolute depth residual | 1.535603 / 45.865154 / 215.749600 mm | 1.407944 / 41.657781 / 225.421508 mm |
| Sensor-normalized squared residual | 1.217913 / 125.774065 / 54,342.663821 | 0.861594 / 94.313212 / 61,865.380922 |

These selection-conditioned residuals are **not** independent innovation NIS or
calibration proof. The large tails and iteration limits are unresolved evidence
of poor correspondence/group-fit behavior. Exact imposed local distances do not
make the groups physically correct. **Neither bounded run is an accepted
reconstruction or a demonstrated converged nonlinear solution for every group.**

Forward stage timers: prediction 0.735674 s, association 37.223731 s, filtering
32.472558 s. Reverse: 0.781705 / 38.015716 / 33.014845 s. Remaining time includes
partition/input handling, storage/export and viewer generation. Spatial kNN and
surface graph operations use CPU; bounded FP64 Lie propagation/information
reductions/corrections use CUDA. No dense global covariance is allocated.

## 4. Partition and failing boundary example

Initial graph: same-camera 4-pixel adjacency intersected with local geometric
neighbors, depth compatibility under the declared sensor covariance; mutual
cross-camera nearest neighbors additionally require covariance-compatible XYZ
and reciprocal valid-foreground projected depth. No fixed metric cutoff is
used. Isolated raw points remain represented. Cross-camera linked provenance
units share a held-out fold. The default target occupancy 64 is explicitly a
computational benchmark, not a moving-part count or physical scale.

Geodesic farthest-point seeds and shortest-path assignment preserve graph
connectivity. Membership/canonical geometry is fixed for a full epoch.
Independent real-data partitions below have median 23 samples per group;
forward sizes range 1–722, including 864 single-sample groups. Occupancy is a
target, not a guarantee of exactly 64 members. Such small groups cannot certify
physical rotation, and information rank is exported separately.

**Mandatory boundary regression fails.** A synthetic RGB-D cabinet/door sensor
fixture supplies 200 raw endpoint observations and 10 real rasterized RGB-D
frames. One deliberately erroneous initial computational group is used; correct
physical membership is never supplied to estimation. Training frames 1–4 propose
119/81-point subpatches. The 119-point patch has **two** surface-graph components;
the other has one. The candidate is rejected, yielding `UNRESOLVED_SPLIT`.

Further diagnostics show accepted raw support falling from 200 to 52 by frame 9.
At that frame the second proposed patch has only **one** independent training
sample, pose information rank **3**, insufficient for a rigid pose. This is a
second reason the proposal cannot be accepted. A complete held-out two-model
comparison and whole-series two-group refit were **not** completed. No split
success or boundary location is claimed.

Preserved evidence: `grouped_20261022_gateB_failure_final/` contains actual
synthetic RGB/depth/mask/valid arrays and camera/time metadata; Phase-1 histories;
raw factors; `rejected_training_proposal.npz`, `rejected_proposal.png`,
`actual_support_by_frame.json`, `gate_b_failure.json`, and the explicitly
**diagnostic-only** unresolved held-out result. This is a synthetic gate fixture,
not a plot falsely presented as a real fridge observation.

The prototype proposes global residual clusters from training measurements and
only then checks connectedness. Post-hoc rejection is safe, but insufficient to
find the required physically valid connected split. **Narrow correction:**
generate connected surface-graph cuts directly from training residual evidence;
refit candidate trajectories from training RGB-D with explicit fixed-association
updates/rematching transactions when support has been lost; evaluate both on
the same independent held-out raw observations. Retain the existing covariance,
observability and complexity checks. Do not relax connectivity/rank or add
motion/residual thresholds. This patch is proposed, not falsely reported as done.

## 5. Moving/static/ambiguous support

No physical memberships, candidate hierarchy or coherence plots are available.
Gate C was not implemented/reached. Viewer colors show computational groups or
source RGB, never inferred physical part labels.

## 6. Shared moving rigid trajectory

No Phase-2 shared rigid trajectory exists. Kabsch in boundary diagnostics is an
initializer for training-only diagnostic fits, not the required fresh measured
RGB-D Phase-2 refinement. There are no accepted Phase-2 results.

## 7. Joint hypotheses and user correction

Section 8 and mandatory test 11.4 of the authoritative contract were updated
exactly to the user's candidate-specific identifiability policy. A valid
prismatic hypothesis may win against an unidentifiable revolute axis; neither
is selected merely because its competitor fails. Static/noise-indistinguishable
or insufficiently excited data must remain unresolved; a prismatic physical
origin is not identifiable and is not required. The runnable fitting/selection
stage is not implemented/reached, so no P/R residuals, winner or axis is reported.

## 8. Viewer and stored artifacts

Each bounded run stores `phase1/partition_v000/` with:
`partition.npz`, full FP64 `group_poses.npy`, `twists.npy`, `covariance.npy`,
`xyz.npy`, actual `point_observed.npy`, group measurement counts and ranks,
chronological metadata, checkpoint, source fingerprints and compressed actual
raw factor identities/UVZ/R/RGB. Full point XYZ is derived from group poses.

`viewer/index.html` streams every stored prefix frame's binary XYZ and observation
mask from `viewer/frames/`, with a two-frame cache. Persistent Three.js buffers,
dark 380px sidebar, playback slider, orbit/pan/zoom, explicit ALL/200k/50k/10k
LOD, RGB/group colors, observed/predicted/outlier toggles and stable measured
camera focus are implemented. ALL displayed **184,274/184,274** forward points;
there is no hidden 512/4096 cap. Frame 7 has 84,819 directly measured point
updates and 99,455 prediction-only points, including 12,122 predictions outside
the canonical measured ROI. ROI/outlier handling is display-only.

Headless Chrome loaded the actual viewer through local HTTP and checked first
and last prefix buffers, point counts, stable camera, explicit LOD, persistent
buffers, outlier toggle, fit-all and restored measured focus. `browser_qa.json`,
`browser_dom.html`, `browser.log`, `browser_verification.png` retain the evidence.
First browser process took about 3.5 s; JS heap about 23 MB. Browser WebGL GPU
memory is not measured. Browser screenshots were inspected. The viewer has no
moving/static/ambiguous classifications, split/refinement layers or P/R axes
because those models have not been established. It is **not** the complete
Section 9 downstream viewer.

`overlays/` includes selected original RGB images, mask boundary, every in-image
predicted point, actual accepted raw pixels, actual valid metric-depth images
and actual matched-depth residual maps. No estimator updates occur during
export or browser QA. FP64 inference histories remain unchanged; binary display
coordinates alone are encoded float32.

## 9. Remaining stop and commands

Gate B must be corrected and revalidated before implementing/running C, D, E or
full-scene F. The outstanding joint-selection policy is the user's corrected
candidate-specific rule; no successful joint fit is claimed. No full 328×3 run,
recovery loop, new-point discovery or Phase 3 was started.

Fresh reproduction output paths are required. Exit **1** from the grouped suite
and boundary audit currently means the preserved real test failure. Exit **3**
from a bounded Phase-1 CLI run means explicitly incomplete, not reconstruction
success. A full-source/default downstream request returns `STOPPED_AT_GATE_B`
before filtering.

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/rgbdekf_mpl
ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf"
SCENE="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314"

python -B tools/test_grouped_rgbd.py --output "$ROOT/grouped_tests_reproduce"
python -B tools/test_pointwise_rgbd.py --output "$ROOT/grouped_baseline_reproduce"
python -B tools/audit_grouped_gate_b.py --output "$ROOT/grouped_boundary_reproduce"

# Bounded diagnostic only. No recovery, no complete joint pipeline.
python -B tools/run_grouped_rgbd_joint.py \
  --scene "$SCENE" --output "$ROOT/grouped_first_reproduce" \
  --anchor first --stop-after phase1 --frame-limit 8 --device cuda
python -B tools/run_grouped_rgbd_joint.py \
  --scene "$SCENE" --output "$ROOT/grouped_last_prefix_reproduce" \
  --anchor last --stop-after phase1 --frame-limit 8 --device cuda

python -B tools/report_grouped_gate_a.py --run "$ROOT/grouped_first_reproduce"
python -B tools/qa_grouped_viewer.py --run "$ROOT/grouped_first_reproduce"

# View the existing actual forward-prefix result at http://127.0.0.1:8765/
python -m http.server 8765 --bind 127.0.0.1 \
  --directory "$ROOT/grouped_20261022_gateA_first01/viewer"
```
