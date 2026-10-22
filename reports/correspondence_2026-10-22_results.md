# Pointwise RGB-D correspondence correction — measured results

22 October 2026. Repository: `SamueleMara/RGBDiekf`, working tree based on
`d9c9772`. These changes are local and have not been committed or pushed.

**Coverage/ownership correction and bounded real-data validation completed.
Unlimited discovery was not initiated. This is not a completed reconstruction.**

## Inputs and reproducibility

Dataset: `/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314`.
All **328 times × 3 cameras**, 640×480, actual 25 Hz timestamps from
`metadata/frame_timestamps.csv:timestamp_s`, **54,970,499** valid foreground
observations. Frame 0 supplies **184,274** initial points. The reused SOLVED
cache includes one previously fitted discovery point: **184,275** trajectories.

Frozen source: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/pointwise_20261022_full_01/pointwise_rgbd`.
Original frame state/covariance digests, appended-track SHA256, calibration,
timestamps, frame identities and RGB/depth/mask input fingerprints were checked.
The initial optimizer was never rerun. No ground-truth parts, rigid clustering,
Gaussian parameters, normals, surfels, mesh or global re-optimization were used.

Output directories, all under the requested SSD results root:

- `correspondence_audit_20261022_full_01`: all-frame comparison and actual visuals.
- `correspondence_incremental_20261022_tests_final`: final CUDA tests.
- `correspondence_discovery_20261022_01/pointwise_rgbd`: one bounded insertion and incremental-reference validation.

Each experiment records actual source hashes. The earlier audit used the same
coverage kernel as the insertion and reference check; its SHA256 is
`0436e83975ca80bbb3bce96fd23051cba22520eef8bbdb1e9f366c8c5d19e122`.

## Source changes and preserved mathematics

The responsible old functions/equations and exact new compatibility equations
are in [AUDIT_CORRESPONDENCE_2026-10-22.md](../AUDIT_CORRESPONDENCE_2026-10-22.md).

| Source | Resulting behavior |
|---|---|
| `pointwise_rgbd/association.py` | Existing exclusive temporal matcher retained unchanged. It is not the surface-coverage definition. |
| `pointwise_rgbd/coverage.py` | Observed foreground XYZ → frozen SOLVED world-XYZ 8NN; position covariance, ray, depth, visibility and RGB compatibility; many samples may share a witness. |
| `pointwise_rgbd/coverage_store.py` | Separate persistent coverage masks/witnesses and completed model horizon; no measurement ownership. |
| `pointwise_rgbd/incremental_coverage.py` | Reusable measured foreground support/UNSOLVED lookup; test only new tracks against still-unexplained samples, retain valid old witnesses. |
| `pointwise_rgbd/controller.py` | One initial forward pass, immutable old histories, one new point fitted independently from its anchor, insertion, restart reverse discovery from T. Coverage calls never filter or claim measurements. |
| `pointwise_rgbd/store.py` | Tracking ownership recognizes actual factors/anchors, excludes old kind-2 bookkeeping; seeds must be unexplained AND not already assimilated. |
| `pointwise_rgbd/discovery.py` | Same 6D independent two-branch optimizer; passes actual tracking ownership rather than legacy geometric bookkeeping. |
| `pointwise_rgbd/frozen.py`, `cached_discovery.py` | Read-only old histories and audited-coverage bootstrap; no initial refit or new initial-history allocation. |
| `pointwise_rgbd/diagnostics.py` | Coverage and assimilations reported separately; old stage timings explicitly labeled cached rather than charged to this run. |
| `tools/run_pointwise_rgbd_discovery.py` | Requires completed audit for cached discovery, separate output and unchanged cached timing/noise model. |
| Audit/report/visualization/reference tools | Numerical comparisons, attribution, actual dataset views and exhaustive insertion-reference checks. |

`filter_iekf.py`, `process6.py`, `projection.py`, `state6.py` and `observations.py`
are byte-for-byte unchanged from the base commit. State remains exactly R6 with
full 6×6 per-point covariance; the original initial forward filtering equations
are unchanged. Backward discovery uses the existing reverse process factor,
not an invented RTS pass or shared rigid motion.

For unavailable trajectory/observation error cross-covariance, coverage uses the
PSD upper bound `2*(P_xyz + Sigma_observation)`, not an assertion of independence.
The analogous RGB bound is `4*R_rgb`. The noise-normalized color distance splits
exactly into angular/cosine and brightness terms; it is counted once. Scores
combine marginal Gaussian negative log likelihoods without an arbitrary metric
vs. color weight. Absolute XYZ cosine is never used.

The approved **1% chi-square tail** remains unchanged: 6.634897 (1D), 9.210340
(2D), 11.344867 (3D). It is an **uncalibrated association-policy assumption**.
Dependent component tests and conservative covariance bounds are not a calibrated
joint 99% confidence or exact global correspondence inference.

## Same cached trajectories, all actual observations: before/after

| Policy | Explained observations | Unexplained observations | Explained |
|---|---:|---:|---:|
| Original exclusive-owner coverage | 36,535,367 | 18,435,132 | 66.463590% |
| Controlled old-family/gates/ties, exclusivity removed | 39,415,824 | 15,554,675 | 71.703595% |
| New geometric kNN coverage | 45,855,798 | 9,114,701 | 83.418923% |
| New geometric + RGB angular/brightness coverage | 44,810,732 | 10,159,767 | 81.517783% |

Original **actual assimilated/anchor ownership is 36,457,603**, distinct from
36,535,367 old coverage owners: 77,764 old kind-2 bookkeeping claims were not
measurements. The new coverage matcher creates **zero** assimilation claims.
Geometric and appearance coverage permit respectively 14,069,515 and 15,128,860
additional many-to-one sample explanations; these are not tracking conflicts.

**4,850,505 / 18,435,132 = 26.311203%** of the original unmatched set have a
compatible edge under the old candidate family/gates/tie convention once
exclusive restrictions are removed. All 4,850,505 have an already-assigned point
as a witness. This is the controlled ownership-only component. The broader new
matcher recovers **9,473,163** old misses, but that number also changes query
direction, covariance treatment and visibility/appearance tests and cannot all
be attributed to ownership.

The controlled net increase is smaller than gross recovery: 1,970,048 old owners
fail those gates when reevaluated at the optimized cached states. The audit
compares the actual saved old decisions with posterior-cache compatibility;
it does not rerun their original pre-update trajectories.

The new appearance rule rejects **1,197,798** observations previously called
explained. Identity: `18,435,132 - 9,473,163 + 1,197,798 = 10,159,767`.
Both new variants explain **100% of frame-0 valid foreground samples** across
all three cameras, without requiring ownership input.

## Distances, RGB and unmatched reasons

All distances are metric world-coordinate distances. Exact global histograms,
quantiles and per-view distributions are saved in `global_distributions.json`,
`distributions/`, and `selected_pair_global_distributions.json`.

| Quantity on the frozen cache | Median | 99th percentile | Maximum |
|---|---:|---:|---:|
| Original actual tracking-owner pair distance | 0.624586 mm | 18.504402 mm | 723.361498 mm |
| Observed → nearest SOLVED distance | 1.382346 mm | 11.546052 mm | 47.959253 mm |
| Selected geometry-compatible distance | 0.972211 mm | 11.056387 mm | 59.924184 mm |
| Selected geometry + appearance distance | 1.041779 mm | 10.762726 mm | 59.924184 mm |

Original actual tracking-pair RGB cosine median: **0.999282** across
36,457,603 factors/anchors (18 zero-color cosines undefined). These exact
posterior-cache pair statistics are in `exclusive_tracking_pair_metrics.json`
and its per-view JSONL; their no-fit extraction took **89.682183 s**. Original
RGB score uses `2*R_rgb`, while new conservative coverage uses `4*R_rgb`; their
Mahalanobis magnitudes should not be compared as if the covariance were equal.
The pair sets have different sizes and selection rules, so smaller selected
distances are not proof of more accurate physical correspondences.

Nearest RGB cosine median: **0.998354**; selected geometric cosine median:
**0.998906**; selected appearance cosine median: **0.999196**. Cosine alone
cannot distinguish gray brightness. Undefined zero-color cosine is recorded as
undefined while Gaussian RGB distance remains valid. Acceptance statistics are
selection-conditioned; they are not independent NIS calibration evidence.

The following disjoint observation-level failures sum exactly to 10,159,767:

| First failing coverage stage | Observations |
|---|---:|
| No verified visible candidate | 8,172 |
| Visible candidates, none passing world-XYZ covariance gate | 8,372,642 |
| 3D-compatible candidates, none passing ray gate | 733,875 |
| Ray-compatible candidates, none passing depth gate | 12 |
| Geometry-compatible candidates, none passing RGB gate | 1,045,066 |

Candidate-edge passing counts (different denominator; not disjoint failures):
439,763,992 tested; 427,506,693 visible; 237,287,807 3D-compatible;
233,150,568 ray-compatible; 233,150,536 depth-compatible; 223,025,799 RGB-compatible.
Invalid covariance edges: zero.

There are **1,179,399** appearance-unexplained samples with immutable frozen
measurement owners; **861,278** also fail geometry-only coverage. They remain
explicit consistency conflicts and cannot seed duplicate independent updates.
There are **8,980,368** unexplained, unassimilated candidate observations before
insertion. These are sample candidates, not a predicted number of new points.
The first eligible reverse seed is frame 327, cam_000, pixel **80240**.

## Incremental architecture and actual bounded insertion

Base candidate family: initial observation-to-SOLVED 8NN. After insertion,
coverage is retained old witnesses OR compatibility with the newly inserted
trajectory. Old compatible pairs cannot become invalid when their state,
covariance, color and measurement-depth visibility stay fixed. Recomputing a
strict top-eight list could eject an old compatible witness after inserting
closer incompatible points; the incremental family explicitly avoids this.

Per-view disk lookup stores real measured foreground support plus baseline
UNSOLVED observations, streamed one view at a time. It does not introduce a
latent all-frame point catalogue. The bounded run's coverage/lookup database
occupies **909,524,992 bytes**. A necessary search radius is derived from the
chi-square gate and covariance eigenvalue bounds, followed by the unchanged
full pair tests; no distance novelty tolerance is introduced.

The real bounded run fitted exactly **one** new 6D point, ID **184275**, from
frame 327/cam_000/pixel 80240. It traversed **327 backward frames**; the forward
branch is correctly empty because the anchor is T. Both nonempty branches are
covered by the pointwise numerical/controller regression fixtures. Anchor used
once; **172** actual RGB-D measurements including the anchor; **110** observed
frames and **218** prediction-only frames. Zero old-point optimizer calls;
zero initial optimizer calls; zero newly allocated initial-history bytes.

After insertion it restarted reverse discovery at T; the next eligible seed
is frame 327/cam_000/pixel **80866**. The explicit one-event limit then stopped
it. Actual status is **EXPLICIT_TEST_EVENT_LIMIT**, exit code **3**, not failure
of a numerical solve and not completed discovery. No unlimited process remains.

| Incremental update measurement | Result |
|---|---:|
| Newly inserted trajectories examined | 1 |
| Old trajectory frame reads inside update | 0 |
| Raw image loads inside update | 0 |
| Views visited / changed | 984 / 351 |
| New-point/observation pairs tested | 691,352 |
| Additional geometry-compatible observations | 381,797 |
| Additional geometry + appearance-compatible observations | 425,893 |
| Incremental update runtime | 77.048102 s |
| All-observation/new-point reference runtime | 137.529598 s |
| Reference raw samples / views | 54,970,499 / 984 |
| Geometry / appearance mask disagreements | 0 / 0 |
| Changed old appearance witnesses | 0 |

The reference independently evaluates **every** observation against the new
trajectory, without the production spatial prefilter, and ORs with the audited
baseline. No old trajectory reads or optimizer calls occur in that reference.
`discovery/incremental_reference_validation.json` records PASS and source hashes.

After the one insertion: **45,236,625** covered (82.292549%), **9,733,874**
unexplained, including **1,115,855** with frozen measurement owners. Actual
assimilated observations increased by only **172**, to **36,457,775**; coverage
increased by 425,893. The distinction is deliberate and verified.

## Runtime, tests and remaining ambiguity

The all-frame before/after audit took **2,034.929728 s** and invoked zero
optimizers. It includes **1,431.359265 s** of controlled legacy ablation, not
just new matching. New matching reported **349.568559 s** across views, including
84.663744 s kNN and 45.669856 s projection. Shared per-frame cloud-index build
time was **57.178939 s**; summing the per-camera reused-index statistic would
incorrectly triple it. These counters exclude some data loading/cache reads and
are not a direct end-to-end per-seed speed comparison.

The bounded run's one-time audited bootstrap/lookup construction took
**224.612093 s**. Its backward fit progress measured about **116 s**; the exact
new coverage-update timer is above. The controller recorded **432.913510 s**
before final export; this is not total process time including final artifact
export. Peak RSS was **5,156,380,672 bytes**, peak CUDA allocation **41,938,432
bytes**. Original stage durations are retained separately under
`cached_initial_stage_seconds_not_this_invocation` rather than charged again.
SciPy spatial indexing is CPU-based; batched compatibility/filter algebra uses
PyTorch FP64 on the RTX 4070 Laptop GPU.

Final regression result: **45 passed, 0 failures, 0 errors, 0 skipped**, CUDA
available, **23.748239 s**. Coverage tests include density/viewpoint changes,
separated-surface discovery, background/invalid-depth invariance, frame-zero
coverage, RGB brightness vs cosine, CPU/CUDA agreement, covariance-bound validity,
world-origin invariance, occlusion, ownership-only ablation, frozen-cache reuse,
no duplicate assimilation, incremental/full-pair equality, forbidden old reads,
interrupted-update resume and monotone witnesses beyond eight new neighbors.

**Remaining limits are material.** The new point's maximum principal position
standard deviation is **0.850152 m**, and its largest adjacent-frame mean jump
is **0.284091 m**. It can statistically cover 425,893 samples under those broad
uncertainties; this is not physical confirmation of that surface extent.
The old already-cached discovery point also had poor motion evidence. Calibration
is marked `CANDIDATE_NOT_VALIDATED`; fixed-calibration and initial-noise guesses
remain assumptions. Texture repetition, illumination changes, partial depth
support, occlusion, finite candidate search and poor frozen trajectories can
produce incorrect or ambiguous compatibility. Temporal association retains its
existing color-ranking behavior. The correction neither retunes noise/gates nor
refits old tracks to hide these issues. Red diagnostic points mean unexplained
under this policy, not ground-truth proof of new geometry.

## Actual dataset visualizations

In `correspondence_audit_20261022_full_01`:

- `coverage_comparison.html`: interactive actual cached XYZ and measured XYZ,
  policy/camera/frame controls, explicit full-prediction view.
- `visuals/<frame>_<camera>/comparison.png`: side-by-side actual RGB overlays.
- `visuals/<frame>_<camera>/world_comparison.png`: side-by-side world-coordinate
  point clouds with identical bounds.
- `frozen_measurement_conflicts.png`: actual owned/unexplained conflicts.
- `browser_verification.png`: dashboard screenshot inspected in headless Chrome.

Representative frames 0, 1, 38, 39, 80, 160, 240 and 327, all cameras. Visual
payloads alone sample 4,096 SOLVED / 6,000 raw coordinates; numerical audit and
reference use every observation. Measured-surface viewports crop outlying
predictions only for display; HTML exposes the full frozen estimates.

## Commands

Use a fresh output path for each reproduction; existing evidence is preserved.

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/rgbdekf_mpl
ROOT="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf"
SCENE="/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314"
CACHE="$ROOT/pointwise_20261022_full_01/pointwise_rgbd"
AUDIT="$ROOT/correspondence_audit_20261022_full_01"

python -B tools/test_pointwise_rgbd.py --output "$ROOT/correspondence_tests_reproduce"

# Audit only: no filtering. A fresh audit path is required.
python -B tools/audit_surface_correspondence.py \
  --cache "$CACHE" --output "$ROOT/correspondence_audit_reproduce" --device cuda
python -B tools/report_surface_correspondence.py \
  --audit "$ROOT/correspondence_audit_reproduce" \
  --tests "$ROOT/correspondence_tests_reproduce/tests.json"
python -B tools/report_surface_correspondence.py \
  --audit "$ROOT/correspondence_audit_reproduce" --tracking-pairs-only
python -B tools/visualize_surface_correspondence.py \
  --audit "$ROOT/correspondence_audit_reproduce"

# One-event validation, using the already completed original audit above.
# Exit 3 is expected for the explicit event limit.
python -B tools/run_pointwise_rgbd_discovery.py \
  --scene "$SCENE" --output "$ROOT/correspondence_discovery_reproduce" \
  --cached-solved "$CACHE" --coverage-audit "$AUDIT" \
  --device cuda --max-events 1

python -B tools/validate_incremental_surface_coverage.py \
  --audit "$AUDIT" \
  --discovery "$ROOT/correspondence_discovery_reproduce/pointwise_rgbd" --device cuda
```

Omitting `--max-events 1` in a fresh cached-discovery command starts unlimited
discovery. It was deliberately not initiated: the correspondence correction
and the documented motion/uncertainty ambiguities must be assessed first.
