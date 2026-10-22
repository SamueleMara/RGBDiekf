# Pointwise RGB-D pipeline, 22 October 2026

The five `*_2026-10-22` files named by the user are the current authority. Start
at `START_HERE_Pointwise_RGBD_2026-10-22.md`. This implementation is isolated in
`pointwise_rgbd/`; use the entry point below. The older `main_iekf.py`,
`main_ukf.py`, and `scripts/run_full_scene.sh` implement older architectures.

Each point has only `[world XYZ, physical world velocity]` and a full 6x6
covariance. One initial forward IEKF tracks every valid masked frame-zero raw
observation. Those histories and measurement references become read-only.
Discovery scans backward in deterministic frame/camera/pixel order, fits one new
point from its actual discovery-frame anchor in two independent directions,
appends that track, and restarts the reverse scan. No old-point filtering is
repeated. There is no rigid clustering, orientation state, renderer, RTS pass,
normal, surfel, mesh, or fixed point catalogue. The optional 6D UKF is not
implemented; the CLI accepts only `iekf`.

From this repository, run the complete scene without test limits:

```bash
conda activate rigeuler3dgs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B tools/run_pointwise_rgbd_discovery.py \
  --scene "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314" \
  --output "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/pointwise_fridge_run_20261022" \
  --estimator iekf --all-cameras --all-frames --debug-numeric --debug-visual
```

The output is `<output>/pointwise_rgbd/`. Choose a fresh output directory; the
runner refuses to overwrite it. To resume an interrupted run, repeat the same
command with `--resume`. Completed initial frames are checkpointed; the initial
filter is not repeated after freezing. A lock protects a live worker. Resume
checks model, calibration, source-code hashes, and the raw input content of
completed initial frames. Do not edit the estimator during a run you intend to
resume. Files from a failed run are preserved, including numerical failures.

Directory, ZIP, and `metadata/` pointers use the same calibrated loader. For a
synthetic scene, replace `--scene` with the actual scene folder, for example:

```text
/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render/blade_103706
```

Timing is read from the selected input; `--fps`, `--dt-seconds`, and
`--timestamps-file` are optional explicit overrides. Missing masks, missing
timing, inconsistent camera clocks, and unrectified nonzero distortion fail
with a diagnostic rather than falling back to invented measurements. The real
scene resolves to 25 Hz from `metadata/frame_timestamps.csv:timestamp_s`.

CUDA FP64 handles projection, process prediction, innovation tests, and batched
6D IEKF information/covariance solves. Image I/O, SciPy kNN/index construction,
ownership, covariance diagnostic eigenspectra, and plotting use the CPU. This is
GPU-accelerated computation, not a GPU-only program.

Noise comes from the approved `configs/example_uncalibrated.json` initial
guesses, with explicit 6D process discretization and provenance in the manifest.
These are not independent calibration. Matching uses the declared local policy
in `configs/pointwise_association.json`: eight candidate neighbors and a 1%
chi-square tail, overridable with `--association-tail-probability`. The documents
require a declared gate but do not supply its numeric tail. This policy is
explicitly an implementation assumption; it is not attributed to the papers or
to sensor calibration. Every unclaimed raw observation remains seed-eligible.
The user subsequently explicitly authorized that 1% tail; see
`configs/pointwise_association_authorization_2026-10-22.md`.

Useful output files:

- `run_manifest.json`, `run.log`: exact sources, units, timing, configuration.
- `initial_forward/summary.json`, `nis_by_frame.csv`, `trajectory_store/`:
  initial tracking, covariance diagnostics, immutable histories and raw factors.
- `discovery/events.jsonl`, `matching_by_frame.csv`, `nearest_residuals.csv`:
  restart order, ownership, residual distributions and empirical match gaps.
- `discovery/events/<point_id>/`: seed, states/covariances/NIS, all inspected
  candidate reasons, actual RGB/mask overlays and depth residual plots.
- `discovery/geometry_growth.html`: offline 3D time and discovery-event sliders;
  `.preview.png` shows the same stored geometry. The viewer samples old points
  for display only; inference retains all observations.
- `final/summary.json`, `point_tracks.json`, `unexplained_observations.sqlite`:
  explicit completion status and exact remaining raw IDs.

Run the numerical, ownership, CUDA, storage and integration tests:

```bash
python -B tools/test_pointwise_rgbd.py \
  --output "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/pointwise_tests_20261022"
```

An explicit smoke test adds `--frame-limit 3 --camera-limit 2 --max-events 1`
and uses its own output folder. Exit code 3 means an explicit test selection or
event limit; it is not a complete reconstruction. Exit code 1 means failure;
only an exhausted, unlimited full sequence can exit 0. A single-observation
new point is accepted as required, with its other states labeled prediction
only. There is no claim of identifiable physical motion from that anchor.

See `AUDIT_POINTWISE_2026-10-22.md` and
`reports/pointwise_2026-10-22_validation.md` for the measured acceptance evidence
and remaining limits. Historical rigid-pipeline reports are not pointwise
validation.


## Correspondence correction and cached discovery

Surface coverage now uses observed-world-XYZ -> transported SOLVED spatial kNN
with uncertainty, visible foreground depth, and noise-normalized RGB distance.
RGB angle and brightness are recorded as the exact decomposition of that color
distance. It is separate from exclusive temporal IEKF measurement ownership.
Coverage never claims a measurement or updates a Kalman state.

Discovery selects unexplained, unassimilated observations in deterministic
reverse order. Previously assimilated but incompatible observations remain
unexplained consistency conflicts. Initial means/P and all previously appended
tracks remain frozen. New coverage is the union of retained old witnesses and
compatible newly inserted point trajectories; updates load no old trajectories.

Use `--cached-solved <existing pointwise_rgbd>` together with
`--coverage-audit <completed full audit>` to skip the initial optimization.
Select a fresh `--output`; the cache and dataset are read-only. Add
`--max-events 1` for bounded validation; removing the limit starts potentially
very expensive finite discovery and is not needed for the correspondence audit.
See [measured results and exact commands](reports/correspondence_2026-10-22_results.md).
