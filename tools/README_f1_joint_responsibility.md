# Final 8444 / 45588 responsibility test

Investigation-only implementation of
`RGBDiekf_Codex_Final_Joint_Responsibility_8444_45588_Test.md`.
No production inference/configuration, existing run/report, or camera/noise prior
is modified. No scene is re-filtered and no GT enters any policy or optimization.

## Run tests, then the actual audit

Both commands may use the **same** output directory. Tests/input hashes are
reused only when unchanged; completed reports and logs are never overwritten.

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
FOR="/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030"
OUT="$FOR/final_joint_responsibility_8444_45588_20261030"

CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_joint_mpl \
python -u -B tools/f1_joint_responsibility_audit.py \
  --forensics-root "$FOR" \
  --fifth-pass-root "$FOR/fifth_pass_source_responsibility_20261030" \
  --out "$OUT" --experiment tests

CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_joint_mpl \
python -u -B tools/f1_joint_responsibility_audit.py \
  --forensics-root "$FOR" \
  --fifth-pass-root "$FOR/fifth_pass_source_responsibility_20261030" \
  --out "$OUT" --experiment all --frame-start 0 --frame-end 10
```

`all` automatically runs the tests if they have not already completed. For a fresh
run, it is sufficient to execute that command alone. For a repeat after a completed
report or changed code, select a new output suffix; never edit saved hashes.

Direct regression suite without an output inventory:

```bash
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_joint_mpl \
python -B -m unittest tests.test_f1_joint_responsibility_audit -v
```

The end-to-end test uses the actual pair and temporary `/tmp` output. It took
about 85 seconds on this machine; it tests the report and figures as well as the
math. The output runner records complete `tests.log`, `tests.json` and exit code.
The main report is `FINAL_JOINT_RESPONSIBILITY_CAUSAL_REPORT.md`.

## What is implemented

- Availability by point/frame 0–10, source reprojection, actual source/owner IDs,
  original and reconstructed frame-1 priors/F/Q, production parity for 8444,
  and explicit `PARITY_NOT_VERIFIABLE` for incomplete 45588 archives.
- Complete local camera-0/1 pixel occupants, not merely a two-point-only scene.
  Other canonical points are frozen external competitors. Initial prediction is
  exact; later local environment assumptions are labelled `TIER_CONTROLLED_LOCAL`.
- Faithful J0 per-point baseline, including its original adequacy/fallback rule.
- Genuine 12-state conditional legacy diagnostics: current-ownership shared-noise
  control, atomic 45588→8444 slot transfer with **both states updated**, current
  referenced control, transferred referenced proxy, and source-only negative control.
- One rounded-pixel owner; bilinear support overlap is separately modelled through
  `L diag(sensor variance) L.T`. No jitter or independent duplicate observation is
  used. Changing owner never overrides masks or physical occlusion.
- Fixed prior, explicit physical/row support failures, deterministic GN updates
  and actual conditional-energy descent checks. These legacy proxies omit
  covariance derivatives and likelihood normalizers and are **not** a replacement
  for the undefined complete fixed-pixel model.
- Full conditional 12×12 covariance, shared-noise cross blocks, proper toy-mixture
  mixed Hessians and assignment-marginal cross covariance, complete covariance
  propagation tests, and proper Gaussian nuisance Schur/GLS equivalence tests.
- An explicit fixed raw foreground-pixel universe chosen before interventions;
  local assignment enumeration, normalized **partial** RGB/Z compatibility and
  proof that unspecified coverage/prior factors can reverse any finite ratio.
- Real `LATENT_REFERENCE` reported `NOT_IDENTIFIED`, with conditional per-channel
  Cauchy–Schwarz variance bounds. Synthetic proper-prior tests never become real
  calibrated nuisance parameters.
- Small archived controls for 60019, USB 67827/87290/85696 and 8416; no population
  factorial runs, GT recentering, static labels or source-pixel freezing.
- All requested applicable CSV/NPZ/provenance artifacts and measured RGB/depth
  ownership/cost-region figures, plus an explicit sequence-blocker file.

## Mathematical and archive limits — not silently replaced

The original archive does **not** trace 45588 or contain its full float64 6D
posterior and 6×6 covariance. Its XYZ is float64, but velocity/variance diagonals
are float32. The first missing checkpoint is
`storage_45135/traced/free_forward/covariance/frame_0001.npy`.
Both initial/frame-1 priors can be reconstructed from canonical source covariance,
recorded independent Pvv and exact F/Q; no missing posterior is invented.

A zero-size point and canonical color do not define the probability that an
arbitrary non-source pixel observes that material point. Source-ray quantization
uncertainty is not a calibrated feature-correspondence likelihood. Therefore the
complete fixed-observation J2/J3/J5 models and calibrated real latent-reference
inference are explicitly `UNDEFINED_MODEL`/`NOT_IDENTIFIED`. Local legacy
counterfactuals are quantified, but are not promoted into assignment probabilities.

Phase D is consequently blocked. No `joint_sequence_0_to_10.csv` or pair trajectory
plot is emitted to imply a valid ten-frame posterior. `joint_sequence_status.json`
records the model and exact data blockers. This is an allowed scientific
unidentifiability result, not a claim that tracking was fixed.

Exit 1 from `--experiment tests` means a failed/skipped test. Exit 3 from `all`
means its baseline/test/calibration gate failed; data can still be collected and
qualified. Exit 0 means the requested frame-1 diagnostic/report completed under
its declared limits, **not** that material identity or a ten-frame reconstruction
was solved. Always inspect the report and `archive_completeness.json`.
