# F1 fifth-pass investigation: implementation and execution

This implements investigation tooling for
`RGBDiekf_Definitive_F1_Static_Drift_Causal_Forensics_Codex.md`.
It does not change or launch production inference. The user runs the tests and
investigation. No tests or fifth-pass experiments were executed during implementation.

## Repository audit at implementation time

- Branch `main`, HEAD `e1e663b` (`mask measurements implmentation`).
- Seven existing tracked modifications, 117 insertions / 22 deletions, were
  already present: `estimator/free_projective_run.py`,
  `estimator/projective_sensor.py`, `estimator/projective_terms.py`,
  `tools/root_cause_oracle.py`, `tools/root_cause_trace.py`,
  `tools/run_status.py`, and `train.py`. They were preserved.
- Existing untracked temporal-estimation modules, tests, configurations and
  earlier forensic reports were preserved. No commit or push was performed.
- Current F1 is a direct projective RGB-D, per-point `[x,v]` filter. The new
  tooling reads its archived production traces and uses the existing independent
  NumPy single-point oracle. It does not invoke grouped estimation, Gaussian
  rendering, RTS, IEKS, EM, training, or a complete scene filter.
- The three archived scenes and earlier second/third/fourth-pass evidence are
  present. Actual Storage/USB baseline directories come from their saved
  `forensics/trace_plan.json`, rather than an invented `scene/baseline` path.
- Read-only inspection confirms the USB 67827 frame-1 accepted set includes
  cameras **5 and 6**. Camera 4 must not be attributed to that update.

The runtime inventory records the actual commit, dirty diff, source hashes,
manifests, policies, point mapping, original RGB/depth/mask/validity files,
calibration and timing metadata. It hashes baseline arrays and existing traces
before numerical work, and checks input/source immutability afterward.

## Run everything

From the repository, using the existing environment:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/rgbdie_f1_mpl \
  python -u -B tools/f1_definitive_drift_forensics.py all
```

Defaults are the specification's existing evidence directory and a **new**
output directory:

```text
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030
└── fifth_pass_definitive_static_drift_20261030/
```

If that output already exists, select a fresh name beginning with `fifth_pass_`
using `--out`. Prior evidence and reports are never overwritten. Example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/rgbdie_f1_mpl \
  python -u -B tools/f1_definitive_drift_forensics.py all \
  --out "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030/fifth_pass_definitive_static_drift_20261030_run2"
```

CPU-only execution is intentional: this is a small-point independent numerical
audit of saved GPU production results. Hashing and image I/O still read substantial
data. Per-file-stage, test and point progress are printed. Test details go to
`tests.log`; no previous 328-frame optimization is restarted.

## Separate test, investigation and report commands

All stages use the same output directory; `preflight` alone creates it.

```bash
python -u -B tools/f1_definitive_drift_forensics.py preflight
python -u -B tools/f1_definitive_drift_forensics.py test
python -u -B tools/f1_definitive_drift_forensics.py analyze
python -u -B tools/f1_definitive_drift_forensics.py report
```

For custom paths, pass the same `--out` to every stage; pass `--forensics-root`
to `preflight` (or `all`). The inventory supplies later stages' source paths.
Earlier legacy tests retain their original archived-data locations; moving
the evidence archive requires adjusting their fixture constants, otherwise their
skips/failures prevent an accepted verdict. The supplied default paths need no edits.

Report generation also has its own executable and never runs replays:

```bash
python -B tools/report_f1_definitive_drift.py \
  --out "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030/fifth_pass_definitive_static_drift_20261030"
```

The report refuses to replace an existing report. The test stage similarly
preserves an existing `tests.log`. `all` stops before analysis if tests fail or
skip; it writes `TEST_GATE_FAILURE.md`, prints the saved test-log path and exits
with code 1. This blocker report never claims an accepted root cause. The explicit
`analyze` command can still produce qualified diagnostics after a test failure,
while blocking causal benefit claims. Exit code 3 from `all`/`report`
means the evidence package failed validation. A scientific `NOT_IDENTIFIABLE`
does not mean a software test failure or permission to invent missing parameters.

The original fifth-pass run encountered a runner indexing bug: some fourth-pass
cases contain five saved frames, while production traces cover ten. The corrected
runner compares every available historical frame and marks later ones
`NO_ARCHIVED_FRAME`; it still checks **all ten** frames against production. The
position/velocity tolerances are unchanged. A regression test verifies that a production
mismatch beyond the historical archive's end still fails.

Covariance diagnostics now include explicit per-block norms, as requested by G0:
`||Pblock - Prefblock||F <= 1e-12 + 1e-9 * ||Prefblock||F`, in each block's
units (xx m², xv/vx m²/s, vv m²/s²). These are additional diagnostics;
**the original stricter entrywise `np.allclose` still decides covariance parity**.
No mean/covariance tolerance, estimator or noise is changed.
`covariance_elementwise_equal` and `covariance_block_norm_equal` distinguish
the two checks, with every block's error, scale and bound.
After fixing the crash, USB 51975 frame 2 exposed this distinction: independent
covariance entries differ by about 3e-11 while state and camera decisions agree;
the recorded production information matrix has condition number about 1e9.
This remains a failed strict parity test; it is not converted to a success.
USB 51974 also exceeds the block-relative covariance bound at frame 10.
A regression verifies that even passing block norms cannot override a failed
original strict covariance check. These discrepancies must be resolved before
an accepted forensic verdict; a failed test is not hidden to let `all` proceed.

After updating the runner, use a fresh output name (for example the `_run2`
command above). The failed run and its input/source inventory remain unchanged;
its source hashes must not be edited to make updated code appear identical.
Powell warnings from invalid-support search trials are now recorded in
`solver_oracles.csv`; a nonfinite final point is explicitly a failed numerical
oracle, never a successful reconstruction or a zero residual.

## What the code computes

- All 19 mandatory cases, continuous A0 frames 1–10, componentwise mean/covariance
  comparisons against production and available fourth-pass arrays, decisions,
  accepted camera sets, prediction closure and first divergence.
- Archived raw/frozen/admitted/solver/final camera lifecycle; exact six-state
  information equations, same-total-Hessian camera/channel force closure,
  prior-whitened and position eigensystems, decimal-70 same-operator solve checks,
  accepted-iterate stationarity and exact position-to-velocity transfer.
- Actual original frame-0/onset pixels, production float32 samples, calibration,
  weights, gradients, branch representation, residuals, noise and innovation
  statistics. Missing references stay missing, never zero.
- Frozen-image, single-camera field replacement/removal, RGB/depth/reference
  arms, frozen physical support, exact offline zero-velocity constraint and
  explicitly labelled prior sensitivity. Each sequence propagates its own state
  and covariance. Later altered priors prevent a fixed-context causal label.
- Same-prior one-update inverse-depth and non-source-camera color sensitivity;
  local inverse-depth plane consistency and conditional displacement components.
  These do not establish physical face identity or change sensor defaults.
- Exact fixed-linearization camera-subset Shapley vectors, separately labelled
  from force closure and nonlinear interventions.
- Actual accepted H/R/F temporal observability segmented at support/cell changes;
  conditional local stationary/moving/nuisance models and held-out scores.
  Proper shared-reference Schur/GLS covariance oracles include correlated-source
  controls. No numerical physical-bias prior is invented.
- One-sided pixel perturbations, comparable-support nonlinear objectives and a
  safeguarded numerical minimization oracle; invalid support is undefined.
- All saved baseline frames' vector drift closure, accepted-camera chronology,
  moving-point evaluation-only GT checks and late camera gate evidence.
- Manifest-seeded random static holdouts selected before residual inspection,
  stratified representatives, measured non-source stable controls, denominators,
  raw temporal holdout samples and cross-point camera gain/offset holdout scores.

## Outputs and limitations

Main report: `DEFINITIVE_F1_STATIC_DRIFT_ROOT_CAUSE.md`.
Disconfirmation: `OPEN_QUESTIONS_AND_DISCONFIRMING_EVIDENCE.md`.
Validation: `report_validation.json`, `tests.json`, `tests.log`,
`evidence_inventory.json`, `output_hashes.json`, `analysis.json`.
The requested CSV tables, indexed information NPZ, reference files, curated raw
image/objective/trajectory/branch figures and numerical objective grids accompany
them. The test command includes the new fifth-pass tests, all 17 fourth-pass
tests and the applicable earlier oracle/unit modules.

Do not call implementation a completed investigation before those commands run.
In particular:

- Known boundary nonparity remains a qualification, never an artificial pass.
- Untraced holdouts lack archived full covariance/owner histories, so their raw
  observations and saved-array statistics cannot certify causal point replays.
- Archived owner lists can be truncated; total occupancy is recorded separately.
  Missing competing inner-cloud states cannot be recovered by inventing them.
- Mode C uses the earlier diagonal competitor-covariance approximation and is
  explicitly qualified; it is not an exact fully coupled cloud counterfactual.
- A latent physical-bias posterior is not identifiable without an independently
  specified proper nuisance prior and source/cross-time correlation model.
  The small synthetic Schur/GLS equivalence check does not supply either.
- A conditional plane fit cannot certify material identity at a crease. Metadata
  defines the archived fixed camera rig; independent camera/calibration accuracy
  and dense face identity remain unverified.
- A failed or incomplete package is reported as such. No root-cause outcome,
  experimental timing, test success, or physical uniqueness is pre-filled.
