# Fifth-pass source responsibility diagnostics

Implementation of `RGBDiekf_Codex_Fifth_Pass_Source_Responsibility_Causal_Experiments.md`.
Everything is CPU-only and investigation-only. No production file, configuration,
noise assumption, saved trajectory or earlier diagnostic report is modified.

## Run the tests with a complete saved log

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_source_mpl \
python -u -B tools/f1_source_responsibility_audit.py \
  --experiment tests \
  --out "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030/fifth_pass_source_responsibility_tests_20261030"
```

This creates a **fresh** directory, inventories/hashes the actual archived inputs,
runs `tests.test_f1_source_responsibility` and the 17 earlier fourth-pass/unit
checks, and writes the full `tests.log`, `tests.json`, exit code and inventory.
It does not run the factorial or population experiment. Each test name is printed
as it starts; complete details remain in the log. Existing output directories are
refused. For a repeat, append a new suffix such as `_run2` to `--out`.

For just the new tests, without a new inventory/output directory:

```bash
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -m unittest tests.test_f1_source_responsibility -v
```

## Run the actual source-responsibility investigation separately

```bash
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/rgbdie_source_mpl \
python -u -B tools/f1_source_responsibility_audit.py \
  --forensics-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030" \
  --fourth-pass-root "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030/fourth_pass_sequential_replay_20261030" \
  --out "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/forensics_F1_inner_drift_20261030/fifth_pass_source_responsibility_20261030" \
  --frames 0:10 --experiment all
```

`all` starts with the complete test gate. On a test failure it writes an explicitly
failed report and starts no experiments. Successful tests do not automatically
qualify every dataset point: each point still has its own production/archived
parity check. Failed points remain in the tables, with their interventions blocked.
Exit 3 means an actual point still fails parity; it does not mean the report was
silently accepted. `--experiment report` regenerates only from saved artifacts
in an existing output directory and refuses to overwrite an existing report.

The population audit is frame-0/1 likelihood/ownership evaluation in bounded
chunks, plus reads of saved endpoints. It is not a scene filter. The optional
`--no-population` is an explicit incomplete-audit option, recorded as a limitation.
The compute chunk size is `--population-chunk-size`; it is not a geometry gate.

## Scope and safeguards

- A–F independently vary eligibility and reference subtraction. Frame-0
  material ambiguity is run as admit/exclude sensitivity; missing references
  have separate retain-original/omit variants. Referenced variants include both
  original R (`ALGEBRA_ONLY`) and R+R0 sensitivity, explicitly lacking shared
  cross-time covariance. No referenced sequential covariance is called exact.
- Frame-1 and first-measurement comparisons use identical saved priors, separately
  from continuous 0–10 arms whose own posteriors propagate. Original common-support
  and modified-model objectives are labelled separately. Invalid support is undefined.
- Source-priority bypasses only a duplicate winner, and still respects raw masks,
  invalid depth, FOV, measured-depth occlusion and self-occlusion. Its named
  displaced owner loses the slot in the counterfactual interpretation. No combined
  independent posterior or globally valid double assimilation is claimed.
- Source IDs stay permanent while pixels reproject at the current state. Depth
  agreement never becomes `VERIFIED_MATERIAL_ID`. Report decisions can remain
  `INDETERMINATE` even if an arm holds a static point still.
- Mode P uses exact archived frozen ownership. Mode C reuses the fourth-pass
  competitor reconstruction, including its diagonal covariance approximation and
  near-tie flags. C interventions are blocked when its own A baseline cannot be
  reproduced. No discarded uncertainty is silently filled with zero.
- Each frame's actual camera IDs, raw reasons, reference status and source probe
  remain available. Frame-0 exact ownership uses the production visibility routine
  on the CPU. Archived front entries give actual winner IDs, depths, uncertainties
  and source provenance; the archive may truncate the remaining occupants.
- The source module never imports GT evaluation. Static-cohort GT selection occurs
  only after image features; moving GT is loaded by the disconnected evaluator
  after replays. Missing material identity remains explicit.
- Saved population velocities and some variances are float32. Untraced population
  force calculations are conditional accepted-state linearizations, not exact
  full-precision accepted-step reconstructions. Full numerical provenance is saved.
- Local RigEuler source is hashed and cited by function/line. Contributor freezing
  is per linear solve, followed by nonlinear re-rendering; it is not permanent
  frame-0 pixel ownership. No renderer is imported into this point audit.

Outputs include the requested responsibility/conflict/arm/state/row/event/population
CSVs, `information_steps.npz`, selection records and exclusions, original input
hashes, owner/image/world-coordinate figures, test logs and
`SOURCE_RESPONSIBILITY_CAUSAL_REPORT.md`. The report does not automatically emit a
candidate production design: material/bias ambiguity must first be resolved.

The full dataset factorial/population investigation is left for the user command;
small fixtures and archived-point integration tests are the implementation checks.
