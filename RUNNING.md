# Current execution status

The full requested reconstruction is **not working/accepted**. Production validates the selected real input, then returns `UNSPECIFIED_MODEL_SELECTION`: theory C3/D7 and implementation Section 10 require a normalized group/association/model-size score that neither the theory nor approved noise config defines. Removed implementation-invented penalties are not used as defaults. This is a mathematical-definition gap; timing and noise guesses load automatically.

Read [LATEST_REQUEST_VALIDATION.md](LATEST_REQUEST_VALIDATION.md) for actual numerical failures and real attempt results. These commands reproduce the current behavior; they do **not** currently yield a completed reconstruction. Use fresh output directories:

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf
python main_iekf.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/iekf_requested_rerun'
python main_ukf.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/ukf_requested_rerun'
```

Both default to CUDA, all frames/cameras, scene timing and approved existing noise guesses. Default artifacts are compact; `--debug-artifacts` explicitly enables detailed exports. Resume never stops a running worker. The completed current test command was:

```bash
python -B scripts/run_parallel_tests.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/validation_requested_rerun'
```

It currently reports 126/127 passing, with the original UKF progressive motion assertion failing. No assertion or covariance was reduced to hide it.

---

Historical commands/experiment notes below describe earlier code and outputs; current behavior above takes precedence.

# Independent masked RGB-D pipelines — v2

`Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md` is authoritative. Both entry
points use its foreground-only observation model and deterministic M3 projection
rules. No visibility policy, confidence argument, timing boilerplate, or new
calibration file is required for the selected scene.

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf

python main_iekf.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/results_iekf_masked_v2'

python main_ukf.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/results_ukf_masked_v2'
```

These are full-scene CUDA invocations. Each reads all timestamps/cameras, auto
loads actual scene timing and `configs/example_uncalibrated.json`, and writes
runtime caches/logs to its output. Use a fresh output directory; old failed runs
are preserved. `--resume` requires an accepted v2 checkpoint with identical
original inputs and priors. Older observation-model checkpoints are incompatible.

The loader records whether acquisition metadata explicitly declares
`object_only_no_external_occluders` (or `no_external_occluders` / the inverse
`external_occluders`). Such a Boolean can be in the scene observation-model,
sequence, metadata, job/config, or conversion metadata. Unknown/false means a
mask-negative projection is undetermined; it supplies no background RGB/depth
measurement. Explicitly true means first-visible negative support is infeasible.
This is the M3 acquisition qualification, not a confidence knob. The selected
real scene currently supplies no such declaration; that limitation is recorded.
No input metadata is edited by the estimator.

Independent raw validity images, if supplied, are loaded from `depth_validity`,
`raw_depth_validity`, `depth_valid`, or `valid_depth`, per camera as frame PNG/NPY
binary images. Without a separate validity export, original finite positive
metric depth defines the recorded validity; this derivation is reported. Native
0/1 and 0/255 masks are accepted without thresholding. Invalid depth and all
mask-zero RGB/depth are excluded from the objective and inference.

Transport/projection, sparse matching, foreground backprojection, IEKF analytical
information reductions, and UKF nonlinear sigma/moment reductions use Torch
CUDA. Small manifold solves, initializer tracks, streaming and orchestration also
use CPU. `--device cpu` selects the explicit numerical reference. No installation
is needed beyond the existing environment. tqdm and the provisional dashboard
`visual_debug/index.html` expose actual progress; a live display is not evidence
of a completed reconstruction.

Run the numerical, CUDA, observation invariance and full input-integrity suite:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B scripts/run_parallel_tests.py \
  --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314' \
  --output '/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/tests_masked_v2_new'
```

The environment uses standard-library unittest; pytest is not installed. Reports
include JUnit, JSON, a source snapshot/hash, raw-ID mask assertions and complete
scene integrity. `MASK_PORT_VALIDATION.json` explicitly separates component
results from real reconstruction acceptance. See `VALIDATION.md` for actual
execution outcomes and outstanding failures. Conditional covariances do not
represent exact map marginals or a hard-truncated Bayesian posterior.

The current full-scene experiments and their exact source ZIPs are under
`parallel_masked_v2_20261020/full_scene_02/iekf` and
`parallel_masked_v2_20261020/full_scene_04/ukf`. These are existing active runs;
starting the examples above would create separate experiments. Current
acceptance is **not passed**: the UKF motion-recovery regression still fails,
and the full real temporal cycles have not finished. A verified-worker status
monitor updates `parallel_masked_v2_20261020/reports/v2_current_acceptance.json`
and its Markdown table. After both workers exit it writes
`reports/final_measured_acceptance.json`, retaining failed gates. Its existence
alone is never a success flag; inspect `all_acceptance_criteria_passed`.
