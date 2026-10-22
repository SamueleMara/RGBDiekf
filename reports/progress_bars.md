# Progress display — 18 October 2026

Tqdm 4.68.4 is already installed in the existing `rigeuler3dgs` environment.
Bars are enabled automatically for full image validation, reprojection
validation, canonical catalogue construction, objective evaluations, forward
IEKF, backward RTS, support-index chunks, ray tiles, PCG iterations, outer cycles
and prediction export. The current commands and SSD output destination remain
unchanged. Each bar shows completed work, elapsed time, rate and estimated time
remaining. Outer-cycle and PCG totals denote iteration budgets; reaching an early
stop below 100% does not imply that the estimator converged.

Bars use stderr, including when the launcher pipes output through `tee`.
The existing JSON cycle records use tqdm-safe writes on stdout; saved JSON/CSV
diagnostics and checkpoints retain their existing contents and semantics.
Nested bars are transient, throttled to 0.5-second refreshes, and delayed 0.2
seconds to avoid displaying very short work. `TQDM_DISABLE=1` disables all bars:

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
bash scripts/run_full_scene.sh --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314'
```

```bash
TQDM_DISABLE=1 bash scripts/run_full_scene.sh --validate-only
```

`tests/test_progress.py` runs the same named small production scene twice with
enabled and disabled bars through redirected (non-TTY) streams. Trajectory,
canonical-map and posterior arrays agree exactly. Covariance queries and cycle
diagnostics also agree exactly, excluding elapsed time and peak memory. The test
verifies visible bars on stderr, parseable JSON records on stdout, and silent
stderr when disabled. The complete numerical test log is `numerical_tests.txt`.

No new full-scene reconstruction was run for this presentation change. Existing
synthetic timing and full-scene convergence limitations remain documented in
`acceptance_status.md`.
