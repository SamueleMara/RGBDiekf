# Restructure check (27 October 2026)

The repository was reorganized into the RigEuler3DGS layout. Code moved from `pointwise_rgbd/` and `rgbd_common/` into `scene/`, `estimator/`, `joint_reduction/`, `debug/`, `utils/` and `rigeuler_ref/`, and `train.py` became the single entry point. Only file locations and import lines changed. To confirm that no behaviour changed, the failing suites were rerun on an untouched copy of the pre-restructure tree (`RGBDekf_backup_before_restructure_20261027.tar.gz`).

| Test | Before (old tree) | After (new tree) |
|---|---|---|
| `test_image_rigid_joint` + `test_image_support` | 16 run, 5 fail | 16 run, the same 5 fail |
| prismatic forward, rotation error (rad) | 0.014129280864487219 | 0.014129280864487219 |
| revolute forward, rotation error (rad) | 0.034737438628860105 | 0.034737438628860105 |
| hinge from zero init, translation (m) | 0.11077879681979197 | 0.11077879681979197 |
| prismatic backward, translation (m) | 0.0021646228094443012 | (same test fails) |
| support recall of uncertain movers | 0.06422018348623854 | 0.06422018348623854 |

The full suite after the restructure is 63 tests: 58 pass and these 5 fail. The failures are pre-existing method failures, not restructure regressions. The default Phase-1 path also reproduces the saved v2 production states bitwise (frames 0–9, `reports/phase1_b1_small_contractions.md`).
