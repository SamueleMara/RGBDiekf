# All-camera corrective implementation — 24 October 2026

Production camera partitioning has been removed. Filtering, shared rigid refinement, both joint hypotheses and endpoint-relative support completion use every available camera. RigEuler coherence retains supernode cross-validation; no cameras are reserved. Earlier reserved-camera performance numbers are superseded.

## Source changes

- `free_phase1.py`: all cameras enter one point IEKF correction and one `factors.sqlite`; no camera-role masking or additional model z-buffer temporal culling. Per-camera attempted/matched diagnostics remain.
- `shared_rigid_rgbd.py`: one shared rigid trajectory fitted from all-camera raw factors, including proper canonical priors and motion/map Schur covariance.
- `canonical_schur.py`: additional endpoint-camera observations inform canonical coordinates while the conditioned endpoint pose has zero stochastic motion columns. Dense-reference covariance and cross-covariance tests include this case.
- `joint_rgbd.py`: both finite-transform joint reductions fit and score the same complete raw RGB-D observations. Production reports actual fitted likelihood/MAP residuals, not independent predictive performance. No reserved-camera veto.
- `joint_bidirectional.py`: consistent world/sign/endpoint-gauge comparison, unique all-camera sensor pooling, no independent posterior product.
- `joint_support.py`: identity versus endpoint-relative joint transport over all available multiview observations. It runs only when a joint is supported.
- Runner/caches/manifests: assert all available production cameras; reject camera-split trajectories and rigid fits. Old files remain preserved. At the source endpoint, only exact prior pixels are excluded; other-camera observations contribute with the pose gauge eliminated.
- Viewer, tests and reports: all-camera policy, per-camera residuals and actual support execution status. The camera selector changes display only.

Baseline process, point IEKF, observation, projection, association and controller source files remain unchanged. The current noise configuration and authorized 1% association tail are unchanged. No physical supernode splitting, alternate estimator or camera validation architecture was added.

## Executed numerical checks

- tests_free_endpoint_covariance_final: 34 tests, 0 failures, 0 errors, 0 skips; CUDA available: True.
- tests_baseline: 45 tests, 0 failures, 0 errors, 0 skips; CUDA available: True.

Tests include both endpoint signed-time updates with eight cameras at every frame, foreground-only unique raw factors, eight-camera shared rigid fitting and support completion, same-set joint selection, camera-split cache rejection, candidate-specific degeneracy, dense-reference multistep covariance/cross-covariance and CPU/CUDA agreement.

## Re-evaluated computational group settings

Same actual blade input, first eight frames, every camera, identical initialization/noise/association settings. This is explicitly a limited TEST benchmark, not a complete reconstruction. The residuals are fitted/conditioned correspondence residuals.

| Occupancy | Groups | Accepted raw factors | u/v RMSE (px) | Depth RMSE (mm) | Filter time (s) |
|---|---:|---:|---:|---:|---:|
| 64 | 582 | 240187 | 0.5358 / 0.4138 | 5.9876 | 23.67 |
| 128 | 315 | 240031 | 0.4985 / 0.3765 | 5.1568 | 21.15 |
| 256 | 182 | 239570 | 0.4702 / 0.3688 | 4.6134 | 21.82 |

## blade_corrected_64

Actual status: **INCOMPLETE_OR_UNRESOLVED**; complete: **False**. Full source: True. Affected-stage rerun runtime: 239.43 s (completed all-camera point filtering and rigid initialization are reused from manifest paths). Peak RSS: 1.995 GB. CUDA allocated peak: 0.091 GB.

Input: 60 timestamps × 8 cameras. FPS: 59.0 Hz, explicit user-authorized override; scene FPS is not fabricated.

### first

34369 individual endpoint points, 582 computational groups. Full 60-frame pass, all-camera participation checked at every step. Accepted raw factors by camera: [329991, 269734, 122228, 268309, 324952, 271386, 111755, 279750].
Minimum 6×6 marginal eigenvalue: 3.55555e-07; maximum checked covariance inverse-solve relative residual: 1.1035e-06. Nonlinear group iteration-limit count: 0; line-search stops: 0.

Coherence: RESOLVED. Shared rigid MAP: FIT, converged: True. Joint: JOINT_TYPE_UNRESOLVED. Support: NOT_EXECUTED.

Endpoint evidence (source priors once plus additional observations): {"prior_raw_count_by_camera": [118, 88, 61, 154, 197, 206, 155, 151], "additional_raw_count_by_camera": [43, 33, 26, 46, 46, 50, 33, 40], "source_prior_raw_pixels_reassimilated": 0}
- PRISMATIC: INCONSISTENT_OR_UNCONVERGED; same-set raw count 44794; raw chi² 5687645.21565885; local conditional p-value 0.0. 
- REVOLUTE: INCONSISTENT_OR_UNCONVERGED; same-set raw count 44794; raw chi² 4462219.718972152; local conditional p-value 0.0. 
### last

38622 individual endpoint points, 671 computational groups. Full 60-frame pass, all-camera participation checked at every step. Accepted raw factors by camera: [334817, 271751, 121090, 270641, 329857, 279951, 131348, 291981].
Minimum 6×6 marginal eigenvalue: 3.57424e-07; maximum checked covariance inverse-solve relative residual: 5.18408e-07. Nonlinear group iteration-limit count: 0; line-search stops: 0.

Coherence: RESOLVED. Shared rigid MAP: FIT, converged: True. Joint: JOINT_TYPE_UNRESOLVED. Support: NOT_EXECUTED.

Endpoint evidence (source priors once plus additional observations): {"prior_raw_count_by_camera": [1, 22, 150, 28, 9, 0, 0, 2], "additional_raw_count_by_camera": [17, 15, 16, 16, 5, 0, 0, 8], "source_prior_raw_pixels_reassimilated": 0}
- PRISMATIC: INCONSISTENT_OR_UNCONVERGED; same-set raw count 8637; raw chi² 383876.5130399039; local conditional p-value 0.0. 
- REVOLUTE: INCONSISTENT_OR_UNCONVERGED; same-set raw count 8637; raw chi² 348757.14016895025; local conditional p-value 0.0. 

World-coordinate directional comparison:
```json
{
  "compatible": false,
  "reason": "At least one direction has no identifiable selected joint",
  "candidate_geometry_diagnostics": {
    "PRISMATIC": {
      "compatible": false,
      "type": "PRISMATIC",
      "axis_sign": 1,
      "axis_angle_radians": 1.4320106200598632,
      "axis_difference_d2": 35670736.971206546,
      "chronological_displacement_difference_d2": 6665054.771931096,
      "displacement_rmse": 0.2231625641357078,
      "world_reference": "fixed calibrated world; q[0]=0 in both directions; no point-ID correspondence assumed",
      "uncertainty": "2*(P_forward+P_backward) upper bound under unknown correlation; never independent-product confidence"
    },
    "REVOLUTE": {
      "compatible": false,
      "type": "REVOLUTE",
      "axis_sign": 1,
      "axis_angle_radians": 0.769393812677751,
      "axis_difference_d2": 407538.7667920694,
      "chronological_displacement_difference_d2": 224101.9835503273,
      "displacement_rmse": 0.35206769136811755,
      "world_reference": "fixed calibrated world; q[0]=0 in both directions; no point-ID correspondence assumed",
      "uncertainty": "2*(P_forward+P_backward) upper bound under unknown correlation; never independent-product confidence",
      "revolute_line_distance_m": 0.8086692180802542,
      "line_difference_d2": 916105.6493006676
    }
  }
}
```
Original full filtering/rigid-initialization execution: 804.73 s, retained at /media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/all_cameras_20261024/blade_full_64. Its endpoint-omitting C/D results are superseded; only compatible point histories and rigid initialization were reused.

Actual viewer and data: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/all_cameras_20261024/blade_corrected_64/viewer/index.html`. Full logs: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/all_cameras_20261024/blade_corrected_64.log`. Covariance, trajectory and raw-factor arrays remain under each direction. An unresolved result is not successful joint-guided support completion.

## blade_corrected_256

Actual status: **INCOMPLETE_OR_UNRESOLVED**; complete: **False**. Full source: True. Affected-stage rerun runtime: 286.09 s (completed all-camera point filtering and rigid initialization are reused from manifest paths). Peak RSS: 3.045 GB. CUDA allocated peak: 1.622 GB.

Input: 60 timestamps × 8 cameras. FPS: 59.0 Hz, explicit user-authorized override; scene FPS is not fabricated.

### first

34369 individual endpoint points, 182 computational groups. Full 60-frame pass, all-camera participation checked at every step. Accepted raw factors by camera: [332111, 270533, 120623, 269583, 330407, 271348, 103565, 276413].
Minimum 6×6 marginal eigenvalue: 3.27196e-07; maximum checked covariance inverse-solve relative residual: 2.45738e-07. Nonlinear group iteration-limit count: 0; line-search stops: 0.

Coherence: RESOLVED. Shared rigid MAP: FIT, converged: True. Joint: JOINT_TYPE_UNRESOLVED. Support: NOT_EXECUTED.

Endpoint evidence (source priors once plus additional observations): {"prior_raw_count_by_camera": [5896, 4717, 2010, 4704, 5770, 4612, 1882, 4737], "additional_raw_count_by_camera": [0, 0, 15, 0, 0, 0, 1, 0], "source_prior_raw_pixels_reassimilated": 0}
- PRISMATIC: INCONSISTENT_OR_UNCONVERGED; same-set raw count 1113135; raw chi² 112205563.47165433; local conditional p-value 0.0. 
- REVOLUTE: INCONSISTENT_OR_UNCONVERGED; same-set raw count 1113135; raw chi² 98160049.10975078; local conditional p-value 0.0. 
### last

38622 individual endpoint points, 221 computational groups. Full 60-frame pass, all-camera participation checked at every step. Accepted raw factors by camera: [338479, 274222, 120922, 271509, 332624, 279391, 129277, 290316].
Minimum 6×6 marginal eigenvalue: 3.80355e-07; maximum checked covariance inverse-solve relative residual: 2.11653e-07. Nonlinear group iteration-limit count: 0; line-search stops: 0.

Coherence: RESOLVED. Shared rigid MAP: FIT, converged: True. Joint: JOINT_TYPE_UNRESOLVED. Support: NOT_EXECUTED.

Endpoint evidence (source priors once plus additional observations): {"prior_raw_count_by_camera": [1, 22, 127, 28, 9, 0, 0, 2], "additional_raw_count_by_camera": [17, 15, 16, 16, 5, 0, 0, 8], "source_prior_raw_pixels_reassimilated": 0}
- PRISMATIC: INCONSISTENT_OR_UNCONVERGED; same-set raw count 11561; raw chi² 1345771.7116086434; local conditional p-value 0.0. 
- REVOLUTE: INCONSISTENT_OR_UNCONVERGED; same-set raw count 11561; raw chi² 1240580.0197008152; local conditional p-value 0.0. 

World-coordinate directional comparison:
```json
{
  "compatible": false,
  "reason": "At least one direction has no identifiable selected joint",
  "candidate_geometry_diagnostics": {
    "PRISMATIC": {
      "compatible": false,
      "type": "PRISMATIC",
      "axis_sign": -1,
      "axis_angle_radians": 0.06028209850157737,
      "axis_difference_d2": 142140.92420070065,
      "chronological_displacement_difference_d2": 15450362.742528291,
      "displacement_rmse": 0.38213198638605267,
      "world_reference": "fixed calibrated world; q[0]=0 in both directions; no point-ID correspondence assumed",
      "uncertainty": "2*(P_forward+P_backward) upper bound under unknown correlation; never independent-product confidence"
    },
    "REVOLUTE": {
      "compatible": false,
      "type": "REVOLUTE",
      "axis_sign": -1,
      "axis_angle_radians": 0.08536629630064581,
      "axis_difference_d2": 76924.81007966239,
      "chronological_displacement_difference_d2": 1181863.1006909243,
      "displacement_rmse": 0.14973121031215733,
      "world_reference": "fixed calibrated world; q[0]=0 in both directions; no point-ID correspondence assumed",
      "uncertainty": "2*(P_forward+P_backward) upper bound under unknown correlation; never independent-product confidence",
      "revolute_line_distance_m": 0.8752424283998513,
      "line_difference_d2": 7493129.329643934
    }
  }
}
```
Original full filtering/rigid-initialization execution: 1811.20 s, retained at /media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/all_cameras_20261024/blade_full_256. Its endpoint-omitting C/D results are superseded; only compatible point histories and rigid initialization were reused.

Actual viewer and data: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/all_cameras_20261024/blade_corrected_256/viewer/index.html`. Full logs: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/all_cameras_20261024/blade_corrected_256.log`. Covariance, trajectory and raw-factor arrays remain under each direction. An unresolved result is not successful joint-guided support completion.

## Requested corrections: evidence and remaining outcome

| Request | Verification |
|---|---|
| Remove production camera splits | One all-camera factor store and no role mask in A/C/D/E/F. Old split caches error. |
| Joint 6D point updates | All eight camera factors enter each grouped point update; independent full-frame factor audit. |
| Supernode cross-validation | Reference 481b2cf2 node folds retained unchanged; no camera folds. |
| One shared rigid trajectory | All-camera motion/map MAP, original endpoint priors and Schur coupling; per-run convergence above. |
| Revolute/prismatic reduction | Uncentred translation PCA, finite rotation-log PCA, gauged finite-axis solve and accumulated relative rotations initialize both hypotheses; joint fits retain fixed geometry and q(t). |
| Same-set candidate comparison | Exact same raw factor list for both models; per-camera counts and residuals saved, no camera assessment gate. |
| Independent complete directions | Actual first/last catalogues and chronological indexing; common world/sign/line-gauge diagnostics saved. |
| All-camera completion | Eight-camera integration regression passes. On any unresolved full scene it remains explicitly NOT_EXECUTED because reliable joint information is unavailable. |
| Configuration / manifests / viewer | Derived all-available camera policy, cache guards, per-camera diagnostics and actual browser checks. Approved noise/timing settings retained. |
| Preserve architecture | No alternate estimator, physical splitting or new camera-validation architecture. Baseline preserved. |

blade_corrected_64 actual browser checks: 4; all passed: True. Display-only source snapshots, when refreshed after estimation, are stored under viewer/source_snapshot and display_only_refresh.json; no estimator array was changed.
blade_corrected_256 actual browser checks: 2; all passed: True. Display-only source snapshots, when refreshed after estimation, are stored under viewer/source_snapshot and display_only_refresh.json; no estimator array was changed.

## Statistical and correspondence limits

Noise values remain approved initial guesses, not independent calibration. All-camera residuals are in-sample and conditioned on finite kNN associations and fixed camera calibration. They cannot establish physical accuracy by themselves. The existing 1% compatibility/observability rules remain explicit; the inherited BIC classifier was removed as required by the verified reference; no tolerance was retuned to obtain an accepted joint.

All actual sensor factors must be valid foreground. The metadata does not declare the object-only/no-external-occluder premise, so negative-mask predictions remain undetermined under the existing contract; no background values are used as explanations. This is separate from the mandatory positive-foreground rule.

## Reproduction

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/run_free_grouped_rgbd_joint.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/blade_103706" \
  --output "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/blade_all_cameras_new_run" \
  --fps 59 --device cuda --anchor both --target-occupancy 64
```

Choose a fresh output folder. No frame or camera limit is required for production. Use occupancy 256 only to reproduce that explicitly reported comparison configuration. The default remains 64.
