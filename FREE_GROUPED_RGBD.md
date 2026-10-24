# Free-point grouped RGB-D correction — 24 October 2026

Authority: the 24 October user instruction to use ALL available cameras in every production stage, and the user correction “RGBDiekf — Correct implementation based on RigEuler3DGS” (attachment `324765a9-6f19-4ec2-aa6e-4aa51f15cbdb/Pasted text.txt`). It supersedes the physical-supernode/boundary-splitting interpretation. The historical audits and results remain preserved.

The current executable is `tools/run_free_grouped_rgbd_joint.py`; `tools/run_grouped_rgbd_joint.py` routes directly to it. This is a point IEKF pipeline. It does not run the older rigid-body IEKF/UKF or discovery controllers.

## Implementation

| Stage | Source | Behavior |
|---|---|---|
| A | `free_phase1.py`, `grouped_point_iekf.py`, `grouped_covariance.py` | All measured endpoint points have individual XYZ/physical-time velocity states and 6×6 marginals. Fixed canonical computational groups restrict each correction, not its subsequent trajectory. Independently initialized first/last endpoint passes cover the full sequence. |
| B | `free_coherence.py`, `reference_motion.py`, `reference_coherence.py` | Separate representative graph; exact reference log-distance strain, graph hierarchy and held-out complement-relative rigid validation. Other points remain unclassified. |
| C | `shared_rigid_rgbd.py`, `canonical_schur.py` | Weighted Kabsch initialization, then common per-frame SE(3) RGB-D fitting. Measured canonical-point uncertainty and shared motion/map coupling are retained. |
| D | `joint_rgbd.py` | Fixed prismatic/revolute geometry and chronological q, candidate-specific identifiability, the same complete all-camera raw factors. No BIC classifier or reserved-camera gate. |
| E | `joint_bidirectional.py` | World/gauge/sign comparison without matching endpoint point IDs. Compatible estimates are refitted using unique raw observations and source priors, rather than multiplying correlated directional posteriors. |
| F | `joint_support.py` | Existing first/last endpoint points only; static versus endpoint-relative joint prediction, common visible raw evidence, nonexclusive surface support. No recovery or new point insertion. |
| G | `free_visualization.py`, `assets/free_viewer.html` | Actual full-resolution RGB-D and point clouds, free trajectories, groups, core, fitted rigid/joint geometry, support labels, residuals and direct observation flags. No hidden display sampling. |

## Covariance and mathematical scope

The physical correction basis expands translation/rotation increments into XYZ and linear/angular velocity increments into individual linear velocities. An invertible normalization on its structural range improves conditioning without changing that range. It is not a persistent physical group pose.

The restricted linearized gain is `K = G [Gᵀ(P⁻¹ + HᵀR⁻¹H)G]⁻¹ GᵀHᵀR⁻¹`, with inversion on the structural range. Covariance is Joseph covariance of this chosen estimator, not expanded inverse reduced Hessian and not a claim of unrestricted Bayes optimality.

Production retains `P = D + G C Gᵀ + G Vᵀ + V Gᵀ` separately within each computational block. `D` consists of 6×6 point blocks; `V` has 12 columns per block. Because fixed canonical `G` satisfies `F G = G T`, prediction preserves this representation. A QR/Cholesky solve in at most 24 dimensions per block retains cross-point correlations without dense point-pair storage or growing history. Actual solve residuals are checked against the documented numerical tolerance (1e-5). No jitter, artificial covariance floor or discarded-correlation projection is used.

The rigid/joint stages jointly estimate shared motion and measured canonical coordinates. Sparse point elimination leaves only a small motion Schur matrix. Source priors are used once. Saved `Sinv`, `U` and shared motion covariance represent pose-point and cross-point coupling without materializing point-point matrices. The production comparison uses fitted raw multiview residuals and the proper MAP objective; it does not present in-sample data as independent posterior prediction.

These are local linearized Gaussian uncertainties, conditional on calibration, selected correspondences, chosen model and approved uncalibrated noise assumptions. They are not independent physical calibration or exact multimodal inference.

## Evidence and foreground rules

Every endpoint observation is mask-positive with valid metric depth. Every available camera jointly contributes to each per-point update and the shared rigid and joint fits. Both hypotheses are scored on the same complete multiview set. Endpoint source pixels enter once as canonical priors. Other valid endpoint-camera pixels still enter the rigid/map and joint objectives; excluding source priors must never exclude an entire frame. The endpoint pose gauge has no stochastic pose columns, while its additional observations correctly inform canonical coordinates and their cross-time coupling. The reference coherence cross-validation splits SUPERNODES, never cameras. Camera-partitioned caches are rejected; their point states and covariance cannot be reused unchanged.

Temporal association retains the original pointwise measured-foreground-depth matcher, without an added model z-buffer culling step. The separate nominal hard negative-mask check uses deterministic point first hits (depth, then point ID). Background RGB/depth cannot explain occlusion or affect association. Under the v2 M3 contract, a mask-negative prediction is a hard contradiction only with an explicit object-only/no-external-occluder acquisition premise; otherwise it is undetermined. The input premise is recorded, never inferred from the object name. Foreground depth can establish observed occlusion using the existing uncertainty policy.

The 1% chi-square tail is the user-authorized, uncalibrated association-policy assumption. It is unchanged. RGB uses the existing covariance-based appearance likelihood; XYZ cosine is never used. Correlated directional evidence is never multiplied as independent data. Support classification uses conditional likelihoods and explicitly does not claim independent point labels.

## Reproduce

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/run_free_grouped_rgbd_joint.py \
  --input "/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/preparation_8cams/partnet_subset_fixed_render_linear/blade_103706" \
  --output "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/blade_all_cameras_run" \
  --fps 59 --device cuda
```

The output must be fresh. `--fps 59` is the user's explicit timing assumption for this synthetic scene; the general loader otherwise reads actual scene timing. Noise guesses load from the approved configuration automatically. `--anchor both` is the default. Explicit frame/camera limits are tests only. The viewer is `OUTPUT/viewer/index.html`, served with `python -m http.server --directory OUTPUT/viewer 8765`.

Numerical tests: `python -B tools/test_free_rgbd.py --output /tmp/rgbdekf_free_tests`. Baseline regression tests: `python -B tools/test_pointwise_rgbd.py --output /tmp/rgbdekf_baseline_tests`.

Unresolved or failed model validation returns a nonzero exit and writes its actual status. It must not be described as completed joint-guided reconstruction. See the measured validation report for experiment outcomes.
