# Audit of the free-point grouped correction

Updated 24 October: ALL available cameras are mandatory at every production frame and stage. No reserved-camera architecture remains in this executable.

The current authority is the user attachment “RGBDiekf — Correct implementation based on RigEuler3DGS”. The persistent physical-supernode interpretation and its boundary-splitting gate are superseded. Baseline pointwise files, input datasets and previous experiment outputs are preserved. No commit or push has been performed by this implementation task.

## Corrections made

| Finding | Actual correction |
|---|---|
| Computational coordinates had become permanent rigid body states. | The new default runner stores individual 6D point states and applies grouped increments only. No group poses are propagated in Phase 1. |
| Keeping only Joseph marginals discarded correlations and eventually lost numerical positive definiteness. | Production now retains the closed block-sparse/rank-24 covariance representation across prediction and correction; inverse applications have checked residuals. The earlier failed experiments remain preserved. |
| Rigid/joint fitting treated measured canonical coordinates as exact. | Actual source-point covariance is retained in the joint motion/map MAP. Sparse Schur factors represent shared point, pose and cross-time uncertainty. The dense reference tests verify their covariance and predictive scores. |
| Background depth had been used to explain occlusion. | Temporal correspondence preserves the existing measured-foreground-depth matcher. The separate hard nominal mask check uses model first hits; neither reads background RGB/depth. Perturbing background RGB/depth cannot affect associations; CPU/CUDA regressions verify this. |
| Endpoint identity was inferred from motion. | Production now passes the actual source endpoint explicitly to coherence and rigid fitting, including static sequences. |
| Directional estimates reuse the same sensor data. | Compare in a common gauge, bound unknown cross-correlation, and refit on unique raw returns with both endpoint priors. Never multiply the directional posteriors as independent evidence. |
| One-to-one tracking ownership could hide represented surface samples. | Final support classification uses nonexclusive geometric correspondence. It does not assimilate an extra Kalman measurement or claim independent labels for duplicated endpoint samples. |
| A degenerate revolute candidate could veto a valid prismatic fit. | Candidate-specific rank, motion observability and measurement consistency. Pure translation, finite rotation, static, negligible-motion and incompatible-direction regressions are included. |
| Camera partitions excluded available information and could veto a meaningful joint. | One all-camera factor stream supplies filtering, rigid fitting, both joint hypotheses and completion. Candidate-specific observability and the same complete multiview residuals determine support; there is no reserved-camera admission gate. |
| Viewer geometry used the wrong canonical means after map refinement. | Measured endpoint geometry and fitted canonical MAP geometry are exported separately. Rigid/joint display uses its actual fitted geometry. |

The new path does not execute the preserved `phase1_grouped_controller.py` or its physical-group filter. `tools/run_grouped_rgbd_joint.py` is a direct entry alias for the free-point runner, not a wrapper around that estimator.

## Reference provenance

Reference checkout: `/home/samuelemara/Joint3DGS/RigEuler3DGS`, current HEAD `79779876f75a6707ad986deec8d36dbf4bbe9478`. Its present `estimator/coherence_extraction.py` is an unrelated plotting implementation. The graph/coherence implementation was read and ported from verified historical revision `481b2cf2d5602932286948cf14114f9caee4de3c`. The other requested mathematical reference files were inspected too. Hashes and narrowly documented adaptations are in `reports/reference_port_provenance.json`; `tools/audit_correction_sources.py` records checkout differences and baseline hashes.

The port uses a fixed canonical sparse kNN graph and the reference log-distance strain/hierarchy, including its Torch median convention and data-scale numerical epsilon. Representative occupancy is separate from computational occupancy. The finite candidate budget is explicit (default 48), not an exhaustive physical segmentation claim.

## Uncertainty and computation

See [the equations and executable commands](FREE_GROUPED_RGBD.md). Default assumptions remain the approved uncalibrated configuration: acceleration spectral density 0.01 m²/s³, position diffusion 0.0001 m²/s, initial velocity covariance I m²/s², pixel covariance I/12 pixel², depth variance 0.0001 m², and RGB covariance 0.0025 I in [0,1]². Actual parameter matrices and sources are serialized, not described as independently calibrated.

The association tail remains the user-authorized 1%. Reuse for measurement compatibility/observable model separation is explicitly an uncalibrated implementation policy, not a calibrated family-wise confidence statement. The verified reference pipeline explicitly removes BIC joint classification; the inherited BIC penalty and separate static-fit gate have been removed. Nonzero-motion observability and same-set measurement consistency remain explicit conditional diagnostics.

CUDA accelerates the point projection, grouped reductions/covariance solves and motion/map information calculations. Spatial kNN, deterministic assignment, graph extraction and some joint derivatives remain on the CPU. This is mixed CPU/GPU computation in one point IEKF framework; it is not an IEKF/UKF hybrid.

There are no production point-pair covariance matrices. The dense matrices are bounded computational-block matrices (at most 24 dimensions) and low-dimensional shared motion Schur matrices (354 dimensions for a free rigid 60-frame trajectory). Motion-map factors scale with points times motion dimension. Large-frame-count scenes may still require further memory engineering; the complete blade sequence is tested directly rather than dismissed as a resource limit.

## What is and is not established

A full temporal pass means every selected frame was actually filtered; it does not imply every nonlinear group reached tolerance. Iteration limits, line-search stops, energies, solve residuals and covariance records are saved. Gaussian uncertainty remains local, conditional on the selected correspondences and model. No unrestricted multimodal Bayesian inference or exact truncated posterior at mask boundaries is claimed.

All actual observation factors must be mask-positive with valid depth. Temporal measured-depth visibility uses the baseline point matcher; nominal hard negative-mask checks use point first hits. Under the existing v2 M3 contract, negative support is conclusive only with an explicit object-only/no-external-occluder premise. The selected blade metadata does not declare it: negative support is undetermined. Zero counted confirmed contradictions must not be presented as proof of unconditional silhouette feasibility.

Production jointly assimilates every available camera. Both joint candidates fit and score the same complete raw measurements, retaining the existing covariance treatment and per-camera diagnostics. Coherence cross-validation partitions supernodes, as in revision 481b2cf2, rather than cameras. Old split-camera histories and reserved-camera performance numbers are superseded; fresh filtering and optimization are necessary. Endpoint source observations initialize their priors once; other endpoint-camera observations remain in the likelihood with the exact pose gauge eliminated. Correspondence selection remains conditioned.

The reference coherence hierarchy proposes a core from estimated free trajectories. This is not automatically a credible physical part. A common six-DOF per-frame fit may explain its selected observations while neither fixed prismatic nor revolute motion explains them. Such a run remains unresolved and cannot justify joint-guided support completion.

Actual experiment metrics, acceptance by stage, remaining failures and browser checks are in [the measured report](reports/free_grouped_all_cameras_2026-10-24_validation.md). Positive synthetic support-completion tests do not replace that required real-data result.
