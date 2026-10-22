> Latest user-request results: [LATEST_REQUEST_VALIDATION.md](LATEST_REQUEST_VALIDATION.md). Production reconstruction is **not accepted**: invented model-count priors were removed; the required model-selection measure is undefined. UKF progressive motion recovery still fails. Earlier experiments and running-status statements below are historical.

> Current authority: `Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md` (20 October).
> Earlier visibility-policy requirements and background-depth ordering below
> are historical, superseded behavior. See the v2 update at the end of this file.

# Covariance audit for the independent point pipelines

All numeric physical scales originate in `configs/example_uncalibrated.json`.
They are authorized initial assumptions, never independently calibrated values.
The left pose chart uses world/spatial translation-first errors; process pose
noise is right/body applied. RGB is encoded [0,1], metric depth is in meters.

| Quantity | Representation and checks | Interpretation |
|---|---|---|
| P0 | Reference pose fixed/eliminated, original 6D twist prior per exchangeable group | Pose zeros are gauge constraints; no tiny invented diagonal |
| Q(dt) | Approved local 12D covariance rate times actual interval | Discrete rate model specified by approved config, not a substituted continuous acceleration law |
| IEKF prediction | Analytical F P F' + G Q G', with nonzero-pose F/G finite differences | Left chart; G=diag(Ad(Tpred),I) |
| IEKF update | Fixed original Pminus each inner iteration, raw 6D R, undamped information inverse | Conditional on map/matches; solver damping never becomes a statistical prior |
| IEKF RTS | Saved analytical F, original predicted/filtered P, both mean-chart transports | Same shared trajectory, no independent inverse filter |
| UKF prediction | 49 augmented sigma points per independent 12D group; joint dimensions if prior cross terms exist | Nonlinear process, same ensemble cross covariance; no F substitution |
| UKF update | 25-state-sigma M/b reductions per conditionally separable group, Woodbury covariance and retraction transport | One-shot moment approximation; a MAP line search is not imposed on this mean |
| UKF smoother | Saved augmented-sigma cross covariance, covariance and chart updates | Unscented; no imported IEKF Jacobians |
| Raw measurement | Full UV/depth/RGB 6x6 blocks, including approved RGB-depth cross terms | No bilinear interpolation in production; each raw owner is unique |
| Point coordinate fusion | Small conditional 6x6 precision blocks plus unique raw CSR provenance | Conditional on frozen poses; these diagonal blocks are not independent map marginals |
| Source pose/point coupling | Sparse original raw and source-time/group keys, Jy/JT and conditional fusion sensitivities | Shared pose/raw errors stay shared; never copied into independent per-ray R |
| Selected covariance queries | Controlled CG helper for supplied gauge-fixed information operator; source augmentation accepts full joint raw/pose covariance | Generic selected solver tested; full-scene joint-map marginal operator/queries are not claimed |
| Final accepted means | Covariances save their own filtered/smoothed means and an accepted-mean compatibility flag | Rejected final trajectory covariance is not relabeled as covariance at the accepted trajectory |

`source_pose_factors/` stores the first-order coordinate-fusion sensitivities
with frozen observation covariance. It is a sparse dependence representation,
not another likelihood or an independent landmark prior. Raw-pose cross moments
must be supplied to a joint augmentation query if motion was inferred from the
same raw data. Treating them as zero would be false independence. The source-only
inverse/forward self-fit has zero source-pose derivative; the test verifies this
cancellation. The inner filter's fixed-map conditional covariance is distinct
from a joint Schur marginal and from these source initialization sensitivities.

The source-fusion derivative linearizes backprojection and the stated covariance
law at observed depth. It does not differentiate a heteroscedastic covariance
with respect to the observation. Its provenance and frozen-covariance convention
are required when interpreting a query.

The UKF uses the fixed external projected competition graph and defers factors
whose sigma points violate the tested frustum/depth ordering. Pixel-bin changes
introduce discrete associations outside this conditional model. They are handled
by the external full-score transaction, not silently made into smooth output.
No unconditional independence of map-marginal group errors is claimed.

Mandatory numerical references include finite differences, Joseph covariance,
linear Gaussian updates and dense small smoothers, Woodbury/dense innovation,
shared-source Monte Carlo verification, gauge changes, CUDA/CPU parity and actual
bidirectional synthetic integration. Read test results for execution status.
Full-scene reconstruction and full-scene covariance acceptance are separate gates.

Selected conditional cross-time smoother blocks are now available through
TrajectoryMoments. In each saved smoothed chart, the backward regression is
J_current C_k inv(J_next); multiplying these regressions into the later
smoothed block gives a selected cross-time covariance without a K-by-K global
allocation. Numerical tests compare every block against a small dense Gaussian
posterior for each filter's own smoother. This is still conditional on the
frozen map and chosen factors. Constant-R factor reuse and batched sigma
projection do not diagonalize R or discard off-diagonal covariance terms.

## Corrections after the initial audit

Source fusion now rebuilds its proper-prior precision and all unique source
factors at the accepted poses. It does not use stale coordinate-fusion blocks
as final covariance. The source Gaussian center and offset from the accepted map
are saved, along with its full conditional 6x6 blocks and source sensitivities.
This is an initialization-reference Gaussian, not the final raw-factor posterior.

`map_uncertainty/conditional_map_covariance.npz` separately accumulates the
original geometry prior once and every selected raw RGB-D likelihood at the
accepted trajectory. It stores its own linearized Gaussian mean, covariance,
offset and error second moment, including unobserved points' proper priors.
This is a fixed-pose/fixed-assignment conditional approximation. Each factor
retains frame/group/point/raw IDs for cross-information reconstruction.
No full-scene joint map-motion marginal covariance is asserted.

Motion covariance files now save both pose and twist Gaussian centers. When
the final proposal is rejected, accepted-reference fields re-express that same
Gaussian with its nonzero error mean; they do not silently move its center to
the accepted trajectory. Shared raw/source-pose queries require explicit cross
moments; missing cross covariance never defaults to independence.

The hard-mask amendment adds a nonsmooth feasible set. Neither small Gaussian
filter covariance represents its complete truncated posterior. IEKF derivatives
are still derivatives of the smooth conditional RGB-D output; mask feasibility
is checked without differentiation. UKF sigma points are not independently
clipped to masks. A rejected infeasible posterior retains prediction covariance.
Visibility decisions under nonzero Gaussian depth uncertainty require an explicit
decision model; no unspecified confidence level is supplied as calibration.

The preserved real IEKF checkpoint audit processed all 328 times / three cameras:
194,717 points, 39,016,829 selected raw factors, 24,774 unobserved points. It
rebuilt the pre-amendment score within 8.35e-7 and used 209,362,432 peak allocated
GPU bytes. This audit is under `checkpoint_audit_v9_iekf/` on the SSD. It predates
the hard-mask correction and is neither a complete real cycle nor acceptance
under the amended objective.

## V2 observation-domain and sigma-support correction

Only `M AND Vraw AND finite(D) AND D>0` contributes to R/factor precision.
Background metric depth and invalid foreground depth provide no factor, even as
an occluder or no-return penalty. The approved likelihood/noise magnitudes are
unchanged. This changes the domain of the comparable sequence score, so old
checkpoints are not resumed as though scores were identical.

UKF sigma support now uses a fresh full-map z-buffer at every sigma pose, including
newly entering competitors, explicit point-ID depth ties, frustum and foreground
ordering events. A changing discrete factor is deferred. No Gaussian covariance
shrinkage can be claimed from a deferred factor: M=I and b=0 retain Pminus when
all factors defer. The same sigma rule is tested against scalar full-output moments
and dense tiny innovation references. Actual physical motion recovery is a
separate test and currently fails for the preserved drawer UKF fixture.

Mask feasibility is a discrete trial acceptance condition outside either update
matrix. The resulting Gaussian remains conditional and does not implement an
exact truncated posterior. An unknown acquisition premise does not become a new
numeric uncertainty assumption. Full numerical execution status is in VALIDATION.md.

The final sigma event check does not turn the fixed associated target's noisy
depth residual into a second occluding object. Its original six-dimensional R
remains in the Woodbury update, with RGB-depth cross terms preserved. An isolated
uncertain-depth regression requires active information and compares posterior
mean/covariance against dense S. Independent other-foreground/model occlusion
continues to defer invalid factors. Cached device observations exist for one
camera sigma ensemble only, with exactly the immutable valid raw IDs.

Any nonzero cross-body covariance now selects the joint sigma ensemble; no
absolute machine-epsilon threshold silently declares statistical independence.
The regression uses variances 1e-18 with a genuine 2e-19 cross-body term and
requires 97 augmented sigma points and preserved cross-covariance. The actual
real initial priors are exactly block diagonal under the declared conditional
model. The first real saved IEKF/UKF smoother proposals have been independently
checked for finite symmetric PSD covariance and their own transition source;
see SSD reports/first_real_smoother_audit.json.
