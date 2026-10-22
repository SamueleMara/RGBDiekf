# Point-cloud correspondence audit and correction

This concerns only the six-state Euclidean point estimator. No rigid-part,
orientation, normal, Gaussian, rendering or motion-model change is introduced.
The cached initial pass and all previously SOLVED tracks remain immutable.
Final measured results are in `reports/correspondence_2026-10-22_results.md` and
the external SSD audit directory named there.

## Existing code and the coverage error

The pre-correction code is committed as `d9c9772`. Relevant sources/functions:

- `pointwise_rgbd/observations.py:22`, `samples`: select raw pixels with
  `M & V & isfinite(D) & (D>0)`, read their RGB, backproject metric optical Z,
  propagate the complete `(u,v,Z)` measurement covariance into world XYZ, and
  construct a SciPy cKDTree over that camera's observed XYZ.
- `pointwise_rgbd/projection.py:16`, `backproject`: for world-to-camera `(R,t)`,
  `y = R^T (D K^-1 [u,v,1]^T - t)` and `Sigma_y = J_b R_z J_b^T`.
- `pointwise_rgbd/store.py`, `TrackStore.frame(k)`: retrieve the optimized
  `state[k,point,:3]`, its full stored 6x6 covariance and source RGB. Append
  separately persisted discovery tracks at the same timestamp. It does NOT
  transform frame-zero points again or integrate their velocity a second time.
- `pointwise_rgbd/association.py:20`, `associate`: query **SOLVED -> observed**
  world-XYZ kNN (`k=8`, line 47). Before querying, line 33 excludes every SOLVED
  point whose ID already appears as an owner in that camera. Line 54 excludes
  already-owned raw pixels. Lines 79-80 run exclusive point/pixel assignment.
- Pre-correction `controller.py:76`, `match_all_frozen`: reuse `associate` for
  surface coverage, claim selected pixels as ownership kind 2, repeat to closure.
- Pre-correction `store.py`, `Ownership.first_unclaimed`: scan backward and
  define missing geometry as `valid_pixel & (owner[pixel] < 0)`.
- Pre-correction `diagnostics.py`, `Diagnostics.final`: count those unowned
  valid pixels as unmatched. This produced the original 18,435,132 count.

The error is using a tracking assignment to define surface coverage. Once a
point has an owner assignment, other nearby observed samples cannot be covered
by it, even if they agree with the same transported surface. Raw depth sample
density also differs across viewpoints; a surface does not require a bijection
between its samples.

### Exact old association equations

For point i, projected measurement `h_i=(fx X/Z+cx, fy Y/Z+cy, Z)`, Jacobian H_i,
raw observation `z_j=(u_j,v_j,D_j)` and covariance R_j:

```text
candidate j in kNN_observed_world(x_i), k=8
e_ij = z_j - h_i
S_ij = H_i P_i H_i^T + R_j
d_ij^2 = e_ij^T S_ij^-1 e_ij
ray_ij^2 = e_uv^T S_uv^-1 e_uv
color_ij^2 = (rgb_j-rgb_i)^T (2 R_rgb)^-1 (rgb_j-rgb_i)
score_ij = (d_ij^2 + log(det S_ij) + color_ij^2)/2
```

The old eligibility gates are `d^2 <= chi2_3(.99)` and
`ray^2 <= chi2_2(.99)`. Projected mean points must be in front/in image; a
closer foreground depth excludes a confidently occluded prediction using
`(Z_pred-Z_obs)^2 > chi2_1(.99)*(var(Z_pred)+var(Z_obs))`. Background depth never
supplies a factor. RGB affects ranking and numerical ties, but has no old
appearance rejection gate. The old Gaussian RGB distance already contains
brightness information implicitly; there was no explicit cosine statistic.
Neither old nor new geometry uses cosine on absolute XYZ.

## Corrected surface matching

`pointwise_rgbd/coverage.py`, `SurfaceIndex`, builds one spatial index per
timestamp over **all finite transported SOLVED world XYZ**, with their stable
point IDs, position covariance marginals and stored RGB. Each valid observed
world point queries its eight nearest SOLVED points. There is no owner input.

For each candidate, Euclidean distance is reported. A predicted mean must have
positive camera Z, lie in the image, have real foreground depth support at its
projected pixel, and agree with that depth under the declared uncertainty rule.
Unknown depth is not fabricated. Occluded/inconsistent candidates cannot explain
the current observation; their trajectories are preserved. The point is also
tested for world-XYZ compatibility, projected ray compatibility and target-depth
compatibility. These derived checks are gates, NOT extra IEKF residual factors.

### Uncertainty and ranking, without invented weights

Frozen posteriors may already depend on a queried observation. Their saved
marginals do not contain the required cross-covariance C. Claiming
`Cov(x-y)=P_position+Sigma_y` to be exact would therefore be false.

For ANY joint covariance with those marginals:

```text
Cov(x-y) = P_position + Sigma_y - C - C^T
Cov(x-y) <= 2*(P_position + Sigma_y) = B_g
```

The difference between the bound and the actual covariance is `Cov(x+y)`, which
is positive semidefinite. This is a derived conservative bound, not a calibrated
new noise parameter. The analogous color bound for two RGB samples with marginal
covariance R_rgb is `B_c=4*R_rgb`. The projected UVZ bound is
`2*(H_position P_position H_position^T + R_j)`.

```text
r_g = y_j - x_i
d_g^2 = r_g^T B_g^-1 r_g
L = chol(B_c)
a = L^-1 rgb_j ; b = L^-1 rgb_i
d_c^2 = ||a-b||^2
      = (||a||-||b||)^2 + 2||a||||b||(1-cos(a,b))
geometry_score = (d_g^2 + log(det B_g))/2
appearance_score = geometry_score + (d_c^2 + log(det B_c))/2
```

The second equality explicitly supplies angular/cosine and brightness/color
information. It is ONE color distance; cosine and RGB distance are not counted
as two independent measurements. Raw RGB cosine is also reported. Cosine for
black/zero RGB is undefined and recorded as such; its finite Gaussian color
distance still works.

The score is the sum of marginal Gaussian negative log likelihoods, used as a
**composite compatibility score**. Both contributions are dimensionless and
noise-normalized; there is no arbitrary metric/appearance weight. No exact
independent joint likelihood is claimed. It assumes the documented encoded
RGB [0,1] convention, a locally Gaussian geometric approximation, meaningful
saved marginal uncertainties and the approved RGB error model. Unmodeled
illumination or calibration error can violate these assumptions.

The authorized tail remains 0.01. Thresholds are approximately 6.634897 (1D),
9.210340 (2D), and 11.344867 (3D). Geometry uses the 3D test plus ray/depth
visibility gates. The appearance variant additionally requires the 3D color
test. These dependent component-wise tests do NOT imply 99% familywise or
calibrated physical correspondence confidence. No gate or noise value is
automatically retuned. Invalid covariance candidates are explicitly ineligible;
temporary solver guard matrices never become accepted uncertainty values.

## Coverage and measurement ownership are separate

`coverage_store.py`, `CoverageIndex`, persists nonexclusive surface coverage by
frame/camera/model version. Many observations can be explained by one point;
ties between compatible surface candidates do not manufacture missing geometry.
`controller.match_all_frozen` now populates only this index. It never calls
`Ownership.claim`, an IEKF, a smoother, or a new transform estimator.

`Ownership.tracking_owners` contains only actual assimilations and initialized
anchors (kinds 1/3). Legacy kind-2 geometric bookkeeping is ignored for tracking.
Temporal `associate` remains exclusive; `track_one_point` still has one anchor
and independent backward/forward branches. Measurement ownership can never be
created by coverage. The 6D process, measurement Jacobian, and IEKF update source
files are unchanged.

Discovery uses `Ownership.first_unexplained` in reverse frame/camera/pixel order,
not a raw-owner test. An important consistency case is explicitly distinguished:
an observation may be geometrically/photometrically unexplained while already
being an immutable measurement of an old point. It cannot simply seed a second
independent point. It remains UNEXPLAINED in the diagnostics but is ineligible
for a second independent assimilation. Discovery selects the first unexplained,
not-yet-assimilated, not-failed seed in the same deterministic reverse order.
If only such owned conflicts remain, the run reports
`FROZEN_MEASUREMENT_COVERAGE_CONFLICTS`, never completed reconstruction.

## Incremental coverage after an insertion

`incremental_coverage.py` retains old valid coverage witnesses. An old pair's
geometry, uncertainty, color and observed-depth visibility are unchanged when a
new trajectory is appended. Therefore coverage is monotone: old coverage OR
compatibility with the new point. Each new trajectory is opened once; no old
`TrackStore.frame` or old point NPZ is used by the incremental matcher.

The candidate family is explicit: initial observation-to-cloud 8NN, followed by
compatibility with each subsequently inserted trajectory. Retaining valid old
witnesses avoids manufacturing novelty when incompatible new nearer points
would displace an old witness from a freshly recomputed top-eight list. This is
an exact union over the declared candidate family, not an exhaustive search of
all possible old pairs or an exact global Bayesian posterior.

A disk-backed per-view lookup retains actual mask-positive valid depth support
and the baseline UNSOLVED observations' XYZ, UVZ and RGB. It holds measured
observations, not a second canonical map or per-pixel dynamic states. The cache
is streamed one view at a time. There is no old-trajectory reload or raw-image
reload in the coverage update; point fitting and diagnostic rendering still
read their required actual observations.

For a new point, a spatial radius prefilter is **derived**, not tuned:

`radius² = chi2_3(0.99) * 2 * (lambda_max(P_new_xyz) + max_j lambda_max(Sigma_j))`.

Weyl's spectral bound makes this a necessary Euclidean bound on a candidate
passing the conservative 3D Mahalanobis gate. The unchanged coverage kernel then
evaluates the actual covariance, ray, depth, foreground support and RGB tests.
Outward floating-point rounding guards the search radius; it is not a physical
distance allowance. Previously covered samples need no new test.

`coverage_store.py` records the completed model horizon and last included point
ID. Changed views commit atomically. Discovery cannot read a partially completed
future horizon. An interrupted update resumes only uncommitted views and
publishes the new horizon after every view has been checked. Unchanged view
blocks can retain their older local version under that completed horizon.

Regression tests compare incremental coverage against baseline OR exhaustive
new-point/all-observation tests, forbid old-cloud/image reads, interrupt and
resume an update, retain a valid witness beyond eight incompatible nearer
insertions, and exercise changing surface sampling density/viewpoint and a new
spatially separated surface.

## Controlled before/after comparison

`tools/audit_surface_correspondence.py` reads the original ownership result,
frozen means/P, the same original RGB-D/mask bytes, and calibrated timestamps.
It verifies original frame digests, appended track SHA256, calibration and raw
input fingerprints. It invokes no point filter or optimizer. All frames and
cameras are the default.

Besides the three requested policies, `legacy_coverage_audit.nonexclusive_legacy`
performs a controlled ownership-only ablation: same old SOLVED -> observed
candidate family, same old UVZ/depth gates, same RGB ranking and numerical-tie
policy, with exclusive ownership restrictions removed. This isolates a supported
component of the original unmatched set without confounding world-coordinate
gating, search direction or the newly explicit appearance rejection. Further
reclassification by the new matcher is reported separately; it is not all
attributed to ownership.

Every raw valid observation is classified and indexed in per-camera NPZ files.
Scalar metric vectors, exact global/per-camera distributions, transition counts,
visibility/gate failures, and permitted many-to-one multiplicity are retained.
Overlays use actual RGB; interactive world views use actual cached XYZ and raw
backprojected observations. Only visual payloads are sampled and are labeled.

`FrozenCache` and `--cached-solved --coverage-audit` allow subsequent discovery
from the old histories without calling the initial optimizer. The original
histories are opened read-only, output goes to a separate directory, and only
new point histories can be appended there. A completed audit is required by the
cached CLI. No 328-frame initial optimization was rerun for this audit.
