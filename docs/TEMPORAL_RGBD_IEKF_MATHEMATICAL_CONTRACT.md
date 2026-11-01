# Depth-sorted temporal RGB-D implementation contract

This documents the implementation, including its limits. It is not a claim that the
production scientific acceptance gates have passed. Entry: `train.py`;
`--observation-model legacy` remains the default. The earlier `--free-solver
 temporal-ieks` is a different model and cannot be combined with this one.

## Common observation operator

Each point has world state `[x,y,z,vx,vy,vz]`. Prediction uses the original
`transition`, `process_noise`, and `predict` functions, including signed dt from
actual acquisition timestamps. Q is not adjusted for this feature. B1 processes
last-to-first and keeps arrays in original chronological order.

`temporal_zbuffer.first_hit_pixel_owner` projects every point in every camera.
Integer pixel coordinates use ties-to-even rounding (NumPy rint / torch round).
Stable sorting by pixel, depth, then canonical point ID gives exactly one winner.
Masks and measured depths do not decide the rendered depth order. Counts and exact
depth ties are separate outputs. The endpoint source identity is retained in
`canonical.npz` and `ReferenceTable.source_identity`; unrestricted cross-camera
rendering never replaces this known source reference.

Only winners over finite, positive, valid foreground RGB-D may have a factor.
Reference interpolation requires all four neighbors valid and foreground, including
neighbors with zero interpolation weight. Depth continuity uses the existing
projective policy, unchanged. Raw background values never enter temporal factors.
Occluded points have no direct camera update. Uncovered foreground and rendered
background violations are separately reported.

A reference belongs to one point AND one camera. First acquisition bootstraps
without a temporal measurement. Only accepted states refresh references. Invalid
support reboots on the current valid observation. Occlusion retains references;
processing-order age >= lag makes a reference unusable (`REFERENCE_OUTSIDE_LAG`).
Backward references are not required to have smaller chronological frame numbers.

For integer current pixel p, current/reference states t/r, define

```
q = p - pi_c(x_t) + pi_c(x_r)
h = [sum_a w_a(q) ell_r,a.RGB,
     sum_a w_a(q) ell_r,a.depth + Z_c(x_t) - Z_c(x_r)]
r = y_current - h
H_t.RGB = -grad_RGB Jpi_t        H_r.RGB = +grad_RGB Jpi_r
H_t.D = JZ_t - grad_D Jpi_t     H_r.D = -JZ_r + grad_D Jpi_r
H_ell,a = w_a I4
```

All direct velocity columns are zero; temporal coupling comes from the process.
Bilinear gradients use the actual four values in h, not a current-image gradient.
The operator is metric-depth interpolation, not inverse-depth interpolation.
Exact cell boundaries are flagged NONSMOOTH. The warp is a local image translation
and depth translation; it does not assume a rigid 3D patch.

## Full path

`temporal_sparse_gpu.TemporalSoAGraph` has unique raw pixel keys
`('pixel', camera, frame, flat_pixel)` and six-dimensional non-anchor state keys.
Every touched pixel gets exactly one sensor likelihood `y-ell ~ N(0,R_sensor)`.
Warp factors are `ell_current - h(states,ell_reference) ~ N(0,R_warp)`.
Process factors use the original F and Q. Reusing a reference does not create a
second independent copy of its sensor error. Sensor factors already marginalized
into the lag prior cannot be reintroduced.

Source initialization eliminates the exact backprojection constraint rather than
adding an independent depth-derived XYZ prior. Free source variables are pixel
coordinate error (2), velocity (3), and the unique clean source RGB-D pixel (4):

```
x_anchor = R_cw.T (D_clean * [(u+epsilon_u-cx)/fx,
                              (v+epsilon_v-cy)/fy, 1] - t_cw)
epsilon_pixel ~ N(0, approved pixel covariance)
v_anchor ~ N(0, approved velocity covariance)
y_source ~ N(ell_source, approved sensor covariance)
```

The pixel covariance is the approved Gaussian moment approximation to uniform
unit-box pixel response. Differentiating this parameterization gives the original
local backprojection covariance AND nonzero Cov(x_anchor, ell_source.depth).
There is no additional independent Pxx likelihood. The product of pixel error
and clean depth is nonlinear; all published Gaussian covariances are local.

Whitened Jacobian J and exact equality C are sorted sparse GPU COO operators,
with sparse Jv/J-transpose-v. Full production never invokes the bounded
`TemporalWindowGraphDenseOracle` or an independent-point update. Current clean
pixels with +I4 hard-constraint pivots are algebraically substituted, including
all consumers and source-depth coupling. Repeated or conflicting pivots fail.

The sparse solve uses invertible source/process/sensor innovation coordinates
built from the actual supplied covariances and F. Algebraically identity prior
rows are eliminated from the augmented residual system into their true Gaussian
information and gradient. This is a numerical change of coordinates, not a new
prior or a loss of correlations. cuDSS factors the remaining augmented KKT
without forming current measurement normal equations, without jitter or pivot
perturbation. Iterative refinement and entrywise primal/backward-error checks
certify each solve. Failure is explicit. Selected joint covariance queries reuse
that factorization; RHS colors combine only genuinely disconnected components.
No global covariance, global dense nullspace or inverse local information block
is substituted for a joint marginal.

Nonlinear acceptance uses the complete stochastic objective and exact equalities,
full depth-competitor rerendering, valid foreground/reference support and a fixed
common measurement domain. Complete true variable components can backtrack at
different rates; no factor is removed. Rejected/frozen components and failure to
converge are reported. The historical no-new-outside-foreground policy is retained:
preexisting outside projections are diagnosed, not retroactively called feasible
foreground measurements. They never supply raw sensor values.

Window sliding consumes factors touching expired variables and retains unaffected
nonlinear warps. Sparse Schur retains every induced information/gradient edge,
linearization, residual constant and surviving equality. A single batched CPU
symbolic bipartite matching identifies constraint pivots; numerical elimination,
fill and selected solves are CUDA. FEJ constructs normal information for this
reduction, so QR parity and conditioning tests remain essential; it is not claimed
to have square-root conditioning. Numeric rank failure is not repaired by jitter
or discarded eigenmodes. Induced fill and workspace can still be resource limits.

This is exact for the chosen local linearized operator and frozen FEJ boundary,
not globally exact nonlinear Bayesian inference or full-sequence smoothing.
Signed physical timestamps determine both forward and backward dynamics. Stored
trajectories are each current filtered marginal, not retroactively smoothed old
publications. Numerical witnesses include source/state/pixel cross-covariance,
shared raw pixels and full Schur fill.

### Resource and acceptance status

The CUDA backend is real and exercised by numerical/integration tests. Production
acceptance remains governed by USB/Blade all-cloud frames 0–2 and actual recorded
component oracle parity; tests or assembly profiles alone cannot pass it. See
`reports/temporal_sparse_gpu_exact_implementation.md` and its JSON for actual runs,
failures, stage timings and limitations. Larger full-mode jobs are blocked pending
source-matched three-frame evidence. The dense oracle has a 4096-scalar hard ceiling
and is never a production fallback.

## Simplified path

`update_points_temporal_simplified` uses precisely the same renderer, reference
lifecycle, patch validity and temporal derivatives. It fixes reference states and
uses measured reference values. Each row uses

```
R_eff = R_current + sum_a w_a^2 R_reference,a + R_warp
A_i = inverse(P_i_minus) + sum H_i.T inverse(R_eff) H_i
```

Iteration fixes the prior. Covariance is inverse(A_i), explicitly conditional:
`fixed_reference_state + marginal_sensor_noise_only; ignores_temporal_and_cross_point_correlations`.
An independent point with no row keeps exactly its predicted mean and covariance,
including integrated nonzero velocity. Initialization uses the legacy Pxx and is
an approximation, not the coupled full initialization.

## Nonlinear acceptance and diagnostic interpretation

Both paths use fixed-prior iterations and numerical backtracking. Owners are
rerendered at proposals. The implemented conservative objective stays on a common
current-pixel/owner/reference-frame domain: an active factor cannot disappear
or change owner to claim lower cost. Reference interpolation cells DO refresh
on nonlinear proposals, with all four neighbors validated. Full mode registers
newly touched raw pixels once, initializes them at their sensor means, and retains
the union of old and new sensor factors. The expanded baseline has zero additional
sensor residual, so a candidate cannot win by deleting old raw observations. The
linear solve freezes contributors; the next accepted relinearization uses the
refreshed clean values and weights. Full mode also checks retained historical warp
domains and owners. This can still stall at current-owner or invalid-support boundaries;
it is a piecewise local approximation, not an optimizer that traverses arbitrary
hard association changes. Changes outside active factors are counted separately.
No new rendered-background point/camera violation is accepted; preexisting
violations are reported, excluded from RGB-D likelihoods and not repaired by an
invented mask force. Such violations remain a scientific acceptance concern.

Monotone cost does not prove convergence or goodness of fit. Iteration history
records cost, trial cost, gradient, backtracking, and convergence. Simplified NIS
uses the point's full stacked conditional innovation covariance (Woodbury
reduction), not independent per-camera sums. Full predictive NIS is evaluated on
the NEW warp batch with all shared latent/state prior covariance retained,
`S = I + J_w P_prior J_w.T` in whitened coordinates. Degrees of freedom are the
rank (full row rank because warp covariance is SPD). Hard selection and empirical
noise fitting mean nominal chi-square calibration is conditional, not guaranteed.
No association tail is retuned. The configured authorized 1% pointwise association
policy remains unchanged for paths that use it; the new temporal observation
specification admits hard-valid first-hit rows and does not silently copy the old
projective appearance gate into that model.

## Noise inputs and provenance

Existing approved P0, Q and raw sensor noise conventions are unchanged. Existing
`sensor-noise measured` is an empirical estimator with its existing static-majority
and cross-camera mismatch assumptions. That mismatch is NOT temporal warp noise.
`temporal_noise.load_warp_noise` loads `metadata/temporal_warp_noise.json`, then a
scene-root equivalent, or an explicit file. It requires a finite SPD 4x4
`covariance_rgbd`, `units`, `assumption_class`, and `method`. Units describe the
variables: three `encoded_rgb_0_1`, then `m`; covariance units are their products.
Measured policy requires empirical-Bayes or independently-calibrated provenance.
An explicit assumption must use config policy and is not called calibration.

No safe validated temporal mismatch calibration was found for the ten actual
scenes. Existing sensor MAD and cross-camera residuals cannot identify it without
additional assumptions. As explicitly permitted by specification section 16,
missing warp noise fails MISSING_TEMPORAL_WARP_NOISE. No illustrative numerical
warp covariance is silently used. The JSON schema defines the required file.

## Outputs and visual diagnostics

Existing `free_forward` / `free_backward` arrays, source IDs, numeric files,
traces, and timeline column names are preserved. Additional temporal directories
contain config/provenance, per-frame objectives, ownership/reference/innovation/
covariance/timing CSVs, traced point numerical records and selected full component
information/covariance NPZs. The live `free_DIRECTION/temporal/index.html` works
with file:// and remains readable after a failure. It has measured RGB/depth/mask,
first-hit ownership, warped predictions, residuals, uncovered foreground, bootstrap
and conflict panels; orbit/pan/zoom and selectable 3D points; prediction/canonical
comparisons; support coloring; frame playback; and convergence records.

The complete existing 3D/RGB-D/numerical viewer is still generated at
`debug/index.html`, with links to temporal pages. Display sampling (12,000 plus
explicit point IDs) and display color quantiles never reduce inference data.
Downstream coherence/rigid/joint stages remain legacy and are labeled
TEMPORAL_FREE_WITH_LEGACY_RIGID. GT is read only by separate evaluation tools.

Square-root elimination identity, used by the implementation and Schur tests:

```
A_old = Qv Rv
B = (I - Qv Qv.T) A_retained
r_perp = (I - Qv Qv.T) r
B = Qb Rb
new FEJ residual = Rb (a_retained - a_linearization) + Qb.T r_perp
Rb.T Rb = Lambda_uu - Lambda_uv inverse(Lambda_vv) Lambda_vu
Cov_joint = inverse(R_joint) inverse(R_joint).T
```

The eliminated precision log determinant is `2 sum log abs(diag(Rv))`.
Residual constants discarded by QR compression are tracked in the MAP cost.
Fixed Gaussian normalization constants do not affect the state solve; reported
MAP costs are not automatically comparable marginal evidences across models.

Full-mode status code 2 means NO DIRECT TEMPORAL ROW: a coupled marginal may still
change through another point's shared evidence. Independent components with no
new temporal factor take an exact prediction-only publication branch. Source
backprojection is numerically centered on the recorded canonical mean, preventing
NumPy/torch roundoff from creating artificial initial image flow. The centered
formula is algebraically the same coupled source model, not an extra XYZ prior.
Sparse unexplained-observation keys are saved per frame for future discovery.

## Next-actions correction: ideal discrepancy and actual validation

`--temporal-warp-noise ideal` is a separate diagnostic model, labeled
`IDEAL_WARP_ASSUMPTION_UNCALIBRATED`. It sets ONLY the additional conditional
warp-model discrepancy to zero. Raw sensor factors remain SPD and are registered
once per raw `(camera,frame,pixel)` key. Reference/current states and cross blocks
are still stochastic. Version-1 `covariance_rgbd` already meant this additional
term; version-2 provenance names that meaning explicitly without reinterpreting
old files. Config/measured files remain strictly SPD; ideal accepts no file.

In full ideal mode the warp is an equality `c(v)=0`. For a linearization with
whitened stochastic rows `A d + r` and equality rows `C d + c`, the solver uses
`d0=-C^+c`, a nullspace basis `Z` of C and QR of `A Z`. Its covariance is
`Z (R^-1 R^-T) Z^T`. No tiny warp variance, inverse of a zero covariance, or
independent copy of reference noise is introduced. Total prior-predictive
innovation covariance includes latent sensor/state uncertainty and must pass
Cholesky. Constraint rank, free dimensions and QR condition are reported.

Marginalization eliminates expired coordinates from the exact equalities first,
then projects their remaining stochastic free coordinates out by QR. Both the
remaining hard constraints and complete Gaussian FEJ boundary factor survive.
Stochastic QR rows are retained without eigenvalue truncation. Numerical rank of
projected hard equations uses the ORIGINAL constraint scale so roundoff of an
exact zero equation cannot become artificial information. Trace archives contain
both the stochastic information matrix and the equality matrix; inverting the
former alone is not the constrained marginal.

The simplified model retains `R_current + sum(w^2 R_reference)`, with positive
sensor noise even in ideal mode. It still discards cross-time/cross-point
correlations. Acceptance is per independent point, followed by a joint depth-owner
check and rollback of interacting ownership changes. A point whose update was
never accepted retains its propagated prior covariance. Full mode backtracks
whole TRUE components, never cuts edges, and reports acceptance per state rather
than marking every eligible row as a successful update. Hard support boundaries
may still stall either model; nonconvergence is not successful tracking.

Iteration ownership can be represented sparsely using the same deterministic
sort and exact winners. Dense images are materialized only for accepted-frame
exports. CPU/CUDA tests compare every sparse pixel query with the dense map.
This is an algebra/storage optimization, not a different visibility model.
The small-matrix eigenvalue diagnostics use bounded independent batches to avoid
cuSOLVER batch/workspace failures; this does not partition the lag graph.

The temporal measured-sensor path reuses the existing same-camera consecutive
pixel MAD estimator only. Its static-majority/quantization assumptions are
explicit; it is not independent sensor calibration. Cross-camera mismatch is
neither computed nor credited as temporal warp uncertainty. The separate
`tools/calibrate_temporal_warp_noise.py` study reports full robust 4x4 temporal
candidate differences, sensor contributions and held-out temporal blocks. It
returns `MODEL_DISCREPANCY_NOT_IDENTIFIABLE` when independent motion/reference
uncertainty is unavailable; it does not clip a residual excess into a purported
calibration or emit a usable covariance file.

See `reports/temporal_rgbd_next_validation.md` for actual estimated-frame evidence,
individual regressions, bounded pilot outcomes and production limits. The earlier
preflight-only report is historical. Focal controls select original canonical IDs
explicitly, preserve the ID/source mapping, and are prominently labeled reduced
occlusion catalogues; they are never full-cloud validation.
