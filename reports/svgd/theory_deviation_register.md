# SVGD theory deviation register (append-only)

| # | date | entry |
|---|---|---|
| 0 | 2026-11-03 | `NO_DEVIATIONS_RECORDED` (initial entry at Gate TF0; **superseded** by entries 1-12 below) |
| 1 | 2026-11-03 | **DEV-0001** MODEL_APPROX / PROPOSED / NON_EQUIVALENT |
| 2 | 2026-11-03 | **DEV-0002** MODEL_APPROX / PROPOSED / NON_EQUIVALENT |
| 3 | 2026-11-03 | **DEV-0003** THEORY_CHANGE / PROPOSED / UNDETERMINED |
| 4 | 2026-11-03 | **DEV-0004** THEORY_CHANGE / PROPOSED / UNDETERMINED |
| 5 | 2026-11-03 | **DEV-0005** THEORY_CHANGE / PROPOSED / UNDETERMINED |
| 6 | 2026-11-03 | **DEV-0006** MODEL_APPROX / PROPOSED / NON_EQUIVALENT |
| 7 | 2026-11-03 | **DEV-0007** MODEL_APPROX / PROPOSED / UNDETERMINED |
| 8 | 2026-11-03 | **DEV-0008** MODEL_APPROX / PROPOSED / NON_EQUIVALENT |
| 9 | 2026-11-03 | **DEV-0009** NUMERICAL_APPROX / PROPOSED / CONTROLLED_APPROXIMATION |
| 10 | 2026-11-03 | **DEV-0010** THEORY_CHANGE / PROPOSED / UNDETERMINED |
| 11 | 2026-11-03 | **DEV-0011** OPTIMIZER_MODIFICATION (TF.5 solver-policy row; no probability-model change) / PROPOSED / EXACT_EQUIVALENT objective / NON_EQUIVALENT map |
| 12 | 2026-11-03 | **TC-0001** THEORY_CONFLICT / OPEN_REQUIRES_USER_DECISION |
| 13 | 2026-11-03 | **DEV-0007 amended** (audit b62c7cee F7): 'conservative' withdrawn -> bias sign UNDETERMINED, conclusion UNDETERMINED |

Authorization: none of the entries is authorized; each is reachable only with `--svgd-theory-fidelity experimental` and is excluded from
theory-fidelity acceptance and from any advertised probability (spec TF.4).

## DEV-0001 — MODEL_APPROX (PROPOSED, NON_EQUIVALENT)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `(18), 8.2`, exact_expression: `h_t over the latent visible set V_t with p(V | X, C)`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `_stages / FrameFeatureTerm`, exact_expression_or_algorithm: `V_t = {1..N} for every t, alpha_i = 1/N`
- **reason**: the visibility law is NOT_DEFINED_IN_THEORY; without some V_t no feature likelihood exists
- **mathematical_difference**: conditions on all canonical points being comparable; ignores occlusion/self-occlusion and visibility-event likelihood
- **affected_outputs**: E_obs, accepted transport, all diagnostics
- **error_or_bias**: units: `-`, measured: `occluded base points flagged by the diagnostic (tests/test_svgd_mask_visibility)`, bound_or_limitation: `unbounded under occlusion (PDF 7.2 warns of false motion evidence)`
- **independent_oracle**: test: `tests/test_svgd_mask_visibility.VisibilityTests.test_occluded_static_behind_blade`, result: `failure mode exposed`, tolerance_basis: `-`
- **correctness_risk**: static points pulled to occluders; false motion
- **alternatives_considered**: point-only ray-visibility law with event likelihood (BLOCKED: not specified)
- **ablation_command**: --svgd-theory-fidelity experimental --config configs/svgd_support_experimental.json
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0002 — MODEL_APPROX (PROPOSED, NON_EQUIVALENT)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `(19)`, exact_expression: `Gamma_t estimated from data incl. measurement, flicker, sampling, correlations`
- **actual_implementation**: file: `estimator/svgd_feature_measurement.py`, function: `iid_sampling_gamma`, exact_expression_or_algorithm: `Gamma_t = C_Q/M_t + C_P/N at the stationary init`
- **reason**: no calibrated feature covariance exists
- **mathematical_difference**: independent i.i.d. sample means; drops ray/camera/time correlation and model mismatch; frozen (no log det dependence on state)
- **affected_outputs**: E_obs scale, step acceptance
- **error_or_bias**: units: `feature^2`, measured: `condition numbers in status.json gates.FEATURES`, bound_or_limitation: `likely overconfident (correlated samples)`
- **independent_oracle**: test: `tests/test_svgd_feature_measurement.FeatureTests.test_normalized_nll_and_logdet`, result: `likelihood exact GIVEN Gamma`, tolerance_basis: `1e-10`
- **correctness_risk**: overweighted data term chases noise
- **alternatives_considered**: repeated-stationary bootstrap of feature means (needs stationary intervals, not available without labels)
- **ablation_command**: edit features.Gamma_model
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0003 — THEORY_CHANGE (PROPOSED, UNDETERMINED)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `(8), 5.4, 14`, exact_expression: `LambdaX, LambdaC declared and validated (not specified)`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `representation_scales`, exact_expression_or_algorithm: `LambdaX_j = Sigma_j + s_t^2 I (median exact NN distance); LambdaC = R_C + diag(1/2 median (d_j - d_nn)^2)`
- **reason**: Codex proposal (EXPERIMENTAL_DEVIATION) to make TARGET runnable
- **mathematical_difference**: a specific representation-scale rule chosen by Codex
- **affected_outputs**: score, Stein field, transport
- **error_or_bias**: units: `m^2 / rgb^2`, measured: `status.json gates.TARGET.representation_scales`, bound_or_limitation: `no sensitivity study yet`
- **independent_oracle**: test: `-`, result: `not validated`, tolerance_basis: `-`
- **correctness_risk**: narrow basin (no motion found) or merged surfaces
- **alternatives_considered**: coarse-to-fine continuation; per-camera footprint scales
- **ablation_command**: edit target_density.Lambda_*
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0004 — THEORY_CHANGE (PROPOSED, UNDETERMINED)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `(13)`, exact_expression: `HX, HC declared interaction bandwidths (not specified)`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `median_heuristic`, exact_expression_or_algorithm: `H = med^2 / (2 log N) I per block (Liu & Wang reference heuristic)`
- **reason**: Codex proposal
- **mathematical_difference**: bandwidth from particle spread
- **affected_outputs**: Stein field
- **error_or_bias**: units: `m^2`, measured: `status.json gates.FEATURES.H_X_m2`, bound_or_limitation: `object-scale coupling across static/moving boundaries`
- **independent_oracle**: test: `-`, result: `not validated`, tolerance_basis: `-`
- **correctness_risk**: boundary contamination (PDF 6.3)
- **alternatives_considered**: fixed physical scale; per-block validation
- **ablation_command**: edit stein.H_*
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0005 — THEORY_CHANGE (PROPOSED, UNDETERMINED)

- **original_theory**: document: `CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md`, equation: `5.1`, exact_expression: `declared psi, B_X, B_C, D`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `_stages`, exact_expression_or_algorithm: `RFF psi with B = H (Stein bandwidths), D = 1024`
- **reason**: Codex proposal
- **mathematical_difference**: feature kernel tied to the transport scale
- **affected_outputs**: E_obs, acceptance
- **error_or_bias**: units: `-`, measured: `-`, bound_or_limitation: `which differences are observable is untested`
- **independent_oracle**: test: `tests/test_svgd_feature_measurement.FeatureTests.test_rff_kernel_and_mmd`, result: `map exact; choice unvalidated`, tolerance_basis: `6/sqrt(D)`
- **correctness_risk**: insensitive to fine motion
- **alternatives_considered**: B = Lambda; multi-scale features
- **ablation_command**: edit features.*
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0006 — MODEL_APPROX (PROPOSED, NON_EQUIVALENT)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `8.3`, exact_expression: `latent unexplained support B_t in the likelihood`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `_stages`, exact_expression_or_algorithm: `B_t omitted`
- **reason**: law not specified
- **mathematical_difference**: all observed feature mass must be explained by P_0
- **affected_outputs**: transport toward newly exposed geometry
- **error_or_bias**: units: `-`, measured: `observed_samples_without_supported_particle in observed_counts.csv`, bound_or_limitation: `unbounded under disocclusion`
- **independent_oracle**: test: `-`, result: `-`, tolerance_basis: `-`
- **correctness_risk**: forced transport onto new surfaces
- **alternatives_considered**: normalized background component with declared mass (BLOCKED)
- **ablation_command**: -
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0007 — MODEL_APPROX (PROPOSED, UNDETERMINED)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `(38)`, exact_expression: `Omega = J P_zeta J^T with the joint covariance of all states`
- **actual_implementation**: file: `estimator/svgd_rigidity.py`, function: `edge_covariance_structured / prior_position_marginals`, exact_expression_or_algorithm: `shared-reference exact (rank 1) + independent prior-propagated marginals per time and point`
- **reason**: joint posterior covariance of 6N(T+1) states infeasible
- **mathematical_difference**: drops cross-time/cross-point correlation of trajectory states and all data information
- **affected_outputs**: rigid_chi2, rigid_loglik
- **error_or_bias**: units: `m^2`, measured: `dense vs structured identical when structure holds; MC agreement 3%; log det bias +/- with the sign of the dropped correlation`, bound_or_limitation: `bias sign UNDETERMINED (corrected 2026-11-03 per audit F7: Var(x_i - x_l) = Var_i + Var_l - 2 Cov; dropping a negative covariance deflates Omega)`
- **independent_oracle**: test: `tests/test_svgd_rigidity.RigidityTests.test_dropped_cross_covariance_bias_sign_undetermined`, result: `opposite log-det bias for rho = +/-0.8`, tolerance_basis: `sign test`
- **correctness_risk**: over- OR under-stated rigidity evidence; no calibrated rho_ij while Omega is unverified
- **alternatives_considered**: selected-inverse posterior covariance (earlier temporal-sparse work)
- **ablation_command**: -
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0008 — MODEL_APPROX (PROPOSED, NON_EQUIVALENT)

- **original_theory**: document: `docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf`, equation: `(35)`, exact_expression: `E = sum E_F - log p_motion + rigidity + support/visibility priors + C_norm`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `SequenceObjective`, exact_expression_or_algorithm: `E_obs + E_time (conditional on frozen V_t; Z, m not in the step objective)`
- **reason**: support/visibility laws not specified
- **mathematical_difference**: conditional trajectory objective (VISIBILITY_CONDITIONAL_APPROXIMATION)
- **affected_outputs**: accepted transport
- **error_or_bias**: units: `nat`, measured: `objective_timeline.csv`, bound_or_limitation: `no global ELBO monotonicity claimed`
- **independent_oracle**: test: `tests/test_svgd_support_runner.RunnerTests.test_objective_monotone_on_accepted_steps`, result: `monotone`, tolerance_basis: `exact comparison`
- **correctness_risk**: self-consistent but non-posterior transport
- **alternatives_considered**: structured variational ELBO with discrete-label entropy (BLOCKED)
- **ablation_command**: -
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0009 — NUMERICAL_APPROX (PROPOSED, CONTROLLED_APPROXIMATION)

- **original_theory**: document: `CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md`, equation: `DEV-0003/DEV-0004 statistics`, exact_expression: `median over all samples / pairs`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `representation_scales / median_heuristic`, exact_expression_or_algorithm: `median over <= 4096 queries evenly spaced in (camera name, pixel) order`
- **reason**: O(M^2) exact median per frame
- **mathematical_difference**: sample median estimate of a declared scale; every component still enters the density
- **affected_outputs**: Lambda, H values
- **error_or_bias**: units: `m`, measured: `exact when M <= 4096`, bound_or_limitation: `order-invariant under camera permutation`
- **independent_oracle**: test: `tests/test_svgd_support_runner.RunnerTests.test_camera_order_invariance`, result: `PASS`, tolerance_basis: `relative 1e-4`
- **correctness_risk**: low
- **alternatives_considered**: exact selection on GPU
- **ablation_command**: -
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0010 — THEORY_CHANGE (PROPOSED, UNDETERMINED)

- **original_theory**: document: `CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md`, equation: `D3`, exact_expression: `Omega_model declared rigidity-model discrepancy`
- **actual_implementation**: file: `configs/svgd_support_experimental.json`, function: `rigid_graph.Omega_model`, exact_expression_or_algorithm: `Omega_model = 0`
- **reason**: not specified
- **mathematical_difference**: no discrepancy term
- **affected_outputs**: rigid_chi2
- **error_or_bias**: units: `m^2`, measured: `-`, bound_or_limitation: `overconfident strain`
- **independent_oracle**: test: `-`, result: `-`, tolerance_basis: `-`
- **correctness_risk**: spurious strain on real data
- **alternatives_considered**: calibrate on static intervals
- **ablation_command**: -
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## DEV-0011 — OPTIMIZER_MODIFICATION (TF.5 solver-policy row; no probability-model change) (PROPOSED, EXACT_EQUIVALENT objective / NON_EQUIVALENT map)

- **original_theory**: document: `CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md`, equation: `B3`, exact_expression: `x <- x + eta Phi (positions only)`
- **actual_implementation**: file: `estimator/svgd_support_run.py`, function: `_stages (velocity_lift = conditional_temporal_lift)`, exact_expression_or_algorithm: `[x, v] <- [x + eta Phi, v + dV(eta Phi)], dV = E37`
- **reason**: audit F2: a positions-only proposal raises E_time through inconsistent velocities and can be rejected although dE_obs < 0
- **mathematical_difference**: different proposal map; SAME accepted objective E
- **affected_outputs**: accepted transport only
- **error_or_bias**: units: `-`, measured: `dE_time of both families logged per backtrack (STEIN_DIAGNOSTICS)`, bound_or_limitation: `not the original SVGD map`
- **independent_oracle**: test: `tests/test_svgd_temporal.TemporalTests.test_subblock_solve_and_velocity_lift_exact`, result: `PASS vs dense autograd minimization`, tolerance_basis: `1e-9 rel`
- **correctness_risk**: none for the objective; changes which proposals are accepted
- **alternatives_considered**: positions-only (default, original)
- **ablation_command**: stein_numerics.step_policy.velocity_lift = 'conditional_temporal_lift' (strict needs velocity_lift_authorization = AUTHORIZED)
- **authorization**: source: `none`, decision: `NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)`

## TC-0001 — THEORY_CONFLICT (OPEN_REQUIRES_USER_DECISION)

- **theory**: PDF (3),(8),(11): reference colour c_i is a fixed vector; the score is evaluated at the fixed query colour.
- **spec**: Spec 1.1/3.2: "Propagate uncertainty of a_i; it is not noise-free"; Cov(a_i) must enter colour comparison once.
- **consequence**: fixed c_i: colour flicker of the reference is attributed to the target bandwidth only; marginalized: N3(a_i; d_j, LambdaC_j + Sigma_a) (exact Gaussian convolution) broadens every colour factor by the same shared reference covariance.
- **implementation**: target_score(query_color_cov=None) = PDF; query_color_cov=Sigma_a = spec. Config key reference.color_reference_uncertainty selects; strict mode refuses to run TARGET while the choice is NOT_DEFINED_IN_THEORY.

