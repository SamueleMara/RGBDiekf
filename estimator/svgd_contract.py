"""Frozen theory-to-code contract of the opt-in SVGD moving-support Stage 1 (Gate TF0).

Single machine-readable source of truth for docs/SVGD_THEORY_TO_CODE_CONTRACT.md, svgd_support/model_contract.json and
the strict-mode validator. Theory: docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf (equation NUMBERS below are
the PDF's printed numbers; the .tex is not in the repository, so the spec's LaTeX labels are mapped to them by content:
TEX_UNAVAILABLE_LABELS_MAPPED_BY_CONTENT). Spec: CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md.

category: A exact algebraic identity | B selected proposed model | C approximation to that model | D numerical only.
classification: EXACT_MODEL | NUMERICAL_APPROX | MODEL_APPROX | DEVIATION | NOT_DEFINED_IN_THEORY | BLOCKED_MODEL_UNSPECIFIED |
                OUT_OF_SCOPE_PHASE1.
role: proposal (optimizer direction, never evidence) | likelihood (the single statistical observation) | prior | evidence |
      diagnostic | decision.
"""

THEORY_PDF = 'docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf'
SPEC = 'CODEX_RGBDiekf_SVGD_Sequence_Moving_Support_CUDA_THEORY_PARITY_2026-11-03.md'
LABEL_MAPPING = 'TEX_UNAVAILABLE_LABELS_MAPPED_BY_CONTENT'
TRANSPORT_LABEL = 'restricted_physical_particle_stein_transport'      # never "SVGD posterior sampling" (PDF 4.2, spec 2.2)

S = 'tests/test_svgd_'
EQUATIONS = [
    # ---- Phase 1A target density / score (operator A)
    dict(id='E01', label='eq:target', pdf='(8)', spec='A1', role='proposal', category='B', classification='EXACT_MODEL',
         expr='p_t(x,c) = sum_j w_j N3(x; y_j, LambdaX_j) N3(c; d_j, LambdaC_j), w_j = 1/M_t',
         invariant='normalized 6D mixture over the actual masked samples, full Gaussian normalizers, uniform 1/M_t selected by the PDF',
         function='estimator/svgd_target_score.py:target_score', tests=[S+'target_score.TargetScoreTests.test_log_density_and_score_match_dense_numpy',
         S+'target_score.TargetScoreTests.test_normalization_integral'], dtype='float64', units='log density of m^-3 * rgb^-3',
         parameters=['target_density.Lambda_X', 'target_density.Lambda_C', 'target_density.source_weights'], deviation='NONE'),
    dict(id='E02', label='eq:responsibility', pdf='(9)', spec='3.1', role='proposal', category='A', classification='EXACT_MODEL',
         expr='r_j = a_j / sum_m a_m (stable log-sum-exp)', invariant='responsibilities sum to one; density contributions only, not correspondences',
         function='estimator/svgd_target_score.py:responsibilities', tests=[S+'target_score.TargetScoreTests.test_log_density_and_score_match_dense_numpy'],
         dtype='float64', units='1', parameters=[], deviation='NONE'),
    dict(id='E03', label='eq:targetscore', pdf='(10)-(12)', spec='A2', role='proposal', category='A', classification='EXACT_MODEL',
         expr='g_t(x,c) = sum_j r_j LambdaX_j^-1 (y_j - x)', invariant='exact gradient of log p_t in x at fixed query colour, heteroscedastic',
         function='estimator/svgd_target_score.py:target_score', tests=[S+'target_score.TargetScoreTests.test_score_five_point_finite_difference',
         S+'cuda_parity.CudaParityTests.test_target_score'], dtype='float64', units='1/m', parameters=[], deviation='NONE'),
    dict(id='E04', label='eq:refcolour-marginal', pdf='(3) vs spec 1.1/3.2', spec='3.2', role='proposal', category='A', classification='EXACT_MODEL',
         expr='int N3(c; d_j, LambdaC_j) N3(c; a_i, Sigma_a) dc = N3(a_i; d_j, LambdaC_j + Sigma_a)',
         invariant='Gaussian convolution identity: reference colour uncertainty enters as LambdaC_j + Sigma_a (THEORY_CONFLICT TC-0001 choice)',
         function='estimator/svgd_target_score.py:target_score(query_color_cov=)', tests=[S+'target_score.TargetScoreTests.test_reference_colour_marginalization'],
         dtype='float64', units='rgb^2', parameters=['reference.color_reference_uncertainty'], deviation='TC-0001'),
    # ---- Phase 1B Stein transport (operator B)
    dict(id='E05', label='eq:physkernel', pdf='(13)', spec='B1', role='proposal', category='B', classification='EXACT_MODEL',
         expr='K(i,l) = exp(-1/2 dx^T HX^-1 dx - 1/2 dc^T HC^-1 dc)', invariant='SPD XYZ-RGB interaction kernel, bandwidths distinct from Lambda and R',
         function='estimator/svgd_particle_operator.py:stein_field', tests=[S+'stein_operator.SteinTests.test_kernel_gradient_finite_difference'],
         dtype='float64', units='1', parameters=['stein.H_X', 'stein.H_C'], deviation='NONE'),
    dict(id='E06', label='eq:repulsion', pdf='(15)', spec='B2', role='proposal', category='A', classification='EXACT_MODEL',
         expr='grad_{x_l} K(l,i) = K(l,i) HX^-1 (x_i - x_l)', invariant='correct sign: pushes x_i away from x_l',
         function='estimator/svgd_particle_operator.py:stein_field', tests=[S+'stein_operator.SteinTests.test_repulsion_sign_two_particles',
         S+'stein_operator.SteinTests.test_kernel_gradient_finite_difference'], dtype='float64', units='1/m', parameters=[], deviation='NONE'),
    dict(id='E07', label='eq:physicalsvgd', pdf='(14)', spec='B2-B3', role='proposal', category='B', classification='EXACT_MODEL',
         expr='Phi_i = 1/N sum_{l=1..N} [K(l,i) g(x_l,c_l) + K(l,i) HX^-1 (x_i - x_l)];  x <- x + eta Phi (accepted only if the common objective decreases)',
         invariant='entire sum over all N particles, global 1/N, attraction/repulsion stored separately, colour fixed',
         function='estimator/svgd_particle_operator.py:stein_field', tests=[S+'stein_operator.SteinTests.test_dense_numpy_oracle_all_terms',
         S+'stein_operator.SteinTests.test_liu_wang_reference_gaussian', S+'cuda_parity.CudaParityTests.test_stein_field'],
         dtype='float64', units='1/m (field), m (accepted step)', parameters=['stein_numerics.step_policy'], deviation='NONE'),
    dict(id='E08', label='eq:svgd-standard', pdf='(5)-(7)', spec='2.1', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='standard SVGD on samples of ONE target (Liu & Wang 2016), reference only',
         invariant='physical particles are not posterior samples: no KL-theorem claim for support inference',
         function='tests/test_svgd_stein_operator.py:liu_wang_svgd (reference)', tests=[S+'stein_operator.SteinTests.test_liu_wang_reference_gaussian'],
         dtype='float64', units='-', parameters=[], deviation='NONE'),
    # ---- Phase 1C feature observation (operator C)
    dict(id='E09', label='eq:featureobs-mean', pdf='(17)', spec='C1', role='likelihood', category='B', classification='EXACT_MODEL',
         expr='y^F_t = 1/M_t sum_j psi(y_j, d_j)', invariant='observed feature mean over all masked samples',
         function='estimator/svgd_feature_measurement.py:RFFFeatures.mean', tests=[S+'feature_measurement.FeatureTests.test_rff_kernel_and_mmd'],
         dtype='float64', units='feature', parameters=['features.feature_method', 'features.D', 'features.fixed_seed', 'features.xyz_rgb_kernel_B'], deviation='NONE'),
    dict(id='E10', label='eq:featurepred', pdf='(18)', spec='C1', role='likelihood', category='B', classification='BLOCKED_MODEL_UNSPECIFIED',
         expr='h_t(X_t, V_t) = 1/|V_t| sum_{i in V_t} psi(x_it, c_i)', invariant='comparable VISIBLE predicted support; V_t law not defined in theory',
         function='estimator/svgd_feature_measurement.py:RFFFeatures.mean', tests=[S+'feature_measurement.FeatureTests.test_jvp_vjp_dot_product_and_fd'],
         dtype='float64', units='feature', parameters=['observations.visibility_model'], deviation='DEV-0001'),
    dict(id='E11', label='eq:featureobs', pdf='(19)', spec='C2', role='likelihood', category='B', classification='BLOCKED_MODEL_UNSPECIFIED',
         expr='y^F_t = h_t + eps_t, eps_t ~ N_D(0, Gamma_t)', invariant='Gamma_t must be estimated from data and the declared feature construction',
         function='estimator/svgd_feature_measurement.py:FeatureGaussian', tests=[S+'feature_measurement.FeatureTests.test_normalized_nll_and_logdet'],
         dtype='float64', units='feature^2', parameters=['features.Gamma_model'], deviation='DEV-0002'),
    dict(id='E12', label='eq:featcost', pdf='(20)', spec='C5', role='likelihood', category='B', classification='EXACT_MODEL',
         expr='E_F,t = 1/2 e^T Gamma^-1 e + 1/2 log det(2 pi Gamma)', invariant='single normalized feature likelihood per frame; never raw + weighted MMD twice',
         function='estimator/svgd_feature_measurement.py:FeatureGaussian.nll', tests=[S+'feature_measurement.FeatureTests.test_normalized_nll_and_logdet'],
         dtype='float64', units='nat', parameters=[], deviation='NONE'),
    dict(id='E13', label='eq:featgrad', pdf='(21)', spec='C3', role='likelihood', category='A', classification='EXACT_MODEL',
         expr='grad_{x_i} log p = 1/|V| J_psi(x_i,c_i)^T Gamma^-1 (y^F - h)', invariant='matrix-free JVP/VJP, conditional on frozen V_t',
         function='estimator/svgd_feature_measurement.py:RFFFeatures.jvp/vjp', tests=[S+'feature_measurement.FeatureTests.test_jvp_vjp_dot_product_and_fd',
         S+'cuda_parity.CudaParityTests.test_feature_operator'], dtype='float64', units='nat/m', parameters=[], deviation='NONE'),
    dict(id='E14', label='eq:mmd', pdf='(22)', spec='5.1 item 1', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='MMD^2 = ||1/N sum psi(x_i,c_i) - 1/M sum psi(y_j,d_j)||^2 (exact Gaussian product kernel, CPU oracle)',
         invariant='finite RFF map is the DECLARED feature model psi (TF.5 exception), not the infinite kernel',
         function='estimator/svgd_feature_measurement.py:exact_mmd2', tests=[S+'feature_measurement.FeatureTests.test_rff_kernel_and_mmd'],
         dtype='float64', units='1', parameters=[], deviation='NONE'),
    # ---- Phase 1D visibility / unexplained support
    dict(id='E15', label='visibility-law', pdf='8.2', spec='6.3/6.7', role='likelihood', category='B', classification='NOT_DEFINED_IN_THEORY',
         expr='p(V | X_0:T, C) (not specified)', invariant='occluded point never measured by its occluder; no V_t=0 from leaving the basin',
         function='estimator/svgd_visibility.py:ray_visibility (diagnostic classification only)', tests=[S+'mask_visibility.VisibilityTests.test_occluded_static_behind_blade'],
         dtype='float64', units='-', parameters=['observations.visibility_model'], deviation='DEV-0001'),
    dict(id='E16', label='unexplained-support', pdf='8.3', spec='6.5', role='likelihood', category='B', classification='NOT_DEFINED_IN_THEORY',
         expr='B_t latent new/unrepresented observed support (not specified)', invariant='newly exposed geometry is not forced onto old IDs',
         function='NONE', tests=[], dtype='-', units='-', parameters=['observations.unexplained_support_model'], deviation='DEV-0006'),
    dict(id='E17', label='sampling-density', pdf='8.4', spec='6.4', role='likelihood', category='B', classification='NOT_DEFINED_IN_THEORY',
         expr='camera-overlap / footprint sampling model (not specified); 1/M_t is the PDF baseline', invariant='duplicate cameras are not independent area',
         function='estimator/svgd_target_score.py:target_score(log_weights=None -> uniform)', tests=[S+'target_score.TargetScoreTests.test_duplicate_identifier_vs_new_exposure'],
         dtype='-', units='-', parameters=['observations.camera_overlap_policy'], deviation='NONE'),
    # ---- temporal prior
    dict(id='E18', label='eq:temporalprior', pdf='(16)', spec='C4', role='prior', category='B', classification='EXACT_MODEL',
         expr='E_time = 1/2 sum_i |s_i0 - s0bar_i|^2_{P_i0^-1} + 1/2 sum_t sum_i |s_it - F_t s_i,t-1|^2_{Q_t^-1}; F,Q white acceleration (point_kalman_cuda)',
         invariant='actual timestamps, shared frame-0 source covariance once, declared velocity prior',
         function='estimator/svgd_temporal.py:TemporalPrior', tests=[S+'temporal.TemporalTests.test_energy_gradient_and_hessian_dense',
         S+'temporal.TemporalTests.test_gauss_newton_equals_rts_smoother'], dtype='float64', units='nat', parameters=['motion_prior.process_model', 'motion_prior.initial_P'], deviation='NONE'),
    # ---- rigidity (operator D)
    dict(id='E19', label='eq:rigid', pdf='(23)-(24)', spec='7.2', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='|x_it - x_lt| = |x_i0 - x_l0| under common SE(3)', invariant='no pivot/axis/part centre',
         function='estimator/svgd_rigidity.py:edge_residuals', tests=[S+'rigidity.RigidityTests.test_rigid_motion_preserves_and_cross_support_changes'],
         dtype='float64', units='m', parameters=[], deviation='NONE'),
    dict(id='E20', label='eq:distdiff', pdf='(25)', spec='D4', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='(x_i - x_l)^T (v_i - v_l) = 0', invariant='check on estimated motion, not a measurement',
         function='estimator/svgd_rigidity.py:differential_rigidity', tests=[S+'rigidity.RigidityTests.test_rigid_motion_preserves_and_cross_support_changes'],
         dtype='float64', units='m^2/s', parameters=[], deviation='NONE'),
    dict(id='E21', label='eq:edgeresid', pdf='(26)', spec='D1', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='e_il,t = |x_it - x_lt| - |x_i0 - x_l0|', invariant='complete history, no self/coincident edges',
         function='estimator/svgd_rigidity.py:edge_residuals', tests=[S+'rigidity.RigidityTests.test_edge_jacobian_finite_difference'],
         dtype='float64', units='m', parameters=[], deviation='NONE'),
    dict(id='E22', label='eq:edge-delta', pdf='(36)-(37)', spec='7.2', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='de = n_t^T (dx_it - dx_lt) - n_0^T (dx_i0 - dx_l0)', invariant='exact first-order Jacobian',
         function='estimator/svgd_rigidity.py:edge_jacobian', tests=[S+'rigidity.RigidityTests.test_edge_jacobian_finite_difference'],
         dtype='float64', units='1', parameters=[], deviation='NONE'),
    dict(id='E23', label='eq:edge-cov', pdf='(38)', spec='D2', role='diagnostic', category='C', classification='MODEL_APPROX',
         expr='Omega_il = J P_zeta J^T; production: shared-reference exact, trajectory states independent across time and points (rank-1 + diagonal)',
         invariant='shared frame-0 reference counted once; Sherman-Morrison inverse/log det exact for that structure',
         function='estimator/svgd_rigidity.py:edge_covariance_structured', tests=[S+'rigidity.RigidityTests.test_covariance_monte_carlo_and_structured_exact',
         S+'rigidity.RigidityTests.test_structured_inverse_equals_dense'], dtype='float64', units='m^2', parameters=['rigid_graph.covariance_approximation'], deviation='DEV-0007'),
    dict(id='E24', label='eq:rigidlik', pdf='9.1', spec='D3', role='evidence', category='B', classification='EXACT_MODEL',
         expr='p(e | H^R) = N_T(e; 0, Omega + Omega_model)', invariant='conditional likelihood only, not a posterior; static pairs also satisfy it',
         function='estimator/svgd_rigidity.py:rigid_edge_loglik', tests=[S+'rigidity.RigidityTests.test_structured_inverse_equals_dense'],
         dtype='float64', units='nat', parameters=['rigid_graph.Omega_model'], deviation='DEV-0010'),
    dict(id='E25', label='eq:edgepost', pdf='(27)', spec='7.1', role='evidence', category='B', classification='BLOCKED_MODEL_UNSPECIFIED',
         expr='w_il = P(z_il = 1 | Q_0:T)', invariant='coherent partition, not independent Bernoulli edges', function='NONE (production); tiny oracle',
         tests=[S+'support_posterior.PosteriorTests.test_partition_enumeration_coherent'], dtype='float64', units='1',
         parameters=['rigid_graph.support_prior'], deviation='NONE'),
    # ---- Phase 1F evidence (operator E)
    dict(id='E26', label='eq:fullpost', pdf='(29)', spec='E1', role='evidence', category='B', classification='BLOCKED_MODEL_UNSPECIFIED',
         expr='p(X,m,Z,V,B | Q) ~ p(Q | X,V,B) p(X | m,Z) p(V,B,m,Z)', invariant='one posterior, one observation model',
         function='estimator/svgd_support_posterior.py:support_posterior (BLOCKED in production)', tests=[S+'support_posterior.PosteriorTests.test_point_marginals_exact_enumeration'],
         dtype='float64', units='-', parameters=['model_evidence.method', 'model_evidence.priors'], deviation='NONE'),
    dict(id='E27', label='eq:marginal', pdf='(30)', spec='E2', role='evidence', category='A', classification='EXACT_MODEL',
         expr='gamma_i = sum_{m_-i,Z} int p(..., m_i=1, ... | Q)', invariant='exact enumeration on tiny hypothesis sets',
         function='estimator/svgd_support_posterior.py:enumerate_marginals', tests=[S+'support_posterior.PosteriorTests.test_point_marginals_exact_enumeration'],
         dtype='float64', units='1', parameters=[], deviation='NONE'),
    dict(id='E28', label='eq:hypotheses', pdf='(31)-(32)', spec='8.2', role='evidence', category='B', classification='EXACT_MODEL',
         expr='H0: x_t = x_0; H1: x_t = R_t x_0 + b_t (latent, identity at frame 0, proper random-walk prior)',
         invariant='same observations and noise under both; extra freedom penalized by integration',
         function='estimator/svgd_support_posterior.py:linear_gaussian_log_evidence/laplace_log_evidence', tests=[S+'support_posterior.PosteriorTests.test_moving_fits_jitter_but_loses_evidence'],
         dtype='float64', units='nat', parameters=['model_evidence.priors'], deviation='NONE'),
    dict(id='E29', label='eq:evidence', pdf='(33)', spec='E3', role='evidence', category='A', classification='EXACT_MODEL',
         expr='L^h = int p(y | Theta, H^h) p(Theta | H^h) dTheta', invariant='closed form (linear Gaussian) vs quadrature oracle',
         function='estimator/svgd_support_posterior.py:linear_gaussian_log_evidence', tests=[S+'support_posterior.PosteriorTests.test_linear_gaussian_evidence_vs_quadrature'],
         dtype='float64', units='nat', parameters=[], deviation='NONE'),
    dict(id='E30', label='eq:bayes', pdf='(34)', spec='E3', role='evidence', category='A', classification='EXACT_MODEL',
         expr='P(H1 | y) = pi L1 / (pi L1 + (1 - pi) L0) (log-sum-exp)', invariant='p_static + p_dynamic = 1 only for the complete binary model',
         function='estimator/svgd_support_posterior.py:binary_posterior', tests=[S+'support_posterior.PosteriorTests.test_no_observation_returns_prior'],
         dtype='float64', units='1', parameters=['model_evidence.priors'], deviation='NONE'),
    dict(id='E31', label='eq:laplace', pdf='10.2 (Laplace named)', spec='E4', role='evidence', category='C', classification='MODEL_APPROX',
         expr='log L ~ -E(theta*) + d/2 log 2 pi - 1/2 log det Hess', invariant='only with all normalizers and PD Hessian; error vs quadrature reported',
         function='estimator/svgd_support_posterior.py:laplace_log_evidence', tests=[S+'support_posterior.PosteriorTests.test_laplace_error_nonlinear_rotation'],
         dtype='float64', units='nat', parameters=[], deviation='NONE'),
    dict(id='E32', label='eq:jointobjective', pdf='(35)', spec='C6', role='evidence', category='C', classification='MODEL_APPROX',
         expr='E = sum_t E_F,t - log p_motion + 1/2 sum z e^T Omega^-1 e - log p(m,Z,V,B) + C_norm; production trajectory step uses E_obs + E_time only',
         invariant='conditional trajectory objective; rigidity/support/visibility terms not in the accepted-step objective',
         function='estimator/svgd_support_run.py:SequenceObjective', tests=[S+'support_runner.RunnerTests.test_objective_monotone_on_accepted_steps'],
         dtype='float64', units='nat', parameters=[], deviation='DEV-0008'),
    dict(id='E33', label='eq:seedrisk', pdf='10.3', spec='E5', role='decision', category='A', classification='EXACT_MODEL',
         expr='FDR(S) = sum_{i in S} (1 - gamma_i) / max(1, |S|) <= alpha', invariant='only on established posteriors; IDs are original IDs',
         function='estimator/svgd_support_posterior.py:select_seeds', tests=[S+'support_posterior.PosteriorTests.test_seed_risk_rule'],
         dtype='float64', units='1', parameters=['seed_selection.posterior_risk_alpha'], deviation='NONE'),
    dict(id='E34', label='eq:hinge', pdf='(28)', spec='14.2', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='|x_t - x_0| = 2 r |sin(theta/2)|', invariant='used only to build evaluation fixtures', function='tests fixtures', tests=[S+'rigidity.RigidityTests.test_hinge_displacement_identity'],
         dtype='float64', units='m', parameters=[], deviation='NONE'),
    dict(id='E35', label='eq:increments', pdf='(4)', spec='3.2', role='diagnostic', category='A', classification='EXACT_MODEL',
         expr='Cov(dz_t, dz_t+1) = -R for i.i.d. errors', invariant='noise study, not a residual', function='scene/pointcloud_sequence.py:increment_noise_study',
         tests=[S+'mask_visibility.VisibilityTests.test_increment_anticorrelation_identity'], dtype='float64', units='m^2, rgb^2', parameters=[], deviation='NONE'),
    dict(id='E36', label='stages-2-4', pdf='(39)-(44)', spec='10.4', role='diagnostic', category='B', classification='OUT_OF_SCOPE_PHASE1',
         expr='rigid recovery, joint identification, support completion', invariant='no fake downstream outputs', function='NONE', tests=[S+'support_runner.RunnerTests.test_no_legacy_stage_outputs'],
         dtype='-', units='-', parameters=[], deviation='NONE'),
]

THEORY_CONFLICTS = [
    dict(id='TC-0001', status='OPEN_REQUIRES_USER_DECISION',
         theory='PDF (3),(8),(11): reference colour c_i is a fixed vector; the score is evaluated at the fixed query colour.',
         spec='Spec 1.1/3.2: "Propagate uncertainty of a_i; it is not noise-free"; Cov(a_i) must enter colour comparison once.',
         consequence='fixed c_i: colour flicker of the reference is attributed to the target bandwidth only; marginalized: N3(a_i; d_j, LambdaC_j + Sigma_a) '
                     '(exact Gaussian convolution) broadens every colour factor by the same shared reference covariance.',
         implementation='target_score(query_color_cov=None) = PDF; query_color_cov=Sigma_a = spec. Config key reference.color_reference_uncertainty selects; '
                        'strict mode refuses to run TARGET while the choice is NOT_DEFINED_IN_THEORY.'),
]

# Statistical / model parameters that strict mode requires to be declared (class must be one of ALLOWED_CLASSES).
REQUIRED_PARAMETERS = ['target_density.Lambda_X', 'target_density.Lambda_C', 'target_density.source_weights', 'reference.color_reference_uncertainty',
                       'stein.H_X', 'stein.H_C', 'stein_numerics.step_policy', 'features.feature_method', 'features.xyz_rgb_kernel_B', 'features.D',
                       'features.fixed_seed', 'features.Gamma_model', 'observations.visibility_model', 'observations.unexplained_support_model',
                       'observations.camera_overlap_policy', 'motion_prior.process_model', 'motion_prior.initial_P', 'rigid_graph.edge_proposal_method',
                       'rigid_graph.covariance_approximation', 'rigid_graph.Omega_model', 'rigid_graph.support_prior', 'model_evidence.method',
                       'model_evidence.priors', 'seed_selection.posterior_risk_alpha']
ALLOWED_CLASSES = ('derived-from-measured-input', 'declared-research-assumption', 'numerical-error-budget', 'hardware-budget')
# Which stage needs which parameters (the runner blocks at the first stage whose parameters are not usable).
STAGE_PARAMETERS = {
    'INIT': [],
    'TARGET': ['target_density.Lambda_X', 'target_density.Lambda_C', 'target_density.source_weights', 'reference.color_reference_uncertainty'],
    'STEIN_PROPOSAL': ['stein.H_X', 'stein.H_C', 'stein_numerics.step_policy', 'features.feature_method', 'features.xyz_rgb_kernel_B', 'features.D',
                       'features.fixed_seed', 'features.Gamma_model', 'observations.visibility_model', 'observations.unexplained_support_model',
                       'motion_prior.process_model', 'motion_prior.initial_P'],
    'TEMPORAL_REFINEMENT': [],
    'RIGID_GRAPH': ['rigid_graph.edge_proposal_method', 'rigid_graph.covariance_approximation', 'rigid_graph.Omega_model'],
    'MODEL_EVIDENCE': ['rigid_graph.support_prior', 'model_evidence.method', 'model_evidence.priors'],
    'JOINT_REFINEMENT': [],
    'SEED_SELECTION': ['seed_selection.posterior_risk_alpha'],
}
STAGES = list(STAGE_PARAMETERS)


def lookup(config, dotted):
    node = config
    for k in dotted.split('.'): node = node.get(k) if isinstance(node, dict) else None
    return node


def parameter_problems(config, names, mode):
    """[(name, reason)] for parameters unusable in `mode` ('strict' | 'experimental'). Strict additionally rejects any
    parameter tied to a deviation that is not AUTHORIZED; experimental accepts PROPOSED deviations (results quarantined)."""
    out = []
    for n in names:
        p = lookup(config, n)
        if not isinstance(p, dict): out.append((n, 'MISSING_FROM_CONFIG')); continue
        if p.get('class') not in ALLOWED_CLASSES or p.get('value') is None:
            out.append((n, f"BLOCKED_MODEL_UNSPECIFIED: class={p.get('class')}; calibration path: {p.get('calibration_path', 'not given')}")); continue
        dev = p.get('deviation')
        if dev and dev != 'NONE' and mode == 'strict' and p.get('authorization') != 'AUTHORIZED':
            out.append((n, f'UNAUTHORIZED_DEVIATION {dev} ({p.get("authorization", "PROPOSED_NOT_AUTHORIZED")})'))
    return out


def open_conflicts(config, mode):
    """THEORY_CONFLICT records still open for the requested computation (strict refuses them)."""
    choice = lookup(config, 'reference.color_reference_uncertainty')
    resolved = isinstance(choice, dict) and choice.get('class') in ALLOWED_CLASSES and choice.get('value') is not None \
        and (mode == 'experimental' or choice.get('authorization') == 'AUTHORIZED')
    return [] if resolved else [c['id'] for c in THEORY_CONFLICTS]


def model_contract(config, mode):
    """svgd_support/model_contract.json: what every output conditions on."""
    blocked = {st: parameter_problems(config, ps, mode) for st, ps in STAGE_PARAMETERS.items()}
    return {'theory_pdf': THEORY_PDF, 'spec': SPEC, 'label_mapping': LABEL_MAPPING, 'transport_label': TRANSPORT_LABEL, 'mode': mode,
            'equations': EQUATIONS, 'theory_conflicts': THEORY_CONFLICTS, 'open_conflicts': open_conflicts(config, mode),
            'stage_parameter_problems': {k: [{'parameter': a, 'reason': b} for a, b in v] for k, v in blocked.items()},
            'probability_semantics': 'NONE_UNTIL_MODEL_EVIDENCE_GATE: no p_dynamic is published while E26/E25 are BLOCKED_MODEL_UNSPECIFIED',
            'provisional_outputs': 'transport_provisional / velocity_provisional / support_edges are INTERNAL_NOT_FINAL_TRACKS diagnostics'}


PROHIBITED = {'E01': 'nearest point, hard correspondence, unnormalized kernels, implicit sample weights',
              'E03': 'different Gaussian kernel, missing covariance inverse or colour factor', 'E05': 'dropping colour, depth-dependent bandwidth without a model',
              'E07': 'degree normalization, neighbour-only sum, repulsion removal, unreported clipping', 'E18': 'independent frame-to-frame registration as sequence inference',
              'E13': 'feature replacement, fixed denominator with changing support, untracked RFF approximation', 'E12': 'unweighted + weighted MMD as two measurements; diagonal Gamma called exact',
              'E23': 'treating distances at different frames or edges as independent', 'E26': 'independent edge probabilities as a partition; energies summed as posterior',
              'E29': 'MAP improvement, sigmoid of displacement/Stein norm, p-value as probability', 'E33': 'near-hinge labels from GT/joint axis; fake legacy outputs'}

EQUIVALENT_OPTIMIZATIONS = [
    dict(id='OPT-01', equations=['E01', 'E02', 'E03'], what='tiled streaming log-sum-exp over ALL components (running max, rescaled sums)', test=S+'target_score.TargetScoreTests.test_log_density_and_score_match_dense_numpy'),
    dict(id='OPT-02', equations=['E05', 'E06', 'E07'], what='tiled all-pairs Stein sums with global 1/N', test=S+'stein_operator.SteinTests.test_dense_numpy_oracle_all_terms'),
    dict(id='OPT-03', equations=['E13'], what='matrix-free JVP/VJP, point-chunked, no N x D Jacobian', test=S+'feature_measurement.FeatureTests.test_jvp_vjp_dot_product_and_fd'),
    dict(id='OPT-04', equations=['E18'], what='exact block-Thomas solve of the temporal Hessian, point-chunked', test=S+'temporal.TemporalTests.test_energy_gradient_and_hessian_dense'),
    dict(id='OPT-05', equations=['E23', 'E24'], what='Sherman-Morrison inverse / log det of rank-1 + diagonal Omega', test=S+'rigidity.RigidityTests.test_structured_inverse_equals_dense'),
    dict(id='OPT-06', equations=['E03', 'E07', 'E13', 'E18', 'E24'], what='torch CUDA float64 execution of the same operators (TF32 off)', test=S+'cuda_parity.CudaParityTests'),
]

_P = 'PROPOSED'; _NA = {'source': 'none', 'decision': 'NOT_AUTHORIZED (experimental mode only; excluded from theory-fidelity acceptance)'}
DEVIATIONS = [
    dict(id='DEV-0001', status=_P, classification='MODEL_APPROX', original_theory={'document': THEORY_PDF, 'equation': '(18), 8.2', 'exact_expression': 'h_t over the latent visible set V_t with p(V | X, C)'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': '_stages / FrameFeatureTerm', 'exact_expression_or_algorithm': 'V_t = {1..N} for every t, alpha_i = 1/N'},
         reason='the visibility law is NOT_DEFINED_IN_THEORY; without some V_t no feature likelihood exists', mathematical_difference='conditions on all canonical points being comparable; ignores occlusion/self-occlusion and visibility-event likelihood',
         affected_outputs='E_obs, accepted transport, all diagnostics', error_or_bias={'units': '-', 'measured': 'occluded base points flagged by the diagnostic (tests/test_svgd_mask_visibility)', 'bound_or_limitation': 'unbounded under occlusion (PDF 7.2 warns of false motion evidence)'},
         independent_oracle={'test': S+'mask_visibility.VisibilityTests.test_occluded_static_behind_blade', 'result': 'failure mode exposed', 'tolerance_basis': '-'},
         correctness_risk='static points pulled to occluders; false motion', alternatives_considered='point-only ray-visibility law with event likelihood (BLOCKED: not specified)',
         ablation_command='--svgd-theory-fidelity experimental --config configs/svgd_support_experimental.json', authorization=_NA, conclusion='NON_EQUIVALENT'),
    dict(id='DEV-0002', status=_P, classification='MODEL_APPROX', original_theory={'document': THEORY_PDF, 'equation': '(19)', 'exact_expression': 'Gamma_t estimated from data incl. measurement, flicker, sampling, correlations'},
         actual_implementation={'file': 'estimator/svgd_feature_measurement.py', 'function': 'iid_sampling_gamma', 'exact_expression_or_algorithm': 'Gamma_t = C_Q/M_t + C_P/N at the stationary init'},
         reason='no calibrated feature covariance exists', mathematical_difference='independent i.i.d. sample means; drops ray/camera/time correlation and model mismatch; frozen (no log det dependence on state)',
         affected_outputs='E_obs scale, step acceptance', error_or_bias={'units': 'feature^2', 'measured': 'condition numbers in status.json gates.FEATURES', 'bound_or_limitation': 'likely overconfident (correlated samples)'},
         independent_oracle={'test': S+'feature_measurement.FeatureTests.test_normalized_nll_and_logdet', 'result': 'likelihood exact GIVEN Gamma', 'tolerance_basis': '1e-10'},
         correctness_risk='overweighted data term chases noise', alternatives_considered='repeated-stationary bootstrap of feature means (needs stationary intervals, not available without labels)',
         ablation_command='edit features.Gamma_model', authorization=_NA, conclusion='NON_EQUIVALENT'),
    dict(id='DEV-0003', status=_P, classification='THEORY_CHANGE', original_theory={'document': THEORY_PDF, 'equation': '(8), 5.4, 14', 'exact_expression': 'LambdaX, LambdaC declared and validated (not specified)'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': 'representation_scales', 'exact_expression_or_algorithm': 'LambdaX_j = Sigma_j + s_t^2 I (median exact NN distance); LambdaC = R_C + diag(1/2 median (d_j - d_nn)^2)'},
         reason='Codex proposal (EXPERIMENTAL_DEVIATION) to make TARGET runnable', mathematical_difference='a specific representation-scale rule chosen by Codex', affected_outputs='score, Stein field, transport',
         error_or_bias={'units': 'm^2 / rgb^2', 'measured': 'status.json gates.TARGET.representation_scales', 'bound_or_limitation': 'no sensitivity study yet'},
         independent_oracle={'test': '-', 'result': 'not validated', 'tolerance_basis': '-'}, correctness_risk='narrow basin (no motion found) or merged surfaces',
         alternatives_considered='coarse-to-fine continuation; per-camera footprint scales', ablation_command='edit target_density.Lambda_*', authorization=_NA, conclusion='UNDETERMINED'),
    dict(id='DEV-0004', status=_P, classification='THEORY_CHANGE', original_theory={'document': THEORY_PDF, 'equation': '(13)', 'exact_expression': 'HX, HC declared interaction bandwidths (not specified)'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': 'median_heuristic', 'exact_expression_or_algorithm': 'H = med^2 / (2 log N) I per block (Liu & Wang reference heuristic)'},
         reason='Codex proposal', mathematical_difference='bandwidth from particle spread', affected_outputs='Stein field', error_or_bias={'units': 'm^2', 'measured': 'status.json gates.FEATURES.H_X_m2', 'bound_or_limitation': 'object-scale coupling across static/moving boundaries'},
         independent_oracle={'test': '-', 'result': 'not validated', 'tolerance_basis': '-'}, correctness_risk='boundary contamination (PDF 6.3)', alternatives_considered='fixed physical scale; per-block validation',
         ablation_command='edit stein.H_*', authorization=_NA, conclusion='UNDETERMINED'),
    dict(id='DEV-0005', status=_P, classification='THEORY_CHANGE', original_theory={'document': SPEC, 'equation': '5.1', 'exact_expression': 'declared psi, B_X, B_C, D'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': '_stages', 'exact_expression_or_algorithm': 'RFF psi with B = H (Stein bandwidths), D = 1024'},
         reason='Codex proposal', mathematical_difference='feature kernel tied to the transport scale', affected_outputs='E_obs, acceptance', error_or_bias={'units': '-', 'measured': '-', 'bound_or_limitation': 'which differences are observable is untested'},
         independent_oracle={'test': S+'feature_measurement.FeatureTests.test_rff_kernel_and_mmd', 'result': 'map exact; choice unvalidated', 'tolerance_basis': '6/sqrt(D)'}, correctness_risk='insensitive to fine motion',
         alternatives_considered='B = Lambda; multi-scale features', ablation_command='edit features.*', authorization=_NA, conclusion='UNDETERMINED'),
    dict(id='DEV-0006', status=_P, classification='MODEL_APPROX', original_theory={'document': THEORY_PDF, 'equation': '8.3', 'exact_expression': 'latent unexplained support B_t in the likelihood'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': '_stages', 'exact_expression_or_algorithm': 'B_t omitted'}, reason='law not specified',
         mathematical_difference='all observed feature mass must be explained by P_0', affected_outputs='transport toward newly exposed geometry', error_or_bias={'units': '-', 'measured': 'observed_samples_without_supported_particle in observed_counts.csv', 'bound_or_limitation': 'unbounded under disocclusion'},
         independent_oracle={'test': '-', 'result': '-', 'tolerance_basis': '-'}, correctness_risk='forced transport onto new surfaces', alternatives_considered='normalized background component with declared mass (BLOCKED)',
         ablation_command='-', authorization=_NA, conclusion='NON_EQUIVALENT'),
    dict(id='DEV-0007', status=_P, classification='MODEL_APPROX', original_theory={'document': THEORY_PDF, 'equation': '(38)', 'exact_expression': 'Omega = J P_zeta J^T with the joint covariance of all states'},
         actual_implementation={'file': 'estimator/svgd_rigidity.py', 'function': 'edge_covariance_structured / prior_position_marginals', 'exact_expression_or_algorithm': 'shared-reference exact (rank 1) + independent prior-propagated marginals per time and point'},
         reason='joint posterior covariance of 6N(T+1) states infeasible', mathematical_difference='drops cross-time/cross-point correlation of trajectory states and all data information', affected_outputs='rigid_chi2, rigid_loglik',
         error_or_bias={'units': 'm^2', 'measured': 'dense vs structured identical when structure holds; MC agreement 3%', 'bound_or_limitation': 'conservative (prior marginals broader than posterior)'},
         independent_oracle={'test': S+'rigidity.RigidityTests.test_covariance_monte_carlo_and_structured_exact', 'result': 'PASS', 'tolerance_basis': 'MC 2e5 samples'},
         correctness_risk='understated rigidity evidence', alternatives_considered='selected-inverse posterior covariance (earlier temporal-sparse work)', ablation_command='-', authorization=_NA, conclusion='CONTROLLED_APPROXIMATION'),
    dict(id='DEV-0008', status=_P, classification='MODEL_APPROX', original_theory={'document': THEORY_PDF, 'equation': '(35)', 'exact_expression': 'E = sum E_F - log p_motion + rigidity + support/visibility priors + C_norm'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': 'SequenceObjective', 'exact_expression_or_algorithm': 'E_obs + E_time (conditional on frozen V_t; Z, m not in the step objective)'},
         reason='support/visibility laws not specified', mathematical_difference='conditional trajectory objective (VISIBILITY_CONDITIONAL_APPROXIMATION)', affected_outputs='accepted transport',
         error_or_bias={'units': 'nat', 'measured': 'objective_timeline.csv', 'bound_or_limitation': 'no global ELBO monotonicity claimed'}, independent_oracle={'test': S+'support_runner.RunnerTests.test_objective_monotone_on_accepted_steps', 'result': 'monotone', 'tolerance_basis': 'exact comparison'},
         correctness_risk='self-consistent but non-posterior transport', alternatives_considered='structured variational ELBO with discrete-label entropy (BLOCKED)', ablation_command='-', authorization=_NA, conclusion='NON_EQUIVALENT'),
    dict(id='DEV-0009', status=_P, classification='NUMERICAL_APPROX', original_theory={'document': SPEC, 'equation': 'DEV-0003/DEV-0004 statistics', 'exact_expression': 'median over all samples / pairs'},
         actual_implementation={'file': 'estimator/svgd_support_run.py', 'function': 'representation_scales / median_heuristic', 'exact_expression_or_algorithm': 'median over <= 4096 queries evenly spaced in (camera name, pixel) order'},
         reason='O(M^2) exact median per frame', mathematical_difference='sample median estimate of a declared scale; every component still enters the density', affected_outputs='Lambda, H values',
         error_or_bias={'units': 'm', 'measured': 'exact when M <= 4096', 'bound_or_limitation': 'order-invariant under camera permutation'}, independent_oracle={'test': S+'support_runner.RunnerTests.test_camera_order_invariance', 'result': 'PASS', 'tolerance_basis': 'relative 1e-4'},
         correctness_risk='low', alternatives_considered='exact selection on GPU', ablation_command='-', authorization=_NA, conclusion='CONTROLLED_APPROXIMATION'),
    dict(id='DEV-0010', status=_P, classification='THEORY_CHANGE', original_theory={'document': SPEC, 'equation': 'D3', 'exact_expression': 'Omega_model declared rigidity-model discrepancy'},
         actual_implementation={'file': 'configs/svgd_support_experimental.json', 'function': 'rigid_graph.Omega_model', 'exact_expression_or_algorithm': 'Omega_model = 0'},
         reason='not specified', mathematical_difference='no discrepancy term', affected_outputs='rigid_chi2', error_or_bias={'units': 'm^2', 'measured': '-', 'bound_or_limitation': 'overconfident strain'},
         independent_oracle={'test': '-', 'result': '-', 'tolerance_basis': '-'}, correctness_risk='spurious strain on real data', alternatives_considered='calibrate on static intervals', ablation_command='-', authorization=_NA, conclusion='UNDETERMINED'),
]
