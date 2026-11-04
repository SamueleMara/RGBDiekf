# SVGD Stage 1 — proposed theory amendments (STATUS: PROPOSED, AWAITING USER APPROVAL)

Response to `RGBDiekf_SVGD_Commit_b62c7cee_Technical_Audit_2026-11-03.md`, recommended-order items 1, 3, 4, 5, 6. Nothing below is implemented as
inference: per the audit ("formally amend [the theory], not a silent code decision") each block needs an explicit decision. Once approved, each
becomes contract rows in `estimator/svgd_contract.py`, an independent oracle, and a named gate, exactly like E01–E38. Baseline theory:
`docs/references/Stein_SVGD_Moving_Support_Pipeline.pdf` (PDF). Notation as in the PDF; c indexes cameras, t frames, i persistent points, j samples.

## A1 — reference colour (resolves THEORY_CONFLICT TC-0001)

- **Option F (PDF (3), (11))**: c_i fixed. Simple; RGB flicker of the frame-0 reference is absorbed by Λ_C only.
- **Option L (recommended)**: one latent true colour a_i per persistent point, prior a_i ~ N(c_i, R_C,i) entered ONCE in the joint objective
  (E_ref = ½ Σ_i |a_i − c_i|²_{R_C^-1} + normalizer). a_i is a shared state across all frames (so frames are not independently re-conditioned on a
  noisy reference); ψ(x_i,t, a_i) uses a_i; the Gauss-Newton state grows from [x, v] to [x, v] ∪ {a}. The KDE proposal may use the per-frame marginal
  N3(a_i; d_j, Λ_C + Σ_a) (already implemented, E04) because a proposal never counts as evidence.
- Across-frame dependence: under L the per-frame feature residuals are conditionally independent given (X, a), dependent marginally — exactly the
  structure the joint objective handles. Option F + marginal-per-frame is NOT allowed in the likelihood (it re-counts the reference at every frame).

## A2 — normalized point-only visibility p(V | X, C) (audit F4)

Per camera c, frame t, point i: angular bin b = π_c(x_i,t) of the calibrated pixel grid, observed front-most masked depth ẑ_c,t(b) (if the bin is
mask-valid), particle depth z_i = z_c(x_i,t), σ²_ic = R_D(ẑ) + R_D(z_i) (measured depth noise, never a free constant).

    P(v_ict = 1 | X) = 1[b inside image, z_i > 0] · 1[b mask-valid] · Φ((ẑ_c,t(b) − z_i)/σ_ic + κ)

with κ ≥ 0 a declared surface-thickness term in σ units (Φ(κ) is the visibility of a point lying exactly on the observed surface; κ = 2 gives
0.977; to be approved, not chosen in code). Event semantics:
`ẑ < z_i` by many σ → occluded (probability → 0: the point is not compared with its occluder, audit F4);
`ẑ > z_i` by many σ → FREE-SPACE contradiction (the camera saw through the point's location): this is evidence, scored by an explicit factor
`log(1 − p_fs)` with declared p_fs, never silently dropped; bin not mask-valid but inside the image → off-foreground contradiction under hard masks
(same factor). Piecewise-smooth in X (Φ is smooth; bin changes are active-set changes logged per step).

**Camera-stratified feature observation** (also fixes camera-overlap duplication, PDF 8.4): for each (t, c)

    y_tc = 1/M_tc Σ_{j in c} ψ(y_j, d_j),   h_tc(X) = Σ_i w_ict ψ(x_i,t, a_i) / Σ_i w_ict,   w_ict = P(v_ict = 1 | X) · f_ic,

f_ic = expected sample footprint of point i in camera c (z_i² / (f_x f_y |n·ray|) — needs a normal: PROPOSED to use the frame-0 local PCA normal of
the canonical neighbourhood, a declared estimate) so that the predicted mean is a sampling-density-consistent average. The derivative of w_ict
w.r.t. x enters the full Jacobian (spec C3 "weight derivatives").

## A3 — unexplained / newly exposed support B_t (audit F4)

Mixture in feature space per (t, c): y_tc = (1 − β_tc) h_tc(X) + β_tc u_tc + ε_tc, with
β_tc ~ Beta(α_β, β_β) (declared prior on the unexplained fraction) and u_tc ~ N(ū_tc, S_u) the mean feature of unexplained surface
(ū_tc = y_tc-independent prior mean = feature mean of a declared broad density over the camera's own masked rays; S_u its sampling covariance).
Marginalizing u: y_tc | X, β ~ N((1 − β) h + β ū, Γ_tc + β² S_u) — normalized; β integrated by 1D quadrature per (t, c) (exact enough, oracle by
dense quadrature). Explaining new geometry by β > 0 costs prior mass; moving old points onto it costs E_time/visibility: the comparison is a
likelihood ratio, not a free choice.

## A4 — coupled rigidity: support assignment, motion prior, partition prior (audit F1)

Latent labels z_i ∈ {S (static), 1..K_max (moving supports), U (unexplained free motion)}; coherent by construction (a labelling induces a transitive
partition). Per moving support k: rigid nuisance T_k(t) ∈ SE(3), T_k(0) = I, prior: twist random walk ξ_k(t) − ξ_k(t−1) ~ N(0, Δτ_t Q_ξ) (declared Q_ξ).
Support-conditional trajectory prior (replaces the label-free E_time for labelled points):

    S: x_i(t) ~ N(x_i^0, Σ_i^0 + Σ_def)         k: x_i(t) ~ N(T_k(t) x_i^0, Σ_i^0 + Σ_def)         U: the current white-acceleration prior (C4)

Σ_def = declared rigid-deformation tolerance (a model term, not damping). Partition prior: Chinese-restaurant process CRP(α) over moving supports
(exchangeable, normalized, unknown count) times π_S / π_U for the static / unexplained labels; NO spatial MRF (its normalizer is intractable) —
spatial locality enters only through the likelihood. Inference: structured variational EM, q(z) = Π_i q_i(z_i) (categorical; transitivity holds per
sample of q), Gauss/Laplace q(T_k), and the trajectory variables X optimized in the same ELBO:

    ELBO = E_q[log p(Q | X, a, V, B)] + E_q[log p(X | z, T)] + log p(T) + E_q[log p(z)] + H[q(z)] + H[q(T)]

The Stein field stays a PROPOSAL for X; acceptance is on the ELBO. Ablations demanded by the audit: (i) rigidity coupled (this) vs post-hoc graph
(current), (ii) same fixtures, (iii) two simultaneously moving bodies and moving/static contact. Existing kNN distance histories (E19–E24) become a
diagnostic of q, not a separate likelihood (no double counting of the same trajectories).

## A5 — feature-noise covariance Γ_tc (audit F5)

Traceable decomposition (each term from measured inputs; no free inflation):

    Γ_tc = J_y diag(R_X,j) J_yᵀ / M_tc² + J_d (R_C ⊗ C_ρ) J_dᵀ / M_tc² + Γ_sample,tc + (reference: once via a_i, A1)

J_y, J_d: ψ Jacobians at the observed samples; C_ρ: same-camera spatial correlation of RGB flicker estimated from neighbouring-pixel increment
covariance (PDF (4) generalized; majority statistics, no labels); Γ_sample: block bootstrap over image-row blocks of the masked set (mask-edge
resampling). Validation: whitened residuals on held-out stationary frames, e_tcᵀ Γ_tc⁻¹ e_tc vs χ²_D coverage (static scenes and the RGB-flicker
fixture); any unidentifiable component reported, never set by hand.

## A6 — separate scales Λ_X, H_X, B_X (audit F6)

- Λ_X,j = Σ_j + (z_j²/(f_x f_y)) · (1/12) I·m_Λ: the measured covariance plus the pixel-footprint representation scale (m_Λ ≥ 1 declared, default 1).
- H_X = r_m² I, r_m = median distance to the m-th canonical neighbour (m declared, e.g. the graph k): part-LOCAL interaction instead of the
  object-scale median heuristic (118 mm on Blade); validated by the post-hoc cross-support mass audit (`tools/audit_svgd_stein_rejection.py`).
- ψ multiscale: concatenated RFF blocks at B ∈ {Λ-scale, H-scale, coarse = object scale}, one Γ for the concatenated feature (no double counting).

## A7 — scalable normalized support posterior (audit F8)

γ_i = q(z_i ∈ {1..K_max}) from the A4 variational solution, with an explicit U alternative and CRP complexity; points with no usable observation
keep their prior. Required before publishing: (1) exact parity with `estimator/svgd_support_posterior.enumerate_marginals` on ≤ 5-point toys
(ELBO vs exact evidence gap reported), (2) calibration (reliability / ECE) on held-out synthetic sequences, (3) no-observation → prior, (4) E5 seed
selection only on these γ. Label `APPROXIMATE_MODEL_POSTERIOR` (variational), never `CALIBRATED` without held-out evidence.

## Decisions requested (tick to authorize; each unlocks the listed implementation)

| # | decision | options | unlocks |
|---|---|---|---|
| D1 | TC-0001 reference colour | F (fixed) / **L (latent shared a_i)** | E04 resolution, a_i state |
| D2 | visibility law A2 incl. κ, p_fs, footprint normals | approve / modify | E10, E15, camera-stratified features |
| D3 | unexplained support A3 incl. Beta prior | approve / modify | E16 |
| D4 | coupled rigidity A4 incl. Q_ξ, Σ_def, CRP α, π_S, π_U, K_max | approve / modify | F1 transport coupling, E25, E26 |
| D5 | Γ decomposition A5 | approve / modify | E11, DEV-0002 replacement |
| D6 | scales A6 incl. m_Λ, m | approve / modify | DEV-0003/0004/0005 replacement |
| D7 | posterior A7 | approve / modify | p_dynamic, seeds (Gate 6) |

Until then the experimental config keeps DEV-0001..0011 as PROPOSED_NOT_AUTHORIZED and strict mode keeps blocking at TARGET.
