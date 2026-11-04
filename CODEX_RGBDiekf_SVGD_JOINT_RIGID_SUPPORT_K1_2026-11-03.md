# CODEX IMPLEMENTATION SPECIFICATION
# Point-cloud-only, jointly inferred moving support and rigid motion using Rao–Blackwellized SVGD (K = 1)

**Date:** 2026-11-03  
**Repository:** `SamueleMara/RGBDiekf` (local checkout has also been called `RGBDekf`)  
**Status:** NEW FRAMEWORK — IMPLEMENT IN NEW MODULES, NOT AS A PATCH TO THE FAILED FREE-POINT SVGD  
**Priority:** physical correctness > Bayesian semantics > regression protection > speed > visual polish.

> **Non-negotiable goal.** Given a time sequence of *3D XYZ–RGB point sets*, discover a high-confidence subset of reference points that (i) is poorly explained by remaining stationary and (ii) is jointly explained by **one shared rigid transformation sequence**. The number of moving parts is supplied as `K=1`. Do **not** supply GT point membership, GT motion, joint type or joint parameters to the estimator. Infer support and motion **together**, rather than transporting individual physical points first and grouping their trajectories afterward.

> **SVGD must be retained.** An SVGD particle in the new algorithm is a **hypothesis about the one rigid-motion sequence and the global moving-support fraction**, not a freely moving point from the scene. All reference points participate in the posterior score of each hypothesis. Binary support labels are *analytically marginalized conditional on the shared motion* (Rao–Blackwellization), which avoids pretending that binary labels are Euclidean SVGD coordinates.

## 0. Why we are replacing the old formulation

Read `reports/svgd/falsification_root_cause_2026-11-03.md` and the original `estimator/svgd_support_run.py`, `estimator/svgd_target_score.py`, `estimator/svgd_feature_measurement.py`, `estimator/svgd_temporal.py` before writing code.

The prior **free-point transport** formulation failed for different, independently verified reasons:

| Experiment | Ground-truth motion vs stationary under OLD objective | Failure |
|---|---|---|
| Blade, 2 frames, 34,369 reference points | `ΔE_obs ≈ -84`, but `ΔE_total* ≈ +1038` | 3,218 points are separately charged for one rigid velocity (`≈ +1122`) |
| Blade, 10 frames | `ΔE_total* ≈ -4641`, but useful SVGD motion rejected | particle-density Stein vector differs from optimized likelihood gradient; unnecessary frame-0 movement |
| Textured translation, 10 frames | GT better by `≈ -1682`, interpolation barrier `≈ +3.6e7` | global feature mismatch / covariance / optimization barrier |
| Hinge rotation, 10 frames | correct rotation *worse* by `≈ +2.96e8` | all 242 reference plate samples counted despite only ~22 appearing in final observed set |

These are findings of the *old experimental model*, not properties to preserve in the new model. Preserve those records and add new, explicitly versioned comparisons. There was good arithmetic parity in the old code; that DOES NOT establish physical validity.

## 1. Hard constraints — test and enforce

1. **Only point sets during inference.** Input frames are `Q_t={(y_j in R³, d_j in R³, Σ^XYZ_j, R^RGB_j)}` in world coordinates, plus times. Reference `P_0={(x_i,c_i,Σ^XYZ_0i,R^RGB_0i)}`. The existing data loader may use cameras to *materialize* calibrated 3D points and uncertainties; after this boundary the new estimator **must not consume camera ID, rays, pixel indices, depth maps, intrinsics, projections, masks or image pixels**. Masks are applied strictly by the existing loader *before* 3D materialization. Absolutely no 2D reprojection residual.
2. **Points only.** No normals, surfels, meshes, silhouette rendering, learned descriptors, per-camera visibility, or GT correspondences. All observations remain full masked 3D point sets. Never restrict inference to a display cap or an arbitrary `30k` subset; a display subsampler is visualization-only.
3. `K=1` is given. Only two physical labels: `z_i=0` static; `z_i=1` one moving body. One common `SE(3)` transform per time, `T_0=I`. **No free per-point 6D pose/velocity states.** Reference geometry/appearance uncertainty is allowed as one *shared* latent anchor per point, not a per-frame trajectory.
4. Same mathematics on synthetic and real data. Calibrate noise, representation uncertainty and temporal assumptions for each dataset; no separate synthetic/real branch, zero-noise shortcut, unreported covariance floor/inflation or tuning to GT labels/poses.
5. No hard-coded motion thresholds, positive/negative pseudo-label gates, rigid-distance cutoffs, static/dynamic heuristics, invented posterior confidences or arbitrary point rejection. A numerical truncation tolerance may be used **only** with a verified numerical error bound. Every modeling assumption, prior, approximation, seed and runtime cap is declared, logged and ablated.
6. **Original code remains untouched.** Add an opt-in new `svgd_joint_k1` pipeline with a separate runner/output directory/config/tests. `train.py` defaults, existing `svgd_support` and old falsification outputs are immutable. Do not delete the 16 older Markdown specs or reset the user's dirty checkout.
7. GT is used only in an **offline, separate-process evaluator** after the inference artifacts have been written and hashes committed. Strict test that an estimator import graph cannot reach `gt_joint`, `gt_transform`, labels or `metadata/frame_values.csv`.
8. Return *uncertainty and unresolved points* when the evidence cannot distinguish static from moving. In particular, a point on a uniform translated surface or close to a hinge need not become a confident seed.

## 2. Mathematical state: just ONE moving body

For each reference material point `i` and each time `t`:

\[
\widehat x_{it}(z_i,T_t;r_i)=\begin{cases}
 r_i,&z_i=0,\\
 R_t r_i+t_t,&z_i=1.
\end{cases}
\]

`r_i` is the *single latent true reference position*, and `a_i` the *single latent reference appearance*. Both are shared across all times. Their frame-0 conditional density `q_0(u_i)` with `u_i=[r_i,a_i] in R^6` is built once from measured frame-0 XYZ–RGB and its calibrated covariance. There is NO free `x_it` for `t>0`.

Let the global continuous state for one SVGD hypothesis be

\[
\theta=(\nu_1,\ldots,\nu_T,\;h),\quad \nu_t\in\mathbb R^6,\quad h=\mathrm{logit}(\pi),\quad \pi\in(0,1).
\]

`π` is the (unknown) prior fraction of reference points belonging to the moving support. Use a declared Beta prior `π ~ Beta(a_π,b_π)`; do not set `π` to the GT fraction. There is only one candidate moving body, but the posterior must be allowed to say **not enough evidence for any moving seeds**. This is distinct from fitting a second independent moving body.

For actual acquisition intervals `Δt_t>0`, define one *body-twist velocity* `ν_t` per frame and compose the rigid poses:

\[
T_0=I,\qquad T_t=T_{t-1}\operatorname{Exp}_{SE(3)}(\Delta t_t\nu_t).
\]

The common motion prior is paid **once per moving body**, not once per point:

\[
\nu_1\sim \mathcal N(0,P_{\nu,0}),\qquad
\nu_t\mid\nu_{t-1}\sim\mathcal N(\nu_{t-1},\Delta t_t Q_\nu),\quad t\ge 2.
\]

`P_ν0` and `Q_ν` are declared SPD `6x6` covariances (translation m/s and rotation rad/s); calibrate or honestly mark assumptions. The process prior models a random walk in **body twist**, not an exact general physical acceleration law. Use analytically correct `SE(3)` exp/Jacobians and compare to an independent Lie-group oracle. Do not reuse the old per-point `TemporalPrior` as the motion prior. The prior remains active with `K=1` even if the moving fraction is small.

## 3. Entirely 3D, uncertainty-aware observation score

### 3.1 Frame-specific normalized KDE from *observed* XYZ–RGB points

Use/reuse the exact chunked Gaussian reductions in `estimator/svgd_target_score.py` as a **density evaluator**, not as the gradient that is directly applied to physical point positions:

\[
p_t(x,c) = \sum_{j=1}^{M_t}w_{jt}
  \mathcal N_3(x;y_{jt},\Lambda^X_{jt})
  \mathcal N_3(c;d_{jt},\Lambda^C_{jt}),
\quad\sum_j w_{jt}=1.
\]

`Λ^X_j=Σ^XYZ_obs,j + Σ^XYZ_repr,j`; `Λ^C_j=R^RGB_obs,j + Σ^RGB_repr,j`. **Never add frame-0 reference covariance to these kernels:** it enters exactly once via `q_0(u_i)` below. Sampling/representation bandwidth and sensor covariance are separate terms; calibrate using **point-cloud spacing/statistics**, not arbitrary global object-size bandwidth or GT trajectories. Each kernel is properly normalized including determinants. Camera order permutations must leave results invariant. RGB units remain `[0,1]` encoded values until a separately declared photometric model is adopted.

Observed-point mixture weights `w_j` must be explicit. Start with normalized uniform weights as an audited baseline; evaluate an inverse-density reweighting (point-cloud-only) to prevent repeated/overlapping sampling from dominating. Never silently change `w_j` between ablations. Numerically compare GPU KDE log-density and XYZ/RGB gradient to an independent dense float64 oracle.

### 3.2 Missing observations: **neutral evidence, not a fake static assignment**

A reference point may not have any comparable observed point in a later cloud. Without images we **cannot know** whether this is occlusion, undersampling, or missing support. We can nevertheless define a point-only *unexplained-observation* factor.

Define the *frozen* background density using **only the reference point cloud**: `b_t ≡ b_0`, where

\[
b_0(x,c)=\mathcal N_3(x;\bar x_0,\operatorname{Cov}_{P_0}(x)+\overline\Sigma_0^X)
\,\mathcal N_3(c;\bar c_0,\operatorname{Cov}_{P_0}(c)+\overline R_0^C).
\]

Its covariances are positive definite because the measured sensor covariances are positive definite; if input covariance is invalid, fail rather than adding an arbitrary floor. Its mean and covariance are calculated once from `P_0` (with the declared reference sampling weights) and then never modified by the estimated motion. Define `ρ_t in (0,1)` by a **point-only, held-out within-frame cross-validation fit**: partition `Q_t` into disjoint reproducible spatial blocks, fit KDE on training blocks, maximize the held-out mixture likelihood `(1-ρ)p_train(q)+ρ b_0(q)` under a declared Beta prior, repeat folds, and report dispersion. This estimates *sampling/model mismatch*, **not a magically observed physical visibility label**. If the result is unidentifiable, report sensitivity and mark support confidence uncalibrated. Never fit `ρ_t` against GT positions, and never set it to a free optimizer variable that can rise to 1 and erase all motion evidence.

Define a dimensionless positive potential:

\[
\ell_t(x,c)=\rho_t+(1-\rho_t)\frac{p_t(x,c)}{b_t(x,c)}.
\]

Consequences:

- If no observed density supports either static or transported location, both get approximately `log ρ_t`: **no fabricated motion evidence**.
- If the transported location fits the observed 3D+RGB cloud better than the static location, it gets a larger score.
- `ρ_t` is a missingness/contamination model parameter. Estimate it by **dataset-specific unsupervised held-out point-cloud likelihood/coverage** or an explicitly documented model prior; never tune it using GT motion/labels and never let it absorb every mismatch silently. Log the fraction of point terms dominated by `ρ_t` and perform a sensitivity analysis.

This is a **conditional evidence potential** and the full model in §4 is a normalized *conditional Gibbs posterior over unknowns*, not an automatically normalized generative likelihood for the observed point set. **Do not misrepresent `ℓ` as a calibrated observation density or its ratio as a physical Bayes factor.** A complete physical generative point-process model would require separately accounting for scene-dependent sampling; avoid that unsupported claim. Calibrate posterior scores empirically before calling them calibrated probabilities.

The candidate `b_t` must be positive over the entire optimization domain (not a bounding box that silently rejects moved points). Calibrate and freeze its parameters without GT; expose distribution-shift sensitivity. `ρ_t` may be a held-out-calibrated fixed value in the initial implementation; do not optimize an unconstrained `ρ_t→1` to trivially suppress all evidence.

### 3.3 Shared reference uncertainty integrated **once**

For each reference point `i`, define `u_i=(r_i,a_i)` and its one frame-0 density `q_0i(u_i)` from measured XYZ–RGB values/covariance. Define static/moving *evidence for a common latent anchor*:

\[
L_i^S = \int q_{0i}(u)\,
  \exp\Big[ a_i^{\rm eff}\sum_{t=1}^T \log\ell_t(r,a)\Big]du,
\]
\[
L_i^M(T) = \int q_{0i}(u)\,
  \exp\Big[a_i^{\rm eff}\sum_{t=1}^T \log\ell_t(R_t r+t_t,a)\Big]du.
\]

The scalar `a_i^eff>0` is a **point-set sampling/evidence weight**, not a GT label or manually tuned moving-point boost. Freeze this explicit, reproducible default construction from reference-point correlations:

\[
K^0_{ij}=\exp\left(-\tfrac12\|x_i-x_j\|^2_{H_X^{-1}}-\tfrac12\|c_i-c_j\|^2_{H_C^{-1}}\right),\quad
\rho_i^0=\sum_jK^0_{ij},\quad
n_{\rm eff}=\frac{N^2}{\sum_{i,j}K^0_{ij}},
\]
\[
a_i^{\rm eff}=n_{\rm eff}\frac{1/\rho_i^0}{\sum_k 1/\rho_k^0}.
\]

`H_X,H_C` are **sampling-correlation scales**, separate from target KDE bandwidths and the motion-hypothesis SVGD kernel; derive them from dataset-measured XYZ/RGB representation/noise and local spacing, with full provenance, not GT motion. The effective-count formula is a **modeling choice** for a conditional composite likelihood (not an exact count of independent physical observations). It is invariant to uniformly duplicating every reference point: `n_eff` stays fixed and each duplicate receives `1/m` of its original weight; verify this algebraically and numerically. Implement tiled `K^0` reductions; never create a dense `N×N` matrix at production scale. Provide the `a_i^eff=1` and nearby correlation-scale alternatives as documented sensitivity controls, not silent switches. Include `a_i^eff` **inside** each shared-anchor evidence integral.

**Key property:** the *same* `r_i,a_i` appears in **all** factors inside each integral. Computing a different per-frame source convolution and multiplying afterwards is **not equivalent**; it counts reference uncertainty repeatedly. Test this with a small Gaussian model where the integral is exactly computable.

For full clouds implement a differentiable **6D per-point Laplace approximation** or adaptive common-random-number/quasi-Monte-Carlo integration; this is the one significant evidence approximation and MUST be documented:

- Independently test against high-precision quadrature/QMC for small synthetic 3D+RGB examples and a representative, *evaluation-only* stratified set of larger-scene reference points.
- Differentiate through the integral approximation (including its determinant/normalizer) with respect to each shared `T_t`; compare gradients by central finite differences.
- If the local Gaussian/Laplace Hessian is indefinite, multimodal or inaccurate, report it and use a validated higher-accuracy integration route. Do not add an arbitrary covariance jitter or return a plausible-looking probability.
- Tune numerical integration accuracy by a **declared error tolerance**, not a physical motion gate. Process **all** points in inference; sampling is permitted only for diagnostics and timing.

A performance-oriented `reference_point_estimate` ablation may evaluate at `u_i=mean(q_0i)` but is **NOT** the main uncertainty-aware model and cannot be presented as a fully marginalized posterior. Run it solely to isolate integration cost/failures.

### 3.4 What individual support evidence actually means

For a candidate common transform sequence `T`, define

\[
B_i(T)=\log L_i^M(T)-\log L_i^S.
\]

- `B_i>0` means this **one common moving model** explains point `i` better than the static hypothesis, under the declared conditional model.
- `B_i≈0` means the point is ambiguous (uniform surfaces, near hinges, missing observations). Do not force a class.
- `B_i<0` means the static hypothesis is better.

The old independent per-point displacement field, old independent velocity penalties, old global feature-mean matching, and old post-hoc rigidity graph **do not enter this score**.

## 4. Joint static/dynamic + shared-motion posterior — the precise target for SVGD

Let the physical labels be `z_i ∈ {0,1}` with `z_i|π ~ Bernoulli(π)`. Conditional on the time-sequence data, define the **new** Gibbs model using the evidences above:

\[
\boxed{
 p(T,\pi,z\mid Q_{0:T}) \propto
 p(T)\,p(\pi)\,
 \prod_i \big[(1-\pi)L_i^S\big]^{1-z_i}
             \big[\pi L_i^M(T)\big]^{z_i}
}
\]

The normalizer of this conditional model over all `z,T,π` exists for a proper motion and Beta prior and bounded positive potentials. This makes the conditional model mathematically explicit; it is **not** evidence that its probabilities are empirically calibrated.

**Exactly sum each binary label; do not optimize a continuous fake label.** Conditional on `T,π`, the labels factorize, giving

\[
\boxed{
F(T,\pi)=\log p(T)+\log p(\pi)
+\sum_{i=1}^N\log\Big[(1-\pi)L_i^S+\pi L_i^M(T)\Big]
}
\]

and

\[
\boxed{
\gamma_i(T,\pi)
=P(z_i=1\mid T,\pi,Q)
=\operatorname{sigmoid}\Big(\operatorname{logit}\pi+B_i(T)\Big).
}
\]

**This is the desired joint inference:** each candidate pose uses the evidence of *every* point; labels change as the shared pose changes; and those label probabilities directly determine which points drive the pose gradient.

The derivative (when `π` is treated as a separate coordinate) is

\[
\boxed{
\nabla_T F
=\nabla_T\log p(T)
 +\sum_i\gamma_i(T,\pi)\nabla_T\log L_i^M(T).
}
\]

For `h=logit π` and the Beta prior `Beta(aπ,bπ)`, include the change-of-variables Jacobian, so the density in `h` has prior log term

\[
\log p(h) = a_\pi\log\pi+b_\pi\log(1-\pi)-\log B(a_\pi,b_\pi)
\]

(up to the chosen base coordinate measure). With this convention,

\[
\frac{\partial F}{\partial h}
=\sum_i\big[\gamma_i(T,\pi)-\pi\big]
 + a_\pi(1-\pi)-b_\pi\pi.
\]

Write exact unit tests for all these identities, in float64, using dense enumeration of `2^N` label states for `N<=5`. At `T_t=I` for all `t`, `L_i^M=L_i^S` **exactly** and `γ_i=π`: no artificial moving evidence. At zero usable evidence, conditional probabilities return the prior.

**Do not conflate conditional `γ_i(T,π)` with final marginalized `γ_i`.** The final quantity is

\[
\gamma_i=P(z_i=1\mid Q)=\mathbb E_{p(T,\pi\mid Q)}[\gamma_i(T,\pi)].
\]

SVGD's ensemble approximates this expectation. Its simple particle average is an **approximation**, not an automatically calibrated posterior. Validate on tractable exact posterior toys and synthetic held-out reliability tests before labeling the artifact `APPROXIMATE_MODEL_POSTERIOR`.

## 5. SVGD: hypotheses about shared rigid motion, not physical points

Create `M` particles `θ^(m)=[ν_1:T^(m),h^(m)]`. The transforms for each particle are computed by the `SE(3)` composition in §2. **All per-point labels have already been marginalized analytically in `F`.**

Work in a *declared whitened coordinate system* `v=W(θ-μ)` based on the shared-motion prior and a stated scale for the `h` coordinate; use chain rule, proper covariance factors and units. The kernel operates on the `M` hypotheses, not on `N` physical points:

\[
 k(v_n,v_m)=\exp\left[-\frac12(v_n-v_m)^\top H^{-1}(v_n-v_m)\right].
\]

\[
\boxed{
\phi(v_m)=\frac1M\sum_{n=1}^M
\left[k(v_n,v_m)\nabla_{v_n}F(v_n)
 +\nabla_{v_n}k(v_n,v_m)\right].
}
\]

**Both terms use the *same* target posterior `F`.** There must be NO separate physical-point KDE Stein score `∇_x log p_t` applied as the motion update. This prevents the proven old score-versus-objective inconsistency. The repulsion is only across **candidate motion hypotheses**, avoiding a 34k-by-34k physical-particle SVGD interaction.

Implement a deterministic seed-controlled SVGD particle optimizer with a declared learning-rate schedule/preconditioner, and an opt-in direct MAP optimization **of exactly the same `F`** as a control. SVGD does not require every particle's `F` to decrease: the repulsive term intentionally spreads particles. Do not reject repulsive updates via the old per-particle objective backtracking rule. Track the mean and spread of `F`, score norm, average kernel mass, particle diversity and variation in `γ`.

- Initialize particles with multiple motion hypotheses from the **declared shared-motion prior**; additionally allow point-cloud-only registration-based pose proposals strictly as an **optimizer initialization ablation**, never as evidence or a source of GT membership.
- Avoid a fixed all-identity initialization: at identity the static/moving distinction is unidentifiable.
- `M` and kernel scale are configurable numerical/algorithmic approximations (initial pilots may use `M=16/32`), not facts about the scene. Test changes in `M`, random seed, and kernel bandwidth; record stability. Never call `M` "number of moving parts" — `K=1` always.
- For larger displacements use the full sequence in the posterior and physically valid body-twist motion hypotheses; do not demand that a small per-frame KDE force track 127 mm from the frame-0 position.
- Treat `SE(3)` composition and differentiation rigorously. Document that SVGD is on the selected **body-twist coordinate density** and include change-of-variable factors if using a different pose measure; do not claim an intrinsic Haar-measure Stein flow without deriving it.

## 6. Reference points' mutual reinforcement and unresolved cases

The joint log posterior is a **sum of contributions from all reference points to one shared `T`**. Informative end/boundary/texture points may move the shared transform; other points then get their own static-versus-moving odds evaluated against that transform.

This is *not* a post-hoc fit: `γ_i(T,π)` enters the pose gradient during every SVGD update. A point with no static/moving discrimination contributes little to the transform (its evidence ratio is near one); it is not automatically promoted just because the common transform passes through it.

Final Phase-1 outputs:

- `gamma_moving.npy` `[N]`: marginalized `P(z_i=1|Q)` only after approximation/calibration tests succeed; otherwise output `gamma_provisional.npy` with `probability_semantics=UNCALIBRATED_APPROXIMATION` and explicit numerical errors.
- `evidence_static.npy`, `evidence_moving_conditional.npy` (posterior ensemble summaries), `log_evidence_ratio.npy`, `identifiability_status.npy`; don't assign a hard label when the ratio is unresolved.
- `pose_particles.npy` `[M,T+1,4,4]`, prior/pseudo-posterior score, relative particle spread and pose-uncertainty report; `T_0` exactly identity for every particle.
- `moving_seed_indices.npy` selected by a documented **posterior expected false-discovery** risk policy, e.g. sorted-high `γ` and `sum_(selected)(1-γ_i)/|selected| <= α` with configurable, explicit `α`. This is a **model-based** risk statement, not a guaranteed frequentist precision without empirical calibration. Do not insert a hidden magnitude/displacement/GT gate.
- Full XYZ–RGB reference catalogue and source IDs remain intact and unchanged. Save traceable support-versus-pose evidence, uncertainty provenance and profiling.

Important distinction: a point that moves together with the body but cannot reject the static explanation is **not** automatically a confident seed; it remains unresolved for the later support-completion stage.

## 7. Concrete implementation layout (follow the codebase, not a rewrite)

Inspect these currently existing files before changes:

- `scene/pointcloud_sequence.py`: produces full reference catalogue and 3D masked cloud materializations. Add a `PointCloudOnlySequence` adapter that **exposes ONLY XYZ, RGB, 3D/RGB covariances, reference IDs, times**. Drop camera/pixel/depth metadata at this interface; reject accidental use in estimator. Existing preprocessor can remain camera-aware upstream.
- `estimator/svgd_target_score.py`: reuse the exact online-logsumexp/streamed Gaussian KDE, its gradients and CPU/CUDA parity tests. Add a clean query API for arbitrary transformed 3D points and latent appearance, plus Hessian/JVP only as needed for integrated evidence.
- `estimator/svgd_support_run.py`, `estimator/svgd_temporal.py`, `estimator/svgd_particle_operator.py`: READ to understand what failed; **do not modify** or use their physical-point transport/independent temporal prior in the new inference.
- `estimator/svgd_support_posterior.py`: reuse exact tiny-model enumeration/oracle patterns, not its currently blocked production path or uncalibrated outputs.

Create these separate modules, or equivalent narrowly named files with identical responsibilities:

```
estimator/svgd_k1_observation.py       # 3D KDE, neutral-missing potential, covariance provenance
estimator/svgd_k1_reference.py         # one shared reference XYZ/RGB uncertainty integral
estimator/svgd_k1_motion.py            # SE(3) body-twist chain and one shared motion prior
estimator/svgd_k1_joint_posterior.py    # exact binary-label marginalization and conditional odds
estimator/svgd_k1_stein.py             # SVGD over shared-motion hypotheses only
estimator/svgd_k1_run.py               # new opt-in staged runner
scene/svgd_k1_cloud_adapter.py         # prevents camera/image metadata crossing solver boundary
tools/run_svgd_joint_k1.py             # CLI; no legacy-default changes
tools/audit_svgd_joint_k1_oracles.py    # offline GT-only, independent verification
configs/svgd_joint_k1.json            # full theory/algorithm/calibration provenance
configs/svgd_joint_k1_experimental.json
reports/svgd_joint_k1/                # new reports; do not overwrite earlier audits
```

Create `tests/test_svgd_k1_{observation,reference,motion,posterior,stein,runner,calibration}.py` and new viewer `debug/svgd_joint_k1_debug.py` reusing existing plot/scatter infrastructure. The old `tools/run_svgd_support.py` CLI must still behave identically.

**Very important:** no global RFF feature objective in this new baseline. The observed-cloud KDE is a **query likelihood potential**; SVGD is over `T,π` with score of the exact corresponding joint Gibbs model. If future RFF approximations are introduced, they must approximate this *same* score with measured error and must not silently replace the model.

### Required CLI (adapt parsing to existing conventions)

```
python -B tools/run_svgd_joint_k1.py \
  --scene /path/to/blade_103706 \
  --config configs/svgd_joint_k1_experimental.json \
  --num-moving-parts 1 \
  --device cuda \
  --frame-limit 10 \
  --output /path/to/new_svgd_joint_k1_blade10 \
  --debug full
```

`--num-moving-parts` must reject values other than `1` in this version with `NOT_IMPLEMENTED_K_GT_1` rather than silently creating more bodies. `--frame-limit` labels the run `PREFIX_DIAGNOSTIC`, not whole-sequence inference. Dataset/output paths must be user arguments; do not hardcode or overwrite existing results.

## 8. Computation: full clouds, bounded GPU memory, auditable approximations

The large loops are **cloud density queries**, not interactions among physical SVGD particles. The SVGD kernel is only `M x M` in shared-motion parameter space.

1. Precompute each reference point's static evidence `log L_i^S` once per dataset/horizon/calibration (including the one shared uncertainty integral). Cache with content hashes of all inputs, config and source commit.
2. For each motion particle, evaluate the moving evidence `log L_i^M(T)` for **all** `N` reference points over all `T` frames using streamed GPU query/candidate tiles, and accumulate the exact Rao–Blackwell score and gradient. Do not allocate `[M,N,T,observed_count]` arrays or a dense 34k×34k kernel.
3. Use `float64` for log probabilities, covariance solves, `SE(3)` Jacobians and score accumulation, or prove a mixed-precision error bound against float64. No silent Cholesky jitter. Use stable log-sum-exp throughout.
4. The existing exact tiled KDE is the mandatory correctness baseline. If it is too slow, add a documented spatial acceleration with a **bounded omitted-density and gradient error** (or an exact GPU acceleration), validated against the full dense/tiled oracle on varied densities, colours and surface overlaps. A fixed nearest-neighbour count or fixed radius with no error certificate is not a valid inference replacement.
5. For larger `N`/longer sequences, stream hypotheses/frames in tiles and recompute rather than silently sub-sample points. Preserve global support-coupled gradients exactly in accumulation. Measure on 34k and ~100k, and use the available 8 GB RTX 4070 and, when available, 32 GB RTX 5090; do not promise a runtime without measuring it.
6. If the shared-reference integral is the bottleneck, profile and ablate its *numerical approximations* while keeping the mathematical model fixed. Do not eliminate measurement uncertainty for speed.
7. Report per-operator CUDA peak allocated/reserved memory and time, and total wall clock. The old profiled 100k KDE score could take ~24 s **per frame/evaluation** on an RTX 4070; estimate the cost of `M×T×iterations` upfront and benchmark on a pilot without dropping full-cloud support.

## 9. Strict correctness gates and physical falsification tests

### Gate 0 — state/data boundaries

- `T_0=I` exactly. No per-point trajectory variables in the new optimizer; no `camera`, `pixel`, `depth_map`, `rays`, `normals` in solver module inputs or imported evaluation paths.
- Full input point count preserved across all frames, only mask-preprocessed/valid-depth points enter. Display caps cannot change inference results (hash test).
- GT-leak AST/import-graph sentinel. The default/old pipeline emits byte-identical output given the same old config.

### Gate 1 — mathematical/equation oracles

- `SE(3)` exp/composition and Jacobian finite differences (translation and hinge 0.9 rad; tiny and nonzero twists). Motion prior score from autograd agrees with independently coded dense Gaussian oracle.
- KDE normalized density, gradients and covariance derivatives agree with independent NumPy dense reductions and finite differences. Reference covariance appears ONCE; compare joint-anchor Gaussian toy to exact multiframe integration and deliberately wrong independent-per-frame marginalization.
- For `N<=5`, enumerate all `2^N` static/moving labels: verify `logsumexp` marginalized `F`, `γ_i(T,π)` and every pose/`h` gradient to near float64 tolerance (derive tolerance from oracle precision; aim `<1e-8` relative when well conditioned).
- Verify positivity/normalization of all declared densities, priors, and the `h↔π` Jacobian. Assert the per-body motion prior is evaluated exactly once regardless of moving support size.
- Verify the SVGD attraction uses `∇F` **exactly**, its repulsion sign and derivative, and the same `F` appears in diagnostics and MAP control. No independent physical-point Stein field in pose updates.
- `T=I` implies `B_i=0` and `γ_i=π`; unobserved/fully ambiguous point -> prior, not fabricated motion confidence.

### Gate 2 — independent-point and surface-density invariance

- Replicate a reference point cloud's sampling density in a controlled test **without adding independent physical information**; compare posterior evidence, pose and seeds with appropriate effective-sample weights. An unbounded increase in confidence with duplicate samples is a failure.
- Permute all reference points and all observed samples; results must be invariant up to numerical reduction ordering. Test 2 overlapping/duplicate 3D samples carrying identical and conflicting RGB, with correctly propagated uncertainty.
- Vary synthetic vs real noise *parameter values* under exactly the same code path. Compare whitening and coverage; no zeros or magic floors.

### Gate 3 — physical toy trials (do not use GT inside estimator)

1. **Static-only** scene: do not hallucinate a confident moving subset. With `K=1` permitted, report no evidence and approximate prior uncertainty.
2. **Prismatic moving plate with static base**, 10 frames, textured and uniform-colour variants. Verify high-confidence seeds are genuinely moving (post-hoc), their spatial distribution supports a rigid pose, and the inferred rigid transforms agree within appropriately calibrated uncertainty.
3. **Hinge rotation ~0.9 rad**, with decreasing observed point counts (the previous stress case was 242→22). Missing data should become low information, *not* a huge penalty for moving to correct positions. No 2D/camera visibility model may be introduced.
4. **Two-frame Blade** (`blade_103706`, 34,369 points, 8-camera clouds materialized first), then **10-frame Blade**, all input points. Report stationary vs GT **only offline**, the optimizer MAP and SVGD candidates, support overlap/precision and pose errors.
5. **Ill-conditioned interior/near-hinge points**: remain unresolved unless their own data reject static and the shared motion agrees. Do not promote every member of a GT part just because it follows the fitted `T`.
6. **One genuinely static object with real-data contour noise**: no false high-confidence moving seeds. Same model and dataset-calibrated `Λ`, `ρ`, and reference covariance.

For every trial, record: `F(T=I)`, best MAP and best SVGD `F`, posterior pose spread, `γ` histogram, posterior seed expected error, GT-only actual precision/recall, rigid pose error, fraction of points whose static-vs-moving evidence is intrinsically ambiguous, and numerical integration error. Do not claim that a successful GT energy comparison implies an optimizer can recover GT: test both separately.

### Gate 4 — exact versus SVGD posterior approximation

- On a tiny problem with one or two frame motions, independently integrate a *low-dimensional* pose posterior numerically (quadrature/grid or calibrated MCMC) and compare conditional label marginals, marginalized `γ_i`, credible pose intervals and evidence ratios.
- Compare at least two SVGD particle counts, two seeds and at least one kernel-scale perturbation. Do not label `γ` calibrated because `M` particles merely exist.
- Perform held-out synthetic coverage/reliability tests before changing `UNCALIBRATED_APPROXIMATION` to `APPROXIMATE_MODEL_POSTERIOR`. Keep all uncertain points marked; no fake confidence threshold.

### Gate 5 — whole sequences, performance, regression

- Full 60-frame sequence with K=1 and all full masked XYZ–RGB points whenever available; no fake `DONE` if only a prefix or diagnostic completed.
- CUDA CPU parity for all new operators; report resource usage on RTX 4070 8GB and/or RTX 5090 32GB.
- Old 71 SVGD operator tests (or current equivalent), all new tests, and legacy source-matched known failures preserved as a documented baseline. Do not modify unrelated test expectations to obtain green badges.
- Render a viewer showing at least the original cloud, static/moving confidence, inferred rigidly transformed moving subset, observed cloud, and uncertainty trace. Viewer may subsample **only for display**, never for inference.

## 10. Reporting and STOP conditions — no silent fallbacks

At each step create `reports/svgd_joint_k1/implementation_and_validation.md`, `equation_to_code_contract.json`, `calibration_provenance.json`, `theory_deviation_register.json`, `unit_test_results.json`, `optimizer_diagnostics.json`, `posterior_calibration.json`, `performance.json`, `run_manifest.json` and a `root_cause.md` for every failed gate. Record git SHA, dirty tree, config hashes, dataset fingerprints, seed and numerical tolerance.

Use statuses:

- `PASS_MATHEMATICAL_ORACLE`: equations and tested operators match.
- `PASS_PHYSICAL_GATE`: specified GT-free inference solved the physical toy, independently verified afterward.
- `UNRESOLVED_IDENTIFIABILITY`: data do not distinguish static and moving; no confidence fabricated.
- `MODEL_MISMATCH`: offline GT motion has worse joint score than the static solution (interpret only within the specified conditional model).
- `OPTIMIZATION_FAILURE`: the model prefers a known correct solution but the optimizer cannot reach comparable evidence.
- `CALIBRATION_FAILURE`: apparent probabilities or credible intervals fail held-out coverage.
- `NUMERICAL_FAILURE`: integral/covariance/gradient fails an oracle; no fake numerical floor.
- `NOT_IMPLEMENTED`: known stage not built; don't publish output files implying it succeeded.

**Do not tune hyperparameters using GT evaluation results to turn a failed test into a pass.** GT is for diagnosis and independent evaluation. If a gate fails, first localize the exact equation or numerical routine, perform an ablation under controlled model assumptions, report failure, and keep other code unchanged.

## 11. Suggested implementation order (Codex must actually implement, test and report)

1. **Snapshot the current checkout:** note `b62c7cee` if still the current SHA, current dirty files, and all previous falsification reports; keep original code immutable. The GitHub main branch previously showed `b62c7cee`, but the local uncommitted changes may be newer; inspect rather than assuming.
2. **Independent synthetic oracle and mathematical definition:** freeze the exact conditional Gibbs posterior above; derive its `γ` and `∇F` analytically. Write dense `N<=5` tests and separate `SE(3)` tests before any CUDA code. Establish which measurement/representation/missingness parameters are measured and which are assumed.
3. **Point-only adapter and exact KDE potential:** implement the no-camera interface, `p_t/b_t`, uncertainty covariance handling, static density cache; prove CPU vs CUDA and toy normalization.
4. **Shared-anchor evidence:** implement and validate Laplace/adaptive integration for one common `u_i` across time, with independent reference-integral oracle. Ensure every point is processed during inference and gradients close numerically.
5. **Rao–Blackwellized posterior:** compute `log L^S`, `log L^M`, exact `γ_i(T,π)`, `F`, gradients, uncertainty and no-evidence outcomes. Compare with explicit tiny-label enumeration.
6. **SVGD pose-hypothesis optimizer:** `M` candidate `SE(3)` twist sequences, matching posterior score, repulsion only across candidate hypotheses, compare with MAP of same objective. Measure candidate diversity and posterior accuracy.
7. **Synthetic K=1 physical tests:** static, translation, hinge, partial-sampling/visibility changes, noise. Assess high-confidence seed recovery and transform identifiability before Blade.
8. **Blade 2/10 frames:** full 34,369 points; use all observed points from the original eight-camera materialization, but no camera inside estimation. Diagnose model preference vs optimization separately. Verify pose/seed results offline.
9. **Full sequence and runtime:** only after the prior gates succeed; profile and optimize numerical kernels without changing `F` or dropping points.
10. **Deliver a final report:** what works, what remains broken, equations-to-code mapping, all deviations and tests, physical metrics, and the exact command to reproduce every case.

## 12. Important intellectual honesty / prohibitions

- Standard SVGD is an inference method, **not a generic high-frequency noise-removal theorem**. Noise reduction here comes from uncertainty-aware KDE, a shared 6D rigid motion and a proper temporal prior. Demonstrate it by ablation against the *same* posterior optimized by MAP, not by assertion.
- The model here is a **proper conditional Gibbs distribution over static labels, common rigid motion and π**, built from point-cloud density potentials. It is **not** the same as a fully normalized, physics-derived generative likelihood of all observed point samples. If empirical calibration is unsuccessful, output honest uncalibrated scores, not Bayes-factor claims.
- The label `K=1` means one moving support's motion is shared. There may be no distinguishable seeds in ambiguous data. It does not reveal GT membership, GT motion or the joint.
- Do not use 2D reprojection, camera observability arguments, normals, surfels, mesh, guessed point identities or individual unconstrained trajectories to "fix" failing results.
- Do not introduce heuristic label thresholds, arbitrary depth-noise inflation, fixed sample caps, GT-dependent bandwidth selection, or unapproved model changes.
- If the proposed conditional model fails to distinguish the correct rigid pose from static on a controlled toy, **report `MODEL_MISMATCH`** instead of falling back to independent point motion or silently adding a new likelihood.

**Acceptance statement:** The work is *not complete* when the CUDA tests turn green or some SVGD step is accepted. It is complete when the full 3D-only joint `K=1` posterior is implemented as specified, its exact conditional-label math and pose gradients pass independent oracles, and the estimator finds a reliable, GT-free, spatially distributed moving support with one common rigid transform on translation and rotation experiments. Any partial result must be explicitly labeled.
