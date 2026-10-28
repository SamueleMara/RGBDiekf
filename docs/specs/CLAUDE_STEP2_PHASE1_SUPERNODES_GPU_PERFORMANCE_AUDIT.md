# Claude Code — STEP 2: Phase-1 computational-supernode parity, numerical audit, and evidence-based GPU acceleration

## Scope and purpose

The completed Step-1 offline motion-coherence audit found **exact parity** with pinned RigEuler3DGS coherence on identical inputs, and a change in selected seed from **6 representative nodes (v1 trajectories)** to **183 nodes (v2 trajectories)**, expanded to 192. **Do not touch that extractor now.** The next dependency is understanding the actual per-point filter corrections in Phase 1 and why the moving trajectories lag.

Investigate Phase-1 runtime **in parallel**, but do not mix algorithm repairs with performance changes. Conduct STEP 2A first (read-only correctness, profiling), STEP 2B second (one optimization at a time, in a separate opt-in path). Do not declare a faster run correct merely because it finishes. Keep production results, output folders, reference code, and previous runs intact. Use real RGB-D images from all cameras, do not read ground truth during inference, and do not use point correspondence measurements instead of rendered RGB-D innovations.

Reference: `SamueleMara/RigEuler3DGS` pinned commit `481b2cf2d5602932286948cf14114f9caee4de3c`, especially `estimator/supernodes.py`, `estimator/sparse_iekf.py` (`group_precond`, `measurement_operators`, `iekf_frame_step`), `estimator/renderer_bridge.py`, `estimator/process.py`, `estimator/covariance.py`. Examine *actual current local* RGBDiekf sources (`pointwise_rgbd/supernodes.py`, `image_iekf.py`, `image_phase1.py`, `rgbd_point_renderer.py`, `image_observation.py`, `phase1_covariance.py`, `restricted_posterior.py`, `process6.py`) and record SHA256 hashes. The new local files are not necessarily committed on GitHub: inspect the working tree and the exact saved v2 `source_snapshot`.

## Facts already established — avoid confusing two different supernode systems

- Blade v2 initial point catalogue: **34,369** points with individual state `s_i=(x_i,v_i) ∈ R^6`.
- Phase-1 correction partition: **537 computational supernodes**, approximately 64 points/group, **12 correction coordinates** per group: 3 translation, 3 angular displacement, 3 linear velocity increment, 3 angular velocity increment. Correction solve dimension **6,444**. These increments are *not* permanent rigid motion constraints.
- Coherence uses a **different canonical partition of 192 representative nodes** and its own graph. Its partition, graph, extraction and replay have already passed pinned-source parity and are out of scope.
- All eight available cameras contribute image-space RGB-D observations to the IEKF. No per-camera filter substitution, no nearest-neighbor 3D measurement association, no GT input.
- Current v2 moving-point median translation at original frame 50 was ~0.2725 m versus synthetic GT ~0.7078 m; static points remain nearly stationary. Many nonlinear frame updates reached iteration budget. These are *outcomes to investigate*, **not evidence that any one kernel is wrong**.
- The structured posterior run earlier showed ~59 seconds/frame inside covariance processing (with substeps ~28, 7.7, 7.7, 9.7 s); the later RigEuler block-information configuration reported **~6.3 minutes for 60 forward Phase-1 frames (~6.4 seconds/frame)**. These numbers come from *different covariance variants and hardware contention* and must not be compared as equal-condition benchmarks.
- Shared workstation is a laptop-class NVIDIA RTX 4070 **8 GB**. Detect memory actually available, the current GPU users, and contention. **Do not kill another user's process.** Avoid concurrent benchmark jobs, oversized caches, GPU OOM and unintended writes to the external SSD.

## STEP 2A — prove grouped correction correctness before optimizing anything

### A1. Independent canonical group basis

From saved v2 arrays (`first/phase1/partition.npz` with `xyz`, `membership`, `centroids`, `first/phase1/state.npy`, per-frame covariance and observation metadata), reconstruct the exact phase-1 group basis `G`, with one group at a time or an explicit sparse oracle on tiny problems. For canonical offset `r_i = x_i^0 - c_{g(i)}^0`:

```
δx_i = δt_g + δθ_g × r_i
δv_i = δu_g + δω_g × r_i
```

With `[δt,δθ,δu,δω]` as the group coordinate ordering, and `[δx,δv]` as the point ordering, verify **exact code ordering and signs** against the true reference and the current point port. (The reference Gaussian has `[δμ,δφ,δv,δω]`; adaptation to XYZ+velocity is not a verbatim state dimension copy.) Check canonical centroids, grouping reuse over all frames, absence of accidental use of *current* rather than canonical offsets, proper handling of empty/singleton/near-collinear groups, and full rank/condition of every actual group block.

Tests: `G(z)` against a small explicit dense `G` in float64; the Euclidean adjoint identity `<Gz,w> = <z,Gᵀw>` for random probes on CPU and CUDA; finite differences for the chosen retraction; shuffled point order invariant when group IDs and offsets are permuted consistently. Report max absolute and relative errors and group rank/condition distributions. Never silently normalize, change the basis or discard directions.

### A2. Exact reduced rendered-image measurement operator

At one saved blade frame and multiple synthetic geometries (disocclusion, depth ordering, same-pixel multiple points, multi-camera overlap), freeze a single true rendered-image linearization `H` and measurement weights `W`. Verify `H`/`Hᵀ` against finite differences/adjoint tests and the grouped operator:

```
A_g(z) = Gᵀ[ Λ_minus Gz + Hᵀ W H Gz ]
b_g    = Gᵀ[ Hᵀ W (y - h(s_j)) - Λ_minus (s_j - s_minus) ]
```

The prior is the **fixed once-per-frame predicted Kalman prior**. All camera terms accumulate into the same state-space operator. Run dense toy oracle `A_dense=Gᵀ(Λ+HᵀWH)G` and compare `A_g(z)` and `b_g`; verify symmetry `<u,A_g(v)>≈<A_g(u),v>`, eigenvalues and PCG residual against the explicit solve. Preserve co-visibility cross-point/cross-group coupling; per-point block Fisher is **preconditioner only**, not a substitute for the full normal operator. Investigate damped/undamped calculations separately.

### A3. Exact one-step IEKF convergence audit; **not just final-image error**

Choose original frames 1, 2, 5, 10 and 20, with identical saved priors and observations. Log for every nonlinear iteration: prior translation and velocity mean/covariance; rendered RGB, metric depth and valid coverage; innovation norms per camera; robust weights and their validity masks; matrix-free operator adjoint residual; PCG iterations/residual; computed correction dimensions and norms; trust-region radius, predicted decrease, **actual decrease under the same physical RGB-D + fixed-prior model**, `rho`, accept/reject; posterior mean and covariance; whether it stopped for convergence, exhaustion or numerical failure. Use true CUDA timing where applicable.

Compare (i) the same group basis with an accurately solved reduced linear step; (ii) current PCG; (iii) a tiny dense equivalent; (iv) reference trust-region code on matched operator inputs. **Do not retune max_iters, forcing tolerance, process noise, robust sigma, radius or measurement weights to make the blade pass.** Do not weaken a failing test. If iteration budget is limiting, show a counterfactual **offline ablation** with larger budget separately, and isolate whether it changes motion recovery without changing the estimator theory.

### A4. Covariance: characterize, do not swap silently

Report actual production method `rigeuler_block_information` and alternative restricted-Joseph/structured methods by name. Compare one-frame and multi-frame covariances on a small dense oracle using the true restricted gain. Diagnose conditioning, rank, symmetrization and numerical failure. A correct dense Joseph update must remain PSD given PSD inputs; a *numerically indefinite factorization* is not proof grouped Bayesian filtering is inherently impossible. Huge variance and negative eigenvalues must be treated separately. Don't make covariance changes in this stage unless a concrete algebraic bug is isolated and fixed with an independent regression test. This item can be documented as unresolved if a full repair exceeds scope.

## STEP 2A-P — performance profiling: explain every second and byte

**Do not optimize until baselines are recorded.** Use a warmed GPU and `torch.cuda.Event`+synchronization or `torch.profiler` with CUDA activity (never CPU `time.time()` alone for async kernels). Record per-frame per-camera `median/p95` wall time, CUDA kernel time and GPU peak allocated/reserved memory, and CPU-to-GPU transfers. Profile frame 1, 10 and 30 on fixed identical inputs; at least one frame with strong motion and overlapping footprints. Report for each:

1. Dataset read/decode/SSD I/O and any blocking transfer; CPU→GPU copies.
2. Canonical group construction (once), group expand `G`, reduce `Gᵀ`, group preconditioner construction.
3. Per-iteration point projection/culling, per-camera tile/pixel contribution binning, depth visibility sorting/front selection, RGB/depth/coverage composition.
4. Residual/robust RGB-D weights and their masks.
5. `H v`, `Hᵀ w` time **per invocation**, Fisher-preconditioner time; number of invocations per PCG step, per nonlinear iteration and per frame; Python dispatch/kernel-launch overhead.
6. PCG/Steihaug solve, preconditioning, trust-region trial rerender and acceptance/rejection.
7. Posterior covariance variant, PSD repair, CPU/GPU sparse operations and conversions, and serialization/debug/page generation.

Counters: `N_points`, groups, 8 cameras, image resolution, visible points, total point-pixel footprints, occlusion candidates/pixel, cache size/bytes, nonzero image Jacobian contributions, Newton iterations, PCG iterations, accepted/rejected steps, and GPU active memory. Include all framework flags and precise CUDA/PyTorch/NVIDIA versions. The 34k-point blade and **one larger scene** (96k–346k points, as available) should be profiled, without running a full trajectory or launching an OOM-prone batch.

Create `reports/phase1_grouped_operator_parity.json`, `reports/phase1_gpu_profile.csv`, `reports/phase1_gpu_profile.md`, and a standalone reproducible `tools/profile_phase1_rgbd.py` that does not modify production outputs.

## STEP 2B — speedups, in measured priority order, one independent feature branch/flag at a time

### B1. Zero-semantic-change optimizations first

- **Cache canonical group metadata once** (point-to-group ID, canonical offset, memory-contiguous index ranges and group Gram/preconditioner structure when valid). Do NOT rerun farthest-point/Lloyd clustering each frame or reconstruct unchanged CSR indices every PCG call.
- Keep point state, per-camera cache and group arrays on GPU for the entire frame/sequence where memory permits; async pinned input staging where useful; no hidden `.cpu()`, `.numpy()`, `.item()` or per-node Python `.tolist()` inside iteration or pixel loops; avoid per-iteration allocations and host synchronization.
- Cache the **current accepted/trial linearization** contributions exactly once for each nonlinear iterate, reusing them for `H`, `Hᵀ`, Fisher and PCG. Invalidate when state, visibility support or linearization parameters change; never reuse a stale Jacobian after a different pose is accepted.
- Express point→pixel contributions as compact tile- or pixel-grouped **CSR/COO only where profiling justifies it**. Sparse is good here because each point affects only its projected footprint, while pixels couple multiple points. JVP/VJP should traverse only actual nonzero contributions and fuse RGB + depth where numerically equivalent. A point-pixel contribution CSR should store index/weight/derivative data for one current linearization, not a dense `(pixels × 3N)` Jacobian or `(12G)^2` matrix.
- Evaluate `A_g(z)=GᵀΛGz+GᵀHᵀWHGz` via cached sparse traversal, with fused GPU reductions if beneficial, all 8 cameras. No explicit dense full `H`, no cross-camera visibility conflation, no omission of cross-group coupling. Preconditioner may use only diagonal group blocks; the actual operator may not.
- Avoid CPU/group Python loops in `group_precond()` if profiling identifies them as significant; batched GPU contraction/segment reduction is preferable. Use a reproducible float64 oracle before considering float32/fused mixed precision.
- Where CUDA kernels are worthwhile, reuse a validated CUDA extension architecture rather than porting the entire pipeline or changing the RGB-D observation physics. GPU memory on this device is only 8 GB; chunk by camera/tile and reuse a bounded scratch arena; never allocate quadratic point×pixel tensors. Use deterministic reductions or explicit numerical tolerances for differing GPU reduction order.

### B2. Controlled point-cloud reduction only after B1 timings

The 34,369-point blade is **not** obviously too large. First show that rendering/JVP/VJP cost scales with points or point-pixel contributions and that reduction saves meaningful wall time. If so, implement an **optional**, default-OFF reduced *tracking representation* built ONCE from each endpoint's canonical colored RGB-D cloud. Do not change the dataset or the observed next-frame RGB-D images; all cameras remain. Keep the full-resolution canonical cloud unchanged for final static/moving support and viewer evaluation.

Choose candidate budgets by an automatic, reproducible voxel-hash/stratified surfel sampler, **not random uniform removal** and not GT/motion labels. Candidate budget settings (e.g. full, ~50%, ~25%) are **diagnostic sweeps**, not accepted production defaults. Enforce multi-camera projection coverage, depth-discontinuity/silhouette retention, sampled RGB diversity/geometry and a spatially distributed set across the whole canonical object. Select one representative per geometric voxel plus extra true high-information points as needed; retain precise source indices/weights/original colors/metric positions. Prevent a sampler from deleting a small moving component merely because it is geometrically small. Never use future frames/GT to choose which points to keep.

When reduced points are used, **recompute the separate IEKF computational grouping on the reduced canonical positions**; do not silently change the 192-node coherence algorithm or claim reduced point membership is identical to the original. Save point mapping and sampling provenance and ensure each coherence representative has enough support. The reference coherence graph and 192 fixed representatives are the *downstream method*, not a license to use the exact old memberships after dropping points. Record how new sampling and group resolution affect near-interface nodes.

Comparison is fair only when the same time stamps, cameras, observed RGB-D images, process/measurement uncertainty **units**, solver convergence rule and held-out evaluation are used. Compare full-vs-reduced **rendered RGB-D coverage**, depth/colour innovation, group operator adjoint/oracle tests, tracking trajectory, rigid-coherence seed/precision, static stability, GPU memory and elapsed time. A reduced cloud that renders visibly worse or worsens moving-core selection is a failed optimization, even if it runs faster. Be alert to duplicated/overlapping point footprints: rendering all points at full resolution may already be wasteful, but that must be shown by measured contribution counts.

### B3. Adaptive active-set (optional research experiment; not baseline)

If static/occluded points waste enormous JVP/VJP time, investigate an observation-supported active set **per camera and nonlinear iteration** based solely on current projected coverage, observed depth validity and proven zero derivative support. Completely occluded points can avoid *measurement* operator work while retaining their process state and uncertainty. DO NOT freeze them, force zero velocity, or remove them from the canonical cloud. Verify the sparse active-set operator against the full renderer/JVP/VJP exactly on the same linearization. Any uncertainty-based approximate culling must be a separate opt-in ablation with disclosed error bounds.

## Acceptance and stop conditions

- Step 2A succeeds only with: (a) proven `G/Gᵀ` and reduced `A_g` oracle parity, (b) controlled IEKF per-iteration diagnostics explaining the *first frame* of divergence from a high-accuracy solve, (c) a **measured** time/VRAM breakdown, and (d) unchanged saved output and source files.
- If the operators differ from RigEuler mathematics, **stop and correct that exact bug alone** before any speed change. Provide explicit failing and passing probes.
- Step 2B may start only after the STEP 2A report exists and the operators are validated. Implement **one** optimized operator behind a flag, with numerical parity tests and equal-input benchmark. Don't simultaneously introduce downsampling, new covariance, modified trust-region rules, and kernel rewrites.
- Performance result must report wall-time **speedup of the same completed frame/update** (not just kernel microbenchmark), GPU peak memory, max Jacobian/adjoint/filtered-state discrepancy and whether the prior 192-node coherence extraction still agrees on unchanged full-resolution inputs.
- No arbitrary reductions in nonlinear iterations/PCG tolerances, no CPU-only fallback hidden as GPU acceleration, no GT-guided sampling, no alternate point correspondence observation, no modifying Phase 2, joint, coherence code, or existing result folders.

## Exact outputs

1. Read-only mathematical parity test files and report; source SHA256; explicit first discrepancy if any.
2. CUDA profiler timeline, per-stage breakdown and a short ranked bottleneck list **with measured percentages**.
3. One **opt-in** performance optimization at a time, each with baseline/equivalence/performance checks.
4. If the data justify it, an opt-in canonical voxel/visibility-aware point sampler with complete source mapping and quality-vs-speed report.
5. A clear final status: **operator verified / not verified**, **speedup verified / not verified**, **moving-motion lag diagnosed / not diagnosed**. These are independent outcomes.

Do not run the 40-scene synthetic batch until the Phase-1 operators and the benchmark are correct on small cases.
