# Exact sparse GPU temporal IEKF — measured validation

**FULL_SPARSE_GPU_NOT_ACCEPTED.** This report does not certify a successful reconstruction.

Actual final blocker: None

Last USB full-cloud attempt: None filtering seconds; verified frames None; largest failed-system component None scalars. Blade remained NOT_RUN because fail-fast stopped the batch. No post-anchor trajectory/covariance was published. Reported cuDSS INFO errors are device execution/factorization errors, not independent rank certificates ([NVIDIA API documentation](https://docs.nvidia.com/cuda/cudss/1.0.0/types.html)).

Actual accepted nonlinear iteration history before failure:

```json
[]
```

Baseline: `f77492dc0c8fe3cf5f28820ba5fe1f8ed959cb58`. Backend: cuDSS 1.0.0.35, Torch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 4070 Laptop GPU.

Focused tests: 75 run, 0 failures, 0 errors, 0 skips. Tests match current hashed sources: True.
USB + Blade three-frame full-cloud gate: False. Eleven-frame accuracy comparison: NOT VALIDATED.

| Scene | Batch | Actual status | Verified frames | Accepted point updates |
|---|---|---|---|---|
| blade_103706 | temporal_rgbd_20261102_011651_b9364f61 | COMPLETE | [0, 1, 2] | 39471 |
| USB_100109 | temporal_rgbd_20261102_011901_75a43156 | COMPLETE | [0, 1, 2] | 92771 |

The JSON beside this report contains exact commands, return codes, numerical certificates, solver dimensions/fill, failures, frame journals, oracle comparisons, profiles and every source function/line/hash. Missing frames are NaN and invalid; frame-0 viewers do not count as motion estimation. Failed outputs and the historical batch have been preserved.

## Measured scaling

Frozen initial linearizations on the same USB observations. These are performance and algebra checks, not accepted nonlinear trajectories. Times are seconds. First-size initialization is cold. Schur and nonlinear acceptance are measured separately in tests/failed pilots, not inferred from assembly speed.

| Points | Source GPU | Ownership/registration GPU | Residual/Jacobian GPU | Native factorization | MAP solve | Joint selected covariance | Largest initial component | Host scalar extractions |
|---|---|---|---|---|---|---|---|---|


Full profiler traces/kernel names, sparse counts, fill, host RSS and memory are in JSON and the referenced SSD trace files. Torch peak allocation excludes cuDSS native allocations; native estimates and device-free snapshots are recorded separately and are not mislabeled as total measured GPU peak. Profile matches current source hashes: False.

Actual bounded components from the full-cloud **frozen benchmark**, compared independently with SciPy square-root QR (these do not pass the filtering gate):

```json
[]
```

## Implementation

Production full mode uses `temporal_sparse_run.py` → typed SoA GPU factors → exact current-pixel equality substitution → invertible source/process/sensor innovation coordinates → sparse augmented KKT factorization. Actual Gaussian prior residuals are eliminated algebraically; no artificial prior is introduced. Current 6×6 and requested state/reference/pixel cross blocks come from selected solves of the full joint operator. Component-colored RHS reuse is confined to truly independent components. FEJ retains information, gradient, linearization, constant cost and remaining hard constraints with all induced fill. Shared raw sensor pixels receive one likelihood; the source chart retains source-depth coupling.

Nonlinear trials re-render all depth competitors, validate hard foreground and four-neighbor reference support, keep the fixed common factor domain, project ideal equalities exactly, and test the complete nonlinear objective. Backtracking applies to complete true variable components; frozen/stalled components are reported and never called converged. No GT enters inference. Legacy and simplified mathematical files unchanged: True.

Output publication verifies sizes/checksums and commits frame validity only after posterior/covariance/debug diagnostics exist. Batch execution uses an OS process-group deadline, fail-fast default and an independent failure journal. More than three full-mode frames or more than two scenes is blocked pending source-matched three-frame acceptance evidence. No source snapshots are copied to results.

## Mathematical parity evidence

Independent CPU QR/SVD and sparse CUDA witnesses (complete tables and tolerances in JSON):

```json
[
  {
    "ideal": false,
    "MAP_abs": 1.1299049099724186e-15,
    "MAP_relative": 8.691572793015491e-14,
    "joint_covariance_abs": 1.4432899320127035e-15,
    "joint_covariance_relative": 2.0389816055885687e-15,
    "certificate": {
      "absolute_residual": 5.551115123125783e-17,
      "componentwise_backward_error": 2.2429539791651097e-16,
      "rhs_columns": 1,
      "refinement_steps": 0,
      "mixed_absolute_relative_certificate": true,
      "relative_tolerance": 2e-12,
      "absolute_tolerance": 2e-13,
      "equality_absolute": 0.0,
      "objective": 4.1617623062148914e-13,
      "normal_equations_formed": false,
      "FEJ_information_present": false
    },
    "profile": {
      "eliminated_pixel_scalars": 0,
      "free_dimension": 86,
      "remaining_constraints": 0,
      "nnz_A_reduced": 342,
      "nnz_C_reduced": 0,
      "nnz_substitution": 114,
      "nnz_KKT": 786,
      "setup_seconds": 0.08382814499782398,
      "source_process_sensor_innovation_coordinates": true,
      "exact_prior_residuals_eliminated": 86,
      "backend": "cuDSS-1.0.0.35",
      "rows": 102,
      "nnz": 444,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "algorithm": "symmetric_nd",
      "matrix_type": "symmetric_indefinite",
      "ordering": "nested_dissection",
      "pivoting": "Bunch_Kaufman",
      "VRAM_free_before_handle": 8197767168,
      "host_threading_layer": "libcudss_mtlayer_gomp.so.1",
      "VRAM_free_before_analysis": 8189378560,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": null,
      "torch_allocated_before_analysis": 11877888,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.0591267179697752,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        392476,
        392476,
        21136,
        25520,
        392828,
        36624,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "native_LU_nnz_reported": 589,
      "factorization_seconds": 0.00023954198695719242,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 589,
      "max_LU_nnz_capacity_source": null,
      "inertia": [
        16,
        86
      ],
      "expected_inertia": [
        16,
        86
      ],
      "solve_certificate": {
        "absolute_residual": 5.551115123125783e-17,
        "componentwise_backward_error": 2.2429539791651097e-16,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13,
        "equality_absolute": 0.0,
        "objective": 4.1617623062148914e-13,
        "normal_equations_formed": false,
        "FEJ_information_present": false
      },
      "selected_marginals": {
        "seconds": 0.30980022699804977,
        "output_blocks": 1,
        "query_width": 86,
        "rhs_colors": 86,
        "rhs_batch": 8,
        "connectivity_iterations": 3,
        "component_count": 7,
        "largest_component_scalars": 80,
        "symmetry_absolute": 1.1102230246251565e-16,
        "method": "same augmented sparse factorization; exact component-colored RHS"
      }
    }
  },
  {
    "ideal": true,
    "MAP_abs": 2.040034807748725e-15,
    "MAP_relative": 1.569256273441623e-13,
    "joint_covariance_abs": 3.1785358734623726e-15,
    "joint_covariance_relative": 4.491397499874284e-15,
    "certificate": {
      "absolute_residual": 6.938893903907228e-18,
      "componentwise_backward_error": 2.0494601319547744e-16,
      "rhs_columns": 1,
      "refinement_steps": 0,
      "mixed_absolute_relative_certificate": true,
      "relative_tolerance": 2e-12,
      "absolute_tolerance": 2e-13,
      "equality_absolute": 2.168404344971009e-19,
      "objective": 8.248401640823478e-13,
      "normal_equations_formed": false,
      "FEJ_information_present": false
    },
    "profile": {
      "eliminated_pixel_scalars": 16,
      "free_dimension": 70,
      "remaining_constraints": 0,
      "nnz_A_reduced": 206,
      "nnz_C_reduced": 0,
      "nnz_substitution": 304,
      "nnz_KKT": 498,
      "setup_seconds": 0.07997807202627882,
      "source_process_sensor_innovation_coordinates": true,
      "exact_prior_residuals_eliminated": 70,
      "backend": "cuDSS-1.0.0.35",
      "rows": 86,
      "nnz": 292,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "algorithm": "symmetric_nd",
      "matrix_type": "symmetric_indefinite",
      "ordering": "nested_dissection",
      "pivoting": "Bunch_Kaufman",
      "VRAM_free_before_handle": 8197767168,
      "host_threading_layer": "libcudss_mtlayer_gomp.so.1",
      "VRAM_free_before_analysis": 8189378560,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": null,
      "torch_allocated_before_analysis": 11942400,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.04781907197320834,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        331764,
        331764,
        20112,
        23152,
        332580,
        31312,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "native_LU_nnz_reported": 422,
      "factorization_seconds": 0.00022243201965466142,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 422,
      "max_LU_nnz_capacity_source": null,
      "inertia": [
        16,
        70
      ],
      "expected_inertia": [
        16,
        70
      ],
      "solve_certificate": {
        "absolute_residual": 6.938893903907228e-18,
        "componentwise_backward_error": 2.0494601319547744e-16,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13,
        "equality_absolute": 2.168404344971009e-19,
        "objective": 8.248401640823478e-13,
        "normal_equations_formed": false,
        "FEJ_information_present": false
      },
      "selected_marginals": {
        "seconds": 0.29522801097482443,
        "output_blocks": 1,
        "query_width": 86,
        "rhs_colors": 86,
        "rhs_batch": 8,
        "connectivity_iterations": 3,
        "component_count": 7,
        "largest_component_scalars": 64,
        "symmetry_absolute": 1.1102230246251565e-16,
        "method": "same augmented sparse factorization; exact component-colored RHS"
      }
    }
  },
  {
    "surviving_equality": true,
    "Schur_profile": {
      "possible_fill_entries": 9,
      "actual_product_nnz": 5,
      "rhs_colors": 3,
      "elimination_component_count": 1,
      "connectivity_iterations": 2,
      "estimated_workspace_bytes": 672,
      "backend": "cuDSS-1.0.0.35",
      "rows": 2,
      "nnz": 2,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "algorithm": "symmetric_nd",
      "matrix_type": "symmetric_indefinite",
      "ordering": "nested_dissection",
      "pivoting": "Bunch_Kaufman",
      "VRAM_free_before_handle": 8197767168,
      "host_threading_layer": "libcudss_mtlayer_gomp.so.1",
      "VRAM_free_before_analysis": 8189378560,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": null,
      "torch_allocated_before_analysis": 11810304,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.013161465001758188,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        19940,
        26804,
        13392,
        13440,
        20036,
        13568,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "native_LU_nnz_reported": 4,
      "factorization_seconds": 0.00012360600521788,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 4,
      "max_LU_nnz_capacity_source": null,
      "inertia": [
        1,
        1
      ],
      "expected_inertia": null,
      "last_solve": {
        "absolute_residual": 1.3877787807814457e-17,
        "componentwise_backward_error": 6.938893903907228e-17,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13
      },
      "total_seconds": 0.03392427496146411,
      "symbolic_matching_seconds": 0.0007839210447855294,
      "eliminated_scalars": 1,
      "retained_scalars": 2,
      "old_constraints": 2,
      "eliminated_constraint_rank_candidate": 1,
      "remaining_hard_constraints": 1,
      "FEJ_information_nnz": 4,
      "FEJ_constraint_nnz": 2,
      "exactness": "complete sparse Schur; numerical certification required; normal-information conditioning audited against QR"
    },
    "posterior_abs": 8.326672684688674e-17,
    "objective_abs": 1.3877787807814457e-17
  },
  {
    "source_marginalized": true,
    "mean_abs": 7.018437225457473e-14,
    "cov_abs": 2.7333690866271354e-13,
    "Schur_profile": {
      "possible_fill_entries": 784,
      "actual_product_nnz": 784,
      "rhs_colors": 28,
      "elimination_component_count": 1,
      "connectivity_iterations": 3,
      "estimated_workspace_bytes": 39296,
      "backend": "cuDSS-1.0.0.35",
      "rows": 74,
      "nnz": 180,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "algorithm": "symmetric_nd",
      "matrix_type": "symmetric_indefinite",
      "ordering": "nested_dissection",
      "pivoting": "Bunch_Kaufman",
      "VRAM_free_before_handle": 8197767168,
      "host_threading_layer": "libcudss_mtlayer_gomp.so.1",
      "VRAM_free_before_analysis": 8189378560,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": null,
      "torch_allocated_before_analysis": 11876864,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.07603297202149406,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        285300,
        285300,
        19088,
        21136,
        286204,
        27440,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      ],
      "native_LU_nnz_reported": 318,
      "factorization_seconds": 0.00024047103943303227,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 318,
      "max_LU_nnz_capacity_source": null,
      "inertia": [
        58,
        16
      ],
      "expected_inertia": null,
      "last_solve": {
        "absolute_residual": 1.7763568394002505e-15,
        "componentwise_backward_error": 3.209076275936337e-16,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13
      },
      "total_seconds": 0.10752467002021149,
      "symbolic_matching_seconds": 0.0007536689518019557,
      "eliminated_scalars": 58,
      "retained_scalars": 28,
      "old_constraints": 16,
      "eliminated_constraint_rank_candidate": 16,
      "remaining_hard_constraints": 0,
      "FEJ_information_nnz": 784,
      "FEJ_constraint_nnz": 0,
      "exactness": "complete sparse Schur; numerical certification required; normal-information conditioning audited against QR"
    }
  }
]
```

## Limits and unfinished acceptance

- Sparse FEJ forms normal information with all fill retained; independent QR parity covers tested conditioning only.
- One batched CPU symbolic bipartite matching occurs for FEJ hard-constraint pivots; numerical Schur work is CUDA.
- cuDSS rank-deficient or resource-limited systems fail; no regularization or fallback.
- Prior-predictive innovation NIS is deferred in sparse output, labeled unavailable, never fabricated.
- Nonlinear acceptance retains the baseline no-new-outside-mask policy and reports any preexisting violations.
- Fixed common measurement domain can obstruct descent at discrete ownership/support boundaries.

Full-cloud three-frame execution, actual component oracles, backward execution, moving/static accuracy and the eleven-frame comparison are separate acceptance checks. A test pass or sparse MAP certificate alone does not satisfy them. No 60-frame or ten-scene batch was launched. SSD free at audit: 1544209563648 bytes.

## Reproduce the bounded gate

```bash
cd /home/samuelemara/Joint3DGS/RGBDekf
conda activate rigeuler3dgs
python -B tools/test_temporal_rgbd.py
bash tools/run_temporal_rgbd_critical.sh --mode full --direction forward --frame-limit 3 \
  --scenes USB_100109,blade_103706 --pcd-subsample-factor 1 \
  --temporal-warp-noise ideal --stop-after free --temporal-runtime-budget-seconds 120
```

This command reproduces an acceptance attempt, not a production-cleared reconstruction. The ideal additional-warp-zero assumption is uncalibrated; sensor and reference uncertainty stay active. Re-run this reporting tool with `--test-report`, `--profile-report`, and `--batch` paths to independently inspect newly saved results. No commit or push was performed.
