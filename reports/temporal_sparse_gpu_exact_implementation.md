# Exact sparse GPU temporal IEKF — measured validation

**FULL_SPARSE_GPU_NOT_ACCEPTED.** This report does not certify a successful reconstruction.

Actual final blocker: cuDSS device-side factorization error INFO=-1 after the first accepted nonlinear iteration; frame 1 has no completed covariance. Negative backend INFO does not establish mathematical singularity. The raw status is retained below.

Last USB full-cloud attempt: 11.882047667982988 filtering seconds; verified frames [0]; largest failed-system component 371164 scalars. Blade remained NOT_RUN because fail-fast stopped the batch. No post-anchor trajectory/covariance was published. Reported cuDSS INFO errors are device execution/factorization errors, not independent rank certificates ([NVIDIA API documentation](https://docs.nvidia.com/cuda/cudss/1.0.0/types.html)).

Actual accepted nonlinear iteration history before failure:

```json
[
  {
    "iteration": 0,
    "objective": 44062961178.87926,
    "proposal_objective": 43743829644.68693,
    "accepted": true,
    "backtracks": 30,
    "true_components": 41225,
    "frozen_components": 1104
  }
]
```

Baseline: `f66d734c8b300c4d69c90e35ccea4fc10615ede3`. Backend: cuDSS 1.0.0.35, Torch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 4070 Laptop GPU.

Focused tests: 60 run, 0 failures, 0 errors, 0 skips. Tests match current hashed sources: True.
USB + Blade three-frame full-cloud gate: False. Eleven-frame accuracy comparison: NOT VALIDATED.

| Scene | Batch | Actual status | Verified frames | Accepted point updates |
|---|---|---|---|---|
| USB_100109 | temporal_rgbd_20261101_172133_a6468f0b | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_172133_a6468f0b | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_172523_3235bbfb | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_172523_3235bbfb | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_172926_9e82c7f4 | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_172926_9e82c7f4 | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_174537_30e2d0c6 | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_174537_30e2d0c6 | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_180202_743cb49b | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_180202_743cb49b | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_180334_1381d3fc | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_180334_1381d3fc | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_180517_b474c9b2 | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_180517_b474c9b2 | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_180725_4e6f9ef6 | FULL_RESOURCE_LIMIT | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_180725_4e6f9ef6 | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_181027_02771684 | FULL_RESOURCE_LIMIT | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_181027_02771684 | NOT_RUN | [] | 0 |
| USB_100109 | temporal_rgbd_20261101_181742_48304ebe | FULL_SOLVER_CERTIFICATE_FAILURE | [0] | 0 |
| blade_103706 | temporal_rgbd_20261101_181742_48304ebe | NOT_RUN | [] | 0 |

The JSON beside this report contains exact commands, return codes, numerical certificates, solver dimensions/fill, failures, frame journals, oracle comparisons, profiles and every source function/line/hash. Missing frames are NaN and invalid; frame-0 viewers do not count as motion estimation. Failed outputs and the historical batch have been preserved.

## Measured scaling

Frozen initial linearizations on the same USB observations. These are performance and algebra checks, not accepted nonlinear trajectories. Times are seconds. First-size initialization is cold. Schur and nonlinear acceptance are measured separately in tests/failed pilots, not inferred from assembly speed.

| Points | Source GPU | Ownership/registration GPU | Residual/Jacobian GPU | Native factorization | MAP solve | Joint selected covariance | Largest initial component | Host scalar extractions |
|---|---|---|---|---|---|---|---|---|
| 1000 | 0.10314873504638672 | 0.30691635131835937 | 0.1017220458984375 | 0.06943781999871135 | 0.010633129975758493 | 0.04176224203547463 | 36.0 | 218 |
| 10000 | 0.007451551914215088 | 0.07851229095458985 | 0.05942582321166992 | 0.4450081009999849 | 0.014088838011957705 | 0.09402792801847681 | 39.0 | 218 |
| 50000 | 0.010636639595031739 | 0.09708863830566407 | 0.12156950378417969 | 1.1520749969640747 | 0.06992042303318158 | 8.746089830994606 | 427.0 | 218 |
| 95967 | 0.011365951538085938 | 0.09671868896484374 | 0.17251113891601563 | 1.6854411660460755 | 0.05607387702912092 | 22.3176537649706 | 828.0 | 218 |

Full profiler traces/kernel names, sparse counts, fill, host RSS and memory are in JSON and the referenced SSD trace files. Torch peak allocation excludes cuDSS native allocations; native estimates and device-free snapshots are recorded separately and are not mislabeled as total measured GPU peak. Profile matches current source hashes: True.

Actual bounded components from the full-cloud **frozen benchmark**, compared independently with SciPy square-root QR (these do not pass the filtering gate):

```json
[
  {
    "path": "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_sparse_gpu_profiles_verified_9wehtewh/USB_assembly/oracle_frame_0001_component_0.npz",
    "complete_component_dimension": 29,
    "rows": 5,
    "hard_rows": 0,
    "QR_condition": 5564.847625322888,
    "mean": {
      "absolute_frobenius": 3.806785946510864e-15,
      "relative_frobenius": 9.575676475792106e-15
    },
    "selected_covariance": {
      "absolute_frobenius": 1.1868747611181924e-15,
      "relative_frobenius": 5.934369764021786e-16
    },
    "condition_based_roundoff_bound_not_acceptance_tolerance": 3.520600454642339e-06,
    "absolute_tolerance": 2e-12,
    "relative_tolerance": 2e-11,
    "tolerance_basis": "FP64 CUDA/QR micrograph and real-component discrepancies; not a relaxation to the larger condition-number bound",
    "passed": true,
    "note": "Saved information includes actual eliminated Gaussian priors and FEJ, if present. No production estimate generated here."
  },
  {
    "path": "/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/temporal_sparse_gpu_profiles_verified_9wehtewh/USB_assembly/oracle_frame_0001_component_1.npz",
    "complete_component_dimension": 27,
    "rows": 3,
    "hard_rows": 0,
    "QR_condition": 3945.509877242724,
    "mean": {
      "absolute_frobenius": 6.241415814134041e-16,
      "relative_frobenius": 1.1816731883229968e-12
    },
    "selected_covariance": {
      "absolute_frobenius": 6.939804658665644e-16,
      "relative_frobenius": 3.1035749924728755e-16
    },
    "condition_based_roundoff_bound_not_acceptance_tolerance": 1.76976848154253e-06,
    "absolute_tolerance": 2e-12,
    "relative_tolerance": 2e-11,
    "tolerance_basis": "FP64 CUDA/QR micrograph and real-component discrepancies; not a relaxation to the larger condition-number bound",
    "passed": true,
    "note": "Saved information includes actual eliminated Gaussian priors and FEJ, if present. No production estimate generated here."
  }
]
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
      "absolute_residual": 8.881784197001252e-16,
      "componentwise_backward_error": 1.8695109930096837e-16,
      "rhs_columns": 1,
      "refinement_steps": 0,
      "mixed_absolute_relative_certificate": true,
      "relative_tolerance": 2e-12,
      "absolute_tolerance": 2e-13,
      "equality_absolute": 0.0,
      "objective": 4.161762306210396e-13,
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
      "setup_seconds": 0.010984104999806732,
      "source_process_sensor_innovation_coordinates": true,
      "exact_prior_residuals_eliminated": 86,
      "backend": "cuDSS-1.0.0.35",
      "rows": 102,
      "nnz": 786,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "VRAM_free_before_analysis": 8191475712,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": 63995904,
      "torch_allocated_before_analysis": 11465216,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.001356135995592922,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        856082692,
        856523332,
        0,
        0,
        856523336,
        0,
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
      "factorization_seconds": 0.0008961510029621422,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 1154,
      "solve_certificate": {
        "absolute_residual": 8.881784197001252e-16,
        "componentwise_backward_error": 1.8695109930096837e-16,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13,
        "equality_absolute": 0.0,
        "objective": 4.161762306210396e-13,
        "normal_equations_formed": false,
        "FEJ_information_present": false
      },
      "selected_marginals": {
        "seconds": 0.30246697401162237,
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
    "MAP_abs": 2.024422296464934e-15,
    "MAP_relative": 1.5572466591040615e-13,
    "joint_covariance_abs": 6.66101631588149e-15,
    "joint_covariance_relative": 9.41228075402626e-15,
    "certificate": {
      "absolute_residual": 1.7763568394002505e-15,
      "componentwise_backward_error": 3.209288903020548e-16,
      "rhs_columns": 1,
      "refinement_steps": 0,
      "mixed_absolute_relative_certificate": true,
      "relative_tolerance": 2e-12,
      "absolute_tolerance": 2e-13,
      "equality_absolute": 4.336808689942018e-19,
      "objective": 8.248401640801998e-13,
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
      "setup_seconds": 0.01446121500339359,
      "source_process_sensor_innovation_coordinates": true,
      "exact_prior_residuals_eliminated": 70,
      "backend": "cuDSS-1.0.0.35",
      "rows": 86,
      "nnz": 498,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "VRAM_free_before_analysis": 8191475712,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": 63995904,
      "torch_allocated_before_analysis": 11520512,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.0014851380256004632,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        882228900,
        882600420,
        0,
        0,
        882600424,
        0,
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
      "factorization_seconds": 0.0007861669873818755,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 1220,
      "solve_certificate": {
        "absolute_residual": 1.7763568394002505e-15,
        "componentwise_backward_error": 3.209288903020548e-16,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13,
        "equality_absolute": 4.336808689942018e-19,
        "objective": 8.248401640801998e-13,
        "normal_equations_formed": false,
        "FEJ_information_present": false
      },
      "selected_marginals": {
        "seconds": 0.302827660983894,
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
      "nnz": 3,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "VRAM_free_before_analysis": 8191475712,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": 63995904,
      "torch_allocated_before_analysis": 11408384,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.0013054019655101001,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        4164,
        12804,
        0,
        0,
        12808,
        0,
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
      "factorization_seconds": 9.182398207485676e-05,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 2,
      "last_solve": {
        "absolute_residual": 0.0,
        "componentwise_backward_error": 0.0,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13
      },
      "total_seconds": 0.018337260000407696,
      "symbolic_matching_seconds": 0.0006910460069775581,
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
    "objective_abs": 0.0
  },
  {
    "source_marginalized": true,
    "mean_abs": 1.3160732926575894e-12,
    "cov_abs": 3.618936933843164e-12,
    "Schur_profile": {
      "possible_fill_entries": 784,
      "actual_product_nnz": 784,
      "rhs_colors": 28,
      "elimination_component_count": 1,
      "connectivity_iterations": 3,
      "estimated_workspace_bytes": 39296,
      "backend": "cuDSS-1.0.0.35",
      "rows": 74,
      "nnz": 302,
      "pivot_epsilon": 0.0,
      "formed_normal_equations": false,
      "VRAM_free_before_analysis": 8191475712,
      "VRAM_total": 8444575744,
      "max_LU_nnz_capacity": 63995904,
      "torch_allocated_before_analysis": 11471360,
      "torch_reserved_before_analysis": 25165824,
      "analysis_seconds": 0.0017026340356096625,
      "memory_estimate_query_status": 0,
      "native_memory_estimates_bytes": [
        956264228,
        956583908,
        0,
        0,
        956583912,
        0,
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
      "factorization_seconds": 0.0008931499905884266,
      "info": 0,
      "perturbed_pivots": 0,
      "factor_nnz": 533,
      "last_solve": {
        "absolute_residual": 5.10702591327572e-15,
        "componentwise_backward_error": 2.4079208827850864e-14,
        "rhs_columns": 1,
        "refinement_steps": 0,
        "mixed_absolute_relative_certificate": true,
        "relative_tolerance": 2e-12,
        "absolute_tolerance": 2e-13
      },
      "total_seconds": 0.03173193597467616,
      "symbolic_matching_seconds": 0.0008739160257391632,
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

Full-cloud three-frame execution, actual component oracles, backward execution, moving/static accuracy and the eleven-frame comparison are separate acceptance checks. A test pass or sparse MAP certificate alone does not satisfy them. No 60-frame or ten-scene batch was launched. SSD free at audit: 1610533044224 bytes.

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
