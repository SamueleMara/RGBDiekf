# Rendered-image RGB-D IEKF on `blade_103706` — 24 October 2026

Spec: `CODEX_RIGEULER3DGS_TO_RGBD_TRUE_IEKF_THEORY_AND_IMPLEMENTATION_2026-10-24.md`. Theory, files and commands: [IMAGE_IEKF_RGBD.md](../IMAGE_IEKF_RGBD.md).

## Outcome

The pipeline ran fresh on all 60 frames and all 8 cameras, in both directions, and finished with status **COMPLETE**.

Ground-truth (GT) scores are computed after inference and are never read by an estimator.

| | forward (anchor frame 0) | backward (anchor frame 59) |
|---|---|---|
| Joint decision | PRISMATIC (revolute axis 38 part scales away: degenerate) | PRISMATIC (54 part scales) |
| Prismatic axis error vs GT | 0.81° | 1.18° |
| Image-refined displacement q, RMSE over 0.835 m | 2.4 mm | 5.6 mm |
| Rigid IEKF translation error, mean / max | 8.4 / 15.9 mm | 19.4 / 33.4 mm |
| Rigid IEKF rotation error, mean / max | 2.6° / 6.9° | 2.5° / 5.1° |
| Phase-1 core Kabsch (reference only), translation | 240 / 480 mm | 393 / 1388 mm |
| Coherence core precision / recall, GT-motion labels* | 0.82 / 0.77 | 0.78 / 0.77 |
| STATIC_COMPATIBLE labels on GT-static points* | 99.4% | 99.6% |
| MOVING_JOINT_COMPATIBLE labels on GT-moving points* | 15% | 16% |

The two directions are **COMPATIBLE**: the axes agree to 1.57° and the displacement histories to 6.2 mm RMSE, within the resolution tolerances. The two directions reuse the same pixels, so no posterior product is formed.

\* GT-motion labels: a point is "moving" if, after the GT travel, it lands within 1.5 cm of a final-frame blade point of the sparse GT cloud, and "static" if it lies within 1.5 cm of a final-frame handle point. The sparse cloud (555 and 633 points) labels only about 40% of the catalogue, so these rates are approximate.

## What did not work

**Support "moving" labels are unreliable on this object.** The blade and most of the handle are uniform dark blue and slide along their own length. A sample translated along that direction usually lands on matching surface, so the image cannot decide it, and about half of each catalogue is labelled AMBIGUOUS. The labels that are decided as moving are mostly wrong (15–16% correct). The contradiction rule and model-error tempering reduced, but did not remove, this. Static labels are reliable.

**Rigid rotation is a few degrees off.** RigEuler reports a mean of 0.74° on this scene with trained 3DGS appearance. The point renderer's model error (RGB RMSE 0.06–0.15, depth edges) and the weak information on the uniform blade limit the 6-DOF fit. The 1-DOF image refinement still recovers q to millimetres.

**Phase 1 lags on untextured blade surfaces**, as expected for a free-point filter with weak in-plane information. The best-observed decile tracks the GT slide (0.70 vs 0.71 m at frame 50); the median lags. Coherence and the rigid stage then pool the evidence.

## Departures from the pinned RigEuler calls, each measured

1. **Coherence member weights and image-supported trajectories.** Member weights are `w_i` = visibility × precision. Trajectories are held at the last frame with direct image evidence. Without this, prediction-only drift (a coherent constant-velocity translation) was once selected as the "moving part" (7% blade precision).
2. **Model-error term only in support.** `R_model` (RigEuler `estimate_model_error`, attached to the points) is used only for support labelling. In the rigid and joint estimation stages it raised the synthetic hinge rotation error from 0.37° to 0.64–0.70° (10–15° without the pyramid).
3. **Coarse-to-fine levels (×2, then full resolution)** are a globalisation device in every image IEKF. The prior stays fixed, and the posterior comes from the full-resolution linearisation.
4. **Phase-1 posterior** follows RigEuler's per-point block contract. On a dense case its position variance is about 20% below the exact restricted-Joseph covariance (`reports/image_iekf_dense_covariance_comparison.json`).

## Viewer

`viewer/index.html`; serve it with `python -m http.server --directory viewer 8765`.

Five headless-Chrome QA checks passed (`viewer/qa_*.json` and `.png`). Each check confirms:

* the panel geometry;
* exact coverage counts against the saved arrays;
* the RGB RMSE recomputed from the displayed pixels, agreeing to 1e-14;
* that the 3D points project onto the observed domain of the selected camera;
* that both candidate axes are drawn.

---

# Rendered-image RGB-D IEKF — measured validation

Run: `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/image_iekf_final_20261024`. Status: **COMPLETE**. Runtime 140.8 min. Peak RSS 7.58 GB. CUDA peak allocated 2.20 GB on NVIDIA GeForce RTX 4070 Laptop GPU.
Measurement model `rendered_rgbd_point_cloud_v1`; association used: False; cameras [0, 1, 2, 3, 4, 5, 6, 7]; frames 60; full source True.

## Executed tests

Executed on 24 October 2026, RTX 4070 Laptop (CUDA available, no skips):

| suite | tests | result |
|---|---|---|
| `tests.test_image_renderer` (Gate A: loop oracle, CPU = CUDA, FD-JVP, adjoint, Fisher blocks, metric depth, RGB sensitivity, pyramid alignment) | 8 | OK |
| `tests.test_image_iekf` (Gate B: forward/backward/revolute tracking with `associate()` patched to raise, signed time, fixed prior + one posterior, information = Kalman gain form, dense covariance comparison, background never read) | 7 | OK |
| `tests.test_image_support` (Gate F: fast path = complete rerender with/without point model error, contradiction labels, world comparison) | 4 | OK |
| `tests.test_image_rigid_joint` (Gates C/D/E: end-to-end prismatic fwd/bwd, revolute, static → UNRESOLVED, rigid from zero init, SE(3) process and prior-error Jacobians vs FD) | 7 | OK |
| baseline `tools/test_pointwise_rgbd.py` | 45 | OK |
| baseline `tools/test_free_rgbd.py` | 34 | OK |

## Direction `first`

### Phase 1 (free points)

59 updated frames; mean 20.9 s/frame; stops {'converged': 59}.

| frame | stop | accepted | RGB RMSE | depth RMSE (m) | covered domain | active contributors | adjoint rel. err |
|---|---|---|---|---|---|---|---|
| 1 | converged | 30 | 0.0859 | 0.0392 | 0.994 | 755741 | 1.7e-16 |
| 5 | converged | 17 | 0.0943 | 0.0440 | 0.989 | 709804 | 1.1e-15 |
| 9 | converged | 35 | 0.1033 | 0.0477 | 0.987 | 683350 | 2.5e-16 |
| 13 | converged | 20 | 0.1134 | 0.0537 | 0.981 | 655759 | 7.0e-16 |
| 17 | converged | 16 | 0.1258 | 0.0649 | 0.974 | 627727 | 3.5e-16 |
| 21 | converged | 18 | 0.1349 | 0.0742 | 0.968 | 607492 | 8.6e-16 |
| 25 | converged | 20 | 0.1423 | 0.0824 | 0.958 | 584438 | 4.0e-16 |
| 29 | converged | 17 | 0.1518 | 0.0910 | 0.947 | 571511 | 7.1e-16 |
| 33 | converged | 20 | 0.1558 | 0.0964 | 0.940 | 565190 | 3.5e-16 |
| 37 | converged | 14 | 0.1598 | 0.1033 | 0.932 | 556898 | 1.7e-16 |
| 41 | converged | 16 | 0.1630 | 0.1096 | 0.924 | 550010 | 8.0e-16 |
| 45 | converged | 12 | 0.1667 | 0.1255 | 0.920 | 546284 | 1.4e-15 |
| 49 | converged | 13 | 0.1689 | 0.1308 | 0.913 | 551433 | 0.0e+00 |
| 53 | converged | 19 | 0.1708 | 0.1349 | 0.906 | 558588 | 1.1e-15 |
| 57 | converged | 13 | 0.1730 | 0.1422 | 0.901 | 564379 | 2.8e-16 |
| 59 | converged | 10 | 0.1712 | 0.1450 | 0.897 | 567960 | 3.2e-16 |

### Phase 2 (rigid)

59 updated frames; mean 12.1 s/frame; stops {'converged': 59}.

| frame | stop | accepted | RGB RMSE | depth RMSE (m) | covered domain | active contributors | adjoint rel. err |
|---|---|---|---|---|---|---|---|
| 1 | converged | 8 | 0.0888 | 0.0396 | 0.994 | 785817 | — |
| 5 | converged | 7 | 0.0920 | 0.0379 | 0.991 | 784239 | — |
| 9 | converged | 10 | 0.0934 | 0.0403 | 0.989 | 784347 | — |
| 13 | converged | 5 | 0.0958 | 0.0413 | 0.983 | 783321 | — |
| 17 | converged | 13 | 0.0970 | 0.0406 | 0.975 | 783976 | — |
| 21 | converged | 4 | 0.0976 | 0.0410 | 0.967 | 783102 | — |
| 25 | converged | 24 | 0.0980 | 0.0415 | 0.960 | 783126 | — |
| 29 | converged | 9 | 0.0974 | 0.0411 | 0.953 | 782474 | — |
| 33 | converged | 9 | 0.0982 | 0.0427 | 0.946 | 784077 | — |
| 37 | converged | 17 | 0.1007 | 0.0424 | 0.939 | 785566 | — |
| 41 | converged | 8 | 0.1019 | 0.0439 | 0.929 | 785425 | — |
| 45 | converged | 8 | 0.0986 | 0.0447 | 0.923 | 785157 | — |
| 49 | converged | 18 | 0.1002 | 0.0458 | 0.915 | 787474 | — |
| 53 | converged | 15 | 0.1024 | 0.0459 | 0.907 | 788145 | — |
| 57 | converged | 12 | 0.1031 | 0.0476 | 0.899 | 787053 | — |
| 59 | converged | 5 | 0.1018 | 0.0471 | 0.894 | 787973 | — |

### Coherence

RESOLVED; 22 of 192 representative nodes; 3544 core points; weighting: w_i = image-visibility fraction x mean Phase-1 position precision (RigEuler theory section 5).
Image evidence: {"core_mean_directly_observed_fraction": 0.7473761376373361, "non_core_mean_directly_observed_fraction": 0.9090841381559231, "core_max_node_displacement_m": 0.892947646356721, "note": "directly observed = point had active rendered-image pixels at that frame; prediction-only drift is not image evidence"}

### Rigid SE(3) IEKF

FIT; 3544 core points; {"max_rotation_rad": 0.12024871925596996, "max_frame_to_frame_rotation_rad": 0.03853770417012189, "translation_range_m": 0.8247476487176999, "kabsch_max_rotation_rad": 0.6311633573572047}

### Joint

**PRISMATIC** — revolute degenerate (axis/part scale 37.6, rank deficient False); cannot veto prismatic
gamma_1dof 0.8747; image NLL {"static_identity": -11999761.8, "free_rigid": -14114936.8, "PRISMATIC": -14085708.6, "REVOLUTE": -12223931.2}; max core displacement 0.826 m; revolute axis/part scale 37.6.

* PRISMATIC: axis [0.0105, -0.9999, -0.0094], image NLL -14085708.6, free-rigid vs joint point discrepancy (image q) rms 0.0095 m
* REVOLUTE: axis [-0.7889, -0.2261, 0.5713], axis point [5.161, 0.56, 7.335], image NLL -12223931.2, free-rigid vs joint point discrepancy (image q) rms 1.0612 m

### Support completion

{"MOVING_CORE": 3544, "MOVING_JOINT_COMPATIBLE": 1072, "STATIC_COMPATIBLE": 15556, "AMBIGUOUS": 14197}
{"moving_joint_compatible_median_contradiction_of_static": 28.92683808216636, "static_compatible_median_contradiction_of_joint": 1098.218561730152, "uninfluenced_points": 0}

## Direction `last`

### Phase 1 (free points)

59 updated frames; mean 23.0 s/frame; stops {'converged': 59}.

| frame | stop | accepted | RGB RMSE | depth RMSE (m) | covered domain | active contributors | adjoint rel. err |
|---|---|---|---|---|---|---|---|
| 0 | converged | 23 | 0.2129 | 0.1582 | 0.950 | 532362 | 4.4e-16 |
| 4 | converged | 19 | 0.2100 | 0.1464 | 0.952 | 532140 | 2.7e-15 |
| 8 | converged | 18 | 0.2054 | 0.1370 | 0.954 | 535484 | 4.2e-16 |
| 12 | converged | 16 | 0.2016 | 0.1273 | 0.954 | 543014 | 2.1e-16 |
| 16 | converged | 20 | 0.1994 | 0.1160 | 0.957 | 549921 | 9.0e-16 |
| 20 | converged | 15 | 0.1938 | 0.1078 | 0.958 | 562268 | 3.4e-15 |
| 24 | converged | 16 | 0.1877 | 0.1001 | 0.964 | 577063 | 1.0e-15 |
| 28 | converged | 22 | 0.1767 | 0.0900 | 0.966 | 600678 | 6.5e-16 |
| 32 | converged | 19 | 0.1667 | 0.0830 | 0.970 | 619781 | 1.8e-15 |
| 36 | converged | 13 | 0.1565 | 0.0787 | 0.976 | 647391 | 7.5e-16 |
| 40 | converged | 24 | 0.1405 | 0.0665 | 0.978 | 671336 | 1.9e-16 |
| 44 | converged | 17 | 0.1228 | 0.0560 | 0.982 | 706901 | 2.4e-16 |
| 48 | converged | 17 | 0.1087 | 0.0562 | 0.986 | 741943 | 6.9e-16 |
| 52 | converged | 25 | 0.0964 | 0.0528 | 0.989 | 779406 | 2.2e-16 |
| 56 | converged | 24 | 0.0914 | 0.0427 | 0.990 | 807463 | 0.0e+00 |
| 58 | converged | 34 | 0.0872 | 0.0421 | 0.993 | 843961 | 1.9e-16 |

### Phase 2 (rigid)

59 updated frames; mean 11.5 s/frame; stops {'converged': 55, 'iteration_budget': 4}.

| frame | stop | accepted | RGB RMSE | depth RMSE (m) | covered domain | active contributors | adjoint rel. err |
|---|---|---|---|---|---|---|---|
| 0 | converged | 12 | 0.1165 | 0.0538 | 0.987 | 876979 | — |
| 4 | iteration_budget | 19 | 0.1188 | 0.0512 | 0.987 | 878244 | — |
| 8 | converged | 7 | 0.1186 | 0.0493 | 0.989 | 878661 | — |
| 12 | iteration_budget | 19 | 0.1166 | 0.0479 | 0.987 | 875346 | — |
| 16 | converged | 9 | 0.1151 | 0.0480 | 0.988 | 877046 | — |
| 20 | converged | 9 | 0.1143 | 0.0466 | 0.990 | 880906 | — |
| 24 | converged | 10 | 0.1119 | 0.0450 | 0.989 | 878958 | — |
| 28 | converged | 6 | 0.1098 | 0.0432 | 0.989 | 877505 | — |
| 32 | converged | 11 | 0.1069 | 0.0435 | 0.990 | 879227 | — |
| 36 | converged | 14 | 0.1056 | 0.0427 | 0.990 | 880159 | — |
| 40 | converged | 12 | 0.1011 | 0.0421 | 0.990 | 880349 | — |
| 44 | converged | 8 | 0.0974 | 0.0405 | 0.990 | 882663 | — |
| 48 | converged | 20 | 0.0953 | 0.0407 | 0.991 | 882721 | — |
| 52 | converged | 12 | 0.0918 | 0.0383 | 0.992 | 882140 | — |
| 56 | converged | 14 | 0.0917 | 0.0401 | 0.991 | 884321 | — |
| 58 | converged | 15 | 0.0901 | 0.0399 | 0.993 | 884149 | — |

### Coherence

RESOLVED; 19 of 192 representative nodes; 3383 core points; weighting: w_i = image-visibility fraction x mean Phase-1 position precision (RigEuler theory section 5).
Image evidence: {"core_mean_directly_observed_fraction": 0.7390416326192786, "non_core_mean_directly_observed_fraction": 0.859580880742095, "core_max_node_displacement_m": 1.0651643035521996, "note": "directly observed = point had active rendered-image pixels at that frame; prediction-only drift is not image evidence"}

### Rigid SE(3) IEKF

FIT; 3383 core points; {"max_rotation_rad": 0.08878959672933678, "max_frame_to_frame_rotation_rad": 0.04144363382892417, "translation_range_m": 0.8326319992801472, "kabsch_max_rotation_rad": 3.1156586075174406}

### Joint

**PRISMATIC** — revolute degenerate (axis/part scale 53.7, rank deficient False); cannot veto prismatic
gamma_1dof 0.9164; image NLL {"static_identity": -13234450.9, "free_rigid": -15100004.0, "PRISMATIC": -14954680.7, "REVOLUTE": -14911041.9}; max core displacement 0.837 m; revolute axis/part scale 53.7.

* PRISMATIC: axis [-0.01, 0.9998, -0.018], image NLL -14954680.7, free-rigid vs joint point discrepancy (image q) rms 0.0090 m
* REVOLUTE: axis [-0.3054, 0.0154, 0.9521], axis point [-8.675, -0.213, -2.71], image NLL -14911041.9, free-rigid vs joint point discrepancy (image q) rms 0.0114 m

### Support completion

{"MOVING_CORE": 3383, "MOVING_JOINT_COMPATIBLE": 1086, "STATIC_COMPATIBLE": 16948, "AMBIGUOUS": 17205}
{"moving_joint_compatible_median_contradiction_of_static": 34.934553695730756, "static_compatible_median_contradiction_of_joint": 1341.2588077439518, "uninfluenced_points": 0}

## Forward/backward

**COMPATIBLE** — PRISMATIC in both directions within resolution tolerances
```json
{
 "PRISMATIC": {
  "type": "PRISMATIC",
  "axis_sign": -1.0,
  "axis_angle_rad": 0.027351442307429447,
  "axis_angle_tolerance_rad": 0.036376622249417394,
  "q_rmse": 0.006152831674255452,
  "q_tolerance": 0.030668992549180984,
  "travel": 0.8430548952496728,
  "compatible": true
 },
 "REVOLUTE": {
  "type": "REVOLUTE",
  "axis_sign": 1.0,
  "axis_angle_rad": 0.6738158313616585,
  "axis_angle_tolerance_rad": 0.5560752115624503,
  "q_rmse": 0.0551550273946368,
  "q_tolerance": 0.12939767622348247,
  "travel": 0.23002703747737635,
  "axis_line_distance_m": 5.561162028223286,
  "axis_line_tolerance_m": 0.13332777261976148,
  "compatible": false
 }
}
```

## Browser QA (headless Chrome, real WebGL page)

| panel | checks | pass |
|---|---|---|
| qa_first_joint_revolute_f030_c1 | {"panel_geometry": true, "coverage_counts": true, "rgb_rmse": true, "projection_onto_observed_domain": true, "both_candidates_drawn": true} | True |
| qa_first_phase1_f030_c0 | {"panel_geometry": true, "coverage_counts": true, "rgb_rmse": true, "projection_onto_observed_domain": true, "both_candidates_drawn": true} | True |
| qa_first_rigid_f030_c4 | {"panel_geometry": true, "coverage_counts": true, "rgb_rmse": true, "projection_onto_observed_domain": true, "both_candidates_drawn": true} | True |
| qa_last_joint_prismatic_f030_c2 | {"panel_geometry": true, "coverage_counts": true, "rgb_rmse": true, "projection_onto_observed_domain": true, "both_candidates_drawn": true} | True |
| qa_last_phase1_f010_c6 | {"panel_geometry": true, "coverage_counts": true, "rgb_rmse": true, "projection_onto_observed_domain": true, "both_candidates_drawn": true} | True |

## Ground-truth evaluation (after inference; never read by the estimator)

```json
{
 "gt_type": "PRISMATIC",
 "gt_axis": [
  0.0,
  -1.0,
  2.220446049250313e-16
 ],
 "gt_q_range": 0.835199995432,
 "directions": {
  "first": {
   "anchor_frame": 0,
   "labelled_points": 6912,
   "phase1_slide": {
    "0": {
     "gt_m": 0.0,
     "moving_p25_p50_p90_m": [
      0.0,
      0.0,
      0.0
     ],
     "static_median_m": 0.0
    },
    "10": {
     "gt_m": 0.14155932126,
     "moving_p25_p50_p90_m": [
      0.0083,
      0.041,
      0.1292
     ],
     "static_median_m": -0.0021333929500421127
    },
    "20": {
     "gt_m": 0.283118642519,
     "moving_p25_p50_p90_m": [
      0.0357,
      0.0969,
      0.2685
     ],
     "static_median_m": -0.0023971915142794744
    },
    "30": {
     "gt_m": 0.424677963779,
     "moving_p25_p50_p90_m": [
      0.0616,
      0.1568,
      0.4123
     ],
     "static_median_m": -0.004111157776698789
    },
    "40": {
     "gt_m": 0.566237285039,
     "moving_p25_p50_p90_m": [
      0.0833,
      0.2161,
      0.5585
     ],
     "static_median_m": -0.006060316142910466
    },
    "50": {
     "gt_m": 0.707796606298,
     "moving_p25_p50_p90_m": [
      0.1019,
      0.2792,
      0.7011
     ],
     "static_median_m": -0.006557928655364726
    }
   },
   "core": {
    "points": 3544,
    "precision_labelled": 0.8993993993993994,
    "recall_labelled": 0.9584
   },
   "rigid": {
    "translation_error_mean_max_m": [
     0.008386165445634843,
     0.01592365316908688
    ],
    "rotation_error_mean_max_deg": [
     2.5669595763697144,
     6.889744105220592
    ]
   },
   "kabsch": {
    "translation_error_mean_max_m": [
     0.2403932874701182,
     0.47956679454305645
    ],
    "rotation_error_mean_max_deg": [
     10.839987645779683,
     36.16299655987522
    ]
   },
   "joint": {
    "status": "PRISMATIC",
    "reason": "revolute degenerate (axis/part scale 37.6, rank deficient False); cannot veto prismatic",
    "PRISMATIC": {
     "axis_error_deg": 0.8081566533392881,
     "q_rmse": 0.0023559235720902287
    },
    "REVOLUTE": {
     "axis_error_deg": 76.93224084174634,
     "q_rmse": null
    }
   },
   "support": {
    "MOVING_CORE": {
     "count": 3544,
     "moving_fraction_labelled": 0.8993993993993994
    },
    "MOVING_JOINT_COMPATIBLE": {
     "count": 1072,
     "moving_fraction_labelled": 0.015228426395939087
    },
    "STATIC_COMPATIBLE": {
     "count": 15556,
     "moving_fraction_labelled": 0.0
    },
    "AMBIGUOUS": {
     "count": 14197,
     "moving_fraction_labelled": 0.008211353088182792
    }
   }
  },
  "last": {
   "anchor_frame": 59,
   "labelled_points": 7948,
   "phase1_slide": {
    "0": {
     "gt_m": -0.835199995432,
     "moving_p25_p50_p90_m": [
      -0.7993,
      -0.5633,
      -0.0812
     ],
     "static_median_m": 0.005150469212036046
    },
    "10": {
     "gt_m": -0.6936406741719999,
     "moving_p25_p50_p90_m": [
      -0.6573,
      -0.4561,
      -0.0705
     ],
     "static_median_m": 0.0042836325250272846
    },
    "20": {
     "gt_m": -0.552081352913,
     "moving_p25_p50_p90_m": [
      -0.5157,
      -0.3515,
      -0.0618
     ],
     "static_median_m": 0.004954586182559508
    },
    "30": {
     "gt_m": -0.41052203165299994,
     "moving_p25_p50_p90_m": [
      -0.3736,
      -0.2542,
      -0.0465
     ],
     "static_median_m": 0.00405010513765747
    },
    "40": {
     "gt_m": -0.268962710393,
     "moving_p25_p50_p90_m": [
      -0.2301,
      -0.1624,
      -0.0253
     ],
     "static_median_m": 0.0024928616192864917
    },
    "50": {
     "gt_m": -0.12740338913399996,
     "moving_p25_p50_p90_m": [
      -0.1022,
      -0.0729,
      -0.0048
     ],
     "static_median_m": 0.0014109985121649276
    }
   },
   "core": {
    "points": 3383,
    "precision_labelled": 0.8706896551724138,
    "recall_labelled": 0.30440024110910185
   },
   "rigid": {
    "translation_error_mean_max_m": [
     0.019393963901696874,
     0.03343244322894366
    ],
    "rotation_error_mean_max_deg": [
     2.543341144899004,
     5.08726915725958
    ]
   },
   "kabsch": {
    "translation_error_mean_max_m": [
     0.3926076340523761,
     1.3876019770480674
    ],
    "rotation_error_mean_max_deg": [
     52.974266077983124,
     178.5140886143564
    ]
   },
   "joint": {
    "status": "PRISMATIC",
    "reason": "revolute degenerate (axis/part scale 53.7, rank deficient False); cannot veto prismatic",
    "PRISMATIC": {
     "axis_error_deg": 1.1769853905898457,
     "q_rmse": 0.005557863524666428
    },
    "REVOLUTE": {
     "axis_error_deg": 89.11816873372295,
     "q_rmse": null
    }
   },
   "support": {
    "MOVING_CORE": {
     "count": 3383,
     "moving_fraction_labelled": 0.8706896551724138
    },
    "MOVING_JOINT_COMPATIBLE": {
     "count": 1086,
     "moving_fraction_labelled": 0.49800796812749004
    },
    "STATIC_COMPATIBLE": {
     "count": 16948,
     "moving_fraction_labelled": 0.0631424375917768
    },
    "AMBIGUOUS": {
     "count": 17205,
     "moving_fraction_labelled": 0.2192887931034483
    }
   }
  }
 }
}
```

