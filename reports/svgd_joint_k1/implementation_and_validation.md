# Joint K = 1 static/moving support + shared rigid motion (SVGD over motion hypotheses) — implementation and validation

Generated 2026-11-04 11:35 from `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/svgd_joint_k1_batch_20261104_0034`. Spec: `CODEX_RGBDiekf_SVGD_JOINT_RIGID_SUPPORT_K1_2026-11-03.md`.

| job | state | F identity | F SVGD best | F MAP | seeds (SVGD) | MAP-control seeds | oracle | GT − identity | best − GT | seed precision / recall | MAP-control precision / recall | rho | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic_static | DONE | 126194.2 | 126163.0 | 126179.3 | 0 | 0 | PASS_PHYSICAL_GATE (static control: no hallucinated seeds) | 0.0 | -14.9 | — / 0.000 | — / 0.000 | 0.759 | 37 s |
| synthetic_prismatic_textured | DONE | 102509.0 | 102477.9 | 102494.3 | 0 | 0 | OPTIMIZATION_FAILURE | 14700.1 | -14714.8 | — / 0.000 | — / 0.000 | 0.801 | 49 s |
| synthetic_prismatic_uniform_colour | DONE | 104013.6 | 103982.3 | 106333.9 | 0 | 173 | OPTIMIZATION_FAILURE | 2535.3 | -215.0 | — / 0.000 | 1.000 / 0.715 | 0.724 | 70 s |
| synthetic_hinge_0p9rad | DONE | 109866.9 | 109835.6 | 112966.1 | 0 | 191 | OPTIMIZATION_FAILURE | 4756.0 | -1656.7 | — / 0.000 | 0.942 / 0.744 | 0.779 | 71 s |
| USB_100109_10f | DONE | 7033057.3 | 7113397.5 | 7908391.4 | 0 | 19884 | — | — | — | — / — | — / — | 0.007 | 17368 s |

Probabilities are `UNCALIBRATED_APPROXIMATION` (Gate 4 not run). Prefix runs are `PREFIX_DIAGNOSTIC`. GT only in the post-hoc oracle.

See `equation_to_code_contract.json`, `theory_deviation_register.json`, `root_cause.md`, `unit_test_results.json`.
