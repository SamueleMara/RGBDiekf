# Run archive (24 October 2026)

All runs are under `/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/`. Each blade run contains `manifest.json` (all settings) and `source_snapshot/` (the exact sources it ran). No run was overwritten.

| Folder | Code / configuration | Outcome |
|---|---|---|
| `image_iekf_final_20261024` | v1: LM + pyramid solver, mixture likelihood, RigEuler block information, visibility×precision coherence weights, χ²/far-axis/excitation joint gates | COMPLETE (baseline) |
| `image_iekf_v2_failed_exact_covariance_frame11_20261024` | v2, `restricted_joseph_structured` | FAILED: `B` not PD at forward frame 11 |
| `image_iekf_v2_failed_assumed_density_divergence_20261024` | v2, `restricted_joseph_marginals` | FAILED: covariance blow-up (1e18 m² by frame 14); `B` not PD at frame 20 |
| `image_iekf_v2_rigorous_20261024` | v2, `rigeuler_block_information`; joint DM tests with the stationary-HAC variance (before the detrending fix); support with contradiction-only labels (before the supported-hypothesis rule) | COMPLETE; coherence selects the whole scene (forward) |
| `image_iekf_v2_synthetic_superseded_*` | Synthetic runs stopped when a stage changed | partial (superseded) |
| `image_iekf_v2_synthetic_20261024` | v2, block information, detrended DM; support module loaded before the supported-hypothesis rule | in progress / superseded for support |

**Open diagnoses**, in the audit's order:

* Covariance PD loss: `tools/diagnostics/covariance_blade_dense.py`.
* Coherence replay: `tools/diagnostics/coherence_replay.py`.
