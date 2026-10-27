# Step 1: RigEuler coherence supernode parity and 192/192 failure localization

**Diagnosis: `REFERENCE_MATCHES_BAD_INPUT`.**

| Item | Value |
|---|---|
| Specification | `CLAUDE_STEP1_RIGEULER_COHERENCE_SUPERNODE_PARITY_AUDIT.md` |
| Reference | RigEuler3DGS `481b2cf2d5602932286948cf14114f9caee4de3c`. Loaded from the Git blobs at run time (`tools/audit_coherence_parity.load_pinned`), not from the modified working tree or the vendored copies. |
| Inputs | Saved forward arrays only. No Phase-1 rerun, no GPU, float64. Ground truth is not read by the audit. |
| Machine-readable results | [coherence_reference_parity.json](coherence_reference_parity.json) and [coherence_reference_parity.csv](coherence_reference_parity.csv) (per-node `E_M`, `E_I`, `E_M−E_I`; per-candidate statistics) |
| Command | `CUDA_VISIBLE_DEVICES="" python tools/audit_coherence_parity.py --v2 <…/image_iekf_v2_rigorous_20261024> --v1 <…/image_iekf_final_20261024> --out <…/coherence_parity_audit_20261025>` (106 s) |
| Tests | `python -m unittest tests.test_coherence_parity`: 4/4 pass |

## Stage A: frozen inputs

| Item | Value |
|---|---|
| Phase-1 arrays (v2 forward) | `state.npy` sha256 `a5349a30…` `(60, 34369, 6)` float64; `point_observed.npy` `bb39cd6e…` `(60, 34369)` bool; `partition.npz` `fdf57175…`; `covariance.npy` `6e0efc68…` |
| Immutable bundle | `parity_inputs.npz` sha256 `2c1c3c14…` |
| Direction and order | Forward. Trajectories are chronological; canonical XYZ = `partition.xyz` = frame 0 (checked exactly). |
| Individual 6D point states | 34,369 |
| Phase-1 computational groups | 537 (size min/median/max 1/70/115). Overlap purity against the coherence nodes is median 0.39: a separate partition, as required. |
| Coherence representative nodes | 192 (size min/median/max 32/186/291) |
| v2 trajectory policy | Image-supported node displacements with node-frame weights (`supported_coherence`). Not raw filtered XYZ, not zero-order hold. |
| Pinned blobs | `supernodes.py` `cc96d0b7` (sha256 `2f0a0915…`), `motion_coherence_support.py` `cb2d1903` (`65e7cec9…`), `coherence_extraction.py` `37c8200f` (`5931ee32…`, identical to the vendored copy) |
| Unchanged after the audit | All 15 watched files (Phase-1 arrays plus IEKF, renderer, covariance, grouping and coherence sources) have identical sha256 before and after. `G/G^T` (`pointwise_rgbd/supernodes.py`) was not touched. |

## Stages B to D: input-identical comparisons, stage by stage

| Check | Result |
|---|---|
| v2 saved 192 partition vs pinned `build_supernodes(mu0.float(), 192, seed=0)` | **Exactly equal** (same IDs) |
| Local graph (`image_coherence.canonical_knn_graph`) vs pinned `canonical_knn_graph(traj[0], 12)` | Equal edge sets (1,269 edges, 1 connected component) |
| Saved v2 node trajectories vs supported-policy replay | Equal (atol 1e-12) |
| v2 production replay vs saved labels | Labels identical; final 192/192 reproduced |
| Local extractor (ω ≡ 1) vs pinned extractor, identical raw inputs, both partitions, refine False and True | Labels, pool IDs, selected candidate, seed and final set all **identical** |
| Raw vs supported node trajectories (time 0) | Identical at time 0; the maximum node difference over time is 0.127 m (the policies differ only off-gauge) |

### The 2×2 matrix

The partition, the graph and all numerical parameters are held identical across all four cells. The results are the same for both partitions, B1 (the v2 partition) and B2 (the pinned partition), because the two partitions are equal.

| Extractor | Trajectories | refine=False (seed) | refine=True (final) | Points |
|---|---|---|---|---|
| pinned | raw | 183 | 192 | 34,369 |
| pinned | supported | 181 | 192 | 34,369 |
| local | raw | 183 | 192 | 34,369 |
| local | supported (v2 production) | 183 | 192 | 34,369 |

### Replay of the earlier trajectories through the identical pinned pipeline

The catalogue is identical between the two runs (checked).

| Input trajectories | Seed | Final |
|---|---|---|
| v1 image-IEKF run (`image_iekf_final_20261024`) | 6 nodes | 20 / 192 |
| v2 run | 183 nodes | 192 / 192 |

The outcome follows the input trajectories, not the implementation.

## First divergent operation

The selection boundaries were instrumented for the pinned extractor on the v2 raw trajectories:

1. **Hierarchy.** 161 candidates; the stratified evaluation pool holds 42 (max 48).
2. **Pool content.** The moving material is present in the pool, but only as 3–4-node fragments (mean final node displacement 0.59 / 0.54 / 0.43 m). On v1 the same construction yields 5- and 6-node moving candidates.
3. **Qualification.** The pinned `cross_validated_rigidity` marks a candidate with fewer than `max(4, folds = 5)` nodes as degenerate. No moving fragment can therefore qualify, and the 30 qualified candidates contain no moving side.
4. **Seed selection.** Among the qualified candidates, the pinned rule names the moving side by world-frame ΔRSS per node-frame. It selects the 183-node candidate: 96 boundary edges, relative η 0.982, relative-motion ratio 0.676, world ΔRSS per node-frame 1.3e-4. This candidate qualifies because it moves relative to its 9-node complement, which is the moving body. Both structural invariants hold: the complement is non-empty and there are more than 0 boundary edges.
5. **Refinement.** `refine_from_core` recruits exactly the 9 remaining nodes, each with `E_M < E_I` under the core's near-identity trajectory, which gives 192/192. All growth is explained by the explicit recruit list.
6. **Expansion to points.** `node_ids` is the identity (all 192 nodes valid), so there is no index-space remapping. Points: 32,796 (seed) and 34,369 (final).

So the first operation that turns the result wrong is (4), seed selection. It is caused by (2)–(3), and that happens because the v2 trajectories of the moving part are non-rigid enough that single linkage fragments the part below the 5-node cross-validation minimum. Refinement (5) only completes a wrong seed.

**Decision tree.** Explanations 1–6 (geometry, trajectory policy, graph/strain, candidate pool, seed selection, recruitment) were each tested and found identical to the reference on identical inputs. Explanation 7 holds: the reference fails identically on the v2 saved trajectories.

## Toy two-rigid-body invariant

A 40-node partition of a static plate and a moving plate (5 cm apart):

* The pinned and local extractors give identical labels.
* All 5 part-majority nodes are selected.
* 2 mixed boundary nodes (48% and 40% part points) are recruited with `E_M < E_I`.
* No purely static node is selected.

## Permitted changes in this step

None: no port mismatch exists. No extractor, partition, `G/G^T`, Phase-1, renderer, IEKF or covariance code was modified.

Files added: `tools/audit_coherence_parity.py`, `tools/write_parity_debug_pages.py` (RigEuler-style coherence pages for the pinned replay on v1 and v2 trajectories), `tests/test_coherence_parity.py`, and this report with its JSON and CSV.

**Next separately scoped task:** the Phase-1 motion quality of the moving part (the non-rigid lag of the v2 trajectories). That covers the solver budget and step control, the overconfident carried covariance, and the equilibrated exact covariance.
