# CLAUDE CODE — STEP 1 ONLY: RigEuler motion-coherence supernode parity and 192/192 failure localization

## Objective and scope

Work in `/home/samuelemara/Joint3DGS/RGBDekf`. The authoritative source is **the pinned Git commit** `481b2cf2d5602932286948cf14114f9caee4de3c` from `https://github.com/SamueleMara/RigEuler3DGS`, **not its modified working tree**.

The current v2 blade forward result says `192/192` motion representatives are `MOVING_RIGID`, expanding to `34,369/34,369` endpoint points, despite a genuinely static base. Find the **first stage** that causes this result using **identical saved Phase-1 trajectories, representative membership, and graph inputs**. Do not guess that Phase-1 lag is the cause; earlier runs had similar lag but a useful core.

**This task is diagnostic FIRST; make only a pinpoint coherence-only fix if the exact offending operation and a reference-parity test prove one is required.** Stop and report if the result is explained by input trajectory quality rather than a code mismatch.

### Strict exclusions

- DO NOT rerun/retrain Phase 1 or change its point states, observation model, renderer, IEKF, process/covariance, or saved outputs.
- DO NOT modify the computational Phase-1 supernode partition or its correction operator `G/G^T`. That is a **separate future task**. Here only inspect its membership as necessary to disambiguate artifacts; coherence representatives must be their own partition.
- DO NOT modify Phase 2, rigid SE(3) filtering, joint geometry/selection, support completion, backward merge or viewer calculations.
- DO NOT use GT labels, joint parameters, object names, hand-chosen moving fractions, or tuned motion thresholds in the inference.
- DO NOT change the reference extractor's settings to make blade score well or weaken regression tests. Do not edit pinned RigEuler source.
- DO NOT let the local `rigeuler_ref/` vendor copy silently stand in for the pinned Git blob without checking its source/hash and any modifications.
- No hours-long GPU runs; perform this on saved arrays, preferably CPU in float64. Do not start parallel runs or monitoring tasks.

## Source mapping (read the actual code before running anything)

Reference:

1. `estimator/supernodes.py`: `build_supernodes(means, num_supernodes=192, seed=0)`; deterministic farthest-point seeding + Lloyd grouping, canonical immutable membership, `SupernodePartition`.
2. `estimator/motion_coherence_support.py`: `supernode_trajectories`, `canonical_knn_graph`, `edge_strain`, `run_motion_coherence`, `propagate_to_gaussians`.
3. `estimator/coherence_extraction.py`: `candidate_hierarchy`, `edge_distributions`, `to_reference_frame`, `cross_validated_rigidity`, `select_moving_component`, `refine_from_core`, `extract_moving_component`.
4. `tools/run_motion_coherence_support.py`: its production flow and defaults: **coherence node count 192**, `coherence-knn=12`, `cv-folds=5`, extractor `v2`; independent of the **Phase-1 IEKF grouping** `--iekf-supernode-size=64`.

Read current local code, particularly `pointwise_rgbd/image_coherence.py`, `pointwise_rgbd/supported_coherence.py`, `pointwise_rgbd/supernodes.py`, `pointwise_rgbd/image_phase1.py`, plus pipeline handoff in `tools/run_rgbd_image_iekf.py`. The uncommitted local code may differ from GitHub RGBDiekf. Document exactly which function each stage actually invokes.

## Stage A — locate and freeze the saved inputs

Use the existing finished **forward Phase-1** outputs, e.g. (verify before loading):

```
/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/image_iekf_v2_rigorous_20261024/first/phase1/
```

Expected artifacts include `partition.npz`, `state.npy`, and `point_observed.npy`; do not assume exact keys until inspected. Also use the saved coherence artifacts under `first/coherence/` if present. The endpoint catalogue should have 34,369 points; check actual input sizes, do not force those counts.

Produce a read-only manifest with:

- absolute path, file SHA256, dtype, shape for XYZ trajectories, canonical endpoint XYZ, membership, representative centers, timestamps and observation-support arrays;
- direction and original-frame temporal order; for a backward diagnostic the canonical geometry comes from frame 59, even if stored trajectories are ordered chronologically;
- number of individual 6D point states, IEKF *computational groups*, coherence *representative nodes*, and each group's size distribution **as separate values**;
- whether the trajectory fed to coherence was raw filtered XYZ, zero-order-held/image-supported XYZ, visibility-weighted XYZ, or something else;
- exact effective code commit/blob hash and parameters for both implementations.

**For this first comparison, run forward only.** Do not use the backward arrays as a substitute for canonical forward geometry.

## Stage B — make an input-identical extractor replay

Construct a single immutable input bundle `parity_inputs.npz` from saved Phase-1 arrays and the actual v2 representative data. No edits or re-filtering; no GT. Save its hashes, then call both implementations on this *exact same bundle*. To isolate partitioning from extraction also perform a second replay in which RigEuler **builds the representative partition itself**, using its canonical-endpoint `build_supernodes(..., num_supernodes=192, seed=0)`.

Two controlled comparisons:

**B1 — identical partition (isolates graph/strain/extractor).** Supply both implementations the exact same 192-node `membership` and the exact same complete point trajectories, same deterministic aggregation weights and temporal support policy. Compute the node trajectories **once**, then pass the *same node-trajectory tensor and same edge/strain arrays* into both extractors. Report how local behavior diverges from the pinned extractor, if it does.

**B2 — reference canonical partition (isolates geometry/partition drift).** Build reference 192-node partition from canonical XYZ. Recompute node trajectories and canonical fixed kNN graph with the pinned functions; run both extractors on this bundle. Compare partition differences (not merely node IDs, which can be permuted). Match groups by canonical point-set overlap, and report membership purity/size and graph differences without GT.

For actual pinned runner parity, call:

```python
# PSEUDOCODE: adapt to actual saved array types and imports.
part = ref_build_supernodes(canonical_xyz, num_supernodes=192, seed=0)
sid = part.supernode_id.long().cpu()
sn = ref_supernode_trajectories(positions_original, sid)  # reference uses uniform weights by default
node_trajectories = sn["trajectories"][:, sn["valid"], :]
graph = ref_canonical_knn_graph(node_trajectories[0], k=12)
strain = ref_edge_strain(node_trajectories, graph["edge_i"], graph["edge_j"], times=times)
reference_unrecruited = ref_extract_moving_component(
    node_trajectories, graph["edge_i"], graph["edge_j"], strain["C"],
    node_trajectories.shape[1], folds=5, refine=False, verbose=False)
reference_recruited = ref_extract_moving_component(
    node_trajectories, graph["edge_i"], graph["edge_j"], strain["C"],
    node_trajectories.shape[1], folds=5, refine=True, verbose=False)
```

**Critical caveat:** the pinned reference runner builds the kNN graph from its **node trajectory at chronological time 0** via `run_motion_coherence`, while the representative **membership** is built from canonical endpoint XYZ. These agree in the forward anchor but need not agree in backward processing. Reproduce the actual pinned behavior; do not silently change that convention.

If the local implementation has a different trajectory/support policy, reproduce that policy as an explicit **third axis** of comparison, and label it as a *methodological departure* rather than quietly attributing output changes to the selector. Run a 2×2 matrix (reference/local extractor × reference/local trajectory policy) while holding partition, graph, and numerical parameters identical; do not mix factors.

## Stage C — instrument every component-selection boundary

For **both** extractors and **both** `refine=False`/`True`, persist all of the following, never only `final_moving_nodes`:

1. `candidate_hierarchy` output: count, node indices, birth/death/persistence, hierarchical level, size stratum and selected evaluation pool (including candidates omitted by `max_candidates=48`).
2. Per-candidate internal/boundary edge counts and quantiles of `C`; *selected seed must have nonzero boundary edges* under the pinned `select_moving_component` rule.
3. Held-out rigid comparison: `eta_cv`, `rss_rigid_cv`, `rss_identity_cv`, `delta_rss_cv`, `delta_rss_cv_per_node_frame`, and `relative_motion_ratio`, both world-frame and **complement-relative**. Include `delta_bic` as a diagnostic, not an invented gate.
4. Selected **seed candidate ID, exact seed-node IDs, node count**, which candidate won, and WHY according to its decision rule.
5. `refine_from_core` recruitment step **separately**: for each nonseed node record `E_i^M` and `E_i^I` from the core-only rigid trajectory; record `E_i^M - E_i^I`, exact recruited IDs, rejected IDs, and final members. Pinned reference recruits when `E_i^M < E_i^I`, not with a tunable threshold.
6. Expansion from representative-node labels to point-level labels: counts, membership map, number of unassigned points and any accidental index-space remapping. Detect whether `node_ids` (valid node indices) are incorrectly treated as original partition IDs.
7. Graph connectivity and connected components; coverage and confidence of the underlying raw vs predicted point trajectories, for **diagnosis only**, not to inject GT or tune selection.

The core structural invariant to check is:

```
refine=False: seed has a nonempty complement AND >0 boundary graph edges.
refine=True: final set may be larger; all growth must be explained by an explicit list of recruits.
```

**Do not assert that the final 192/192 alone proves the selector chose a full-graph candidate**: the pinned code first selects a proper seed and can expand it in `refine_from_core`.

## Stage D — exact parity assertions and controlled ablations

Implement one deterministic offline test script, e.g. `tools/audit_coherence_parity.py`, and focused tests under `tests/`. It should run without importing the GPU renderer or launching the production runner. Tests must:

- Assert the same canonical point catalogue and chronological trajectory ordering; never use Phase-1 computational group IDs as 192 representative node IDs.
- Compare node-trajectory arrays, edge sets (as canonical undirected sorted index pairs), edge lengths, `C`, candidate pools, seed, recruited and final memberships **stage by stage**. Allow only measured numerical floating-point tolerances for continuous quantities; compare integer/index sets exactly after deterministic relabeling.
- Verify that `G/G^T` is not touched by this task, and that no source code or saved Phase-1 file changed (hash check).
- Run extractor with `refine=False` and `refine=True` and report which call first turns the selection into 192/192.
- Replay two earlier saved trajectories, if accessible (old successful image-IEKF forward run and latest v2 forward), through the **same pinned 192-representative construction and extractor**. This answers whether the difference follows the input motion trajectories or the extractor implementation. Do not compare runs with different partitions without controlling that factor.
- Give a toy two-rigid-body canonical scene (one static and one moving) as a basic invariant test; if it fails, use it to localize the issue, not to tune blade-specific thresholds.

Do not require any particular GT F1 or a predetermined moving fraction to pass the parity audit. Returning `NO_MOVING`/`UNRESOLVED` is legitimate when observations provide insufficient evidence.

## Stage E — decision tree and permitted changes

Evaluate these explanations **in order**, with evidence:

1. **Group geometry mismatch**: independent 192 partition not used, wrong source canonical geometry, wrong member IDs or incorrect representative node averaging.
2. **Trajectory policy mismatch**: raw predicted drift vs observation-supported trajectories or swapped chronological direction.
3. **Graph or strain mismatch**: edges rebuilt per frame, different kNN, reference-length formula/weighting difference, wrong `C` or invalid node map.
4. **Candidate-pool mismatch**: altered hierarchy, persistence stratification or omission of relevant candidates.
5. **Seed-selection mismatch**: missing complement-relative validation, missing `boundary_edges>0` requirement, size-biased ranking, or GT-dependent selection.
6. **Recruitment mismatch**: seed is acceptable but `refine_from_core` recruits static nodes; inspect its explicit `E_M - E_I` and the trajectories producing those values.
7. **Input-quality failure shared with reference**: both selectors fail on v2 saved trajectories when held strictly identical. In this case STOP after reporting evidence; any corrective change to Phase-1 motion/uncertainty is the next separately scoped task.

Only fix a verified **translation/porting bug in stages 1–6**, with a minimal change and before/after input-identical regression. Do not redesign the extractor or change reference semantics in this task.

## Required deliverables

- `reports/coherence_reference_parity.md`: a measured, line-by-line comparison, with root-cause verdict and *first divergent operation*.
- `reports/coherence_reference_parity.json`: machine-readable full parameter/manifests/candidate/seed/recruitment stats.
- `reports/coherence_reference_parity.csv`: per-candidate and per-node `E_M`, `E_I`, `E_M-E_I`, selected/recruited/ambiguous membership, source labels (not GT).
- `tools/audit_coherence_parity.py`: rerunnable read-only command with explicit input paths, source commit, no Phase-1 rerun.
- Focused tests demonstrating a reproducible failing case before any surgical fix and the result after.
- A `git diff --stat` plus exact list of touched files and SHA256 proofs that the IEKF/renderer/covariance/Phase-1 artifacts were not modified.

**Conclude with one of exactly three diagnoses:**

- `REFERENCE_PARITY_RESTORED`: a proven port mismatch was corrected, and parity tests now pass.
- `REFERENCE_MATCHES_BAD_INPUT`: reference and local algorithms behave alike; source trajectories/observability are the failing dependency; no extractor change is justified.
- `UNRESOLVED`: evidence insufficient or tests disagree; preserve inputs, list failing checks and the next *one* diagnostic.

Do not say `COMPLETE` simply because a run ends or outputs a plausible support. Do not proceed to IEKF group-basis/covariance, rigid Phase 2, joint inference, or support correction until this step is reported and reviewed.
