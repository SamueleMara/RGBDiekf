> **Correspondence correction:** the coverage/ownership conflation in this earlier version is superseded by [AUDIT_CORRESPONDENCE_2026-10-22.md](AUDIT_CORRESPONDENCE_2026-10-22.md) and [the measured real-data results](reports/correspondence_2026-10-22_results.md).

# Implementation audit: pointwise bidirectional discovery

Audit date: 22 October 2026. Parent repository HEAD:
`0b8158abc812e158037030a5bdd2e3f4939e76f9`. The RGBDekf directory is untracked
in that parent checkout, so the commit alone cannot identify these new bytes.
Every experiment records SHA256 hashes of the actual pointwise source files.
Unrelated sibling changes/deletions were preserved.

All five requested documents were read completely before implementation,
including both PDFs (extracted locally for reading). The normative contract is
`CODEX_EXACT_POINTWISE_BIDIRECTIONAL_DISCOVERY_2026-10-22.md`. It supersedes the
previous fixed-K/group/SE(3)/RTS architecture. The following files are newly
implemented rather than wrappers around that estimator.

| Responsibility | Implementation and audited convention |
|---|---|
| Configuration | `pointwise_rgbd/config.py:42`: approved numerical guesses, units and explicit local association policy |
| Input | `pointwise_rgbd/dataset.py:8`: streamed native RGB-D/masks, original IDs, actual timing, calibration and distortion guard |
| Raw samples | `pointwise_rgbd/observations.py:22`: M AND raw V AND finite positive metric Z; RGB only at those pixels |
| Initialization | `pointwise_rgbd/observations.py:31`: every frame-zero camera observation retained, no downsampling or optional duplicate fusion |
| State | `pointwise_rgbd/state6.py:21`: anchor posterior `[backprojected XYZ,0]`, propagated anisotropic position covariance, nonzero velocity prior |
| Process | `pointwise_rgbd/process6.py:5`: signed F and integrated forward/reverse Q; full covariance carry-forward |
| Measurement | `pointwise_rgbd/projection.py:5`: `(u,v,Z)` and analytic 3x6 H; no RGB/XYZ duplicate factors |
| Association | `pointwise_rgbd/association.py:20`: finite spatial kNN, ray/depth innovation tests, RGB affinity, deterministic exclusive raw ownership |
| IEKF | `pointwise_rgbd/filter_iekf.py:15`: fixed predicted prior, nonlinear MAP energy line search, final-linearization information inverse |
| Initial pass | `pointwise_rgbd/controller.py:14`: exactly one vectorized forward pass, file-backed histories |
| Frozen matching | `pointwise_rgbd/controller.py:76`: reverse geometric bookkeeping with no state/covariance updates |
| New point | `pointwise_rgbd/discovery.py:10`: one-point optimizer, independent backward/forward anchor copies, source used once |
| Controller | `pointwise_rgbd/controller.py:95`: first reverse unmatched seed, immediate append, model-version failure registry, restart at last frame |
| Storage | `pointwise_rgbd/store.py:16` and `:96`: transactional raw ownership, temporary seed reservations, immutable measurement refs and separate append-only tracks |
| Diagnostics | `pointwise_rgbd/diagnostics.py:23`, `visualization.py:19`: recorded numerical quantities and actual calibrated RGB-D/3D previews |
| Entry point | `tools/run_pointwise_rgbd_discovery.py`: independent CLI, live lock, source/content/model checks on resume, preserved failures |

Line numbers identify the audited implementation at delivery and can shift with
future edits. The source hashes in each run are the precise version reference.

## Verified reuse and exclusions

`rgbd_common/dataloader.py:174` and `timing.py:116` provide native directory/ZIP,
metadata-root resolution, camera conventions and actual acquisition timing.
`dataloader.py:18` retains original raw validity and binary masks.
`rgbd_common/config.py:44` reads the previously approved numerical configuration;
only the relevant translational, sensor and numerical terms are used. Its old
symbolic group-model-prior warning is not an execution dependency: the current
algorithm has no group count or global model-selection criterion.
`rgbd_common/backends/torch_cuda.py` supplies deterministic Torch/device memory
utilities. `rgbd_common/projection_cuda.py:10` supplies only the Camera data
structure; rigid transport functions are not used. `rgbd_common/array_store.py:8`
provides transactional compressed arrays in SQLite. Shared modules were not
changed for this pipeline.

The sibling `RigEuler3DGS` source audit inspected
`tools/run_iekf_scene.py:236`, `estimator/state.py:232`,
`estimator/process.py:113`, `estimator/covariance.py:97`,
`estimator/sparse_iekf.py:424`, `estimator/masked_observation.py:115`, and the
debug/plotting and pipeline-map documents. Only transferable data, covariance,
fixed-prior IEKF and real-data visualization conventions were retained.
No 3DGS trainer/rasterizer, Gaussian state, alpha/background residual or
ground-truth joint-loading path is imported into the new estimator.

The supplied narrative says the sibling bootstrap resets P0 each frame. The
actual inspected `tools/kpart_pipeline.py:1767,1791` already carries P forward
unless `phase1_covariance_reset` is explicitly enabled. Its bootstrap still
requests `use_supernodes=True` at line 1750. That architecture is not reused.
This audit reports the current bytes, not the historical narrative as fact.
Neither sibling implementation nor existing result directories were modified.

## Equations and uncertainty interpretation

For positive physical interval d, F has identity diagonal blocks and signed
`F_xv = +/- d I`. With white acceleration intensity Qc and direct position
diffusion Qx:

```text
Q_forward = [[d^3 Qc/3 + d Qx,  d^2 Qc/2],
             [d^2 Qc/2,        d Qc    ]]
Q_reverse = [[d^3 Qc/3 + d Qx, -d^2 Qc/2],
             [-d^2 Qc/2,       d Qc    ]]
P_minus = F P_previous_posterior F^T + Q
```

The reverse is the reoriented stochastic process factor. It does not negate
physical velocity, reuse a forward filter backward, or claim an RTS posterior.
For fixed actual measurement references, IEKF information is
`P_minus^-1 + sum(H^T R^-1 H)`; its right-hand side includes the displacement
from the same predicted prior at every iteration. Accepted steps decrease the
actual nonlinear fixed-association energy. The undamped final-linearization
information inverse is the reported local Gaussian covariance. Cholesky
failure is explicit; there is no jitter, clipping or silent P0 reset.

Cross-position/velocity covariance is retained. Cameras observing one point
share its prediction error: joint multiview NIS uses the equivalent 6D
information identity and is tested against a dense tiny reference innovation
covariance. Distinct point blocks are a conditional approximation with fixed
calibration and chosen raw associations. No exact marginalized cross-point,
association or shared-calibration covariance is asserted.

`configs/example_uncalibrated.json` supplies the original translation diffusion
and velocity prior, sensor covariance and unit-pixel response. The new theory's
white-acceleration integration is applied explicitly rather than copying its
old 12D discrete process. Effective initial values and units are recorded in
the manifest. The 1% gate tail and eight-neighbor candidate family come from
`configs/pointwise_association.json`, an explicitly declared implementation
policy, not a paper-supplied numerical value or independently calibrated model.
No noise hyperparameter calibration is claimed.

## Ownership, memory and failure boundaries

Initial frame-major histories have shapes `[K,N0,6]` and `[K,N0,6,6]` on disk.
Only a frame's blocks and bounded CUDA chunks are processed at once. There is
no `[6N,6N]`, all-frame pixel catalogue or `[N,K,C,H,W]` allocation. Initial
measurement references are stored separately and closed at freeze. Appended
tracks are independent NPZ records with source IDs, immutable factors and full
diagnostic candidate records. Read-only memory maps and per-frame SHA256
digests guard initial state/covariance; tests hash old bytes across insertion
and resume. The hot-path immutability assertion also checks file signatures
and append-only track hashes rather than rereading 21 GB after every point.

SQLite transactions bind new raw claims to track registration and model
version. A pre-transaction interrupted track file can remain orphaned;
monotonic allocated IDs are never reused and only registered tracks load on
resume. Temporary reservations are cleared on resume and on numerical failure.
Completed frame checkpoints overwrite an interrupted uncommitted frame.
Committed track diagnostics can be regenerated from their saved records.

Owner kind 1 denotes an actual filter factor; kind 3 an initialized anchor;
kind 2 denotes compatible geometric coverage only. Kind 2 never modifies an
old observed flag or historical filter factor. It is counted separately from
assimilated measurements. Already owned raw IDs and frozen point/camera
assignments are never overwritten. All candidate sources remain eligible;
failed seeds are suppressed only at the same model version.

Computational limits remain real. Exact schedule cost grows with the number of
new seeds and repeated scans. No unlimited full-discovery runtime bound is
claimed. Local finite-k association is not a proof of true correspondence or
observable physical motion. Empirical match/nonmatch distributions and spacing
are reported; failure of separation is not hidden by a geometric novelty
threshold. Unrectified distortion is rejected instead of approximated silently.

## Validation

See `reports/pointwise_2026-10-22_validation.md` for the criterion-by-criterion
matrix, failed trial and actual completed fractions. Historical 12D pipeline
unit tests or initialization-only outputs do not count as this validation.
