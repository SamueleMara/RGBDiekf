# Grouped RGB-D implementation status

The new isolated implementation is **incomplete and stopped at Gate B**, as the
contract requires. It does not replace the reproducible independent pointwise
baseline. Recovery is paused. Do not start a full-source reconstruction.

[Source audit, all-section map, actual tests, real prefix results and commands](reports/grouped_2026-10-22_validation.md).

The implemented core is a 12D spatial-left group pose/twist IEKF with exact
shared rigid generation of point XYZ, source-connected geodesic supernodes,
signed-time continuous covariance propagation, fixed-prior manifold correction,
chronological histories, actual raw measurement identities, and a streamed
Three.js diagnostic viewer. Both bounded real prefix directions were executed.

The measured joint-boundary regression currently produces a disconnected
training proposal and loses training observability later in time. It is rejected,
not promoted to physical rigid support. Coherence, shared Phase 2 and joint
reduction are not yet implemented or accepted. Section 8's policy text now
reflects the user's candidate-specific degeneracy clarification.

The grouped launcher accepts optional `--fps`, `--dt-seconds`, and
`--timestamps-file` overrides through the existing shared timing resolver, with
provenance and disagreement reporting. Metadata timing remains automatic when
no override is supplied. A timestamps file must cover the full source sequence,
even when `--frame-limit` selects a test prefix.

The actual `partnet_subset_fixed_render_linear/blade_103706` scene's
`metadata/job_config_resolved.yaml` has `fps: null`, `duration_sec: null`, and
`video_fps: null`. Automatic resolution currently finds no acquisition timing.
Running it requires the intended physical timing supplied explicitly; video
playback rate is not evidence of acquisition timing. Missing timing now returns
exit 2 with a concise message and preserves the full search in `failure.log`.
Choose a fresh output after a failed attempt. This input fix does not resolve
the Gate B implementation failure or enable full reconstruction.
