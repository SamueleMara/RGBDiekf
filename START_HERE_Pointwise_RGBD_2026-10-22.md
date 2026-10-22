# Pointwise bidirectional RGB-D reconstruction — 22 October 2026

## For Codex

Start with **`CODEX_EXACT_POINTWISE_BIDIRECTIONAL_DISCOVERY_2026-10-22.md`**. That file is the normative, actionable implementation contract. Implement it in an isolated point-only pipeline alongside `RigEuler3DGS` without altering the proven 3DGS static training/rasterizer. Consult **`Pointwise_Bidirectional_RGBD_Theory_2026-10-22.pdf`** for complete dynamical derivations and mathematical limitations, and **`Pointwise_Bidirectional_RGBD_Implementation_2026-10-22.pdf`** for a printable version of the full execution and debug contract.

Do not substitute group-motion discovery, fixed-K clustering, global IEKS or 12D/SE(3) point states for the requested 6D per-point estimator.

### Required algorithm

- One initial vectorized full forward IEKF `0 -> T`, for **all** frame-0 valid masked foreground points.
- Freeze all those state and covariance histories.
- Reverse scan `T -> 0`, pick the **first** raw valid unmatched RGB-D observation, initialize exactly one new 6D point state *at its discovery frame*.
- Estimate that new point's motion **only**: from its discovery frame backward to zero and separately forward to T. One anchor observation, not two; actual RGB-D correspondence updates where observed; predictions with uncertainty where occluded.
- Append that point's full computed trajectory to SOLVED immediately; old tracks remain bitwise identical.
- Update observation ownership and geometric matching, restart the reverse scan from T, repeat until none can be newly seeded (with explicit failure retry safeguards).
- No moving/static rigid part clustering at this stage. Meaningful real-data numerical and visual debug is mandatory.

### Contents of this package

1. `CODEX_EXACT_POINTWISE_BIDIRECTIONAL_DISCOVERY_2026-10-22.md` — definitive Codex task and tests.
2. `Pointwise_Bidirectional_RGBD_Theory_2026-10-22.pdf` — derivations and precise interpretation of the two-sided six-state IEKF.
3. `Pointwise_Bidirectional_RGBD_Implementation_2026-10-22.pdf` — complete printable Codex execution and integration contract.
4. `Pointwise_Bidirectional_RGBD_Theory_2026-10-22.md` — editable theory source.
5. `Pointwise_Bidirectional_RGBD_Implementation_2026-10-22.md` — editable implementation source.

All equations and debug requirements in this package supersede the conflicting fixed-K point-SLAM materials from earlier revisions.
