# Authoritative user correction — 24 October 2026

Production uses ALL available cameras at EVERY frame in both independent endpoint directions. The previous camera sets (0/3/6 fit, 1/4/7 select, 2/5 assess) must not control filtering, rigid fitting, joint selection or support completion.

Sequence: individual free 6D point IEKF with grouped correction coordinates → RigEuler trajectory coherence → one shared rigid SE(3) trajectory → revolute/prismatic joint reduction → world-coordinate forward/backward comparison → joint-guided completion of existing uncertain endpoint points.

RigEuler revision 481b2cf2 cross-validates SUPERNODES for out-of-sample rigid prediction; it does not hold out cameras. Preserve that mechanism. Both joint candidates fit and score the same complete multiview RGB-D factors, with existing covariance treatment and per-camera diagnostics. Use endpoint-relative joint transport for support completion only once joint information is reliable.

Do not add estimators, camera-validation architectures, physical supernode splitting or unrelated theoretical stages. Reuse completed trajectories only if their estimator inputs are identical. Camera-split states/covariances are not mathematically reusable under all-camera filtering.

Update tests, manifests and viewer; assert production processes every available camera. Preserve prior files and rerun changed filtering/optimization. Reserved-camera residuals are superseded, not production performance. Re-evaluate computational group settings using all cameras under otherwise identical settings.
