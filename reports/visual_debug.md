# Visual evolution, convergence and rigid-support playback — 18 October 2026

Visual exports are enabled by default in the actual Python pipeline. Existing
scene loading, physical priors, filtering/smoothing equations, RGB-D likelihood,
sparse information, convergence tests and full numerical outputs are unchanged.
Display options are excluded from the prior hash and solver/checkpoint settings.

## Files in each run's results directory

* `visual_debug/index.html`: live offline dashboard. Initialization, motion,
  geometry, joint refinement, posterior and final states are saved separately;
  accepted versus retained updates are identified. The latest saved-update time
  and solver/export statuses are displayed. Auto-refresh can be disabled while
  inspecting an older stage.
* `visual_debug/convergence.html`: live objective and numerical diagnostics,
  including actual per-stage ELBO, objective components, relative objective
  change, continuous gradient, variational KKT, sparse-solve residuals, conditional
  observation ranks, qualified covariance summaries, elapsed time and peak RSS.
  Log plots explicitly omit zero values; the latest values, including zeros,
  remain printed numerically. Stalls, failures and iteration limits stay visible.
* `final_3d.html`: single-file offline geometry playback and convergence plots,
  written when the solver stops, before full prediction export. Plotly and the
  bounded canonical geometry, support edges, discrete scene assignments,
  probabilities, all frame poses and actual times are embedded locally.
* `visual_debug/rgbd/<camera>/frame_*.png`: observed and actual model-predicted
  RGB/depth, RGB error, valid-return signed depth error, no-return probability and
  expected occlusion. These appear in the live dashboard during prediction export.

Optimization stage and acquisition time have separate controls. Play uses
actual nonuniform acquisition intervals and supports a playback-speed adjustment.
The browser transforms each finite support quad by its saved shared rigid-body
SE(3) pose; there are no per-point motion states or fabricated intermediate poses.
Orbit/zoom, support/center visibility, trajectories, camera centers, scene selection
and rigid-component/RGB/existence coloring are interactive.

At most 5,000 deterministic catalogue IDs are shown by default. Sampling affects
display only and is labeled with sample/total counts. Use `--visual-max-points`
to change this display budget or `--no-visual-debug` to disable visual exports.
Snapshots retain component-conditioned canonical hypotheses. A displayed scene
is one complete configuration from the actual variational family, initially the
MAP scene, not independently thresholded labels or a mean-existence renderer.
Absent proposals stay absent. Alternative scene choices retain their real
posterior weights, including zero. Geometry is transformed in the browser;
per-frame meshes are not duplicated, and only three stage snapshots are cached.

Viewer geometry is distinct from the probabilistic first-hit RGB-D predictions.
Neither the views nor the residual panels add likelihood factors. No physical
noise or geometry parameters are introduced. Calibration and synchronization
qualifications are retained. Uniform display opacity, colors and sampling are
presentation choices, not inferred sensor/geometry uncertainty.

## Verification

The complete 61-test numerical suite passed. New tests verify bounded sampling,
unchanged original arrays, actual probabilities and nonuniform times, multiple
body support corners versus NumPy SE(3) transforms, categorical invalid depth,
neutral zero depth error, resume, script-safe JSON, offline HTML packaging and
honest stall/failure status. The end-to-end comparison enables visualization in
one run and disables it in the other; trajectory/map/posterior arrays and
covariance diagnostics agree exactly, excluding runtime/memory from cycle logs.

Headless Chrome 152 with software WebGL rendered the standalone viewer. A
separate known-transform browser fixture verified lazy-loaded optimization-stage
switching, support motion, the actual frame-time label, RGB-D panels, convergence
charts, Play visiting both frames, RGB coloring and fit controls. The saved DOM
has `data-browser-test=passed`, `data-playback-test=passed` and no test error.
The screenshot is **a known-transform rendering test, not a reconstruction**.
Measured checks and actual production smoke diagnostics are in
`visual_debug_results.json`.

## Actual production-pipeline preview on the requested SSD

`/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf/visual_debug_smoke_20261018`

This explicitly named **two-frame, 2×2-pixel test** contains real pipeline outputs:
11 stage snapshots, two cycles, two RGB-D panels and a standalone final HTML.
It is not a blade/fridge reconstruction. Its actual status is
`STALLED_LINE_SEARCH`, final ELBO 10.924155231629136, gradient infinity norm
37.347007587168015. It is not converged. The preview's MAP configuration contains
static support; select variational scene 2 (weight about 0.2803) to inspect the
moving-body hypothesis. Selecting a scene changes only the view.

Open `results/visual_debug/index.html` for the full preview or
`results/final_3d.html` for standalone geometry playback. The same files appear
automatically in a full run. An older already-running process needs to restart
or resume with the updated code. Existing validation-only runs have no estimated
map/trajectory to visualize. Full-dataset optimization/convergence and Section 14
acceptance remain unverified; no full reconstruction was run for this visual change.

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf
bash scripts/run_full_scene.sh --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314'
```

The launcher prints the live dashboard path and saves all outputs below the
requested SSD root. Synthetic blade metadata still lacks validated acquisition
timing; visual output does not invent a capture clock or bypass this input contract.
