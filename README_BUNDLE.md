# Two independent sparse RGB-D SLAM pipelines — deliverables

- `Parallel_RGBD_IEKF_UKF_Theory_CUDA.pdf`: 19-page mathematical document. The common solved/unsolved algorithm, independent IEKF and UKF state-space equations, their distinct temporal smoothers, deterministic backward geometry discovery, GPU implementation and uncertainty/validation contracts.
- `CODEX_IMPLEMENT_PARALLEL_IEKF_UKF_RGBD_SLAM.md`: single authoritative Codex implementation assignment, including explicit modular file structure, algorithms, tests, full-scene acceptance, and prohibitions on previous catalogue shortcuts.
- `00_COMMON_SOLVED_UNSOLVED_SPEC.md`: common outer-loop source theory.
- `01_IEKF_THEORY.md`: analytical IEKF and RTS source theory.
- `02_UKF_THEORY.md`: unscented UKF and unscented RTS source theory.
- `03_CUDA_AND_IMPLEMENTATION_THEORY.md`: GPU and covariance source theory.
- `MASTER_THEORY_SOURCE.md`: editable master Markdown used to compile the PDF.

The two pipelines share the outer SLAM code but have separate IEKF and UKF inner motion modules. No hybrid switching, normals, surfels, TSDF, per-pixel persistent state, or dense pixel covariance matrix.

Suggested Codex instruction: "Read the Codex Markdown and the theory PDF completely. Audit the current RGBDekf repository, implement the specified independent `main_iekf.py` and `main_ukf.py`, and run the mandatory full forward/backward/frontier/second-forward tests for both. Do not report success until the actual full-data runs complete and the objective/covariance diagnostics are saved."
