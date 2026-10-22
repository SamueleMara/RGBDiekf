# Real and synthetic loader audit — 17 October 2026

The selected scene determines metadata, camera names, image dimensions and the
complete frame/camera set. No scene basename, joint labels, part annotations or
fixed frame/camera counts are used for loading or timing. Directory, metadata
directory and single-scene ZIP inputs share the same loader.

| Supplied scene | Full image scan | Timing | Outcome |
|---|---|---|---|
| `real/fridge_plastic_20261016_163314` | 328×3, 302,284,800 pixels; 197,919,833 valid depths | `metadata/frame_timestamps.csv:timestamp_s`, 25 Hz, 0–13.08 s | `INPUTS_VALIDATED`; no reconstruction performed |
| `preparation_8cams/partnet_subset_fixed_render/blade_103706` | 60×8, 147,456,000 pixels; 2,166,457 valid depths | No valid timestamps/FPS in searched acquisition metadata; resolved config timing fields null | Full integrity scan passed; production remains `MISSING_ACQUISITION_TIMING` |

Real output indices 0–327 correspond to source indices 0,2,…,654. The raw capture
rate is 50 Hz; applying that rate directly to consecutively renumbered images
would give the wrong time intervals and process covariance. The loader prefers
the output timestamp CSV. The producer's documented index/FPS conversion can
reproduce this clock if that file is absent. Raw recording FPS without this
mapping is insufficient. Frame-specific intervals propagate to the existing
f/F/G/Q and smoothing pipeline. Conflicting explicit user overrides, missing
timestamp frames, duplicates, unknown/missing timestamp cameras, or asynchronous
camera clocks without a supported synchronization model are rejected explicitly.

Real camera poses declare `opencv_optical_camera_to_world` and are inverted
directly. Synthetic camera poses/look-at vectors use legacy SAPIEN axes and the
optical conversion. Intrinsics use active rectified pinhole calibration or the
synthetic FOV model. Unknown pose conventions fail. Missing renderer clipping
planes mean Z>0 and an unbounded far domain; no numerical sensor cutoff is
invented. Homogeneous image-frustum clipping handles supports at the Z=0 horizon
without adding a near threshold. Floating-point NPY depths are metric optical Z;
unscaled integer depths are rejected. Data remain streamed one observation at a
time, with every paired image included by default.

The real source reprojection maximum error was 7.74e-13 pixels; maximum extrinsic
discrepancy against COLMAP was 2.78e-16. The synthetic errors were 1.14e-13 pixels
and 4.03e-16 respectively. These verify export/convention consistency, not
independently measured accuracy. The real metadata explicitly says
`CANDIDATE_NOT_VALIDATED`, and hardware synchronization is not independently
validated. Both qualifications are preserved. Rectified per-ray independence
remains an assumption of the selected likelihood. Initial uncertainty values
remain documented guesses, not independent calibration.

See `data_loading_validation.json` for measured counts, memory, runtime, metadata
qualifications and exact SSD run directories. Each run contains the full
manifest, timing array, dataset audit and uncertainty provenance. These scans
are not optimization/convergence results. At nearly 198 million source-depth
candidates, the real reconstruction's memory and throughput are unbenchmarked.
The known small-scene line-search stall remains documented in
`acceptance_status.md`; Section 14 full reconstruction acceptance is still open.

## Execution commands

The launcher uses the existing `rigeuler3dgs` environment and writes logs,
temporary/cache files and results into a fresh folder under:

`/media/samuelemara/Extreme SSD/Samuele/results_11_10_26/RGBDekf`

Real scene (timing and authorized initial noise guesses are automatic):

```bash
conda activate rigeuler3dgs
cd /home/samuelemara/Joint3DGS/RGBDekf
bash scripts/run_full_scene.sh --input '/media/samuelemara/Extreme SSD/Samuele/Backup_09_10_26/Sapien/output/real/fridge_plastic_20261016_163314'
```

Synthetic scene (default path updated to the folder specified on 17 October):

```bash
bash scripts/run_full_scene.sh
```

The latter reports missing acquisition timing with the current metadata. Supply
a verified `--timestamps-file`, `--fps` or `--dt-seconds` only when known. Do not
borrow the real scene's 25 Hz output rate. `--frame-time` is available only as an
explicit different model using frame units, never silently selected.

Append `--validate-only` to either command for input validation alone. Root
scene paths may equivalently be replaced with their `metadata/` directories.

## Regression evidence

`reports/numerical_tests.txt` contains the complete numerical suite (now including
the progress-display invariance test). Thirteen loader
regressions cover the producer formats, frame renumbering, timestamp precedence,
process-noise intervals, nonuniform time, explicit overrides, synchronization,
ZIP and metadata-directory inputs, camera conventions, positive-Z support
derivatives, missing timing, metric-depth validation and output placement.
The existing covariance/filtering/RTS/sparse-information tests also pass.
