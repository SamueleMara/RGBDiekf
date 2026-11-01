# Critical synthetic scenes: actual status

No temporal reconstruction was run on these real scenes. Missing values are not imputed.

| Scene | Frames | Cameras | FPS | Warp covariance | Temporal filtering / accuracy |
|---|---:|---:|---:|---|---|
| USB_100109 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| blade_103706 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| foldchair_102255 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| fridge_10905 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| laptop_10211 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| oven_101917 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| scissor_11100 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| stapler_103111 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| storage_45135 | 60 | 8 | 59.0 | MISSING | NOT RUN |
| washer_103776 | 60 | 8 | 59.0 | MISSING | NOT RUN |

Storage has a complete-point-count geometry/ownership preflight at frames 0/1/2. It is NOT a motion estimate.
No GT enters inference. Static drift, moving-point error, empirical uncertainty coverage and NEES on these real scenes are NOT measured.
Use evaluate_temporal_rgbd.py and compare_temporal_modes.py after actual completed runs; GT evaluation is opt-in and separate.
