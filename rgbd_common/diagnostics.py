"""Observability, runtime, physical units and honest acceptance diagnostics."""
import json
import resource
import time

import numpy as np

from .progress import progress


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def runtime_summary(start):
    return {"runtime_seconds": time.monotonic()-start,
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024}


def validate_full_dataset(dataset):
    from .dataloader import scan
    from scipy.spatial.transform import Rotation
    counts = scan(dataset)
    reprojection_errors, vertical_displacements, cross_view_depth_errors = [], [], []
    for k in progress(range(len(dataset.frame_ids)), desc="Validate reprojections", unit="frame", leave=True):
        for ci, camera in enumerate(dataset.cameras):
            source=dataset.observation(k,ci);depth=source.depth
            yy, xx = np.nonzero(source.valid)
            if not len(xx):
                continue
            # Diagnostic samples only; validation scan above retains ALL pixels.
            pick = np.linspace(0, len(xx)-1, min(64, len(xx)), dtype=int)
            x, y, z = xx[pick], yy[pick], np.asarray(depth[yy[pick], xx[pick]], dtype=float)
            fx, fy, cx, cy = camera.intrinsics
            q = np.c_[z*(x-cx)/fx, z*(y-cy)/fy, z]
            world = (q-camera.world_to_camera[:3, 3])@camera.world_to_camera[:3, :3]
            reconstructed = world@camera.world_to_camera[:3, :3].T+camera.world_to_camera[:3, 3]
            uv = reconstructed[:, :2]/reconstructed[:, 2, None]*[fx, fy]+[cx, cy]
            reprojection_errors.extend(np.linalg.norm(uv-np.c_[x, y], axis=1))
            vertical_displacements.append(float(z[0]/fy))
            target = dataset.cameras[(ci+1) % len(dataset.cameras)]
            qt = world@target.world_to_camera[:3, :3].T+target.world_to_camera[:3, 3]
            fxt, fyt, cxt, cyt = target.intrinsics
            inside = qt[:, 2] > target.near
            uvtarget = np.zeros((len(qt), 2))
            uvtarget[inside] = qt[inside, :2]/qt[inside, 2, None]*[fxt, fyt]+[cxt, cyt]
            indices = np.rint(uvtarget).astype(int)
            inside &= ((indices[:, 0] >= 0) & (indices[:, 0] < target.width) &
                       (indices[:, 1] >= 0) & (indices[:, 1] < target.height))
            target_frame=dataset.observation(k,(ci+1)%len(dataset.cameras));target_depth=target_frame.depth
            sample = target_depth[indices[inside, 1], indices[inside, 0]]
            foreground=target_frame.valid[indices[inside,1],indices[inside,0]]
            cross_view_depth_errors.extend((sample-qt[inside, 2])[foreground])
    report = {**counts, "frames": len(dataset.frame_ids), "cameras": len(dataset.cameras),
              "image_sizes": [[c.width, c.height] for c in dataset.cameras],
              "timestamps_available": dataset.times is not None,
              "resolved_fps": dataset.timing_report.get("fps"),
              "calibration_metadata": dataset.metadata_report,
              "timing_qualifications": dataset.timing_report.get("acquisition_qualifications", {}),
              "source_reprojection_max_pixel_error": max(reprojection_errors, default=0.),
              "camera_y_down_verified": bool(all(v > 0 for v in vertical_displacements)),
              "cross_view_depth_residual_quantiles_m": np.quantile(cross_view_depth_errors, [0, .25, .5, .75, 1]).tolist()
              if cross_view_depth_errors else [],
              "cross_view_residual_semantics": "includes occlusion, rasterization and geometry differences; NOT noise calibration",
              "inference_ground_truth_used": False,
              "integrity_validation_only": True,
              "expected_original_ray_factors": counts["rays"]}
    calibration_checks = []
    relative = "colmap/sparse/0/images.txt"
    if dataset.files.exists(relative):
        lines = dataset.files.read(relative).decode().splitlines()
        data_lines = [line for line in lines if not line.startswith("#")]
        for line in data_lines[::2]:
            fields = line.split()
            if len(fields) < 10:
                continue
            camera_name = fields[9].split("/")[0]
            camera = next((c for c in dataset.cameras if c.name == camera_name), None)
            if camera is None:
                continue
            qw, qx, qy, qz = map(float, fields[1:5])
            rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            translation = np.asarray(fields[5:8], dtype=float)
            error = max(float(np.max(np.abs(rotation-camera.world_to_camera[:3, :3]))),
                        float(np.max(np.abs(translation-camera.world_to_camera[:3, 3]))))
            calibration_checks.append({"camera": camera_name, "extrinsic_max_absolute_error": error})
        if calibration_checks and max(c["extrinsic_max_absolute_error"] for c in calibration_checks) > 1e-7:
            raise ValueError("Camera image-down/world convention disagrees with the COLMAP calibration export")
    report["colmap_calibration_consistency"] = calibration_checks
    report["colmap_check_semantics"] = "export consistency only; not independent physical calibration validation"
    if report["source_reprojection_max_pixel_error"] > 1e-7 or not report["camera_y_down_verified"]:
        raise ValueError("World/camera calibration convention failed reprojection validation")
    return report
