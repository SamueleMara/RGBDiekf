"""Producer-format regressions for real recordings and synthetic SAPIEN scenes."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from numpy.testing import assert_allclose

from rgbd_common.dataloader import camera_from_metadata, load_dataset
from rgbd_common.diagnostics import validate_full_dataset
from rgbd_common.timing import MissingAcquisitionTiming, SceneFiles, resolve_frame_times
from data_fixtures import make_dataset


def real_scene(root, frames=3, cameras=2):
    specs = make_dataset(root, frames, cameras)
    for spec in specs:
        spec.pop("near"); spec.pop("far")
        spec["pose_convention"] = "opencv_optical_camera_to_world"
    (root/"metadata/cameras.json").write_text(json.dumps({"cameras": specs,
        "coordinate_frame": "base_link", "calibration_status": "CANDIDATE_NOT_VALIDATED"}))
    (root/"metadata/raw").mkdir()
    (root/"metadata/raw/info.json").write_text('{"fps":50}')
    (root/"metadata/real_conversion.json").write_text(json.dumps({
        "source_frames": list(range(0, frames*2, 2)), "output_frames": list(range(frames)),
        "timestamps": "source_frame_index / recording FPS; not hardware timestamps",
        "frame_matching": "recorded frame indices; hardware synchronization not independently validated",
        "depth_output_unit": "float32 meters, rectified color-camera Z; zero means missing",
        "calibration_status": "CANDIDATE_NOT_VALIDATED", "calibration_warning": "candidate extrinsics"}))
    (root/"metadata/frame_timestamps.csv").write_text("frame_index,source_frame_index,timestamp_s\n"+
        "".join(f"{i},{2*i},{i/25}\n" for i in range(frames)))
    return specs


class DatasetLoadingTests(unittest.TestCase):
    def test_real_rectified_optical_poses_and_recording_timestamp_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = real_scene(root)
            data = load_dataset(root)
            self.assertEqual((len(data.frame_ids), len(data.cameras)), (3, 2))
            assert_allclose(data.times, [0, .04, .08])
            self.assertEqual(data.timing_report["fps"], 25)
            self.assertEqual(data.timing_report["time_source"], "metadata/frame_timestamps.csv:timestamp_s")
            self.assertIn("not hardware", data.timing_report["acquisition_qualifications"]["timestamps"])
            from rgbd_common.config import load_config
            noise = load_config()
            rate = noise.rate
            for dt in np.diff(data.times):
                assert_allclose(noise.q(dt), np.asarray(rate)*dt)
            assert_allclose(data.cameras[0].world_to_camera, np.linalg.inv(specs[0]["pose_matrix_4x4"]), atol=1e-14)
            manifest = data.manifest()
            self.assertEqual(manifest["metadata"]["calibration_status"], "CANDIDATE_NOT_VALIDATED")
            self.assertEqual(manifest["cameras_calibration"][0]["near"], 0)
            self.assertIsNone(manifest["cameras_calibration"][0]["far"])
            json.dumps(manifest, allow_nan=False)
            audit = validate_full_dataset(data)
            self.assertEqual(audit["rays"], 24)
            self.assertEqual(audit["resolved_fps"], 25)
            self.assertIn("not independent", audit["colmap_check_semantics"])

    def test_metadata_directory_pointer_and_zip_preserve_real_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"scene"; root.mkdir(); real_scene(root)
            data = load_dataset(root/"metadata")
            self.assertEqual(data.root, root)
            archive = root.parent/"unrelated_scene_name.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for path in root.rglob("*"):
                    if path.is_file(): z.write(path, "arbitrary_prefix/"+str(path.relative_to(root)))
            zipped = load_dataset(archive)
            assert_allclose(zipped.times, data.times)
            assert_allclose(zipped.cameras[0].world_to_camera, data.cameras[0].world_to_camera)
            for k in range(3):
                for ci in range(2):
                    for actual, expected in zip(zipped.load(k, ci), data.load(k, ci)):
                        assert_allclose(actual, expected)

    def test_raw_capture_fps_requires_conversion_mapping_and_is_not_output_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            (root/"metadata/frame_timestamps.csv").unlink()
            times, report = resolve_frame_times(root)
            assert_allclose(times, [0, .04, .08])
            self.assertIn("source_frames/output_frames", report["time_source"])
            self.assertEqual(report["fps"], 25)
            (root/"metadata/real_conversion.json").unlink()
            with self.assertRaises(MissingAcquisitionTiming): resolve_frame_times(root)

    def test_nonuniform_output_times_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            conversion = json.loads((root/"metadata/real_conversion.json").read_text())
            conversion["source_frames"] = [0, 2, 7]
            (root/"metadata/real_conversion.json").write_text(json.dumps(conversion))
            (root/"metadata/frame_timestamps.csv").write_text(
                "frame_index,source_frame_index,timestamp_s\n2,7,.14\n0,0,0\n1,2,.04\n")
            times, report = resolve_frame_times(root)
            assert_allclose(times, [0, .04, .14]); self.assertIsNone(report["fps"])
            overridden, report = resolve_frame_times(root, fps_override=10)
            assert_allclose(overridden, [0, .1, .2])
            self.assertTrue(report["metadata_disagreements"])
            with self.assertRaisesRegex(ValueError, "inconsistent user"):
                resolve_frame_times(root, fps_override=10, dt_seconds=.2)

    def test_duplicate_missing_or_conflicting_timestamp_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            for content, message in [
                ("0,0,0\n1,2,.04\n1,4,.08\n", "frame IDs"),
                ("0,0,0\n1,2,.04\n", "frame IDs"),
                ("0,0,0\n1,2,.05\n2,4,.08\n", "Conflicting dataset")]:
                (root/"metadata/frame_timestamps.csv").write_text("frame_index,source_frame_index,timestamp_s\n"+content)
                with self.subTest(content=content), self.assertRaisesRegex(ValueError, message):
                    resolve_frame_times(root)
            times, report = resolve_frame_times(root, fps_override=25)
            assert_allclose(times, [0, .04, .08])
            self.assertTrue(report["metadata_disagreements"])

    def test_async_camera_time_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            (root/"metadata/frame_timestamps.csv").write_text("camera_name,frame_index,timestamp_s\n"+
                "".join(f"cam_{c:03d},{i},{i/25+c*.001}\n" for c in range(2) for i in range(3)))
            with self.assertRaisesRegex(ValueError, "ASYNCHRONOUS_CAMERAS"):
                resolve_frame_times(root)

    def test_per_camera_times_require_complete_identified_camera_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            for cameras, message in [(["cam_000"], "every camera"),
                                     (["cam_000", "unknown"], "unknown camera")]:
                (root/"metadata/frame_timestamps.csv").write_text("camera_name,frame_index,timestamp_s\n"+
                    "".join(f"{c},{i},{i/25}\n" for c in cameras for i in range(3)))
                with self.subTest(cameras=cameras), self.assertRaisesRegex(ValueError, message):
                    resolve_frame_times(root)
            (root/"metadata/frame_timestamps.csv").write_text("camera_name,frame_index,timestamp_s\n"+
                "".join(f"cam_{c:03d},{i},{i/25}\n" for c in range(2) for i in range(3)))
            assert_allclose(resolve_frame_times(root)[0], [0, .04, .08])

    def test_synthetic_look_at_fov_metadata_and_no_stale_backup_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = make_dataset(root, frames=4, cameras=3)
            for spec in specs:
                for key in ("fx", "fy", "cx", "cy", "pose_matrix_4x4"): spec[key] = None
                spec.update(fov_y_deg=45., position=[4., 0., .7], target=[0., 0., .3], up=[0., 0., 1.])
            (root/"metadata/cameras.json").write_text(json.dumps({"cameras": specs}))
            (root/"metadata/job_config_resolved.yaml").write_text("fps: null\nvideo_fps: null\nduration_sec: null\n")
            backup = root/"metadata/frame_renumbering_backup"; backup.mkdir()
            (backup/"frame_timestamps.csv").write_text("frame_index,timestamp_s\n0,0\n1,.1\n2,.2\n3,.3\n")
            with self.assertRaises(MissingAcquisitionTiming): load_dataset(root)
            data = load_dataset(root, integrity_only=True)
            self.assertEqual((len(data.frame_ids), len(data.cameras)), (4, 3))
            camera = data.cameras[0]
            target = np.array([0., 0., .3])
            optical = camera.world_to_camera[:3, :3]@target+camera.world_to_camera[:3, 3]
            assert_allclose(optical[:2], [0, 0], atol=1e-14)
            self.assertGreater(optical[2], 0)
            self.assertIsNone(data.times)
            (root/"metadata/job_config_resolved.yaml").write_text("fps: 30\n")
            assert_allclose(load_dataset(root).times, np.arange(4)/30)

    def test_optical_world_to_camera_K_and_unknown_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = real_scene(root)[0]
            inverse = np.linalg.inv(spec["pose_matrix_4x4"])
            spec["pose_matrix_4x4"] = inverse.tolist()
            spec["pose_convention"] = "opencv_optical_world_to_camera"
            for key in ("fx", "fy", "cx", "cy"): spec.pop(key)
            spec["K"] = [[3, 0, 1], [0, 3, 1], [0, 0, 1]]
            camera = camera_from_metadata(spec)
            assert_allclose(camera.world_to_camera, inverse)
            assert_allclose(camera.intrinsics, [3, 3, 1, 1])
            spec["pose_convention"] = "unknown"
            with self.assertRaisesRegex(ValueError, "pose_convention"): camera_from_metadata(spec)

    def test_integer_depth_cannot_silently_be_interpreted_as_meters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            np.save(root/"depth_npy/cam_000/frame_000000.npy", np.full((2, 2), 1000, dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "floating-point meters"):
                load_dataset(root).load(0, 0)

    def test_zip_glob_does_not_collect_nested_stale_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"scene"; root.mkdir(); real_scene(root)
            archive = root.parent/"scene.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for path in root.rglob("*"):
                    if path.is_file(): z.write(path, "scene/"+str(path.relative_to(root)))
                z.writestr("scene/rgb/cam_000/backup/frame_999999.png", b"not an observation")
            self.assertEqual(len(SceneFiles(archive).glob("rgb/cam_000", "frame_*.png")), 3)
            self.assertEqual(len(load_dataset(archive).frame_ids), 3)

    def test_metadata_pointer_keeps_output_outside_original_scene(self):
        from main import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); real_scene(root)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--input", str(root/"metadata"), "--output", str(root/"accidental_output"), "--validate-only"])
            self.assertFalse((root/"accidental_output").exists())

    def test_joint_value_export_is_never_opened_for_timing(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);real_scene(root)
            (root/'metadata/frame_values.csv').write_text('frame_index,timestamp_s,joint_angle\n0,0,1\n1,100,2\n2,200,3\n')
            original=SceneFiles.read
            def guarded(files,path):
                if Path(path).name=='frame_values.csv':raise AssertionError('Ground-truth joint export read by estimation')
                return original(files,path)
            with patch.object(SceneFiles,'read',guarded):
                data=load_dataset(root)
            assert_allclose(data.times,[0,.04,.08])

    def test_auto_fps_30_for_all_60_frames_and_alternate_metadata(self):
        for relative,key in (('metadata/job_config_resolved.yaml','fps'),('metadata/sequence.json','acquisition_fps')):
            with self.subTest(relative=relative),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);make_dataset(root,60,1)
                (root/relative).write_text(json.dumps({key:30}))
                data=load_dataset(root)
                assert_allclose(np.diff(data.times),1/30,atol=1e-14)
                self.assertAlmostEqual(data.times[-1]-data.times[0],59/30)
                self.assertIn(relative,data.timing_report['time_source'])


if __name__ == "__main__":
    unittest.main()
