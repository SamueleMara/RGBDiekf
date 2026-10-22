"""Resolve acquisition timing from the selected directory/ZIP, never video playback by guess."""
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import yaml


class MissingAcquisitionTiming(ValueError):
    pass


class SceneFiles:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        # Accept the metadata directory as a scene pointer, using its sibling images.
        if self.path.name == "metadata" and (self.path/"cameras.json").is_file():
            self.path = self.path.parent
        self.archive = self.path.is_file() and zipfile.is_zipfile(self.path)
        self.prefix = ""
        if self.archive:
            with zipfile.ZipFile(self.path) as z:
                names = z.namelist()
            matches = [name[:-len("metadata/cameras.json")] for name in names if name.endswith("metadata/cameras.json")]
            if len(matches) != 1:
                raise ValueError("ZIP must contain one scene with metadata/cameras.json")
            self.prefix, self.names = matches[0], set(names)
        elif not self.path.is_dir():
            raise ValueError("--input must be a dataset directory or ZIP")

    def exists(self, relative):
        return self.prefix+relative in self.names if self.archive else (self.path/relative).is_file()

    def read(self, relative):
        if self.archive:
            with zipfile.ZipFile(self.path) as z:
                return z.read(self.prefix+relative)
        return (self.path/relative).read_bytes()

    def glob(self, relative, pattern):
        if self.archive:
            from fnmatch import fnmatch
            prefix = self.prefix+relative.rstrip("/")+"/"
            return [name[len(self.prefix):] for name in sorted(self.names)
                    if name.startswith(prefix) and "/" not in name[len(prefix):]
                    and fnmatch(name[len(prefix):], pattern)]
        return [str(p.relative_to(self.path)) for p in sorted((self.path/relative).glob(pattern))]


def _timestamps(value, count, frame_ids, name):
    if isinstance(value, dict):
        if "units" in value and value["units"] not in ("s", "seconds"):
            raise ValueError(f"{name}: timestamp units must explicitly be seconds")
        if "timestamps_seconds" in value:
            value = value["timestamps_seconds"]
        elif "times_seconds" in value:
            value = value["times_seconds"]
        elif "values" in value:
            value = value["values"]
        else:
            value = [value[str(i)] for i in frame_ids]
    times = np.asarray(value, dtype=float)
    if times.shape != (count,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}: require {count} finite strictly increasing per-frame times")
    return times


def _camera_keys(keys, aliases, name):
    if any(str(key) not in aliases for key in keys):
        raise ValueError(f"{name}: timestamps identify an unknown camera")
    canonical = [aliases[str(key)] for key in keys]
    if len(set(canonical)) != len(canonical) or set(canonical) != set(aliases.values()):
        raise ValueError(f"{name}: timestamps must cover every camera exactly once")


def _timestamp_file(data, suffix, count, frame_ids, name, camera_aliases=None):
    if suffix == ".npy":
        return _timestamps(np.load(io.BytesIO(data), allow_pickle=False), count, frame_ids, name)
    if suffix == ".json":
        return _timestamps(json.loads(data), count, frame_ids, name)
    text = data.decode()
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        key = next((key for key in ("timestamp_seconds", "time_seconds", "timestamp_s", "time_s", "timestamp", "time")
                    if rows and key in rows[0]), None)
        if not key:
            return None  # Joint-value CSV has no acquisition time column.
        index_key = next((key for key in ("frame_index", "frame_id", "frame") if key in rows[0]), None)
        def indexed_times(group):
            if index_key:
                lookup = {int(row[index_key]): float(row[key]) for row in group}
                if len(lookup) != len(group) or set(lookup) != set(frame_ids):
                    raise ValueError(f"{name}: timestamp frame IDs must match the image frames exactly, without duplicates")
                return _timestamps([lookup[i] for i in frame_ids], count, frame_ids, name+":"+key)
            return _timestamps([row[key] for row in group], count, frame_ids, name+":"+key)
        camera_key = next((name for name in ("camera", "camera_id", "camera_name") if name in rows[0]), None)
        if camera_key:
            grouped = {}
            for row in rows:
                grouped.setdefault(row[camera_key], []).append(row)
            if camera_aliases is not None:
                _camera_keys(grouped, camera_aliases, name)
            per_camera = []
            for group in grouped.values():
                per_camera.append(indexed_times(group))
            if any(not np.allclose(t, per_camera[0], atol=1e-12, rtol=0) for t in per_camera[1:]):
                raise ValueError("ASYNCHRONOUS_CAMERAS: explicit observation synchronization model required")
            return per_camera[0]
        return indexed_times(rows)
    return _timestamps(np.loadtxt(io.StringIO(text)), count, frame_ids, name)


def resolve_frame_times(input_path, fps_override=None, timestamps_override=None, dt_seconds=None,
                        frame_ids=None, frame_time=False):
    files = SceneFiles(input_path)
    specs = json.loads(files.read("metadata/cameras.json"))["cameras"]
    aliases = {}
    for spec in specs:
        for key in ("name", "id", "source_camera_name"):
            if spec.get(key) is not None:
                alias = str(spec[key])
                if alias in aliases and aliases[alias] != spec["name"]:
                    raise ValueError(f"Ambiguous camera identifier in metadata: {alias}")
                aliases[alias] = spec["name"]
    if frame_ids is None:
        paths = files.glob("rgb/"+specs[0]["name"], "frame_*.png")
        frame_ids = sorted(int(Path(p).stem.removeprefix("frame_")) for p in paths)
    count = len(frame_ids)
    searches, candidates = [], []
    names = ["job_config_resolved.yaml", "config.yaml", "config.yml", "config.json", "metadata.json",
             "metadata.yaml", "scene_metadata.json", "sequence.json", "sequence.yaml",
             "acquisition_manifest.json", "acquisition_manifest.yaml", "dataset_manifest.json",
             "dataset_manifest.yaml", "manifest.json", "manifest.yaml", "timing.json", "timing.yaml"]

    def inspect_document(document, location):
        def visit(node, keypath=""):
            if not isinstance(node, dict):
                return
            # Honor scene-specific ancestor manifests without reading another scene's rate.
            scene = files.path.stem
            if scene in node and isinstance(node[scene], dict):
                visit(node[scene], keypath+scene+".")
            for key in ("timestamps_seconds", "times_seconds", "frame_times_seconds"):
                if node.get(key) is not None:
                    candidates.append(("timestamps", _timestamps(node[key], count, frame_ids, location+":"+keypath+key),
                                       location+":"+keypath+key))
            per_camera = node.get("camera_timestamps_seconds")
            if per_camera is not None:
                _camera_keys(per_camera, aliases, location+":"+keypath+"camera_timestamps_seconds")
                arrays = [_timestamps(v, count, frame_ids, location+":"+keypath+"camera_timestamps_seconds")
                          for v in per_camera.values()]
                if any(not np.allclose(a, arrays[0], rtol=0., atol=1e-12) for a in arrays[1:]):
                    raise ValueError("ASYNCHRONOUS_CAMERAS: supply an explicit synchronization/continuous-time observation model")
                candidates.append(("timestamps", arrays[0], location+":"+keypath+"camera_timestamps_seconds"))
            for key in ("fps", "capture_fps", "acquisition_fps", "capture_rate_hz", "frame_rate_hz", "frame_rate", "framerate"):
                value = node.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0:
                    candidates.append(("fps", float(value), location+":"+keypath+key))
            if node.get("video_fps_is_capture_fps") is True or node.get("timing_semantics") == "playback_equals_acquisition":
                value = node.get("video_fps")
                if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                    candidates.append(("fps", float(value), location+":"+keypath+"video_fps[documented capture equivalence]"))
            for key in ("dt_seconds", "frame_interval_seconds", "capture_interval_seconds"):
                value = node.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                    candidates.append(("fps", 1/float(value), location+":"+keypath+key))
            duration = node.get("duration_sec", node.get("duration_seconds"))
            if duration is not None and node.get("duration_convention") in ("first_to_last", "frame_count_intervals"):
                divisor = count-1 if node["duration_convention"] == "first_to_last" else count
                if not np.isfinite(duration) or duration <= 0 or divisor < 1:
                    raise ValueError(f"{location}: invalid explicit acquisition duration")
                candidates.append(("fps", divisor/duration, location+":"+keypath+"duration_sec["+node["duration_convention"]+"]"))
            for key in ("acquisition", "timing", "sequence", "capture", "sensor", "sequences", "scenes"):
                if key in node:
                    visit(node[key], keypath+key+".")
        visit(document)

    for folder in ("metadata/", ""):
        for name in names:
            relative = folder+name
            searches.append(relative)
            if files.exists(relative):
                inspect_document(yaml.safe_load(files.read(relative)), relative)
        for name in [stem+suffix for stem in ("timestamps", "frame_timestamps", "frame_times")
                     for suffix in (".csv", ".json", ".txt", ".npy")]+["frame_values.csv"]:
            relative = folder+name
            searches.append(relative)
            if files.exists(relative):
                # Joint/part exports are evaluation-only, even if they contain a time-like column.
                if name == "frame_values.csv":
                    continue
                times = _timestamp_file(files.read(relative), Path(name).suffix, count, frame_ids, relative, aliases)
                if times is not None:
                    key = ""
                    if Path(name).suffix == ".csv":
                        header = next(csv.reader(io.StringIO(files.read(relative).decode())))
                        key = ":"+next(k for k in ("timestamp_seconds", "time_seconds", "timestamp_s", "time_s", "timestamp", "time") if k in header)
                    candidates.append(("timestamps", times, relative+key))
    # Raw recording FPS applies to source indices, not consecutively renumbered
    # output images. Only use it with the producer's explicit conversion mapping.
    timing_notes = {}
    conversion_file, recording_file = "metadata/real_conversion.json", "metadata/raw/info.json"
    if files.exists(conversion_file):
        conversion = json.loads(files.read(conversion_file))
        timing_notes = {key: conversion[key] for key in ("timestamps", "frame_matching") if key in conversion}
        searches.extend([conversion_file, recording_file])
        if files.exists(recording_file) and str(conversion.get("timestamps", "")).startswith("source_frame_index / recording FPS"):
            recording = json.loads(files.read(recording_file))
            rate = recording.get("fps")
            output_ids, source_ids = conversion.get("output_frames"), conversion.get("source_frames")
            if output_ids is not None and source_ids is not None and rate is not None:
                if (not isinstance(rate, (int, float)) or isinstance(rate, bool) or not np.isfinite(rate) or rate <= 0
                        or len(output_ids) != len(source_ids) or len(set(output_ids)) != len(output_ids)
                        or set(output_ids) != set(frame_ids)):
                    raise ValueError(f"{conversion_file}: invalid recording FPS or source/output frame mapping")
                lookup = dict(zip(output_ids, source_ids))
                source = _timestamps([lookup[i] for i in frame_ids], count, frame_ids, conversion_file+":source_frames")
                if np.any(source < 0) or np.any(source != np.floor(source)):
                    raise ValueError(f"{conversion_file}: source frame indices must be nonnegative integers")
                times = source/rate
                candidates.append(("timestamps", times, conversion_file+":source_frames/output_frames + "+recording_file+":fps"))
    # Only documented sequence metadata names in nearby ancestors. Package/export
    # versions and arbitrary FPS-looking text are deliberately not timing evidence.
    if not files.archive:
        for ancestor in list(files.path.parents)[:4]:
            for name in names:
                for path in (ancestor/name, ancestor/"metadata"/name):
                    searches.append(str(path))
                    if path.is_file():
                        inspect_document(yaml.safe_load(path.read_text()), str(path))
    timestamp_candidates = [c for c in candidates if c[0] == "timestamps"]
    fps_candidates = [c for c in candidates if c[0] == "fps"]
    metadata = timestamp_candidates[0] if timestamp_candidates else fps_candidates[0] if fps_candidates else None
    if (timestamp_candidates and fps_override is None and dt_seconds is None and timestamps_override is None
            and any(not np.allclose(c[1], metadata[1], rtol=1e-10, atol=1e-12) for c in timestamp_candidates[1:])):
        raise ValueError("Conflicting dataset acquisition timestamp arrays")
    overrides = []
    for value, name in [(fps_override, "--fps"), (None if dt_seconds is None else 1/dt_seconds if dt_seconds > 0 else -1, "--dt-seconds")]:
        if value is not None:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            overrides.append(("fps", float(value), name))
    if timestamps_override is not None:
        p = Path(timestamps_override)
        times = _timestamp_file(p.read_bytes(), p.suffix, count, frame_ids, str(p), aliases)
        if times is None:
            raise ValueError(f"{p}: no recognized timestamp column in explicit --timestamps-file")
        overrides.append(("timestamps", times, "--timestamps-file:"+str(p)))
    def array(candidate):
        return candidate[1] if candidate[0] == "timestamps" else (np.asarray(frame_ids)-frame_ids[0])/candidate[1]
    if overrides and any(not np.allclose(np.diff(array(c)), np.diff(array(overrides[0])), rtol=1e-10, atol=1e-12) for c in overrides[1:]):
        raise ValueError("Mutually inconsistent user timing overrides")
    selected = overrides[-1] if overrides else metadata
    disagreements = []
    if selected is not None:
        for candidate in candidates:
            if not np.allclose(np.diff(array(candidate)), np.diff(array(selected)), rtol=1e-8, atol=1e-12):
                disagreements.append({"source": candidate[2], "selected_source": selected[2],
                                       "reason": "acquisition intervals disagree; explicit override or actual timestamps take precedence"})
    if selected is None:
        if frame_time:
            selected = ("timestamps", np.asarray(frame_ids, dtype=float)-frame_ids[0], "explicit --frame-time authorization")
        else:
            error = MissingAcquisitionTiming("MISSING_ACQUISITION_TIMING: no valid acquisition timestamps/FPS; searched "+
                ", ".join(searches)+"; keys: timestamps_seconds,times_seconds,fps,capture_fps,acquisition_fps,dt_seconds. "
                "Supply --timestamps-file, --fps, --dt-seconds, or explicitly authorize --frame-time.")
            error.searched_files = searches
            error.searched_keys = ["timestamps_seconds", "times_seconds", "fps", "capture_fps", "acquisition_fps", "dt_seconds"]
            raise error
    times = array(selected)
    _timestamps(times, count, frame_ids, selected[2])
    differences = np.diff(times)
    report = {"scene": str(files.path), "time_source": selected[2],
              "time_units": "frames" if frame_time and not (metadata or overrides) else "seconds",
              "fps": float(1/differences[0]) if len(differences) and np.allclose(differences, differences[0]) and not (frame_time and not (metadata or overrides)) else None,
              "timestamps_seconds": times.tolist() if not (frame_time and not (metadata or overrides)) else None,
              "timestamps_frame_units": times.tolist() if frame_time and not (metadata or overrides) else None,
              "interval_count": len(differences), "interval_min_max_seconds": [float(differences.min()), float(differences.max())] if len(differences) and not (frame_time and not (metadata or overrides)) else None,
              "source_kind": "explicit-user-override" if overrides else "dataset-acquisition-metadata" if metadata else "user-authorized-frame-time-assumption",
              "metadata_disagreements": disagreements, "files_searched": searches,
              "acquisition_qualifications": timing_notes}
    return times, report
