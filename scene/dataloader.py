"""Streaming native SAPIEN directory/ZIP RGB-D, timing and calibration."""
import io,hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from scene.camera import Camera
from utils.progress import progress
from scene.timing import SceneFiles, resolve_frame_times, MissingAcquisitionTiming


def exact_binary_mask(values,source='foreground mask'):
    values=np.asarray(values)
    if values.ndim!=2:raise ValueError(f'{source}: expected a two-dimensional binary foreground mask')
    unique=np.unique(values)
    if not (np.isin(unique,[0,1]).all() or np.isin(unique,[0,255]).all()):
        raise ValueError(f'{source}: NON_BINARY_FOREGROUND_MASK values {unique[:20].tolist()}; expected {{0,1}} or {{0,255}}, no thresholding')
    return values!=0


@dataclass
class Dataset:
    root: Path
    cameras: list
    frame_ids: list
    times: object
    entries: list
    files: object
    timing_report: dict
    metadata_report: dict = field(default_factory=dict)
    masks: object = None
    use_masks: bool = True
    validity_paths: object = None
    observation_fingerprints: dict = field(default_factory=dict,init=False,repr=False)

    def load(self, k, ci):
        rgbpath, depthpath = self.entries[k][ci]
        rgb_bytes=self.files.read(rgbpath)
        with Image.open(io.BytesIO(rgb_bytes)) as image:
            if image.mode not in ("RGB", "RGBA"):
                raise ValueError(f"Expected RGB image: {rgbpath}")
            rgb = np.asarray(image.convert("RGB"), dtype=float)/255.
        if self.files.archive:
            depth = np.load(io.BytesIO(self.files.read(depthpath)), allow_pickle=False)
        else:
            depth = np.load(self.root/depthpath, mmap_mode="r", allow_pickle=False)
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"depth_npy must store floating-point meters, not unscaled integer depth: {depthpath}")
        camera = self.cameras[ci]
        if rgb.shape != (camera.height, camera.width, 3) or depth.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/depth dimensions mismatch: {depthpath}")
        self.observation_fingerprints.setdefault(f'{k:06d}_{ci:03d}',{}).update(rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),depth_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(depth))).hexdigest(),depth_dtype=str(depth.dtype),depth_shape=list(depth.shape))
        return rgb, depth

    def observation(self, k, ci):
        rgb, depth = self.load(k, ci)
        path = None if self.masks is None else self.masks[k][ci]
        if self.use_masks and path is not None:
            mask_bytes=self.files.read(path)
            with Image.open(io.BytesIO(mask_bytes)) as image:
                mask = exact_binary_mask(np.asarray(image),path)
            if mask.shape != depth.shape:
                raise ValueError(f"Foreground mask shape mismatch: {path}")
            self.observation_fingerprints[f'{k:06d}_{ci:03d}']['mask_sha256']=hashlib.sha256(mask_bytes).hexdigest()
        else:
            raise ValueError(f'MISSING_FOREGROUND_MASK: frame {k}, camera {ci}; a real aligned binary mask is required')
        raw_valid=np.isfinite(depth)&(depth>0)
        if self.validity_paths is not None:
            validpath=self.validity_paths[k][ci];content=self.files.read(validpath)
            if validpath.endswith('.npy'):values=np.load(io.BytesIO(content),allow_pickle=False)
            else:
                with Image.open(io.BytesIO(content)) as image:values=np.asarray(image)
            raw_valid=exact_binary_mask(values,validpath)
            if raw_valid.shape!=depth.shape:raise ValueError(f'Raw depth validity shape mismatch: {validpath}')
            self.observation_fingerprints[f'{k:06d}_{ci:03d}']['raw_validity_sha256']=hashlib.sha256(content).hexdigest()
        return RGBDFrame(rgb,depth,mask,self.cameras[ci],k,ci,raw_valid,
            self.metadata_report['mask_visibility_premise']['object_only_no_external_occluders'])

    def manifest(self):
        return {"root": str(self.root), "frames": len(self.frame_ids), "cameras": len(self.cameras),
                "frame_ids": self.frame_ids,
                "times_seconds": None if self.times is None or self.timing_report.get("time_units") != "seconds" else self.times.tolist(),
                "timing": self.timing_report,
                "metadata": self.metadata_report,
                "foreground_masks": self.masks, "use_foreground_masks": self.use_masks,
                "raw_depth_validity_paths":self.validity_paths,
                "depth_convention": "metric optical Z; zero means categorical no-return",
                "cameras_calibration": [dict(name=c.name,width=c.width,height=c.height,
                    intrinsics=c.intrinsics.tolist(),world_to_camera=c.world_to_camera.tolist(),
                    near=c.near,far=c.far if np.isfinite(c.far) else None)
                    for c in self.cameras],
                "observations": self.entries,
                "observation_fingerprints": [[self.observation_fingerprints.get(f'{k:06d}_{c:03d}') for c in range(len(self.cameras))] for k in range(len(self.frame_ids))]}


def camera_from_metadata(spec, default_pose_convention=None):
    w, h = int(spec["width"]), int(spec["height"])
    if min(w, h) <= 0:
        raise ValueError("Camera dimensions must be positive")
    if all(spec.get(key) is not None for key in ("fx", "fy", "cx", "cy")):
        intrinsics = np.array([spec[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
    elif spec.get("K") is not None:
        matrix = np.asarray(spec["K"], dtype=float)
        if matrix.shape != (3, 3) or not np.allclose(matrix, [[matrix[0, 0], 0, matrix[0, 2]],
                [0, matrix[1, 1], matrix[1, 2]], [0, 0, 1]]):
            raise ValueError("Expected rectified PINHOLE intrinsic K; distortion/skew must be resolved by the producer")
        intrinsics = matrix[[0, 1, 0, 1], [0, 1, 2, 2]]
    else:
        fov = float(spec["fov_y_deg"])
        if not 0 < fov < 180:
            raise ValueError("Invalid vertical field of view")
        fy = h/(2*np.tan(np.deg2rad(fov)/2))
        intrinsics = np.array([fy, fy, w/2, h/2])
    if not np.all(np.isfinite(intrinsics)) or min(intrinsics[:2]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    convention = spec.get("pose_convention") or default_pose_convention or "sapien_camera_to_world"
    allowed = ("sapien_camera_to_world", "opencv_optical_camera_to_world", "opencv_optical_world_to_camera")
    if convention not in allowed:
        raise ValueError(f"Unsupported pose_convention {convention!r}; expected one of {allowed}")
    if spec.get("pose_matrix_4x4") is not None:
        c2w = np.asarray(spec["pose_matrix_4x4"], dtype=float)
    else:
        if convention != "sapien_camera_to_world":
            raise ValueError("Optical poses require an explicit pose_matrix_4x4")
        pos, target, up = [np.asarray(spec[key], dtype=float) for key in ("position", "target", "up")]
        forward = target-pos
        if np.linalg.norm(forward) == 0 or np.linalg.norm(np.cross(up, forward)) == 0:
            raise ValueError("Degenerate camera look-at vectors")
        forward /= np.linalg.norm(forward)
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        c2w = np.eye(4)
        c2w[:3, :3] = np.column_stack([forward, left, np.cross(forward, left)])
        c2w[:3, 3] = pos
    if (c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)) or
            not np.allclose(c2w[3], [0, 0, 0, 1]) or
            not np.allclose(c2w[:3, :3].T@c2w[:3, :3], np.eye(3), atol=1e-7) or
            not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-7)):
        raise ValueError(f"Invalid rigid camera transform ({convention})")
    # SAPIEN: X forward, Y left, Z up. Optical: X right, Y down, Z forward.
    conversion = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    g = np.eye(4)
    if convention == "opencv_optical_world_to_camera":
        g = c2w.copy()
    else:
        g[:3, :3] = c2w[:3, :3].T
        if convention == "sapien_camera_to_world":
            g[:3, :3] = conversion@g[:3, :3]
        g[:3, 3] = -g[:3, :3]@c2w[:3, 3]
    # Missing renderer clipping planes mean the natural positive-Z domain,
    # not a guessed physical sensor range. JSON exports represent unbounded far as null.
    near = 0. if spec.get("near") is None else float(spec["near"])
    far = np.inf if spec.get("far") is None else float(spec["far"])
    if not np.isfinite(near) or not 0 <= near < far:
        raise ValueError("Invalid camera clipping planes")
    return Camera(spec["name"], w, h, intrinsics, g, near, far)


def resolve_mask_premise(files):
    """Only an explicit acquisition declaration establishes M3's premise."""
    declarations=[]
    def visit(value,source,key=''):
        if not isinstance(value,dict):return
        for name,item in value.items():
            path=f'{key}.{name}' if key else name
            if name in ('object_only_no_external_occluders','no_external_occluders','external_occluders'):
                if not isinstance(item,bool):raise ValueError(f'{source}:{path} must be a Boolean acquisition premise')
                declarations.append({'source_file':source,'section_or_key':path,'value':not item if name=='external_occluders' else item})
            elif isinstance(item,dict):visit(item,source,path)
    for name in ('metadata/observation_model.json','metadata/sequence.json','metadata/metadata.json','metadata/job_config_resolved.yaml','metadata/config.yaml','metadata/real_conversion.json','sequence.json','metadata.json','config.yaml'):
        if files.exists(name):visit(yaml.safe_load(files.read(name)),name)
    values={record['value'] for record in declarations}
    if len(values)>1:raise ValueError(f'Conflicting object-only/no-external-occluder metadata: {declarations}')
    return {'object_only_no_external_occluders':next(iter(values)) if values else None,'sources':declarations,
            'semantics':'V2 M3: negative support is infeasible only when the premise is explicitly true; otherwise visibility is undetermined. Background depth is never evidence.'}


def load_dataset(input_path, camera_subset=None, frame_subset=None, dt_seconds=None,
                 fps=None, timestamps_file=None, integrity_only=False, frame_time=False, use_masks=True):
    files = SceneFiles(input_path)
    calibration = json.loads(files.read("metadata/cameras.json"))
    specs = calibration["cameras"]
    cameras = [camera_from_metadata(spec, calibration.get("pose_convention")) for spec in specs]
    metadata = {"calibration_source": "metadata/cameras.json",
                "coordinate_frame": calibration.get("coordinate_frame"),
                "calibration_status": calibration.get("calibration_status", "not-declared"),
                "camera_pose_conventions": {spec["name"]: spec.get("pose_convention") or
                    calibration.get("pose_convention") or "sapien_camera_to_world (legacy SAPIEN schema)" for spec in specs},
                "clipping_semantics": "declared renderer planes where present; otherwise Z>0 with no finite far cutoff",
                "depth_storage": "floating-point metric optical Z in depth_npy; zero means no return"}
    metadata['mask_visibility_premise']=resolve_mask_premise(files)
    if files.exists("metadata/real_conversion.json"):
        conversion = json.loads(files.read("metadata/real_conversion.json"))
        metadata["conversion_source"] = "metadata/real_conversion.json"
        metadata["conversion"] = {key: conversion[key] for key in
            ("depth_output_unit", "coordinate_frame", "calibration_status", "calibration_warning", "frame_matching", "timestamps")
            if key in conversion}
        metadata["sensor_independence_qualification"] = (
            "The conditional per-ray likelihood is an assumption. Rectification/resampling can correlate errors; "
            "neither physical independence nor extrinsic/synchronization accuracy is established by loading this scene.")
    if not cameras or len({c.name for c in cameras}) != len(cameras):
        raise ValueError("Dataset must contain uniquely named calibrated cameras")
    frames, by_camera, mask_by_camera, valid_by_camera = None, [], [], []
    for camera in cameras:
        if Path(camera.name).name != camera.name or camera.name in (".", ".."):
            raise ValueError("Camera name must be one directory name")
        rgb = {Path(p).stem: p for p in files.glob("rgb/"+camera.name, "frame_*.png")}
        depth = {Path(p).stem: p for p in files.glob("depth_npy/"+camera.name, "frame_*.npy")}
        if rgb.keys() != depth.keys():
            raise ValueError(f"Missing paired RGB/depth acquisitions in {camera.name}")
        ids = sorted(rgb, key=lambda name: int(name.removeprefix("frame_")))
        if frames is not None and frames != ids:
            raise ValueError("Cameras have different acquisition sets; synchronization is not established")
        frames = ids
        by_camera.append([(rgb[name], depth[name]) for name in ids])
        mask = {}
        for folder in ("mask/", "masks/"):
            found = {Path(p).stem: p for p in files.glob(folder+camera.name, "frame_*.png")}
            if found:
                mask = found; break
        if mask and mask.keys() != rgb.keys():
            raise ValueError(f"Incomplete foreground mask sequence in {camera.name}")
        mask_by_camera.append([mask.get(name) for name in ids])
        validity={}
        for folder in ('depth_validity/','raw_depth_validity/','depth_valid/','valid_depth/'):
            for suffix in ('png','npy'):
                found={Path(p).stem:p for p in files.glob(folder+camera.name,'frame_*.'+suffix)}
                if found:
                    validity=found;break
            if validity:break
        if validity and validity.keys()!=rgb.keys():raise ValueError(f'Incomplete raw depth validity sequence in {camera.name}')
        valid_by_camera.append([validity.get(name) for name in ids])
    if not frames:
        raise ValueError("Dataset is empty")
    if not use_masks or any(path is None for row in mask_by_camera for path in row):raise ValueError('MISSING_FOREGROUND_MASK: all selected acquisitions require an aligned binary mask; unmasked fallback is prohibited')
    frame_ids = [int(name.removeprefix("frame_")) for name in frames]
    try:
        times, timing = resolve_frame_times(files.path, fps, timestamps_file, dt_seconds, frame_ids, frame_time)
    except MissingAcquisitionTiming as error:
        if not integrity_only:
            raise
        times, timing = None, {"status": "MISSING_ACQUISITION_TIMING", "reason": str(error),
            "scene": str(files.path), "time_source": None, "time_units": None, "fps": None,
            "timestamps_seconds": None, "interval_count": len(frame_ids)-1,
            "interval_min_max_seconds": None, "source_kind": None,
            "files_searched": error.searched_files, "keys_searched": error.searched_keys}
    entries = [list(row) for row in zip(*by_camera)]
    masks = [list(row) for row in zip(*mask_by_camera)]
    validity=[list(row) for row in zip(*valid_by_camera)]
    any_valid=any(p is not None for row in validity for p in row)
    if any_valid and any(p is None for row in validity for p in row):raise ValueError('Incomplete raw depth validity across cameras')
    if not any_valid:validity=None
    metadata['raw_depth_validity']='explicit independent validity images' if any_valid else 'finite positive metric depth; no separate validity images provided'
    if camera_subset is not None:
        selection = [i for i, c in enumerate(cameras) if c.name in camera_subset]
        cameras, entries = [cameras[i] for i in selection], [[row[i] for i in selection] for row in entries]
        masks = [[row[i] for i in selection] for row in masks]
        if validity is not None:validity=[[row[i] for i in selection] for row in validity]
    if frame_subset is not None:
        selection = [i for i, frame in enumerate(frame_ids) if frame in frame_subset]
        frame_ids, entries = [frame_ids[i] for i in selection], [entries[i] for i in selection]
        times = None if times is None else times[selection]
        masks = [masks[i] for i in selection]
        if validity is not None:validity=[validity[i] for i in selection]
    if not cameras or not frame_ids:
        raise ValueError("Empty dataset selection")
    return Dataset(files.path,cameras,frame_ids,times,entries,files,timing,metadata,masks,use_masks,validity)


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    camera: object
    k: int
    ci: int
    raw_depth_validity: object = None
    object_only_no_external_occluders: object = None

    @property
    def raw_valid(self):
        physical=np.isfinite(self.depth)&(self.depth>0)
        return physical if self.raw_depth_validity is None else np.asarray(self.raw_depth_validity,dtype=bool)&physical

    @property
    def prepared_rgb(self):
        result=np.zeros((*self.depth.shape,3),float)
        mask=np.asarray(self.mask);result[mask]=self.rgb[mask]
        return result

    @property
    def valid(self):
        return self.mask & self.raw_valid


def scan(dataset):
    counts, foreground_counts, foreground_valid, total = [], [], [], 0
    for k in progress(range(len(dataset.frame_ids)), desc="Validate RGB-D images", unit="frame", leave=True):
        row, fg, fgv = [], [], []
        for ci in range(len(dataset.cameras)):
            observation = dataset.observation(k, ci)
            depth = observation.depth
            row.append(int(np.count_nonzero(np.isfinite(depth)&(depth > 0))))
            fg.append(int(observation.mask.sum()))
            fgv.append(int(observation.valid.sum()))
            total += depth.size
        counts.append(row)
        foreground_counts.append(fg); foreground_valid.append(fgv)
    return {"valid_depth_counts": counts, "candidates": sum(map(sum, counts)),
            "rays": total, "no_return_rays": total-sum(map(sum, counts)),
            "foreground_counts": foreground_counts, "foreground_valid_counts": foreground_valid,
            "foreground_observations": sum(map(sum, foreground_counts)),
            "foreground_valid_observations": sum(map(sum, foreground_valid)),
            "background_valid_observations": sum(map(sum, counts))-sum(map(sum, foreground_valid)),
            "masks_available": bool(dataset.masks and all(all(p is not None for p in row) for row in dataset.masks))}
