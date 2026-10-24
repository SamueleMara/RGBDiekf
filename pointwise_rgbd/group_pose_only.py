# SUPERSEDED persistent-physical-supernode experiment; not used by the corrected free-point pipeline.
"""Separate 6D left-world pose random-walk ablation; no velocity state.

Tminus=Tplus, F=I, Q(h)=abs(h)*Q_pose. The initial endpoint pose is conditioned.
Numeric pose diffusion comes from the already documented local pose block;
this is an explicit ablation assumption, never calibrated physical accuracy.
"""
from dataclasses import dataclass
import numpy as np
from rgbd_common.config import load_config


@dataclass
class PoseNoise:
    pose_spectral_density:np.ndarray
    provenance:dict


def load_pose_noise(path=None):
    source=load_config(path);q=source.rate[:6,:6].copy();np.linalg.cholesky(q)
    return PoseNoise(q,{'model_class':'pose_only_6d_random_walk_ablation','source_file':str(source.source.resolve()),
        'section_or_key':'process_covariance_rate[0:6,0:6]','assumption_class':'documented-initial-guess' if path is None else 'explicit-user-assumption',
        'pose_spectral_density':q.tolist(),'units':'outer([metres,radians])/second; translation-first spatial left tangent',
        'dynamics':'T_next_minus=T_current_plus; F=I6; Q=abs(signed_dt_seconds)*Q_pose',
        'anchor':'conditioned identity pose; P0 eliminated',
        'velocity_state':'ABSENT','canonical_geometry':'conditioned measured endpoint; source covariance retained',
        'group_covariance':'conditional blocks; fixed calibration and associations; not independently calibrated'})


def initialize(count,noise,backend):
    pose=backend.tensor(np.repeat(np.eye(4)[None],count,axis=0))
    return pose,None,pose.new_zeros((count,6,6))


def predict(poses,no_velocity,covariance,h,noise):
    if no_velocity is not None:raise ValueError('Pose-only model has no velocity state')
    if not np.isfinite(h) or h==0:raise ValueError('Nonzero signed interval in seconds required')
    q=covariance.new_tensor(noise.pose_spectral_density)*abs(float(h))
    return poses.clone(),None,covariance+q
