# SUPERSEDED persistent-physical-supernode experiment; not used by the corrected free-point pipeline.
"""Exact linearized continuous spatial acceleration factor, either time direction.

A = [[ad(xi), I], [0,0]], B=[0,I]^T, Qc acceleration spectral density.
F(h)=exp(A h). For d>0, Q(-d)=F(-d) Q(d) F(-d)^T, i.e. the
forward Gaussian process factor reoriented, NOT a reverse Bayesian smoother.
Physical spatial velocity retains its sign when iteration reverses.
"""
from dataclasses import dataclass
import numpy as np
import torch
from rgbd_common.config import load_config as approved_noise
from .group_pose import ad,exp


@dataclass
class GroupNoise:
    acceleration_spectral_density: np.ndarray
    initial_twist_covariance: np.ndarray
    provenance: dict


def load_group_noise(path=None):
    old=approved_noise(path)
    # Numeric magnitudes explicitly documented for 6D twist diffusion; NEW
    # continuous spatial-acceleration model, not the old discrete 12D Q.
    qc=old.rate[6:,6:].copy();p=old.twist_covariance.copy()
    np.linalg.cholesky(qc);np.linalg.cholesky(p)
    return GroupNoise(qc,p,{'source_file':str(old.source.resolve()),'numeric_keys':['process_covariance_rate[6:12,6:12]','initial_twist_covariance'],'assumption_class':'documented-initial-guess' if path is None else 'explicit-user-assumption','model_class':'declared-group-process-assumption','acceleration_spectral_density':qc.tolist(),'initial_twist_covariance':p.tolist(),'Qc_units':'outer([m/s,rad/s])/s; spatial world translation-first','Ptwist_units':'outer([m/s,rad/s])','discretization':'continuous constant nominal spatial twist; exact Van Loan controllability Gramian of local error A, signed factor reorientation','direct_pose_diffusion':'none; old pose diffusion entries are not copied','anchor_pose_covariance':'eliminated conditioned endpoint gauge, exactly zero','canonical_geometry':'condition on measured endpoint XYZ; its sensor covariance is retained as source provenance, not independently added once per group member','group_covariance':'conditional independent 12x12 blocks; shared calibration/association uncertainty not marginalized'})


def matrices(twist,h,qc):
    if not np.isfinite(h) or h==0:raise ValueError('A nonzero signed physical interval in seconds is required')
    n=len(twist);a=twist.new_zeros((n,12,12));a[:,:6,:6]=ad(twist);a[:,:6,6:]=torch.eye(6,device=twist.device,dtype=twist.dtype)
    w=torch.zeros_like(a);w[:,6:,6:]=torch.as_tensor(qc,device=twist.device,dtype=twist.dtype)
    generator=a*(1 if h>0 else -1)
    block=twist.new_zeros((n,24,24));block[:,:12,:12]=generator;block[:,:12,12:]=w;block[:,12:,12:]=-generator.transpose(-1,-2)
    e=torch.linalg.matrix_exp(abs(float(h))*block);f=e[:,:12,:12];q=e[:,:12,12:]@f.transpose(-1,-2)
    return f,(q+q.transpose(-1,-2))/2


def predict(poses,twists,covariance,h,noise,chunk=512):
    result=exp(float(h)*twists)@poses;posterior=torch.empty_like(covariance)
    for lo in range(0,len(poses),chunk):
        hi=lo+chunk;f,q=matrices(twists[lo:hi],h,noise.acceleration_spectral_density)
        p=f@covariance[lo:hi]@f.transpose(-1,-2)+q
        posterior[lo:hi]=(p+p.transpose(-1,-2))/2
    return result,twists.clone(),posterior


def initialize(count,noise,backend):
    poses=backend.tensor(np.repeat(np.eye(4)[None],count,axis=0));twists=poses.new_zeros((count,6));p=poses.new_zeros((count,12,12));p[:,6:,6:]=backend.tensor(noise.initial_twist_covariance)
    return poses,twists,p
