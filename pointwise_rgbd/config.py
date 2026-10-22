from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2
from rgbd_common.config import load_config as approved_noise


@dataclass
class Config:
    qc: np.ndarray
    qx: np.ndarray
    velocity_covariance: np.ndarray
    pixel_covariance: np.ndarray
    depth_variance: float
    depth_squared_variance: float
    color_covariance: np.ndarray
    provenance: dict
    association_tail_probability: float | None = None
    neighbors: int = 8
    iterations: int = 12
    line_search_steps: int = 24
    tolerance: float = 1e-5

    def gate(self, dimensions=3):
        if self.association_tail_probability is None:
            raise ValueError('MISSING_ASSOCIATION_TAIL_PROBABILITY: Section 6 requires a declared tail; supply --association-tail-probability or an explicit pointwise config value')
        if not 0 < self.association_tail_probability < 1:
            raise ValueError('Association tail probability must be in (0,1)')
        return float(chi2.isf(self.association_tail_probability, dimensions))

    def measurement_covariance(self, depth):
        depth=np.asarray(depth,dtype=float)
        out=np.zeros((*depth.shape,3,3));out[...,:2,:2]=self.pixel_covariance
        out[...,2,2]=self.depth_variance+self.depth_squared_variance*depth**2
        return out

    def manifest(self):
        return {key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in vars(self).items()}


def load_config(path=None, tail_probability=None):
    original=approved_noise(path)
    policy_path=Path(__file__).resolve().parents[1]/'configs/pointwise_association.json'
    policy=json.loads(policy_path.read_text());overridden=tail_probability is not None
    if not overridden:tail_probability=policy['association_tail_probability']
    # The approved translational velocity diffusion is acceleration intensity;
    # integrate it with position, retaining all cross blocks (theory Eq. 7).
    q=original.rate
    config=Config(q[6:9,6:9].copy(),q[:3,:3].copy(),original.twist_covariance[:3,:3].copy(),
        original.pixel_covariance.copy(),float(original.rgbd_covariance[3,3]),
        float(original.rgbd_depth_squared[3,3]),original.rgbd_covariance[:3,:3].copy(),{},tail_probability,
        iterations=original.numerical['inner_iterations'],line_search_steps=original.numerical['line_search_steps'],tolerance=original.numerical['gradient_tolerance'])
    for name,source,units in [('qc','process_covariance_rate[6:9,6:9]','m^2/s^3'),('qx','process_covariance_rate[0:3,0:3]','m^2/s'),('velocity_covariance','initial_twist_covariance[0:3,0:3]','m^2/s^2'),('pixel_covariance','pixel_response=uniform_unit_box','pixel^2'),('depth_variance','sensor.covariance[3,3]','m^2'),('depth_squared_variance','sensor.covariance_depth_squared[3,3] if supplied; otherwise zero term','dimensionless'),('color_covariance','sensor.covariance[0:3,0:3]','encoded RGB [0,1]^2')]:
        value=getattr(config,name)
        config.provenance[name]={'source_file':str(original.source.resolve()),'section_or_key':source,'units':units,'value':value.tolist() if isinstance(value,np.ndarray) else value,'class':'documented-initial-guess' if path is None else 'explicit-user-assumption'}
    config.provenance['qc']['discretization']='White acceleration integrated exactly with Eq. 7; this supersedes the earlier 12D discrete process, not a copied diagonal Q'
    for name in ('qc','qx','velocity_covariance','pixel_covariance','color_covariance'):
        value=getattr(config,name)
        if not np.isfinite(value).all() or not np.allclose(value,value.T):raise ValueError(f'{name} is not finite symmetric')
        if np.linalg.eigvalsh(value).min()<0:raise ValueError(f'{name} is not PSD')
    np.linalg.cholesky(config.velocity_covariance)
    config.neighbors=int(policy['neighbors'])
    config.provenance['association']={'source_file':'--association-tail-probability' if overridden else str(policy_path),'tail_probability':tail_probability,'class':'explicit-user-assumption' if overridden else policy['assumption_class'],'qualification':policy['qualification'],'candidate_family':f'{config.neighbors} Euclidean kNN candidates per predicted point/camera; finite local association, not global correspondence inference','policy':'3D innovation chi-square gate, 2D ray gate, foreground depth-order gate; RGB affinity only; deterministic exclusive assignment; numerical ties deferred'}
    config.provenance['surface_coverage']={'candidate_family':f'Each observed foreground world XYZ queries {config.neighbors} transported SOLVED world XYZ samples','geometry_covariance':'2*(P_position + Sigma_observation), PSD upper bound with unknown cross-correlation','rgb_covariance':'4*R_rgb, PSD upper bound with unknown cross-correlation','score':'sum of marginal Gaussian negative log likelihoods; whitened RGB angular + brightness equals one RGB distance','ownership':'many observations may share a surface point; coverage never assimilates measurements','tail_probability':tail_probability,'qualification':'Authorized uncalibrated component-wise policy, not a calibrated joint confidence; RGB is matching only'}
    config.gate()
    return config
