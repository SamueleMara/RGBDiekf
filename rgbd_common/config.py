"""Approved numeric noise, explicit structural priors, and acquisition provenance."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import yaml


@dataclass
class Config:
    raw:dict
    source:Path
    rate:np.ndarray
    twist_mean:np.ndarray
    twist_covariance:np.ndarray
    rgbd_covariance:np.ndarray
    rgbd_depth_squared:np.ndarray
    pixel_covariance:np.ndarray
    map_mean:np.ndarray
    map_covariance:np.ndarray
    sensor:dict
    match_prior:float
    numerical:dict
    provenance:dict

    def q(self,dt,groups=1):
        if not np.isfinite(dt) or dt<=0:raise ValueError('Process interval must be positive finite seconds')
        return np.kron(np.eye(groups),self.rate*dt)
    def p0(self,groups=1):
        p=np.zeros((12*groups,12*groups))
        for g in range(groups):p[12*g+6:12*g+12,12*g+6:12*g+12]=self.twist_covariance
        return p


def spd(value,n,name,semidefinite=False):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,n) or not np.isfinite(a).all() or not np.allclose(a,a.T,rtol=1e-12,atol=1e-14):raise ValueError(f'{name}: expected finite symmetric {n}x{n}')
    if semidefinite:
        if np.linalg.eigvalsh(a).min() < -np.finfo(float).eps*n*max(1.,np.linalg.norm(a)):raise ValueError(f'{name}: not PSD')
    else:np.linalg.cholesky(a)
    return a


def load_config(path=None):
    documented=Path(__file__).resolve().parents[1]/'configs/example_uncalibrated.json'
    source=Path(path) if path else documented
    raw=yaml.safe_load(source.read_text());sensor=raw.get('sensor',{})
    if 'visibility' in raw:raise ValueError('OBSOLETE_VISIBILITY_POLICY: remove visibility; v2 M3 uses foreground likelihood assignment and the dataset acquisition premise, with no configurable confidence rule')
    required=['process_covariance_rate','process_noise_units','initial_twists','initial_twist_covariance','map_prior_mean','map_prior_covariance','existence_prior','pixel_response']
    missing=[key for key in required if key not in raw]+['sensor.'+key for key in ('covariance','no_return','background_mean','background_covariance','background_no_return') if key not in sensor]
    if missing:raise ValueError(f'MISSING_NOISE_MODEL: {source}: '+', '.join(missing))
    if raw['process_noise_units']!='local_covariance_per_second':raise ValueError('Expected documented local discrete covariance rate; no alternative Q is invented')
    if raw['pixel_response']!='uniform_unit_box':raise ValueError('Specify pixel localization covariance for a different response law')
    if np.shape(raw['initial_twist_covariance'])!=(6,6) or np.shape(raw['process_covariance_rate'])!=(12,12):raise ValueError('Supply exchangeable per-group 6D twist and 12D process priors; predeclared static/dynamic blocks are unsupported')
    p0=spd(raw['initial_twist_covariance'],6,'initial_twist_covariance');rate=spd(raw['process_covariance_rate'],12,'process_covariance_rate')
    r=spd(sensor['covariance'],4,'sensor.covariance');rd=spd(sensor.get('covariance_depth_squared',np.zeros((4,4))),4,'sensor.covariance_depth_squared',True)
    spd(sensor['background_covariance'],4,'sensor.background_covariance')
    mc=spd(raw['map_prior_covariance'],6,'map_prior_covariance')
    for name,value,shape in [('initial_twists',raw['initial_twists'],(1,6)),('map_prior_mean',raw['map_prior_mean'],(6,)),('sensor.background_mean',sensor['background_mean'],(4,))]:
        array=np.asarray(value,dtype=float)
        if array.shape!=shape or not np.isfinite(array).all():raise ValueError(f'{name}: expected finite {shape}; group priors are exchangeable, not predefined body patterns')
    for key in ('no_return','background_no_return'):
        if not 0<sensor[key]<1:raise ValueError(f'sensor.{key} must be in (0,1)')
    if not 0<raw['existence_prior']<1:raise ValueError('existence_prior must be in (0,1)')
    numerical={key:(int(raw[key]) if key in ('inner_iterations','outer_iterations','line_search_steps') else float(raw[key])) for key in ('inner_iterations','outer_iterations','line_search_steps','damping','armijo','gradient_tolerance','objective_tolerance')}
    if any(not np.isfinite(value) or value<=0 for value in numerical.values()):raise ValueError('Numerical solver budgets/tolerances must be positive finite')
    numerical.update({'sigma_alpha':1.,'sigma_beta':2.,'sigma_kappa':0.})
    kind='documented-initial-guess' if source.resolve()==documented.resolve() else 'explicit-user-assumption'
    provenance={key:{'source_file':str(source.resolve()),'section_or_key':key,'initial_value':raw.get(key),'assumption_class':kind} for key in required}
    for key,value in sensor.items():provenance['sensor.'+key]={'source_file':str(source.resolve()),'section_or_key':'sensor.'+key,'initial_value':value,'assumption_class':kind}
    provenance['pixel_localization']={'source_file':str(source.resolve()),'section_or_key':'pixel_response','initial_value':[[1/12,0],[0,1/12]],'units':'pixel^2','assumption_class':'documented-initial-guess','derivation':'Variance of the documented unit pixel box; Gaussian conditional model required by the parallel specification'}
    for key,record in provenance.items():
        if 'units' in record:continue
        if key=='process_covariance_rate':record['units']='outer([m,rad,m/s,rad/s],[m,rad,m/s,rad/s])/s in right-body-pose/world-twist noise chart'
        elif key=='initial_twists':record['units']='spatial/world [m/s,rad/s], translation-first'
        elif key=='initial_twist_covariance':record['units']='outer([m/s,rad/s],[m/s,rad/s])'
        elif key in ('map_prior_mean','map_prior_covariance'):record['units']='[m,m,m,RGB,RGB,RGB] or its outer product; encoded RGB [0,1]'
        elif 'covariance' in key or key=='sensor.background_mean':record['units']='[RGB,RGB,RGB,m] or its outer product; encoded RGB [0,1]'
        else:record['units']='dimensionless or named model'
    provenance['structural_priors']={'source_file':'Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md','section_or_key':'M6 / A7 / D7; symbolic group/model complexity term','initial_value':None,'units':'dimensionless negative log probability under a specified common measure','assumption_class':'unresolved-mathematical-definition','qualification':'No numerical group/point-count penalty is supplied by the approved noise document. Full variable-model acceptance is undefined until this mathematical term is specified; no production default is invented.'}
    return Config(raw,source,rate,np.asarray(raw['initial_twists'][0]),p0,r,rd,np.eye(2)/12,np.asarray(raw['map_prior_mean']),mc,sensor,float(raw['existence_prior']),numerical,provenance)
