"""Temporal model mismatch is not sensor noise or cross-camera mismatch."""
from pathlib import Path
import json
import numpy as np


class MissingTemporalWarpNoise(ValueError):
    status='MISSING_TEMPORAL_WARP_NOISE'


def temporal_sensor_only(dataset,config,policy):
    """Reuse the existing conditional MAD sensor estimator, never cross-camera mismatch."""
    import copy
    from scene.noise_calibration import temporal_sensor_noise
    cfg,pol=copy.deepcopy(config),copy.deepcopy(policy);sensor=temporal_sensor_noise(dataset)
    cfg.color_covariance=np.diag(sensor['rgb_variance']);cfg.depth_variance=sensor['depth_variance_m2'];cfg.depth_squared_variance=sensor['depth_squared_variance']
    for name,units in [('color_covariance','encoded RGB [0,1]^2'),('depth_variance','m^2'),('depth_squared_variance','dimensionless')]:
        value=getattr(cfg,name);cfg.provenance[name]={'source_file':str(dataset.root),'section_or_key':'scene/noise_calibration.py:temporal_sensor_noise',
             'units':units,'value':value.tolist() if isinstance(value,np.ndarray) else value,'class':'estimated-from-input-under-assumptions',
             'qualification':'static-majority consecutive-pixel MAD and quantization; not independent calibration; may contain motion/selection bias'}
    return cfg,pol,{'sensor':sensor,'ground_truth_used':False,'cross_camera_mismatch':'NOT_COMPUTED_OR_USED_FOR_TEMPORAL_MODEL',
                    'qualification':'conditional sensor estimate only; additional warp discrepancy separate','result':{
                      'rgb_variance':sensor['rgb_variance'],'depth_variance_m2':cfg.depth_variance,'depth_squared_variance':cfg.depth_squared_variance}}


def load_warp_noise(scene, mode='measured', path=None):
    if mode=='ideal':
        if path is not None:raise ValueError('ideal warp assumption cannot also load a covariance file')
        return np.zeros((4,4)),{'model_version':2,'covariance_semantics':'additional_conditional_warp_model_discrepancy',
            'policy':'ideal','assumption_class':'IDEAL_WARP_ASSUMPTION_UNCALIBRATED','method':'exact conditional local translation warp; raw sensor and joint state uncertainty remain active',
            'units':['encoded_rgb_0_1']*3+['m'],'independently_calibrated_warp':False,'covariance_rgbd':np.zeros((4,4)).tolist()}
    candidates = [Path(path)] if path else [Path(scene)/'metadata'/'temporal_warp_noise.json',Path(scene)/'temporal_warp_noise.json']
    selected=next((p for p in candidates if p.is_file()),None)
    if selected is None:
        raise MissingTemporalWarpNoise('MISSING_TEMPORAL_WARP_NOISE: no safely validated temporal-warp calibration is available. '
                         'Sensor MAD and cross-camera residual variance are different quantities. Provide '
                         '--temporal-warp-noise config --temporal-warp-noise-config PATH with covariance_rgbd '
                         '(4x4 SPD; encoded RGB [0,1], metric depth m), assumption_class, method, units. Searched: '
                         +', '.join(str(p) for p in candidates))
    doc=json.loads(selected.read_text())
    if doc.get('covariance_semantics','additional_conditional_warp_model_discrepancy')!='additional_conditional_warp_model_discrepancy':
        raise ValueError('warp configuration must describe ADDITIONAL conditional discrepancy, not total innovation covariance')
    missing=[key for key in ('covariance_rgbd','assumption_class','method','units') if key not in doc]
    if missing:raise ValueError(f'MISSING_TEMPORAL_WARP_NOISE_FIELDS: {missing} in {selected}')
    cov=np.asarray(doc['covariance_rgbd'],dtype=float)
    if cov.shape!=(4,4) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,rtol=1e-12,atol=0):raise ValueError('warp covariance must be finite symmetric 4x4')
    np.linalg.cholesky(cov)
    for key in ('assumption_class','method','units'):
        if not doc.get(key):raise ValueError(f'warp config missing provenance field: {key}')
    if doc['units']!=['encoded_rgb_0_1']*3+['m']:raise ValueError('warp units must be [encoded_rgb_0_1, encoded_rgb_0_1, encoded_rgb_0_1, m]')
    if mode=='measured' and doc['assumption_class'] not in ('empirical-Bayes','independently-calibrated'):
        raise ValueError('measured warp policy requires empirical-Bayes or independently-calibrated provenance; use config for explicit assumptions')
    return cov,{**doc,'model_version':2,'input_model_version':doc.get('model_version',1),'covariance_semantics':'additional_conditional_warp_model_discrepancy','legacy_v1_interpretation':'additional warp discrepancy; unchanged',
                'source_file':str(selected.resolve()),'policy':mode,'condition_number_in_declared_units':float(np.linalg.cond(cov)),
                'independently_calibrated_warp':doc['assumption_class']=='independently-calibrated'}
