"""Numerical audit of the actual first full real filter/smoother proposals."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing audit; select a fresh output')
    rows=[]
    for name,trial in [('IEKF','full_scene_02'),('UKF','full_scene_04')]:
        path=args.root/trial/name.lower()/'outer_000/filter/forward_smooth.npz'
        with np.load(path) as data:
            cov=data['smoothed_covariance'];eigen=np.linalg.eigvalsh(cov);poses=data['smoothed_poses']
            rotation=poses[:,:,:3,:3];identity=np.eye(3)
            rows.append({'pipeline':name,'trace':str(path),'frames':len(poses),'groups':poses.shape[1],'covariance_shape':list(cov.shape),'finite_covariance':bool(np.isfinite(cov).all()),'max_symmetry_error':float(np.max(abs(cov-cov.swapaxes(-1,-2)))),'min_eigenvalue':float(eigen.min()),'max_eigenvalue':float(eigen.max()),'max_rotation_orthogonality_error':float(np.max(abs(rotation.swapaxes(-1,-2)@rotation-identity))),'saved_analytical_transition_count':len(data['transitions']),'saved_sigma_cross_covariance_count':len(data['sigma_cross_covariances']),'covariance_mean':'own saved smoothed proposal, not necessarily final accepted map/trajectory','qualification':'Local conditional fixed-map/fixed-factor covariance, not calibrated physical accuracy, hard-set truncation or joint map marginal'})
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps({'first_real_smoother_proposals':rows,'completed_reconstruction':False},indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
