"""Full 12-column numerical reference; never imported by production."""
import torch
from iekf.measurement import h_H
from rgbd_common.observation_model import apply_precision


def reduce_information(problem,state):
    backend=problem.backend;groups=len(state.poses)
    information=torch.zeros((groups,12,12),device=backend.device,dtype=torch.float64)
    score=torch.zeros((groups,12),device=backend.device,dtype=torch.float64)
    for match in problem.chunks():
        group=problem.group[match.point]
        predicted,h=h_H(state,problem.xyz[match.point],problem.rgb[match.point],group,match.camera,backend)
        weighted_h=apply_precision(match.covariance,h)
        residual=apply_precision(match.covariance,(match.observed-predicted)[...,None])[...,0]
        for g in range(groups):
            select=group==g;hh=h[select].reshape(-1,12)
            information[g]+=hh.T@weighted_h[select].reshape(-1,12)
            score[g]+=hh.T@residual[select].reshape(-1)
    return information,score
