"""UKF-only moment consistency diagnostics; no analytical transition substitute."""
import numpy as np


def sigma_covariance(deviations,weights):
    value=deviations.T@(deviations*np.asarray(weights)[:,None]);return (value+value.T)/2


def source_cross_covariance(state_deviations,point_deviations,weights):
    return state_deviations.T@(point_deviations*np.asarray(weights)[:,None])
