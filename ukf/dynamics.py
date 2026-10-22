"""Nonlinear augmented-sigma prediction. No analytical F/G or IEKF imports."""
import numpy as np
from scipy.linalg import block_diag
from rgbd_common.lie import exp,State
from .sigma_points import sigma_points,moments


def nonlinear_transition(state,dt,noise):
    w=np.asarray(noise).reshape(-1,12)
    poses=np.array([exp(dt*nu)@pose@exp(e[:6]) for pose,nu,e in zip(state.poses,state.twists,w)])
    return State(poses,state.twists+w[:,6:])


def augmented_predict(state,covariance,dt,q,parameters=(1.,2.,0.)):
    n=state.dim;eta,wm,wc=sigma_points(block_diag(covariance,q),*parameters)
    previous=[state.plus(e[:n]) for e in eta]
    propagated=[nonlinear_transition(s,dt,e[n:]) for s,e in zip(previous,eta)]
    predicted,pminus,dy=moments(propagated,wm,wc)
    dx=np.array([s.minus(state) for s in previous]);cross=dx.T@(dy*wc[:,None])
    return predicted,pminus,cross,{'sigma_count':len(eta),'cross_source':'same augmented nonlinear sigma ensemble'}
