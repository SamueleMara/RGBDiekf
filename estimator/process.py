import numpy as np
import torch


def transition(dt, reverse=False):
    if not np.isfinite(dt) or dt<=0:raise ValueError('Expected positive physical interval in seconds')
    f=np.eye(6);f[:3,3:]=np.eye(3)*(-dt if reverse else dt)
    return f


def process_covariance(dt,qc,qx=None,reverse=False):
    transition(dt,reverse)
    qc=np.asarray(qc);qx=np.zeros((3,3)) if qx is None else np.asarray(qx)
    sign=-1 if reverse else 1
    return np.block([[dt**3/3*qc+dt*qx,sign*dt**2/2*qc],[sign*dt**2/2*qc,dt*qc]])


def predict(state,covariance,dt,config,reverse=False):
    f=state.new_tensor(transition(dt,reverse));q=state.new_tensor(process_covariance(dt,config.qc,config.qx,reverse))
    p=f@covariance@f.T+q
    return state@f.T,(p+p.transpose(-1,-2))/2
