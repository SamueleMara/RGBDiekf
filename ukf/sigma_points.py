"""Deterministic PSD sigma construction and left-chart manifold moments."""
import numpy as np


def sigma_points(covariance,alpha=1.,beta=2.,kappa=0.):
    p=(covariance+covariance.T)/2;n=len(p);scale=alpha*alpha*(n+kappa)
    if scale<=0:raise ValueError('Invalid sigma scaling')
    wm=np.full(2*n+1,1/(2*scale));wc=wm.copy();wm[0]=(scale-n)/scale;wc[0]=wm[0]+1-alpha*alpha+beta
    if np.any(wc<0):raise ValueError('Woodbury sigma correction requires nonnegative covariance weights')
    try:root=np.linalg.cholesky(p)
    except np.linalg.LinAlgError:
        eigen,vectors=np.linalg.eigh(p);tolerance=np.finfo(float).eps*n*max(1.,np.linalg.norm(p,2))
        if eigen.min() < -tolerance:raise ValueError(f'INDEFINITE_SIGMA_COVARIANCE: {eigen.min()}')
        root=vectors*np.sqrt(np.maximum(eigen,0))[None,:]
    root*=np.sqrt(scale);eta=np.r_[np.zeros((1,n)),root.T,-root.T]
    return eta,wm,wc


def manifold_mean(states,weights,tolerance=1e-12,max_iterations=100):
    mean=states[0]
    for _ in range(max_iterations):
        delta=np.einsum('i,ij->j',weights,np.array([s.minus(mean) for s in states]))
        if np.linalg.norm(delta,np.inf)<=tolerance:return mean
        mean=mean.plus(delta)
    raise RuntimeError('MANIFOLD_MEAN_NOT_CONVERGED: sigma set may leave a common logarithm chart')


def moments(states,wm,wc):
    mean=manifold_mean(states,wm);deviation=np.array([s.minus(mean) for s in states])
    covariance=deviation.T@(deviation*wc[:,None])
    return mean,(covariance+covariance.T)/2,deviation
