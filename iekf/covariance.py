"""Small conditional covariance and sparse selected solves, never dense inverse."""
import numpy as np
from scipy.sparse.linalg import cg


def selected_covariance(information,indices,rtol=1e-9,maxiter=2000):
    columns=[];residuals=[]
    for index in indices:
        rhs=np.zeros(information.shape[0]);rhs[index]=1
        x,status=cg(information,rhs,rtol=rtol,atol=0,maxiter=maxiter)
        error=np.linalg.norm(information@x-rhs)
        if status or error>rtol*10:raise RuntimeError(f'SELECTED_COVARIANCE_NOT_SOLVED status={status}, residual={error}')
        columns.append(x[indices]);residuals.append(float(error))
    return np.column_stack(columns),residuals
