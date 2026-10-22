"""Analytical RTS using saved F, with both covariance chart transports."""
import numpy as np


def smooth(steps):
    states=[s.filtered for s in steps];cov=[s.filtered_covariance.copy() for s in steps];gains=[]
    gains=[None]*(len(steps)-1)
    for k in range(len(steps)-2,-1,-1):
        current,nxt=steps[k:k+2]
        gain=np.linalg.solve(nxt.predicted_covariance,nxt.transition@current.filtered_covariance).T
        difference=states[k+1].minus(nxt.predicted);back=np.linalg.inv(nxt.predicted.chart(difference))
        correction=gain@difference;transport=current.filtered.chart(correction)
        cov[k]=transport@(current.filtered_covariance+gain@(back@cov[k+1]@back.T-nxt.predicted_covariance)@gain.T)@transport.T
        cov[k]=(cov[k]+cov[k].T)/2;states[k]=current.filtered.plus(correction);gains[k]=gain
    return states,cov,{'gains':gains,'method':'analytical RTS; conditional on fixed points and raw matches'}
