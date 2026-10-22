"""Unscented smoother driven exclusively by saved sigma cross-covariances."""
import numpy as np


def smooth(steps):
    states=[s.filtered for s in steps];covariances=[s.filtered_covariance.copy() for s in steps]
    gains=[None]*(len(steps)-1)
    for k in range(len(steps)-2,-1,-1):
        current,nxt=steps[k:k+2]
        if nxt.cross_covariance is None:raise ValueError('Unscented smoothing requires the saved augmented-sigma cross covariance')
        gain=np.linalg.solve(nxt.predicted_covariance,nxt.cross_covariance.T).T
        difference=states[k+1].minus(nxt.predicted);back=np.linalg.inv(nxt.predicted.chart(difference))
        correction=gain@difference;chart=current.filtered.chart(correction)
        covariance=chart@(current.filtered_covariance+gain@(back@covariances[k+1]@back.T-nxt.predicted_covariance)@gain.T)@chart.T
        covariances[k]=(covariance+covariance.T)/2;states[k]=current.filtered.plus(correction);gains[k]=gain
    return states,covariances,{'gains':gains,'method':'unscented smoother; saved augmented nonlinear sigma cross covariance; conditional map/matches'}
