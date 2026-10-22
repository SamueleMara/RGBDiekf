"""Selected cross-time CONDITIONAL smoother moments, no dense global matrix."""
import numpy as np


class TrajectoryMoments:
    def __init__(self,steps,smoothed):
        states,self.covariance,report=smoothed;self.regressions=[]
        for k,gain in enumerate(report['gains']):
            current,nxt=steps[k:k+2]
            correction=states[k].minus(current.filtered)
            difference=states[k+1].minus(nxt.predicted)
            self.regressions.append(current.filtered.chart(correction)@gain@np.linalg.inv(nxt.predicted.chart(difference)))
    def block(self,k,t):
        if k>t:return self.block(t,k).T
        value=self.covariance[t].copy()
        for j in range(t-1,k-1,-1):value=self.regressions[j]@value
        return value


def error_moments_at_reference(means,covariance,references):
    """First-order moments of (state boxminus reference), with nonzero mean.

    Re-express the SAME saved Gaussian; this does not move its posterior mean
    to an accepted MAP nominal or turn a rejected proposal into an update.
    """
    offsets=[];central=[];second=[]
    for mean,value,reference in zip(means,covariance,references):
        offset=mean.minus(reference);back=np.linalg.inv(reference.chart(offset))
        transported=back@value@back.T;transported=(transported+transported.T)/2
        offsets.append(offset);central.append(transported);second.append(transported+np.outer(offset,offset))
    return np.array(offsets),np.array(central),np.array(second)
