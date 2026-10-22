"""Standalone analytical sparse IEKF; no unscented imports or fallbacks."""
import numpy as np
from scipy.linalg import block_diag
from rgbd_common.backends.torch_cuda import numpy
from .dynamics import transition
from .measurement import reduce_information
from .smoother import smooth


def newton_decrement(gradient,information):
    # Dimensionless local Gaussian objective metric, invariant to linear units.
    root=np.linalg.cholesky((information+information.T)/2)
    return float(np.linalg.norm(np.linalg.solve(root,gradient)))


class IEKF:
    name='IEKF'
    def __init__(self,config,backend,information_reducer=reduce_information):
        self.config=config;self.backend=backend;self.information_reducer=information_reducer
    def predict(self,state,covariance,dt):
        pred,f,g,q=transition(state,dt,self.config.q(dt,len(state.poses)))
        p=f@covariance@f.T+q
        return pred,(p+p.T)/2,{'transition':f,'injection':g}
    def correct(self,base,pminus,problem,reference=False,nominal=None):
        n=base.dim;free=np.array([12*g+j for g in range(len(base.poses)) for j in range(6,12)]) if reference else np.arange(n)
        precision=np.linalg.inv(pminus[np.ix_(free,free)])
        delta=np.zeros(len(free));settings=self.config.numerical;records=[];stalled=False
        def evaluate(d,linearize=False):
            full=np.zeros(n);full[free]=d;state=base.plus(full)
            value=.5*d@precision@d+problem.evaluate(state)
            if not linearize:return value
            blocks,score=self.information_reducer(problem,state);local=block_diag(*numpy(blocks));g=numpy(score).ravel()
            chart=base.chart(full)[:,free]
            return value,precision@d-chart.T@g,precision+chart.T@local@chart
        if nominal is not None:
            proposed=nominal.minus(base)[free]
            if evaluate(proposed)<evaluate(delta):delta=proposed
        for _ in range(settings['inner_iterations']):
            value,gradient,information=evaluate(delta,True);norm=float(np.max(np.abs(gradient)))
            metric=newton_decrement(gradient,information)
            records.append({'objective':float(value) if np.isfinite(value) else None,'gradient_inf':norm,'newton_decrement':metric})
            if metric<=settings['gradient_tolerance']:break
            direction=-np.linalg.solve(information+settings['damping']*np.eye(len(free)),gradient)
            accepted=False
            for line in range(settings['line_search_steps']):
                step=2.**(-line)*direction
                trial=evaluate(delta+step)
                if np.isfinite(trial) and trial<=value+settings['armijo']*(gradient@step):delta+=step;accepted=True;break
            if not accepted:stalled=True;break
        value,gradient,information=evaluate(delta,True);full=np.zeros(n);full[free]=delta
        if not np.isfinite(value):return base,pminus.copy(),{'history':[],'objective':None,'inner_solve_metric':None,'unresolved_mask_visibility_conflict':True,'covariance':'Prediction retained; no feasible conditional posterior nominal found'}
        chart=base.chart(full)[:,free];posterior=chart@np.linalg.solve(information,np.eye(len(free)))@chart.T
        return base.plus(full),(posterior+posterior.T)/2,{'history':records,'objective':float(value),'gradient_inf':float(np.max(abs(gradient))),'inner_solve_metric':newton_decrement(gradient,information),'inner_stationarity_kind':'Undamped conditional Gauss-Newton decrement; dimensionless local solve criterion, not global association stationarity','stalled':stalled,'covariance':'undamped conditional fixed-map/fixed-match IEKF information; left chart'}
    def smooth(self,steps):return smooth(steps)
    def refine_reverse_hypothesis(self,replay):
        # Replay ORIGINAL stochastic priors and each current raw factor once.
        # The callback performs an actual forward trace with added hypotheses.
        return self.smooth(replay())
