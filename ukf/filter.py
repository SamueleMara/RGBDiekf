"""Standalone Lie UKF; gain damping uses its own sigma moments only."""
import numpy as np
from scipy.linalg import block_diag
from rgbd_common.lie import State,exp
from rgbd_common.backends.torch_cuda import numpy
from .dynamics import augmented_predict
from .measurement import measurement_moments,woodbury
from .smoother import smooth


def block_separable(covariance,groups):
    diagonal=block_diag(*[covariance[12*g:12*g+12,12*g:12*g+12] for g in range(groups)])
    return bool(np.array_equal(covariance,diagonal))


class GroupChart:
    """12D chart for one genuinely independent conditional group, shared map."""
    dim=12
    def __init__(self,full,group):self.full=full;self.group=group
    @property
    def poses(self):return self.full.poses
    @property
    def twists(self):return self.full.twists
    def plus(self,delta):
        d=np.asarray(delta);poses=self.full.poses.copy();twists=self.full.twists.copy()
        poses[self.group]=exp(d[:6])@poses[self.group];twists[self.group]+=d[6:]
        return GroupChart(State(poses,twists),self.group)
    def minus(self,base):
        return State(self.poses[self.group:self.group+1],self.twists[self.group:self.group+1]).minus(State(base.poses[self.group:self.group+1],base.twists[self.group:self.group+1]))
    def chart(self,delta):return State(self.poses[self.group:self.group+1],self.twists[self.group:self.group+1]).chart(delta)


class UKF:
    name='UKF'
    def __init__(self,config,backend,sigma_reducer=measurement_moments):
        self.config=config;self.backend=backend
        self.sigma_reducer=sigma_reducer
        self.parameters=tuple(config.numerical[key] for key in ('sigma_alpha','sigma_beta','sigma_kappa'))
    def predict(self,state,covariance,dt):
        groups=len(state.poses)
        if groups>1 and block_separable(covariance,groups):
            results=[]
            for g in range(groups):results.append(augmented_predict(State(state.poses[g:g+1],state.twists[g:g+1]),covariance[12*g:12*g+12,12*g:12*g+12],dt,self.config.q(dt),self.parameters))
            predicted=State(np.concatenate([r[0].poses for r in results]),np.concatenate([r[0].twists for r in results]))
            return predicted,block_diag(*[r[1] for r in results]),{'cross_covariance':block_diag(*[r[2] for r in results]),'sigma_count_per_group':[49]*groups,'cross_source':'same independent per-group augmented sigma ensembles; conditional independent process/prior blocks'}
        predicted,pminus,cross,report=augmented_predict(state,covariance,dt,self.config.q(dt,len(state.poses)),self.parameters)
        return predicted,pminus,{'cross_covariance':cross,**report}
    def correct(self,base,pminus,problem,reference=False,nominal=None):
        groups=base.dim//12
        if groups>1 and block_separable(pminus,groups) and hasattr(problem,'for_group'):
            states=[];covariances=[];reports=[]
            for g in range(groups):
                candidate,cov,report=self._correct(GroupChart(base,g),pminus[12*g:12*g+12,12*g:12*g+12],problem.for_group(g),reference)
                states.append(candidate.full);covariances.append(cov);reports.append(report)
            result=State(np.array([state.poses[g] for g,state in enumerate(states)]),np.array([state.twists[g] for g,state in enumerate(states)]))
            if hasattr(problem,'feasible') and not problem.feasible(result):return base,pminus.copy(),{'group_diagnostics':reports,'proposal_inf':0.,'inner_solve_metric':None,'unresolved_mask_visibility_conflict':True,'covariance':'Prediction retained after infeasible combined UKF candidate; no sigma clipping or hybrid correction'}
            return result,block_diag(*covariances),{'group_diagnostics':reports,'sigma_count_per_group':[25]*groups,'proposal_inf':max(r['proposal_inf'] for r in reports),'inner_solve_metric':None if any(r.get('inner_solve_metric') is None for r in reports) else max(r['inner_solve_metric'] for r in reports),'inner_stationarity_kind':'Relative backward error of each sigma moment system; no MAP-stationarity claim','active_factors':sum(r.get('active_factors',0) for r in reports),'deferred_visibility_factors':sum(r.get('deferred_visibility_factors',0) for r in reports),'separability':'Conditional fixed-map/fixed-assignment model with independent group prior/process blocks and one raw owner per factor. Coupled covariance uses joint sigma points instead; map-marginal coupling is not discarded or claimed independent.'}
        return self._correct(base,pminus,problem,reference)
    def _correct(self,base,pminus,problem,reference=False):
        a,m,b,report=self.sigma_reducer(problem,base,pminus,self.parameters)
        matrix=numpy(m);rhs=numpy(b)
        delta,pfull=woodbury(a,pminus,matrix,rhs)
        solved=np.linalg.solve(matrix,rhs)
        denominator=np.linalg.norm(matrix,np.inf)*np.linalg.norm(solved,np.inf)+np.linalg.norm(rhs,np.inf)
        error=float(np.linalg.norm(matrix@solved-rhs,np.inf)/max(np.finfo(float).tiny,denominator))
        report.update(inner_solve_metric=error,inner_stationarity_kind='Relative backward error of sigma moment system; no MAP-stationarity claim')
        free=np.array([12*g+j for g in range(base.dim//12) for j in range(6,12)]) if reference else np.arange(base.dim)
        precision=np.linalg.inv(pminus[np.ix_(free,free)])
        before=problem.evaluate(base);candidate=base.plus(delta)
        value=.5*delta[free]@precision@delta[free]+problem.evaluate(candidate)
        if not np.isfinite(value):return base,pminus.copy(),{**report,'inner_solve_metric':None,'proposal_inf':float(np.max(abs(delta))),'accepted':False,'objective':float(before) if np.isfinite(before) else None,'unresolved_mask_visibility_conflict':True,'covariance':'prediction retained after invalid nonlinear sigma proposal'}
        # Canonical one-shot UKF moment update. Its mean is not a MAP step;
        # the common OUTER score accepts/rejects the resulting trajectory.
        # Rejecting every small Jensen-bias correction here would also discard
        # valid measurement information and make a stationary group diverge.
        chart=base.chart(delta);posterior=chart@pfull@chart.T;posterior=(posterior+posterior.T)/2
        minimum=np.linalg.eigvalsh(posterior).min();tolerance=np.finfo(float).eps*base.dim*max(1.,np.linalg.norm(pminus,2))
        if minimum < -tolerance:raise ArithmeticError(f'UKF posterior covariance is indefinite: {minimum}')
        return candidate,posterior,{**report,'proposal_inf':float(np.max(abs(delta))),'accepted':True,'conditional_objective_before':float(before) if np.isfinite(before) else None,'objective':float(value),'conditional_score_increased':bool(value>before),'correction_inf':float(np.max(abs(delta))),'moment_matrix':numpy(m).tolist(),'state_sigma_factor':a.tolist(),'moment_rhs':numpy(b).tolist(),'covariance':'canonical one-shot conditional sigma moments with left-chart transport; hard-mask feasibility checked at nominal, truncated posterior approximated locally; trajectory acceptance is external; not a map marginal'}
    def smooth(self,steps):return smooth(steps)
    def refine_reverse_hypothesis(self,replay):return self.smooth(replay())
