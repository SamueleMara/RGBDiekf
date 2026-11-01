"""Independent small CPU oracles for fifth-pass F1 forensics, never inference."""
import math
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import chi2

LEVELS=('EXACT_CODE_AND_ARITHMETIC','COUNTERFACTUAL_CAUSAL_WITH_FIXED_CONTEXT','OBSERVATIONALLY_SUPPORTED','NOT_IDENTIFIABLE','FALSIFIED','INDETERMINATE')
# Reproduction resolution, not sensor noise, a motion threshold, or tuned policy.
PARITY_X_M=2e-12
PARITY_V_M_S=1e-10
PARITY_COV_ATOL=1e-12
PARITY_COV_RTOL=1e-9


def covariance_blocks(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float);out={}
    for name,sl,unit in [('xx',(slice(0,3),slice(0,3)),'m2'),('vx',(slice(3,6),slice(0,3)),'m2_s'),('vv',(slice(3,6),slice(3,6)),'m2_s2')]:
        e=a[sl]-b[sl];out[f'P{name}_linf_{unit}']=float(abs(e).max());out[f'P{name}_fro_{unit}']=float(np.linalg.norm(e))
    return out


def covariance_agreement(a,b):
    """Compare covariance blocks in their own units, with explicit norm bounds.

    Entrywise relative errors at nearly zero cross-correlations are not a block
    norm. Retain that stricter diagnostic, rather than silently discarding it.
    """
    a,b=np.asarray(a,float),np.asarray(b,float);checks={}
    for name,sl,unit in [('xx',(slice(0,3),slice(0,3)),'m2'),('xv',(slice(0,3),slice(3,6)),'m2_s'),('vx',(slice(3,6),slice(0,3)),'m2_s'),('vv',(slice(3,6),slice(3,6)),'m2_s2')]:
        error=float(np.linalg.norm(a[sl]-b[sl]));scale=float(np.linalg.norm(b[sl]));bound=PARITY_COV_ATOL+PARITY_COV_RTOL*scale
        checks[name]={'frobenius_error':error,'reference_frobenius':scale,'bound':bound,'units':unit,'equal':bool(error<=bound)}
    return {'covariance_equal':bool(np.allclose(a,b,atol=PARITY_COV_ATOL,rtol=PARITY_COV_RTOL)),
            'covariance_block_norm_equal':all(v['equal'] for v in checks.values()),'covariance_block_checks':checks,
            'covariance_elementwise_equal':bool(np.allclose(a,b,atol=PARITY_COV_ATOL,rtol=PARITY_COV_RTOL)),
            'covariance_comparison_policy':'Original strict entrywise allclose gate retained; per-block Frobenius diagnostics do not override it. atol=1e-12, rtol=1e-9'}


def parity(state,reference,P,P_reference,rows,rows_reference,decision,decision_reference):
    a,b=np.asarray(state,float),np.asarray(reference,float)
    dx,dv=a[:3]-b[:3],a[3:]-b[3:]
    out={'position_error_l2_m':float(np.linalg.norm(dx)),'position_error_linf_m':float(abs(dx).max()),'velocity_error_l2_m_s':float(np.linalg.norm(dv)),'velocity_error_linf_m_s':float(abs(dv).max()),**covariance_blocks(P,P_reference),
         'rows_equal':sorted(rows)==sorted(rows_reference),'decision_equal':decision==decision_reference,**covariance_agreement(P,P_reference)}
    ok=out['position_error_linf_m']<=PARITY_X_M and out['velocity_error_linf_m_s']<=PARITY_V_M_S and out['rows_equal'] and out['decision_equal'] and out['covariance_equal']
    out['status']='PARITY_PASS' if ok else 'PARITY_FAIL';return out


def exact_information(s,sm,P,rows):
    """rows are already admitted; excluded candidate samples must not be supplied."""
    s,sm,P=map(lambda a:np.asarray(a,float),(s,sm,P));Lam=np.linalg.solve(P,np.eye(6));Lam=(Lam+Lam.T)/2
    A=Lam.copy();forces=[-Lam@(s-sm)];labels=['prior'];Hs=[];Rs=[];residuals=[]
    prior_cost=float((s-sm)@Lam@(s-sm));costs={}
    for row in rows:
        if row.get('reason',0)!=0 or row.get('excluded_by_variant'):raise ValueError('Excluded camera cannot enter causal information attribution')
        H=np.c_[np.asarray(row['H'],float),np.zeros((4,3))];var=np.asarray(row['var'],float);r=np.asarray(row['r'],float)
        if (var<=0).any() or not np.isfinite(var).all():raise ValueError('Nonpositive measurement covariance')
        A+=H.T@(H/var[:,None]);Hs.append(H);Rs.extend(var);residuals.extend(r)
        for j,ch in enumerate(('R','G','B','D')):
            labels.append(f"camera{row['camera']}_{ch}");forces.append(H[j]*r[j]/var[j]);costs[labels[-1]]=float(r[j]**2/var[j])
    b=np.sum(forces,axis=0);delta=np.linalg.solve(A,b);parts=np.linalg.solve(A,np.asarray(forces).T).T
    H=np.vstack(Hs) if Hs else np.zeros((0,6));R=np.diag(Rs);r=np.asarray(residuals)
    eig,V=np.linalg.eigh(A);meas=A[:3,:3]-Lam[:3,:3];ev,U=np.linalg.eigh(meas)
    L=np.linalg.cholesky(P);Aw=L.T@A@L;ew,Vw=np.linalg.eigh(Aw);Jx=np.linalg.inv(P[:3,:3])+meas;ep,Vp=np.linalg.eigh(Jx)
    S=H@P@H.T+R;nu=r+H@(s-sm)
    return {'A':A,'b':b,'P':P,'H':H,'R':R,'S':S,'nu':nu,'r':r,'delta':delta,'forces':np.asarray(forces),'contributions':parts,'labels':labels,
            'eigenvalues_raw_mixed_units_do_not_interpret':eig,'eigenvectors_raw_mixed_units':V,'whitened_information':Aw,'dimensionless_eigenvalues':ew,'dimensionless_eigenvectors':Vw,'position_posterior_information_m_inv2':Jx,'position_posterior_eigenvalues_m_inv2':ep,'position_posterior_eigenvectors':Vp,'position_measurement_eigenvalues':ev,'position_measurement_eigenvectors':U,
            'normal_equation_residual':float(np.linalg.norm(A@delta-b)),'cholesky_reconstruction_residual':float(np.linalg.norm(np.linalg.cholesky(A)@np.linalg.cholesky(A).T-A)),'attribution_closure':float(abs(parts.sum(0)-delta).max()),'condition':float(np.linalg.cond(A)),
            'prior_cost':prior_cost,'channel_cost':costs,'cost':prior_cost+sum(costs.values())}


def high_precision_step(A,b,precision=70):
    """Decimal elimination on the SAME binary64 A,b: isolates solve roundoff.

    Does not recompute projection, choose a different pixel cell or certify a
    minimum across nondifferentiable branches.
    """
    from decimal import Decimal,localcontext
    with localcontext() as ctx:
        ctx.prec=precision
        a=[[Decimal.from_float(float(x)) for x in row] for row in np.asarray(A)];y=[Decimal.from_float(float(x)) for x in b];n=len(y)
        for j in range(n):
            pivot=max(range(j,n),key=lambda i:abs(a[i][j]));a[j],a[pivot]=a[pivot],a[j];y[j],y[pivot]=y[pivot],y[j]
            if not a[j][j]:raise np.linalg.LinAlgError('Singular decimal normal equation')
            for i in range(j+1,n):
                f=a[i][j]/a[j][j]
                for k in range(j,n):a[i][k]-=f*a[j][k]
                y[i]-=f*y[j]
        x=[Decimal(0)]*n
        for i in reversed(range(n)):x[i]=(y[i]-sum(a[i][j]*x[j] for j in range(i+1,n)))/a[i][i]
        return np.array([float(v) for v in x])


def subset_attribution(s,sm,P,rows):
    """Exact Shapley vector for camera removal in ONE fixed linearized solve.

    No gate, cell, gradient or prior is refit. This is explicitly separate from
    nonlinear removal experiments and from same-total-Hessian force attribution.
    """
    if len(rows)>8:raise ValueError('Offline subset budget is eight admitted cameras')
    n=len(rows);values={};s,sm=np.asarray(s),np.asarray(sm);Lam=np.linalg.inv(P);Lam=(Lam+Lam.T)/2
    for mask in range(1<<n):values[mask]=exact_information(s,sm,P,[r for j,r in enumerate(rows) if mask&(1<<j)])['delta']
    parts=[]
    for j in range(n):
        total=np.zeros(6)
        for mask in range(1<<n):
            if mask&(1<<j):continue
            m=mask.bit_count();weight=math.factorial(m)*math.factorial(n-m-1)/math.factorial(n)
            total+=weight*(values[mask|(1<<j)]-values[mask])
        parts.append(total)
    return {'camera_ids':[r['camera'] for r in rows],'contributions':np.asarray(parts),'prior_only_step':values[0],'full_step':values[(1<<n)-1],
            'closure':float(np.linalg.norm(np.sum(parts,axis=0)+values[0]-values[(1<<n)-1])) if n else 0.,
            'scope':'FIXED_LINEARIZATION_AND_ROWS_ONLY; NOT nonlinear causal attribution'}


def velocity_transfer(P,dx,dv):
    P=np.asarray(P,float);gain=np.linalg.solve(P[:3,:3].T,P[3:,:3].T).T
    return gain,np.asarray(dv)-gain@np.asarray(dx)


def fixed_cost(prob,s,cameras):
    """Ungated physical support; invalid samples are UNDEFINED, never zero."""
    from tools import root_cause_oracle as O
    cost=prob.prior_cost(s);lost=[];values={}
    for c in cameras:
        policy={**prob.policy,'row_innovation_gate':'depth_front_only','visibility_n_sigma':float('inf')}
        row=O.row(prob.imgs[c],prob.cams[c],s[:3],prob.a,prob.noise,policy,prob.P_minus[:3,:3],s[:3]-prob.s_minus[:3])
        if row['reason']!=O.VALID or 'r' not in row:lost.append({'camera':c,'reason':O.NAMES[row['reason']]});continue
        q=float(np.sum(prob.ch*row['r']**2/row['var']));values[str(c)]=q;cost+=q
    return {'cost':None if lost or not cameras else float(cost),'status':'UNDEFINED_SUPPORT' if lost else ('DEFINED' if cameras else 'NO_ROWS'),'lost_rows':lost,'camera_cost':values}


def observability(Hs,Fs,B=None):
    H=np.vstack([np.asarray(h)@np.asarray(f) for h,f in zip(Hs,Fs)]) if Hs else np.zeros((0,6))
    design=H if B is None else np.c_[H,B]
    _,s,V=np.linalg.svd(design,full_matrices=True);tol=(s[0] if len(s) else 0)*max(design.shape,default=0)*np.finfo(float).eps;rank=int((s>tol).sum())
    return {'singular_values':s,'rank':rank,'columns':design.shape[1],'nullspace':V[rank:].T,'status':'NOT_IDENTIFIABLE' if rank<design.shape[1] else 'LOCALLY_FULL_RANK','linearized_matrix':design}


def nuisance_gls(H,y,R,B,C,prior):
    """Proper Gaussian nuisance prior C; Schur equals GLS R+B C B^T."""
    H,y,R,B,C,prior=map(lambda a:np.asarray(a,float),(H,y,R,B,C,prior));W=np.linalg.solve(R,np.eye(len(R)))
    A=np.linalg.inv(prior)+H.T@W@H;cross=H.T@W@B;D=np.linalg.inv(C)+B.T@W@B
    rhs=H.T@W@y;z=B.T@W@y;J=A-cross@np.linalg.solve(D,cross.T);g=rhs-cross@np.linalg.solve(D,z)
    cov=np.linalg.solve(J,np.eye(len(J)));mu=cov@g
    Sigma=R+B@C@B.T;WiH=np.linalg.solve(Sigma,H);Jg=np.linalg.inv(prior)+H.T@WiH;Cg=np.linalg.inv(Jg);mg=Cg@H.T@np.linalg.solve(Sigma,y)
    full=np.block([[A,cross],[cross.T,D]]);joint=np.linalg.solve(full,np.r_[rhs,z]);jointcov=np.linalg.inv(full)
    return {'mean':mu,'covariance':cov,'gls_mean':mg,'gls_covariance':Cg,'joint_mean':joint,'joint_covariance':jointcov,'shared_covariance':Sigma,
            'mean_closure':float(abs(mu-mg).max()),'covariance_closure':float(abs(cov-Cg).max()),'joint_closure':float(abs(mu-joint[:len(mu)]).max())}


def differenced_covariance(raw_covariance,times,channels):
    """L C L^T works for correlated source noise too; no independence invented."""
    C=np.asarray(raw_covariance,float);L=np.zeros((times*channels,(times+1)*channels))
    for t in range(times):L[t*channels:(t+1)*channels,:channels]=-np.eye(channels);L[t*channels:(t+1)*channels,(t+1)*channels:(t+2)*channels]=np.eye(channels)
    return L@C@L.T


def plane_patch(depth,mask,raw_valid,u0,v0,size,cam,noise,tail):
    """Fit one inverse-depth plane, preserving uncertainty and face caveats."""
    half=size//2;x0,y0=u0-half,v0-half
    result={'size':size,'origin':[x0,y0],'face_status':'FACE_IDENTITY_UNPROVEN'}
    if x0<0 or y0<0 or x0+size>cam.W or y0+size>cam.H:return {**result,'status':'OUTSIDE_IMAGE','normal_status':'NORMAL_NOT_IDENTIFIABLE'}
    d=np.asarray(depth[y0:y0+size,x0:x0+size],float);ok=mask[y0:y0+size,x0:x0+size]&raw_valid[y0:y0+size,x0:x0+size]&np.isfinite(d)&(d>0)
    result.update(depth_m=d,mask=mask[y0:y0+size,x0:x0+size],raw_valid=raw_valid[y0:y0+size,x0:x0+size],valid_count=int(ok.sum()))
    yy,xx=np.mgrid[y0:y0+size,x0:x0+size];A=np.c_[np.ones(ok.sum()),xx[ok]-u0,yy[ok]-v0]
    if len(A)<=3 or np.linalg.matrix_rank(A)<3:return {**result,'status':'INSUFFICIENT_SINGLE_PLANE_SUPPORT','normal_status':'NORMAL_NOT_IDENTIFIABLE'}
    var=noise.sigma2(d[ok])/d[ok]**4
    if (var<=0).any():return {**result,'status':'UNDEFINED_VARIANCE','normal_status':'NORMAL_NOT_IDENTIFIABLE'}
    W=1/var;coef=np.linalg.solve(A.T@(W[:,None]*A),A.T@(W/d[ok]));e=1/d[ok]-A@coef;q=float(np.sum(e*e/var));p=float(chi2.sf(q,len(A)-3))
    ncam=np.array([cam.fx*coef[1],cam.fy*coef[2],coef[0]+coef[1]*(cam.cx-u0)+coef[2]*(cam.cy-v0)]);nw=cam.R.T@ncam;nw/=np.linalg.norm(nw)
    single=p>=tail and bool(ok.all())
    return {**result,'status':'CONDITIONAL_SINGLE_PLANE_FIT' if single else 'SINGLE_PLANE_NOT_SUPPORTED','coefficients_inverse_depth':coef,'condition':float(np.linalg.cond(A*np.sqrt(W)[:,None])),
            'chi2':q,'p_value':p,'residual_inverse_depth':e,'normal_world':nw if single else None,'normal_status':'CONDITIONAL_LOCAL_MODEL_ONLY' if single else 'NORMAL_NOT_IDENTIFIABLE'}


def timeline(x,v,t):
    x,v,t=map(lambda a:np.asarray(a,float),(x,v,t));dt=np.diff(t);prediction=x[:-1]+dt[:,None]*v[:-1];correction=x[1:]-prediction
    integrated=np.cumsum(dt[:,None]*v[:-1],axis=0);cumcorr=np.cumsum(correction,axis=0);net=x[1:]-x[0]
    return {'prediction':prediction,'correction':correction,'velocity_increment':np.diff(v,axis=0),'integrated_velocity':integrated,'cumulative_correction':cumcorr,'net':net,'closure_m':net-integrated-cumcorr}


def branch_coordinates(u,v):
    def one(a):
        n=round(float(a));return {'value':float(a),'hex':float(a).hex(),'nearest_integer':n,'ulp_to_integer':float((a-n)/abs(np.spacing(float(n) or 1.))),'floor':math.floor(a),'nextafter_left':float(np.nextafter(a,-np.inf)),'nextafter_right':float(np.nextafter(a,np.inf))}
    return {'u':one(u),'v':one(v),'neighbor_cells':[[math.floor(u)+du,math.floor(v)+dv] for du in (-1,0) for dv in (-1,0)]}
