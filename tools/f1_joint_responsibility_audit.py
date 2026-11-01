#!/usr/bin/env python3
"""Final 8444/45588 CPU audit. Production is read-only; no scene refilter.

LEGACY_PROJECTION is a conditional diagnostic. A fixed-pixel point-identity
likelihood is explicitly UNDEFINED without a correspondence/footprint model.
"""
import argparse
import contextlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
import numpy as np
from scipy.linalg import block_diag
from scipy.special import logsumexp

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']=''  # CPU-only CLI; keep production/test importers' CUDA environment intact
from tools import root_cause_oracle as O
from tools.f1_second_pass import Scene,problem,ACC,ShiftCam
from tools.f1_source_responsibility import source_record,source_reprojection
from tools.f1_temporal_reference_diagnostics import references,ref_row
from tools.f1_definitive_evidence import DEFAULT_ROOT,write_json,read_json,csv_write,digest,file_record,PROTECTED
from tools.f1_definitive_math import velocity_transfer,nuisance_gls
from tools.f1_diagnostic_units import compare_state6,compare_cov6

PAIR=(8444,45588)
SCENE='storage_45135'
TABLES=('baseline_joint_parity','source_collision_candidates','fixed_observation_pixels','candidate_likelihoods',
        'joint_assignment_enumeration','per_point_per_camera_factors','joint_iteration_history','reference_bias_uncertainty',
        'ownership_changes','heldout_controls','model_comparison_identifiability')


def joint_predict(s,P,dt,qc):
    """Propagate the complete covariance; cross-point blocks are never reset."""
    _,_,f,q=O.predict(np.zeros(6),np.eye(6),dt,qc);F=block_diag(f,f);Q=block_diag(q,q)
    return F@np.asarray(s),F@P@F.T+Q,F,Q


def mixture_moments(weights,means,covariances):
    w=np.asarray(weights,float);mu=np.asarray(means,float);C=np.asarray(covariances,float)
    if (w<0).any() or not np.isclose(w.sum(),1):raise ValueError('Mixture weights must be normalized')
    mean=w@mu;e=mu-mean
    return mean,np.einsum('k,kij->ij',w,C)+np.einsum('k,ki,kj->ij',w,e,e)


def toy_mixture(s,y,variance,weights=(.5,.5)):
    """Proper synthetic single-observation mixture; NOT a real pixel model."""
    s=np.asarray(s);w=np.asarray(weights)
    if variance<=0 or (w<=0).any() or not np.isclose(w.sum(),1):raise ValueError('Proper variance/normalized positive priors required')
    d=s[[0,6]]-y;logs=np.log(w)-.5*d*d/variance-.5*np.log(2*np.pi*variance);E=-logsumexp(logs);tau=np.exp(logs+E)
    g=np.zeros(12);H=np.zeros((12,12));g[[0,6]]=tau*d/variance
    for i,j in itertools.product(range(2),repeat=2):
        H[(0,6)[i],(0,6)[j]]=tau[i]*tau[j]*d[i]*d[j]/variance**2
        if i==j:H[(0,6)[i],(0,6)[j]]+=tau[i]/variance-tau[i]*d[i]**2/variance**2
    return float(E),g,H,tau


def frame1_prior(sc,pid):
    P=np.zeros((6,6));P[:3,:3]=sc.P0[pid];P[3:,3:]=np.asarray(sc.run.fman['prior']['velocity_covariance_m2_s2'])
    s=np.r_[sc.x0[pid],np.zeros(3)];dt=float(sc.t[1]-sc.t[0]);qc=np.asarray(sc.run.fman['prior']['q_c_m2_s3'])
    return O.predict(s,P,dt,qc)


def footprint(row):
    sample=row.get('sample',{})
    if sample.get('reason')!=O.VALID or 'w' not in sample:return {}
    x,y=sample['pixel00'];c=row['camera'];coords=((x,y),(x+1,y),(x,y+1),(x+1,y+1))
    return {(c,u,v):float(w) for (u,v),w in zip(coords,sample['w'])}


def shared_sensor_covariance(rows,images,noise,policy):
    """L diag(sensor variance) L^T; overlap is correlation, not double ownership."""
    if noise.interp!='sum_w2':raise ValueError('UNDEFINED_SHARED_NOISE_MODEL: sensor-mode row covariance lacks a raw-pixel decomposition')
    if noise.rgb_model or noise.depth_model:raise ValueError('UNDEFINED_SHARED_NOISE_MODEL: cross-row model-error covariance was not supplied')
    m=len(rows);R=np.zeros((4*m,4*m));supports=[footprint(r) for r in rows]
    for i,j in itertools.product(range(m),repeat=2):
        for key in supports[i].keys()&supports[j].keys():
            c,u,v=key;wi,wj=supports[i][key],supports[j][key];depth=images[c].depth[v,u]
            ai=wi;aj=wj
            if policy.get('depth_interpolation','metric')=='inverse':
                ai*=rows[i]['sample']['depth']**2/depth**2;aj*=rows[j]['sample']['depth']**2/depth**2
            for ch in range(3):R[4*i+ch,4*j+ch]+=wi*wj*noise.rgb_var[ch]
            R[4*i+3,4*j+3]+=ai*aj*noise.sigma2(depth)
    return R


def joint_information(s,sm,P,rows,R):
    H=np.zeros((4*len(rows),12));r=np.concatenate([o['r'] for o in rows]) if rows else np.zeros(0)
    for j,row in enumerate(rows):H[4*j:4*j+4,6*row['pair_index']:6*row['pair_index']+3]=row['H']
    Lam=np.linalg.inv(P);Lam=(Lam+Lam.T)/2;e=s-sm
    if len(rows):
        scale=np.sqrt(np.diag(R))
        if not np.isfinite(scale).all() or (scale<=0).any():raise np.linalg.LinAlgError('INVALID_SHARED_SENSOR_VARIANCE')
        if np.linalg.matrix_rank(R/scale[:,None]/scale[None,:])<len(R):raise np.linalg.LinAlgError('SHARED_SENSOR_NOISE_RANK_DEFICIENT; no diagonal jitter added')
        np.linalg.cholesky(R);W_H=np.linalg.solve(R,H);W_r=np.linalg.solve(R,r)
        A=Lam+H.T@W_H;b=-Lam@e+H.T@W_r;q=float(r@W_r)
    else:A=Lam;b=-Lam@e;q=0.
    d=np.linalg.solve(A,b);cov=np.linalg.inv(A);cov=(cov+cov.T)/2
    return {'A':A,'b':b,'delta':d,'H':H,'r':r,'R':R,'covariance':cov,'criterion':float(e@Lam@e+q),'prior_cost':float(e@Lam@e),
            'cross_information_norm':float(np.linalg.norm(A[:6,6:])),'cross_covariance_norm':float(np.linalg.norm(cov[:6,6:])),
            'covariance_kind':'FROZEN_LINEARIZATION_GAUSSIAN; nonlinear use is a conditional GN approximation, not a mixture posterior'}


class Environment:
    """All canonical competitors, frozen; exact frame-1 prediction only."""
    def __init__(self,sc,pids=PAIR):
        self.sc=sc;self.pids=tuple(pids);self.tables=[];self.n_sigma=sc.policy['visibility_n_sigma']
        source=source_record(sc,self.pids[0]);self.slot=(source['source_camera'],source['source_u'],source['source_v'])
        dt=float(sc.t[1]-sc.t[0]);v=np.asarray(sc.run.fman['prior']['velocity_covariance_m2_s2']);q=np.asarray(sc.run.fman['prior']['q_c_m2_s3'])
        for cam in sc.cams:
            X=sc.x0@cam.R.T+cam.t;z=X[:,2];safe=np.where(z!=0,z,1.);uv=np.c_[cam.fx*X[:,0]/safe+cam.cx,cam.fy*X[:,1]/safe+cam.cy]
            pixel=np.rint(uv).astype(np.int64);valid=np.isfinite(uv).all(1)&(z>cam.near)&(pixel[:,0]>=0)&(pixel[:,0]<cam.W)&(pixel[:,1]>=0)&(pixel[:,1]<cam.H)
            ids=np.flatnonzero(valid);keys=pixel[ids,1]*cam.W+pixel[ids,0];order=np.lexsort((ids,z[ids],keys));ids=ids[order];keys=keys[order]
            r=cam.R[2];var=np.einsum('i,nij,j->n',r,sc.P0,r)+r@(dt*dt*v+dt**3/3*q)@r
            self.tables.append((keys,ids,z,uv,var))

    def occupants(self,c,u,v):
        cam=self.sc.cams[c];keys,ids,z,uv,var=self.tables[c];key=v*cam.W+u
        lo,hi=np.searchsorted(keys,key,'left'),np.searchsorted(keys,key,'right')
        return [{'point_id':int(i),'z_m':float(z[i]),'uv':uv[i],'sigma_z_m':float(np.sqrt(var[i])),
                 'source':source_record(self.sc,int(i))} for i in ids[lo:hi]]

    def query(self,c,x,Pxx,pair_states,transfer=False):
        cam=self.sc.cams[c];g=O.geometric_visibility(cam,x)
        if g!=O.VALID:return {'owner':None,'reason':g}
        u,v,Z,*_=cam.project(x);pu,pv=int(np.rint(u)),int(np.rint(v));occupants=[r for r in self.occupants(c,pu,pv) if r['point_id'] not in self.pids]
        for j,pid in enumerate(self.pids):
            point=pair_states[j,:3];uu,vv,z,*_=cam.project(point)
            if O.geometric_visibility(cam,point)==0 and (round(uu),round(vv))==(pu,pv):
                occupants.append({'point_id':pid,'z_m':float(z),'sigma_z_m':float(np.sqrt(cam.R[2]@Pxx[j]@cam.R[2]))})
        occupants.sort(key=lambda r:(r['z_m'],r['point_id']));front=occupants[0] if occupants else None
        original=front['point_id'] if front else None
        # Only this fixed collision slot is transferred, and true depth occlusion wins.
        if transfer and (c,pu,pv)==self.slot and front and front['point_id']==self.pids[1]:
            source=next((r for r in occupants if r['point_id']==self.pids[0]),None)
            if source and source['z_m']-front['z_m']<=self.n_sigma*np.hypot(source['sigma_z_m'],front['sigma_z_m']):front=source
        return {'owner':front['point_id'] if front else None,'original_owner':original,'z_front_m':front['z_m'] if front else None,
                'sigma_front_m':front['sigma_z_m'] if front else None,'rounded_pixel':(pu,pv),'reason':0,'external':front is not None and front['point_id'] not in self.pids}

    def frozen_codes(self,pid,sm,Pm):
        pair=np.stack([frame1_prior(self.sc,i)[0] for i in self.pids]);Pxx=np.stack([frame1_prior(self.sc,i)[1][:3,:3] for i in self.pids]);codes=[]
        for c,cam in enumerate(self.sc.cams):
            q=self.query(c,sm[:3],Pxx,pair);g=q['reason']
            if g==0 and q['owner']!=pid:
                z=cam.project(sm[:3])[2];sigma=np.sqrt(cam.R[2]@Pm[:3,:3]@cam.R[2]);g=O.SELF_OCC if z-q['z_front_m']>self.n_sigma*np.hypot(sigma,q['sigma_front_m']) else O.DUP
            codes.append(g)
        return codes


def baseline(sc,env):
    results={};problems={};parity=[]
    for pid in env.pids:
        sm,Pm,F,Q=frame1_prior(sc,pid);frozen=env.frozen_codes(pid,sm,Pm)
        if str(pid) in sc.trace(1)['final']:
            p=problem(sc,1,pid);verified=True
        else:
            saved=sc.run.mm('camera_reason')[1,:,pid]
            p=O.Problem(sc.cams,sc.images(1),sc.run.can['canonical_rgb'][pid],sm,Pm,frozen,sc.noise,sc.policy,sc.run.fman['max_iekf_iters'],sc.run.fman['step_tolerance'],[int(r)==O.SELF_OCC for r in saved]);verified=False
        r=p.run(sc.policy['update_acceptance']);results[pid]=r;problems[pid]=p
        if verified:
            t=sc.trace(1)['final'][str(pid)];errors=compare_state6(r['s_plus'],t['s_plus']);cov=compare_cov6(r['P_plus'],t['P_plus']);rows=[o['camera'] for o in p.used(r['rows_accepted'])] if r['observed'] else []
            expected=[o['camera'] for o in t.get('rows',[])] if t['acceptance'].startswith('ACCEPTED') else []
            ok=errors['position_linf_m']<=1e-10 and errors['velocity_linf_m_s']<=1e-9 and np.allclose(r['P_plus'],t['P_plus'],rtol=1e-9,atol=1e-12) and rows==expected and ACC[r['acceptance_code']]==t['acceptance']
            prior_error=float(abs(Pm-np.asarray(t['P_minus'])).max());mean_prior_error=float(abs(sm-np.asarray(t['s_minus'])).max());codes_match=frozen==p.frozen
            pred=sc.trace(1)['prediction'];F_equal=np.array_equal(F,pred['F']);Q_error=float(abs(Q-np.asarray(pred['Q'])).max())
            ok &= prior_error<=1e-12 and mean_prior_error<=1e-12 and codes_match and F_equal and Q_error<=1e-15 and r['converged']==t['converged']
            parity.append({'point_id':pid,'status':'PARITY_PASS' if ok else 'PARITY_FAIL',**errors,**cov,'prior_covariance_max_error':prior_error,'prior_mean_max_error':mean_prior_error,
                'F_equal':F_equal,'Q_max_error':Q_error,'dt_seconds':float(sc.t[1]-sc.t[0]),'converged_equal':r['converged']==t['converged'],
                'reconstructed_frozen_codes_equal':codes_match,'accepted_cameras':rows,'acceptance':ACC[r['acceptance_code']]})
        else:
            parity.append({'point_id':pid,'status':'PARITY_NOT_VERIFIABLE','saved_xyz_error_m':float(np.linalg.norm(r['s_plus'][:3]-sc.run.mm('posterior_xyz')[1,pid])),
                'reason':'Missing 45588 full-precision v and complete 6x6 production posterior; float32 velocity/diagonal arrays do not verify full state/covariance.',
                'accepted_cameras_local':[o['camera'] for o in p.used(r['rows_accepted'])] if r['observed'] else [],'production_acceptance_code':int(sc.run.mm('update_acceptance')[1,pid]),'local_acceptance':ACC[r['acceptance_code']]})
    return problems,results,parity


def observation_universe(sc,problems,results):
    U=set()
    for pid,p in problems.items():
        src=source_record(sc,pid);U.add((src['source_camera'],src['source_u'],src['source_v']))
        states=[p.s_minus,*[h['s'] for h in results[pid]['history']],results[pid]['s_plus']]
        for state in states:
            for c,cam in enumerate(sc.cams):
                row=O.row(sc.images(1)[c],cam,state[:3],p.a,p.noise,p.policy,p.P_minus[:3,:3],state[:3]-p.s_minus[:3]);row['camera']=c;U.update(footprint(row))
    return frozenset((c,u,v) for c,u,v in U if sc.images(1)[c].mask[v,u] and sc.images(1)[c].raw_valid[v,u])


class PairProblem:
    """Coupled legacy-projection diagnostic with one owner and shared raw noise.

    The criterion is conditional on the sampled rows. It is not the undefined
    full fixed-pixel likelihood and is never used to select material identity.
    """
    def __init__(self,sc,env,problems,U,transfer=False,referenced=False,source_only=False):
        self.sc,self.env,self.U=sc,env,U;self.transfer,self.referenced,self.source_only=transfer,referenced,source_only
        self.sm=np.concatenate([problems[i].s_minus for i in env.pids]);self.P=block_diag(*[problems[i].P_minus for i in env.pids]);self.Pxx=np.stack([problems[i].P_minus[:3,:3] for i in env.pids])
        self.refs={pid:references_for_untraced(sc,env,pid) for pid in env.pids};self.rows_initial=None;self.events=[]
        self.rows_initial=self.rows(self.sm,initial=True)

    def rows(self,s,initial=False):
        states=np.asarray(s).reshape(2,6);rows=[];seen=set()
        requested=None if initial else {(r['pair_index'],r['camera']) for r in self.rows_initial}
        for j,pid in enumerate(self.env.pids):
            for c,cam in enumerate(self.sc.cams):
                if requested is not None and (j,c) not in requested:continue
                if self.source_only and c!=int(self.sc.src[pid,1]):continue
                q=self.env.query(c,states[j,:3],self.Pxx,states,self.transfer)
                if q['reason']!=0 or q['owner']!=pid:
                    if requested is not None:raise ValueError('UNDEFINED_SUPPORT: ownership changes or external collider '+repr(q))
                    continue
                key=(c,*q['rounded_pixel'])
                if key in seen:raise ValueError('DUPLICATE_OBSERVATION_ASSIGNMENT')
                r=ref_row(self.sc.images(1)[c],cam,states[j,:3],self.sc.run.can['canonical_rgb'][pid],self.sc.noise,self.sc.policy,self.Pxx[j],states[j,:3]-self.sm[6*j:6*j+3],self.refs[pid][c] if self.referenced else None)
                r.update(camera=c,pair_index=j,point_id=pid,rounded_observation=key)
                if r['reason']!=0:
                    if requested is not None:raise ValueError('UNDEFINED_SUPPORT: '+O.NAMES[r['reason']])
                    continue
                if not set(footprint(r))<=self.U:
                    if requested is not None:raise ValueError('UNDEFINED_SUPPORT: interpolation leaves fixed raw pixel universe')
                    continue
                seen.add(key);rows.append(r)
        return rows

    def evaluate(self,s):
        rows=self.rows(s);R=shared_sensor_covariance(rows,self.sc.images(1),self.sc.noise,self.sc.policy)
        info=joint_information(s,self.sm,self.P,rows,R);info['rows']=rows
        return info

    def optimize(self):
        s=self.sm.copy();history=[];status='ITERATION_LIMIT';maxiters=self.sc.run.fman['max_iekf_iters'];tol=self.sc.run.fman['step_tolerance']
        if not self.rows_initial:return {'state':s,'covariance':self.P.copy(),'status':'UNOBSERVED_COAST','history':[],'rows':[],'criterion':None}
        try:info=self.evaluate(s)
        except (ValueError,np.linalg.LinAlgError) as e:return {'state':s,'covariance':None,'status':'UNDEFINED_MODEL','reason':str(e),'history':[],'rows':self.rows_initial,'criterion':None}
        for j in range(maxiters):
            d=info['delta'];step=float(d@info['A']@d);record={'iteration':j,'s':s.copy(),'criterion_before':info['criterion'],'A':info['A'],'b':info['b'],'delta':d,'step':step,'accepted':False,'support_failures':[]}
            if step<tol:status='CONVERGED_LOCAL_GN';history.append(record);break
            alpha=1.
            while alpha>np.finfo(float).eps:
                proposal=s+alpha*d
                try:new=self.evaluate(proposal)
                except (ValueError,np.linalg.LinAlgError) as e:record['support_failures'].append(str(e));alpha*=.5;continue
                if new['criterion']<info['criterion']:
                    record.update(accepted=True,alpha=alpha,criterion_after=new['criterion']);s=proposal;info=new;break
                alpha*=.5
            history.append(record)
            if not record['accepted']:status='NO_DESCENDING_COMPARABLE_STEP';break
        return {'state':s,'covariance':info['covariance'],'status':status,'history':history,'rows':info['rows'],'criterion':info['criterion'],
            'information':info,'covariance_kind':'CONDITIONAL_GN_APPROXIMATION' if not self.referenced else 'ALGEBRAIC_DIAGNOSTIC_NOT_PROBABILISTIC_POSTERIOR'}


def references_for_untraced(sc,env,pid):
    own=sc.frame0_ownership([pid])[pid];out={}
    for c,cam in enumerate(sc.cams):
        row=O.row(sc.images(0)[c],cam,sc.x0[pid],sc.run.can['canonical_rgb'][pid],sc.noise,sc.policy,sc.P0[pid],np.zeros(3));valid='r' in row and row['sample']['reason']==0 and int(own[c])!=O.SELF_OCC
        out[c]={'valid':valid,'status':'VALID_REFERENCE' if valid else 'NO_VALID_REFERENCE','b':row['r'] if valid else None,'var0':row['var'] if valid else None,
                'source_correlation':'Source pixel/color and x0 are correlated; conditional differencing is not calibrated bias inference.'}
    return out


def archive_completeness(sc,end):
    free=sc.run.free;frames=[];can=sc.run.can;traces={k:sc.trace(k) for k in range(1,end+1)}
    point_archive=sc.fdir/'f1_state_full_covariance.npz';cov_ids=[]
    if point_archive.exists():
        with np.load(point_archive,allow_pickle=False) as z:cov_ids=list(map(int,z['point_id']))
    for k in range(end+1):
        for pid in PAIR:
            traced=k>0 and traces[k] is not None and str(pid) in traces[k]['final'];checkpoint=free/'covariance'/f'frame_{k:04d}.npy'
            frames.append({'frame':k,'point_id':pid,'full_xyz_available':True,'full_6D_state_available':bool(k==0 or traced),
                'full_6x6_covariance_available':bool(k==0 or traced or checkpoint.exists() or pid in cov_ids),
                'state_source':str(free/'numeric'/f'trace_frame_{k:04d}.json') if traced else ('canonical measured XYZ, zero velocity' if k==0 else 'XYZ float64; velocity float32 only'),
                'full_covariance_missing_path':str(checkpoint) if not traced and k else None,
                'missing_array':'P_plus[45588,6,6] and v_plus[45588,3] float64' if pid==45588 and k>0 and not traced and not checkpoint.exists() else None,
                'external_world_snapshot':'canonical XYZ exact at prior' if k<=1 else 'posterior XYZ float64; velocity float32 and incomplete covariance',
                'external_inner_iterate_states':'NOT_ARCHIVED_FOR_ALL_POINTS',
                'tier':'TIER_EXACT_ARCHIVE' if k==0 else 'TIER_CONTROLLED_LOCAL' if k==1 else 'TIER_UNAVAILABLE',
                'scope':'Tier is for a validated joint reconstruction, not availability of scalar diagnostics.'})
    return {'scene':sc.name,'frames':frames,'sources':[source_record(sc,pid) for pid in PAIR],
        'frame1_prior_reconstruction':'Exact original block initial priors and F/Q from canonical source covariance and recorded Pvv/qc; verified against 8444 trace.',
        'initial_cross_point_prior':'Zero by original per-point production initialization, conditional on fixed calibration and the declared factorized model; no physical source/calibration independence is inferred.',
        'sequence_status':'BLOCKED: no normalized fixed-pixel correspondence/OTHER likelihood; complete 45588 posterior and external inner iterates/covariance absent.',
        'fixed_pixel_model_status':'UNDEFINED_MODEL: a zero-size point supplies RGB/Z but no probability of covering an arbitrary image pixel. Source pixel response is source-ray uncertainty, not calibrated non-source material correspondence.'}


def prepare(root,fifth,out,end):
    """Idempotent test->all handoff, preserving every completed artifact."""
    out=Path(out);path=out/'provenance.json'
    if out.exists():
        if not path.exists():raise FileExistsError('Existing directory is not this audit: '+str(out))
        inv=read_json(path)
        if inv['forensics_root']!=str(Path(root).resolve()) or inv['fifth_pass_root']!=str(Path(fifth).resolve()):raise ValueError('Output belongs to different inputs')
        if inv.get('missing_required_evidence'):raise FileNotFoundError('MISSING_REQUIRED_EVIDENCE recorded at frame 1; use a fresh output after supplying: '+', '.join(inv['missing_required_evidence']))
        verify(inv);return inv
    out.mkdir(parents=True);(out/'figures').mkdir()
    git=lambda *a:subprocess.check_output(['git','-C',str(REPO),*a],text=True)
    inv={'forensics_root':str(Path(root).resolve()),'fifth_pass_root':str(Path(fifth).resolve()),'git_commit':git('rev-parse','HEAD').strip(),
        'git_status':git('status','--short'),'dirty_diff':git('diff','--binary'),'files':{},'protected':{},'cpu_only':True,'seed':None,'seed_reason':'No random initializations; prior-only deterministic local starts.'}
    for p in [*PROTECTED,REPO/'tools/f1_joint_responsibility_audit.py',REPO/'tools/report_f1_joint_responsibility.py',REPO/'tests/test_f1_joint_responsibility_audit.py',
              REPO/'RGBDiekf_Codex_Final_Joint_Responsibility_8444_45588_Test.md',*REPO.glob('tools/*f1*.py'),REPO/'tools/root_cause_oracle.py',REPO/'tools/root_cause_trace.py']:
        if p.is_file():inv['files'][str(p)]=file_record(p)
    inv['protected']={str(p):digest(p) for p in PROTECTED if p.is_file()}
    for folder in (Path(root)/'second_pass_20261030',Path(root)/'fourth_pass_sequential_replay_20261030',Path(fifth)):
        print('Hashing prior evidence:',folder,flush=True)
        for p in folder.iterdir() if folder.exists() else []:
            if p.is_file():inv['files'][str(p)]=file_record(p)
    required=[Path(root)/'second_pass_20261030/SECOND_PASS_ROOT_CAUSE.md',Path(root)/'fourth_pass_sequential_replay_20261030/FOURTH_PASS_CAUSAL_REPORT.md',
              Path(fifth)/'SOURCE_RESPONSIBILITY_CAUSAL_REPORT.md',Path(fifth)/'source_owner_conflicts.csv',Path(fifth)/'sequence_state.csv']
    for f in required:
        if not f.is_file():inv['files'][str(f)]=file_record(f)
    inv['missing_required_evidence']=[str(f) for f in required if not f.is_file()]
    for scene in (SCENE,'USB_100109'):
        sc=Scene(root,scene);print('Hashing actual archive:',sc.run.root,flush=True)
        if scene==SCENE:
            from estimator.projective_camera import CameraRig
            import torch
            current=CameraRig(sc.run.data.cameras,'cpu',torch.float64).calibration_hash()
            write_json(out/'noise_and_prior_metadata.json',{'camera_geometry_hash_current':current,'camera_geometry_hash_archived':sc.run.fman['camera_geometry_hash'],
                'camera_hash_equal':current==sc.run.fman['camera_geometry_hash'],'noise':sc.run.man['noise'],'prior':sc.run.fman['prior'],'policy':sc.policy,
                'note':'Preserved original assumptions/provenance; no guess is relabelled as laboratory-calibrated.'})
        from tools.f1_definitive_evidence import paths_for
        for location in (sc.run.root,paths_for(root,scene)['baseline']):
            for p in [location/'run_manifest.json',* (location/'free_forward').glob('*.npy'),* (location/'free_forward').glob('*.npz'),location/'free_forward/run_manifest.json',
                      *(location/'free_forward/numeric').glob('trace_frame_*.json')]:
                if p.is_file():inv['files'][str(p)]=file_record(p)
        base=sc.run.data.base
        for k in range(end+1):
            for c in range(len(sc.cams)):
                validity=base.validity_paths[k][c] if base.validity_paths is not None else None
                for name in (*base.entries[k][c],base.masks[k][c],validity):
                    if name:source=Path(sc.run.man['arguments']['scene'])/name;inv['files'][str(source)]=file_record(source)
        for p in (Path(sc.run.man['arguments']['scene'])/'metadata').glob('*'):
            if p.is_file() and p.suffix in ('.json','.yaml','.csv','.txt'):inv['files'][str(p)]=file_record(p)
        if scene==SCENE:write_json(out/'archive_completeness.json',archive_completeness(sc,end))
    for name in ('renderer_bridge.py','sparse_iekf.py'):
        p=REPO.parent/'RigEuler3DGS/estimator'/name;inv['files'][str(p)]=file_record(p)
    write_json(path,inv);write_json(out/'inputs_sha256.json',{p:r.get('sha256') for p,r in inv['files'].items()})
    (out/'README.md').write_text('# Final joint responsibility audit\n\nCPU float64, no production writes and no whole-scene filtering.\nTests then all may use this SAME output directory: provenance/hashes must match. Completed logs/reports are preserved.\nCommand: '+repr(sys.argv)+'\nNo sequence is exported unless the joint likelihood/data gates pass.\n')
    if inv['missing_required_evidence']:raise FileNotFoundError('MISSING_REQUIRED_EVIDENCE (earliest requested frame 1): '+', '.join(inv['missing_required_evidence']))
    return inv


def verify(inv):
    for p,h in inv['protected'].items():
        if digest(p)!=h:raise RuntimeError('Protected production source changed: '+p)
    for p,record in inv['files'].items():
        if record['status']=='AVAILABLE' and digest(p)!=record['sha256']:raise RuntimeError('Input/code changed; use a fresh output: '+p)


def run_tests(out,root):
    if (out/'tests.json').exists():
        old=read_json(out/'tests.json')
        if digest(out/'tests.log')!=old['log_sha256']:raise RuntimeError('Saved test log changed')
        return old
    if (out/'tests.log').exists():raise FileExistsError('Incomplete old test log preserved; use a fresh output')
    os.environ['F1_JOINT_FORENSICS_ROOT']=str(root);records=[];console=sys.stdout
    class Result(unittest.TextTestResult):
        def startTest(self,t):super().startTest(t);print('CPU test:',t.id(),file=console,flush=True)
        def addSuccess(self,t):super().addSuccess(t);records.append({'test':t.id(),'status':'PASS'})
        def addFailure(self,t,e):super().addFailure(t,e);records.append({'test':t.id(),'status':'FAIL','detail':self._exc_info_to_string(e,t)})
        def addError(self,t,e):super().addError(t,e);records.append({'test':t.id(),'status':'ERROR','detail':self._exc_info_to_string(e,t)})
        def addSkip(self,t,r):super().addSkip(t,r);records.append({'test':t.id(),'status':'SKIP','reason':r})
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_f1_joint_responsibility_audit')
    with (out/'tests.log').open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        f.write('CPU float64. Command: '+repr(sys.argv)+'\n');result=unittest.TextTestRunner(stream=f,verbosity=2,resultclass=Result).run(suite)
        successful=result.wasSuccessful() and not result.skipped;f.write(f'\nexit_code={0 if successful else 1}\n')
    value={'successful':successful,'exit_code':0 if successful else 1,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
           'records':records,'seconds':time.monotonic()-start,'log_sha256':digest(out/'tests.log')};write_json(out/'tests.json',value);return value


def fixed_pixel_compatibility(sc,c,u,v,pid):
    """Normalized RGB/Z term only; the missing location likelihood is explicit."""
    im=sc.images(1)[c];cam=sc.cams[c]
    if not (0<=u<cam.W and 0<=v<cam.H and im.mask[v,u] and im.raw_valid[v,u]):return {'status':'UNDEFINED_SUPPORT'}
    x=sc.x0[pid];Y=np.r_[im.rgb[v,u],im.depth[v,u]];h=np.r_[sc.run.can['canonical_rgb'][pid],cam.project(x)[2]]
    R=np.r_[sc.noise.rgb_var,sc.noise.sigma2(im.depth[v,u])];residual=Y-h;H=np.zeros((4,6));H[3,:3]=cam.R[2]
    nll=.5*float(np.sum(residual**2/R+np.log(2*np.pi*R)))
    return {'status':'PARTIAL_CONDITIONAL_RGB_Z_COMPATIBILITY_ONLY','residual':residual,'R_diag':R,'H':H,'negative_log_RGB_Z_component':nll,
        'observation_ids':[(c,u,v)],'mean_h':h,'observed_Y':Y,'location_likelihood':'UNDEFINED_MODEL','complete_assignment_evidence':None,
        'proof':'At fixed observed pixels the point color is constant: H_RGB=0. Legacy RGB gradients instead come from moving the sampled pixel. A pixel-coverage/material-correspondence model is missing.'}


def enumerate_assignments(sc,env,keys):
    choices=[];candidate_rows=[];total_counts=[];eliminated=[]
    for key in keys:
        c,u,v=key;occupants=env.occupants(c,u,v);total_counts.append(len(occupants));front=occupants[0] if occupants else None;eligible=[]
        for r in occupants:
            if front and r['z_m']-front['z_m']>sc.policy['visibility_n_sigma']*np.hypot(r['sigma_z_m'],front['sigma_z_m']):
                eliminated.append({'pixel':key,'point_id':r['point_id'],'reason':'SELF_OCCLUDED_UNDER_PRODUCTION_RULE'});continue
            eligible.append(r['point_id'])
            candidate_rows.append({'camera':c,'pixel':key,'point_id':r['point_id'],**fixed_pixel_compatibility(sc,c,u,v,r['point_id'])})
        choices.append(eligible)
    count=math.prod(map(len,choices));rows=[]
    # Actual local conflicts, not all scene labels. No null likelihood is invented.
    for index,assignment in enumerate(itertools.product(*choices)):
        values=[fixed_pixel_compatibility(sc,*key,pid) for key,pid in zip(keys,assignment)]
        nll=sum(v['negative_log_RGB_Z_component'] for v in values) if all('negative_log_RGB_Z_component' in v for v in values) else None
        rows.append({'assignment_index':index,'assignment':dict(zip(map(str,keys),assignment)),'observation_ids':keys,'partial_RGB_Z_nll':nll,
            'status':'UNDEFINED_MODEL','reason':'No normalized point-to-pixel coverage/OTHER likelihood or data-supported assignment prior. Partial RGB/Z scores cannot select material identity.',
            'complete_log_likelihood_ratio_range':'(-infinity,+infinity): missing positive location factors can reverse any finite RGB/Z ratio',
            'required_external_cohort':sorted(set(assignment)-set(env.pids))})
    return rows,candidate_rows,{'tested_assignments':count,'pixel_candidate_counts':list(map(len,choices)),'all_pixel_occupant_counts':total_counts,
        'eliminated_physical_candidates':eliminated,'eliminated_assignment_count':math.prod(total_counts)-count,
        'source_bonus':None,'normalized_real_assignment_prior':None,'null_model':'NOT_IDENTIFIED'}


def heldout_controls(root):
    rows=[]
    for scene,pid,frames in [(SCENE,60019,range(1,11)),('USB_100109',67827,(1,4,7)),('USB_100109',87290,(2,3,8)),('USB_100109',85696,(2,3,4,5)),(SCENE,8416,(1,))]:
        sc=Scene(root,scene);own=sc.frame0_ownership([pid])[pid];refs=references_for_untraced(sc,None,pid)
        for k in frames:
            p=problem(sc,k,pid);fin=sc.trace(k)['final'][str(pid)]
            accepted=[r['camera'] for r in fin.get('rows',[])] if fin['acceptance'].startswith('ACCEPTED') else []
            for c,cam in enumerate(sc.cams):
                raw=O.row(sc.images(k)[c],cam,p.s_minus[:3],p.a,p.noise,p.policy,p.P_minus[:3,:3],np.zeros(3));probe=sc.trace(k)['iterations'][0]['points'][str(pid)]['cameras'][c]
                rows.append({'scene':scene,'point_id':pid,'other_point_id':probe.get('pixel_occupants_front_to_back',[{}])[0].get('point') if probe.get('pixel_occupants_front_to_back') else None,
                    'frame':k,'camera':c,'arm':'HELDOUT_ARCHIVED_NO_PAIR_INTERVENTION','accepted_state_row':c in accepted,'source':source_record(sc,pid),
                    'frozen':O.NAMES[p.frozen[c]],'row_if_sampled':raw,'owner_candidates':probe.get('pixel_occupants_front_to_back'),'frame0_reference':refs[c],
                    'canonical_drift_m':float(np.linalg.norm(np.asarray(fin['s_plus'])[:3]-sc.x0[pid])),
                    'scope':'Pair mechanism cannot recover many external competitors or supply missing pixel/material likelihood.'})
            if pid==8416:
                original=p.run(sc.policy['update_acceptance'])
                for eps in (-1e-12,1e-12):
                    src=int(sc.src[pid,1]);cams=[ShiftCam(cam,eps,eps) if j==src else cam for j,cam in enumerate(sc.cams)]
                    changed=problem(sc,k,pid,cams=cams).run(sc.policy['update_acceptance'])
                    rows.append({'scene':scene,'point_id':pid,'frame':k,'camera':src,'arm':'KNIFE_EDGE_DIAGNOSTIC','epsilon_px':eps,
                        **compare_state6(changed['s_plus'],original['s_plus']),'verdict':'NUMERICALLY_INDETERMINATE','production_parity':compare_state6(original['s_plus'],fin['s_plus'])})
    return rows


def analyze(root,fifth,out,tests,inv,end):
    if (out/'FINAL_JOINT_RESPONSIBILITY_CAUSAL_REPORT.md').exists():raise FileExistsError('Completed report preserved; select a new output directory')
    sc=Scene(root,SCENE);env=Environment(sc);problems,original,parity=baseline(sc,env);rows={k:[] for k in TABLES};arrays={}
    rows['baseline_joint_parity']=parity
    for pid in PAIR:write_json(out/f'source_{pid}.json',source_reprojection(sc,pid))
    joint_prior=block_diag(*[problems[p].P_minus for p in PAIR]);joint_mean=np.concatenate([problems[p].s_minus for p in PAIR])
    arrays['prior_mean']=joint_mean;arrays['prior_covariance']=joint_prior
    # One fixed raw universe, constructed only from pre-existing J0/source data.
    U=observation_universe(sc,problems,original);write_json(out/'fixed_universe.json',{'observation_ids':sorted(U),'chosen_before_interventions':True,'source':'Union of source/prior and original J0 inner sampling cells, hard foreground/valid depth only.'})
    users={key:[] for key in U}
    for j,pid in enumerate(PAIR):
        for row in original[pid]['rows_accepted']:
            if row['reason']==0:
                for key in footprint(row):
                    if key in users:users[key].append(pid)
    for c,u,v in sorted(U):
        im=sc.images(1)[c];rows['fixed_observation_pixels'].append({'camera':c,'frame':1,'observation_id':[c,u,v],'RGB':im.rgb[v,u],'depth_m':im.depth[v,u],
            'mask':bool(im.mask[v,u]),'raw_valid':bool(im.raw_valid[v,u]),'J0_bilinear_users':users[(c,u,v)],'overlap_is_independent_reuse':False})
    keys=set()
    for pid in PAIR:
        src=source_record(sc,pid);keys.add((src['source_camera'],src['source_u'],src['source_v']))
        for c in (0,1):
            u,v,*_=sc.cams[c].project(sc.x0[pid]);keys.add((c,int(np.rint(u)),int(np.rint(v))))
    for c,u,v in sorted(keys):
        occupants=env.occupants(c,u,v)
        for rank,r in enumerate(occupants):rows['source_collision_candidates'].append({'camera':c,'frame':1,'observation_id':[c,u,v],'rank':rank,'owner_point_id':occupants[0]['point_id'],**r,'tier':'TIER_EXACT_ARCHIVE','scope':'Exact canonical frame-1 prior prediction; not later inner/global posterior positions.'})
    enumeration,candidates,counts=enumerate_assignments(sc,env,sorted(keys));rows['joint_assignment_enumeration']=enumeration;rows['candidate_likelihoods']=candidates
    rows['model_comparison_identifiability'].append({'arm':'J2_J3_J5','status':'UNDEFINED_MODEL','representation':'FIXED_OBSERVATION_ASSOCIATION',**counts,
        'missing':'Normalized pixel coverage/material association, source-prior strength, external/OTHER likelihood, and source/reference cross-correlations.',
        'proof':'p(RGB,Z|point) alone supplies no p(observed pixel|point); multiplying by unconstrained positive coverage factors reverses any finite partial likelihood ratio. Point RGB is constant at fixed pixels, so H_RGB_position=0.',
        'evidence_level':'UNIDENTIFIABLE'})
    for arm,reason in [('J2','No calibrated point-to-fixed-pixel coverage or external/OTHER likelihood; partial RGB/Z terms cannot normalize the complete pixel universe.'),
        ('J3','A point-camera reference at projected xref is not automatically the reference for a different fixed-pixel latent association; material continuity and source/reference covariance are missing.'),
        ('J5','No data-supported provenance prior/strength exists. Equal weights are a labelled synthetic sensitivity, not measured real assignment priors.')]:
        rows['model_comparison_identifiability'].append({'arm':arm,'status':'UNDEFINED_MODEL','reason':reason,'representation':'FIXED_OBSERVATION_ASSOCIATION','evidence_level':'UNIDENTIFIABLE'})
    outcomes={};states={}
    for pid in PAIR:
        p=problems[pid];res=original[pid];dx=res['s_plus'][:3]-p.s_minus[:3];dv=res['s_plus'][3:]-p.s_minus[3:];gain,error=velocity_transfer(p.P_minus,dx,dv)
        outcomes[f'J0_{pid}']={'arm':'J0','point_id':pid,'dx_m':dx,'dv_m_s':dv,'dx_mm':1000*np.linalg.norm(dx),'dv_norm_m_s':float(np.linalg.norm(dv)),
            'rows':[r['camera'] for r in p.used(res['rows_accepted'])] if res['observed'] else [],'converged':res['converged'],'acceptance':ACC[res['acceptance_code']],
            'velocity_transfer_matrix_per_s':gain,'velocity_transfer_residual_m_s':error,'full_production_parity':next(r['status'] for r in parity if r['point_id']==pid)}
        for j,h in enumerate(res['history']):
            arrays[f'J0_{pid}_{j}_A']=h['A'];arrays[f'J0_{pid}_{j}_b']=h['b'];arrays[f'J0_{pid}_{j}_delta']=h['delta']
            rows['joint_iteration_history'].append({'arm':'J0','point_id':pid,'iteration':j,'state':h['s'],'criterion':h['J'],'step':h['step'],'threshold':h['threshold'],
                'support':[r['camera'] for r in p.used(h['rows'])],'attribution':p.attribution(h['s'],h['rows'],h['A']),'acceptance_rule':'ORIGINAL_CHI2_ADEQUACY_NOT_MONOTONE_DESCENT'})
            for r in p.used(h['rows']):rows['per_point_per_camera_factors'].append({'arm':'J0','point_id':pid,'other_point_id':PAIR[1] if pid==PAIR[0] else PAIR[0],'frame':1,'iteration':j,'camera':r['camera'],**r})
    arrays['J0_covariance_original_factorized_model']=block_diag(*[original[p]['P_plus'] for p in PAIR])
    calibration=read_json(out/'noise_and_prior_metadata.json')
    pairgate=parity[0]['status']=='PARITY_PASS' and tests['successful'] and calibration['camera_hash_equal']
    if pairgate:
        for arm,transfer,referenced,source_only in [('J0_SHARED_NOISE_CONTROL',False,False,False),('J1_ATOMIC_LEGACY',True,False,False),
            ('J4_CURRENT_ALGEBRAIC',False,True,False),('J3_ATOMIC_ALGEBRAIC_PROXY',True,True,False),('J6_SOURCE_ONLY',False,False,True)]:
            print('Local pair diagnostic:',arm,flush=True)
            model=PairProblem(sc,env,problems,U,transfer,referenced,source_only);result=model.optimize();states[arm]=result['state']
            if result['covariance'] is not None:arrays[arm+'_covariance']=result['covariance']
            arrays[arm+'_state']=result['state']
            for j,pid in enumerate(PAIR):
                state=result['state'][6*j:6*j+6];dx=state[:3]-joint_mean[6*j:6*j+3];dv=state[3:]-joint_mean[6*j+3:6*j+6]
                n=sum(r['pair_index']==j for r in result['rows'])
                outcomes[f'{arm}_{pid}']={'arm':arm,'point_id':pid,'dx_m':dx,'dv_m_s':dv,'dx_mm':float(1000*np.linalg.norm(dx)),'dv_norm_m_s':float(np.linalg.norm(dv)),
                    'rows':[r['camera'] for r in result['rows'] if r['pair_index']==j],'status':result['status'],'observation_assessment':'UNOBSERVED_COAST' if n==0 else 'OBSERVATIONS_RETAINED_NOT_MATERIAL_PROOF',
                    'criterion':result['criterion'],'criterion_scope':'CONDITIONAL_PROJECTIVE_ROW_ENERGY_ONLY; NOT fixed-raw-pixel evidence or comparable across different supports',
                    'covariance_kind':result.get('covariance_kind'),'complete_fixed_pixel_model':'UNDEFINED_MODEL','tier':'TIER_CONTROLLED_LOCAL'}
            for record in result['history']:rows['joint_iteration_history'].append({'arm':arm,**record,'representation':'LEGACY_PROJECTION','tier':'TIER_CONTROLLED_LOCAL'})
            for row in result['rows']:rows['per_point_per_camera_factors'].append({'arm':arm,**row,'frame':1,'covariance_scope':'Shared raw pixel covariance accounted for jointly; reference versions are algebraic diagnostics.'})
            if transfer:rows['ownership_changes'].append({'arm':arm,'camera':env.slot[0],'observation_id':env.slot,'old_owner':PAIR[1],'new_owner':PAIR[0],
                'displaced_owner_updated_together':True,'duplicate_independent_assimilation':False,'scope':'Only the designated observation slot changes owner; subsequent physical visibility is re-evaluated.'})
        names=('J0_SHARED_NOISE_CONTROL','J1_ATOMIC_LEGACY','J4_CURRENT_ALGEBRAIC','J3_ATOMIC_ALGEBRAIC_PROXY')
        if all(n in states for n in names):
            interaction=states[names[3]]-states[names[1]]-states[names[2]]+states[names[0]]
            rows['model_comparison_identifiability'].append({'arm':'OWNERSHIP_X_REFERENCE_INTERACTION','delta_x_a_m':interaction[:3],'delta_v_a_m_s':interaction[3:6],
                'delta_x_b_m':interaction[6:9],'delta_v_b_m_s':interaction[9:],'interpretation':'Conditional legacy response contrast, not global Bayesian evidence; different support/local convergence can confound it.'})
    else:rows['model_comparison_identifiability'].append({'arm':'J1_J6','status':'CLAIMS_BLOCKED_GATE_A0_OR_TESTS','reason':'No replacement baseline is substituted.'})
    for pid in PAIR:
        refs=references_for_untraced(sc,env,pid)
        for c,ref in refs.items():
            var=ref['var0'];row={'point_id':pid,'camera':c,'arm':'LATENT_REFERENCE','status':'NOT_IDENTIFIED','reference':ref,
                'missing':'Joint covariance of source x0/canonical RGB/reference and current pixels; calibrated nuisance prior is not supplied.',
                'variance_sensitivity_scope':'Marginal Cauchy-Schwarz bounds conditional on fixed geometry, not a calibrated joint posterior.'}
            if var is not None:
                now=O.row(sc.images(1)[c],sc.cams[c],sc.x0[pid],sc.run.can['canonical_rgb'][pid],sc.noise,sc.policy,sc.P0[pid],np.zeros(3))
                if 'var' in now:row['difference_variance_lower_upper']=[(np.sqrt(now['var'])-np.sqrt(var))**2,(np.sqrt(now['var'])+np.sqrt(var))**2]
            rows['reference_bias_uncertainty'].append(row)
    rows['heldout_controls']=heldout_controls(root)
    write_json(out/'joint_sequence_status.json',{'status':'NOT_RUN_UNDEFINED_FIXED_OBSERVATION_MODEL','earliest_unverified_pair_posterior_frame':1,
        'missing_path':str(sc.run.free/'covariance/frame_0001.npy'),'missing_array':'P_plus[45588,6,6] and exact external inner/global state','coast_is_success':False,
        'reason':'Phase D requires a mathematically meaningful complete pixel likelihood; legacy conditional diagnostics do not supply it. No ten-frame coupled posterior is fabricated.'})
    from tools.report_f1_joint_responsibility import figures,generate
    figures(sc,env,problems,original,U,outcomes,out)
    for name,records in rows.items():
        for r in records:
            r.setdefault('scene',SCENE);r.setdefault('point_id',list(PAIR));r.setdefault('other_point_id',PAIR[1] if r['point_id']==PAIR[0] else PAIR[0] if r['point_id']==PAIR[1] else None);r.setdefault('frame',1);r.setdefault('tier','TIER_CONTROLLED_LOCAL')
            if isinstance(r['point_id'],int) and r['point_id'] in PAIR:r.setdefault('source',source_record(sc,r['point_id']))
            r.setdefault('likelihood_comparability','LEGACY conditional row support only; full fixed-observation evidence is UNDEFINED_MODEL')
        csv_write(out/(name+'.csv'),records,('scene','point_id','other_point_id','camera','frame','arm','tier'))
    np.savez_compressed(out/'joint_covariance_12x12.npz',**arrays)
    write_json(out/'matrix_units.json',{'state_order':'[x_a(m),v_a(m/s),x_b(m),v_b(m/s)]','covariance':'block units: xx m2, xv/vx m2/s, vv m2/s2','joint_information':'GN conditional linearization; not exact mixture Hessian','shared_R':'Raw-pixel L diag(var) L^T, no invented diagonal jitter.'})
    result={'outcomes':outcomes,'baseline_gate':pairgate,'tests_successful':tests['successful'],'scope':'FRAME1_CONDITIONAL_PAIR; FIXED_PIXEL_MATERIAL_MODEL_UNIDENTIFIABLE',
        'canonical_separation_m':float(np.linalg.norm(sc.x0[PAIR[0]]-sc.x0[PAIR[1]])),
        'source_collision_depth_gap_m':float(sc.cams[0].project(sc.x0[PAIR[0]])[2]-sc.cams[0].project(sc.x0[PAIR[1]])[2]),
        'complete_fixed_pixel_model':'UNDEFINED_MODEL','sequence_exported':False,'source_independence_proof':'Production initializes independent per-point covariances from different source camera pixels, conditional on fixed calibration. Later joint covariance is never reset.',
        'joint_solver_scope':'Mean-Jacobian GN with covariance frozen per step and actual conditional row-energy line search; state-dependent variance derivatives/log-normalizers are not part of this legacy proxy. No complete fixed-pixel MAP/assignment evidence is inferred from it.',
        'missing':read_json(out/'joint_sequence_status.json')}
    write_json(out/'results.json',result);verify(inv);print(generate(out),flush=True)
    write_json(out/'outputs_sha256.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='outputs_sha256.json'})
    return 0 if pairgate else 3


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--forensics-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--fifth-pass-root',type=Path)
    p.add_argument('--out',type=Path,default=DEFAULT_ROOT/'final_joint_responsibility_8444_45588_20261030');p.add_argument('--experiment',choices=('tests','all'),default='all')
    p.add_argument('--frame-start',type=int,default=0);p.add_argument('--frame-end',type=int,default=10);a=p.parse_args(argv)
    if a.frame_start!=0 or a.frame_end!=10:p.error('The requested archive audit covers frames 0..10; the coupled sequence is gated, never silently shortened.')
    fifth=a.fifth_pass_root or a.forensics_root/'fifth_pass_source_responsibility_20261030'
    inv=prepare(a.forensics_root,fifth,a.out,a.frame_end);tests=run_tests(a.out,a.forensics_root);print({k:v for k,v in tests.items() if k!='records'},flush=True)
    if a.experiment=='tests':verify(inv);return tests['exit_code']
    return analyze(a.forensics_root,fifth,a.out,tests,inv,a.frame_end)


if __name__=='__main__':raise SystemExit(main())
