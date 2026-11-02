#!/usr/bin/env python3
"""Capture a real rejected run's S0 and S2 linearizations; replay/analyze without RGB-D/filtering.

Capture reuses the cached source catalogue and recorded noise/policy. It archives the first (S0) and the
relinearized second (S2) operator with compact key maps, stops before the second factorization and never
publishes a posterior or valid frame. Replay always uses the saved exact operator, including shared
pixels/equalities. Analyze reports dimensions, true components, key integrity and the provenance of every
cross-point coupling (S0 -> S2), and reproduces the general BTF capacity failure under a controlled bound.

    python -B tools/replay_temporal_sparse_failure.py capture --cached-run RUN/USB_100109_full/free_forward --output OUT/capture
    python -B tools/replay_temporal_sparse_failure.py analyze --linearization OUT/capture --output OUT/analysis
    python -B tools/replay_temporal_sparse_failure.py replay --linearization OUT/capture/linearization.pt --output OUT/replay_nd --backend symmetric_nd
"""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.temporal_output import atomic_json
from estimator.temporal_sparse_gpu import TemporalSoAGraph,SparseLinearizedProblem


def tensor_tree(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,list):return [tensor_tree(v,device) for v in value]
    if isinstance(value,tuple):return tuple(tensor_tree(v,device) for v in value)
    if isinstance(value,dict):return {k:tensor_tree(v,device) for k,v in value.items()}
    return value


class CaptureComplete(Exception):
    """Intentional diagnostic stop before any new numerical solve."""


def graph_record(graph):
    """Compact key maps of the current linearization (no dense matrix): raw keys, source keys, warp rows and the
    exact bilinear weights/q used by the assembled operator at this state."""
    from estimator.temporal_rgbd_terms import temporal_prediction
    graph.layout()
    record={'N':graph.N,'anchor_frame':graph.anchor_frame,'anchor_active':graph.anchor_active,'state_offsets':dict(graph.state_offsets),
            'pixel_offset':graph.pixel_offset,'scalar_dimension':graph.scalar_dimension,'pixel_keys':graph.pixels.keys.cpu(),
            'active_likelihood':graph.pixels.active_likelihood.cpu(),'source_keys':graph.source_keys.cpu(),
            'encoding':[graph.pixels.frames,graph.pixels.cameras,graph.pixels.pixels],'warps':{}}
    for current,w in graph.warps.items():
        weights=torch.full((len(w['point']),4),float('nan'),device=graph.device,dtype=torch.float64);q=torch.full((len(w['point']),2),float('nan'),device=graph.device,dtype=torch.float64)
        for ref in [graph.anchor_frame,*graph.states]:
            ix=torch.where(w['reference_frame']==ref)[0]
            if not len(ix):continue
            p=w['point'][ix]
            now=graph.state_and_blocks_gpu(current,p)[0];old=graph.state_and_blocks_gpu(ref,p)[0]
            values=graph.pixels.clean[graph.pixels.lookup(w['reference_keys'][ix])]
            _,_,_,qq,ww,_,_=temporal_prediction(graph.rig,now,old,w['camera'][ix],w['uv'][ix],values,w['cell'][ix])
            weights[ix]=ww;q[ix]=qq
        record['warps'][current]={**{name:w[name].cpu() for name in ('point','camera','reference_frame','reference_keys','current_keys','cell','uv')},
                                  'weights':weights.cpu(),'q':q.cpu()}
    return record


def capture(cached,output):
    from scene.config import Config
    from scene.dataset import Dataset
    from estimator.projective_camera import CameraRig
    from estimator.projective_sensor import MeasuredFrame,SensorNoise
    from estimator.temporal_reference import ReferenceTable
    from estimator.temporal_zbuffer import first_hit_pixel_owner
    from estimator.temporal_rgbd_terms import raw_values,sensor_covariance,build_temporal_rows
    from estimator.temporal_sparse_nonlinear import solve_nonlinear_sparse_iekf_gpu
    from estimator.temporal_sparse_run import signature
    from estimator.point_kalman_cuda import transition,process_noise
    manifest=json.loads((cached.parent/'run_manifest.json').read_text())
    cfg=dict(manifest['noise'])
    for name in ['qc','qx','velocity_covariance','pixel_covariance','color_covariance']:cfg[name]=np.asarray(cfg[name])
    config=Config(**cfg);policy=manifest['policy']
    dataset=Dataset(Path(manifest['dataset']['input']),frame_limit=3)
    rig=CameraRig(dataset.cameras,'cuda');noise=SensorNoise.from_config(config,policy,'cuda',torch.float64)
    with np.load(cached/'canonical.npz') as source:
        xyz=source['canonical_xyz'];provenance=source['original_source_ids'];ids=source['point_id']
    if not np.array_equal(ids,np.arange(len(ids))):raise ValueError('This full-cloud reproduction requires every original point')
    frames={k:MeasuredFrame(dataset,k,'cuda',torch.float64) for k in [0,1]}
    provenance=torch.as_tensor(provenance,device='cuda',dtype=torch.long);N=len(xyz);C=rig.count
    value=raw_values(frames[0],provenance[:,1],provenance[:,2])
    graph=TemporalSoAGraph(rig,3,N,provenance[:,1],provenance[:,2],torch.as_tensor(xyz,device='cuda'),value,
        sensor_covariance(value,noise),torch.as_tensor(config.pixel_covariance,device='cuda'),torch.as_tensor(config.velocity_covariance,device='cuda'),0)
    graph.layout();state=graph.state_and_blocks_gpu(0)[0]
    refs=ReferenceTable(C,N,'cuda',torch.float64)
    owners=first_hit_pixel_owner(state[:,:3],rig,frames[0]);refs.commit_accepted_references(owners,frames[0],state,0)
    refs.initialize_from_source_catalogue(provenance.cpu().numpy(),0,state)
    warp=torch.zeros((4,4),device='cuda',dtype=torch.float64)
    dt=float(dataset.times[1]-dataset.times[0]);F=transition(dt,device='cuda');mean=state@F.T
    owners=first_hit_pixel_owner(mean[:,:3],rig,frames[1],dense=False)
    rows,_,_=build_temporal_rows(mean,owners,frames[1],refs,frames,rig,noise,policy,warp,1,3)
    sig=signature(rows)
    graph.pack_source_process_warp_factors_gpu(0,1,F,process_noise(dt,torch.as_tensor(config.qc,device='cuda')),mean,rows,frames,noise,warp)
    def same_domain():
        state=graph.state_and_blocks_gpu(1)[0]
        old=graph.state_and_blocks_gpu(0)[0];cc,pp=torch.where(refs.frame==0);refs.state[cc,pp]=old[pp]
        owners=first_hit_pixel_owner(state[:,:3],rig,frames[1],dense=False)
        current,_,_=build_temporal_rows(state,owners,frames[1],refs,frames,rig,noise,policy,warp,1,3)
        if torch.equal(signature(current),sig):return True
        before=torch.full((C,N,2),-1,device='cuda',dtype=torch.long);after=torch.full_like(before,-1)
        before[rows.camera,rows.point,0]=rows.pixel;before[rows.camera,rows.point,1]=rows.reference_frame
        after[current.camera,current.point,0]=current.pixel;after[current.camera,current.point,1]=current.reference_frame
        graph.domain_bad_points=torch.where((before!=after).any(2).any(0))[0]
        return False
    original=graph.assemble_sparse_linearized_problem_gpu;calls=0
    def archive():
        nonlocal calls
        problem=original();calls+=1
        if calls==1:
            torch.save({f.name:tensor_tree(getattr(problem,f.name),'cpu') for f in fields(problem)},output/'linearization_S0.pt')
            torch.save(graph_record(graph),output/'graph_S0.pt')
        if calls==2:
            record={f.name:tensor_tree(getattr(problem,f.name),'cpu') for f in fields(problem)}
            torch.save(record,output/'linearization.pt')
            torch.save(graph_record(graph),output/'graph_S2.pt')
            torch.save({'candidate_state':graph.state_and_blocks_gpu(1)[0].cpu(),'predicted_state':mean.cpu(),'original_point_id':ids.tolist()},output/'uncommitted_candidate.pt')
            atomic_json(output/'capture.json',{'status':'CAPTURED_LINEARIZATION_NOT_FILTERING_SUCCESS','cached_source':str(cached),
                'points':N,'cameras':C,'dt':dt,'iteration':1,'scalar_dimension':problem.scalar_dimension,
                'nonlinear_history':graph.nonlinear_history,'published_frames':[],
                'sha256':hashlib.sha256((output/'linearization.pt').read_bytes()).hexdigest()})
            raise CaptureComplete()
        return problem
    graph.assemble_sparse_linearized_problem_gpu=archive
    try:solve_nonlinear_sparse_iekf_gpu(graph,frames,noise,policy,2,1e-8,same_domain)
    except CaptureComplete:return
    raise RuntimeError('The requested second linearization was not reached')


def replay(saved,output,backend,benchmark_queries):
    from estimator.temporal_sparse_solve import SparseJointFactorization,sparse_rows,connected_columns_gpu
    from estimator.temporal_sparse_gpu import sparse_coo
    from estimator.temporal_cudss import CudaSparseFactor
    problem=SparseLinearizedProblem(**tensor_tree(torch.load(saved,map_location='cpu',weights_only=True),'cuda'))
    def native(matrix):
        torch.save(matrix.cpu(),output/'KKT.pt')
        return CudaSparseFactor(matrix,algorithm=backend)
    started=time.perf_counter()
    try:
        with patch('estimator.temporal_sparse_solve.CudaSparseFactor',side_effect=native):factor=SparseJointFactorization(problem)
        try:
            step,certificate=factor.solve_constrained_sparse_joint_gpu()
            torch.save(step.cpu(),output/'step.pt')
            status={'status':'REPLAY_LINEAR_SOLVE_CERTIFIED','certificate':certificate,'profile':factor.profile}
            if benchmark_queries:
                capture_info=json.loads((saved.parent/'capture.json').read_text());N=capture_info['points']
                Q=sparse_rows(factor.T,torch.arange(5*N,11*N,device='cuda'))
                labels,_=connected_columns_gpu(factor.A,factor.C,factor.information)
                row,col=Q.indices();pair,inverse=torch.unique(row*factor.A.shape[1]+labels[col],sorted=True,return_inverse=True)
                component=pair%factor.A.shape[1];order=torch.argsort(component,stable=True)
                starts=torch.ones(len(order),device='cuda',dtype=torch.bool);starts[1:]=component[order[1:]]!=component[order[:-1]]
                positions=torch.arange(len(order),device='cuda');rank=positions-torch.cummax(torch.where(starts,positions,torch.zeros_like(positions)),0).values
                colors=torch.empty_like(rank);colors[order]=rank
                total=int(colors.max())+1
                bench={'required_exact_RHS_colors':total,'point_count':N,'covariance_computed':False,'batches':[]}
                atomic_json(output/'covariance_query_benchmark.json',bench)
                for width in [8,32]:
                    take=colors[inverse]<width
                    rhs=sparse_coo(torch.stack((factor.A.shape[0]+col[take],colors[inverse[take]])),Q.values()[take],(factor.K.shape[0],width)).to_dense()
                    torch.cuda.synchronize();t0=time.perf_counter();answer=factor.factor.solve(rhs);torch.cuda.synchronize();seconds=time.perf_counter()-t0
                    bench['batches'].append({'width':width,'seconds':seconds,'linear_solve_only_extrapolation_seconds':seconds*total/width,
                        'certificate':dict(factor.factor.profile['last_solve'])})
                    del rhs,answer
                    atomic_json(output/'covariance_query_benchmark.json',bench)
                status['query_benchmark']=bench
        finally:factor.close()
    except Exception as error:
        status={'status':'REPLAY_FAILED','error':str(error),'profile':getattr(error,'profile',None)}
    status.update(backend=backend,seconds=time.perf_counter()-started,filtering_success=False,
        saved_operator=str(saved),saved_operator_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
    atomic_json(output/'replay.json',status);print(json.dumps(status,indent=2))
    return 0 if status['status']=='REPLAY_LINEAR_SOLVE_CERTIFIED' else 1


def load_problem(path,device):
    return SparseLinearizedProblem(**tensor_tree(torch.load(path,map_location='cpu',weights_only=True),device))


def canonical(labels):
    """Partition -> label = smallest member index (independent of the labelling algorithm)."""
    labels=np.asarray(labels);smallest=np.full(labels.max()+1,np.iinfo(np.int64).max,dtype=np.int64)
    np.minimum.at(smallest,labels,np.arange(len(labels)));return smallest[labels]


def cpu_union_find(n,edges):
    """Independent CPU oracle: scipy connected components of an explicit edge list over n nodes."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    a,b=(np.concatenate(x) if x else np.empty(0,dtype=np.int64) for x in zip(*edges)) if edges else (np.empty(0,dtype=np.int64),)*2
    graph=sp.coo_matrix((np.ones(len(a),dtype=np.int8),(a,b)),shape=(n,n))
    return canonical(connected_components(graph,directed=False)[1])


def column_nodes(reduced,record):
    """Reduced (innovation) column -> semantic node: point i (anchor chart or any state of i) or N + raw-key index."""
    problem=reduced.problem;N=record['N']
    pivots=torch.cat([p for _,p in problem.pivot_groups]).cpu() if problem.pivot_groups else torch.empty(0,dtype=torch.long)
    free=torch.ones(problem.scalar_dimension,dtype=torch.bool);free[pivots]=False
    original=torch.where(free)[0].numpy()
    node=np.empty(len(original),dtype=np.int64)
    anchor=5*N if record['anchor_active'] else 0;pixel=record['pixel_offset']
    a=original<anchor;st=(original>=anchor)&(original<pixel);px=original>=pixel
    node[a]=original[a]//5
    node[st]=((original[st]-anchor)%(6*N))//6
    node[px]=N+(original[px]-pixel)//4
    return node


def row_node_pairs_actual(reduced,node):
    """(row block, node) pairs of every nonzero coefficient of the reduced stochastic operator."""
    ix=reduced.A.coalesce().indices().cpu().numpy();values=reduced.A.coalesce().values().cpu().numpy()
    take=values!=0
    return np.unique(np.stack((ix[0,take]//4,node[ix[1,take]]),1),axis=0)


def row_node_pairs_predicted(record):
    """Independent prediction from key maps only: the sensor row of current pixel (c,t,p) substituted by point i's
    warp touches {i (x_t, x_r and their anchor/process innovations), i's source pixel, corners with weight != 0}."""
    N=record['N'];keys=record['pixel_keys'].numpy();index=lambda k:np.searchsorted(keys,k)
    current=[];sets=[]
    for w in record['warps'].values():
        p=w['point'].numpy();src=index(record['source_keys'].numpy()[p])+N
        corners=index(w['reference_keys'].numpy())+N;nz=w['weights'].numpy()!=0
        cur=w['current_keys'].numpy();current.append(cur)
        sets.append((cur,np.stack((p,src),1),corners,nz))
    order=np.argsort(np.concatenate(current));block=np.empty_like(order);block[order]=np.arange(len(order))
    pairs=[];offset=0
    for cur,ps,corners,nz in sets:
        b=block[offset:offset+len(cur)];offset+=len(cur)
        pairs.append(np.stack((np.repeat(b,2),ps.reshape(-1)),1))
        rows,cols=np.nonzero(nz);pairs.append(np.stack((b[rows],corners[rows,cols]),1))
    return np.unique(np.concatenate(pairs),axis=0)


def trace_sets(actual,predicted,N):
    """Every ACTUAL nonzero must be predicted by an analytical factor (no untraceable coupling). Predicted pairs with
    no actual coefficient are exact cancellations, e.g. at x_t = x_r the current/reference Jacobians cancel on columns
    shared by both states (anchor chart, source pixel); they are reported, never pruned."""
    key=lambda x:x[:,0]*(1<<32)+x[:,1]
    ka,kp=key(actual),key(predicted)
    untraced=~np.isin(ka,kp);cancelled=~np.isin(kp,ka)
    c=predicted[cancelled]
    return {'every_actual_nonzero_traced_to_predicted_factor':bool(not untraced.any()),'untraced_actual_pairs':int(untraced.sum()),
            'actual_row_node_pairs':int(len(actual)),'predicted_row_node_pairs':int(len(predicted)),
            'predicted_without_actual_coefficient':{'total':int(cancelled.sum()),'point_nodes':int((c[:,1]<N).sum()),'raw_pixel_nodes':int((c[:,1]>=N).sum())}}


def key_integrity(record):
    F,C,P=record['encoding'];keys=record['pixel_keys'].numpy();N=record['N']
    frame,rest=keys//(C*P),keys%(C*P);camera,pixel=rest//P,rest%P
    out={'unique_keys':int(len(np.unique(keys))),'registered_keys':int(len(keys)),
         'encoding_injective_roundtrip':bool(np.array_equal((frame*C+camera)*P+pixel,keys)),
         'source_keys_unique':int(len(np.unique(record['source_keys'].numpy()))),'points':N}
    used=[record['source_keys'].numpy()]
    for k,w in record['warps'].items():
        cur=w['current_keys'].numpy();ref=w['reference_keys'].numpy()
        out[f'frame{k}_warp_rows']=int(len(cur));out[f'frame{k}_current_keys_unique']=bool(len(np.unique(cur))==len(cur))
        out[f'frame{k}_all_keys_registered']=bool(np.isin(cur,keys).all() and np.isin(ref,keys).all())
        out[f'frame{k}_cross_camera_key_collision']=int(((ref//P)%C!=w['camera'].numpy()[:,None]).sum()+((cur//P)%C!=w['camera'].numpy()).sum())
        used+=[cur,ref.reshape(-1)]
    out['orphan_registered_keys_used_by_no_factor']=int((~np.isin(keys,np.concatenate(used))).sum())
    return out


def coupling_provenance(record,weights_override=None):
    """Point graph through shared raw keys: who couples whom, and through which analytical role."""
    N=record['N'];keys=record['pixel_keys'].numpy();index=lambda k:np.searchsorted(keys,k)
    src=index(record['source_keys'].numpy())
    edges=[];corner_users=[];observed=np.zeros(N,dtype=bool)
    for k,w in record['warps'].items():
        p=w['point'].numpy();observed[p]=True
        nz=(w['weights'].numpy()!=0) if weights_override is None else weights_override[k]
        rows,cols=np.nonzero(nz);c=index(w['reference_keys'].numpy())[rows,cols]
        edges.append((p[rows],N+c));corner_users.append(np.stack((c,p[rows]),1))
    o=np.where(observed)[0];edges.append((o,N+src[o]))
    labels=cpu_union_find(N+len(keys),edges)[:N]
    _,sizes=np.unique(labels[observed],return_counts=True)
    cu=np.unique(np.concatenate(corner_users),axis=0) if corner_users else np.empty((0,2),dtype=np.int64)
    k_,n_=np.unique(cu[:,0],return_counts=True)
    shared=set(k_[n_>=2].tolist());is_src_of=dict(zip(src[o].tolist(),o.tolist()))
    corner_and_source=sum(1 for key,pt in cu.tolist() if key in is_src_of and is_src_of[key]!=pt)
    return {'observed_points':int(observed.sum()),'largest_point_component':int(sizes.max()) if len(sizes) else 0,
            'point_components_observed':int(len(sizes)),'keys_shared_as_nonzero_corner_by_2plus_points':len(shared),
            'corner_uses_of_another_points_source_pixel':int(corner_and_source)}


def analyze(capture_dir,output):
    from estimator.temporal_sparse_solve import ReducedLinearizedSystem,connected_columns_gpu
    from estimator.temporal_cudss import CudaSparseFactor,SparseNumericalFailure
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report={'capture':str(capture_dir),'inputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(capture_dir.glob('*.pt'))},
            'statistical_semantics':'connectivity = actual nonzero coefficients of the reduced linearized operator; no motion graph, no symbolic ties'}
    records={}
    for stage,problem_file in (('S0','linearization_S0.pt'),('S2','linearization.pt')):
        record=torch.load(capture_dir/f'graph_{stage}.pt',weights_only=False);records[stage]=record
        reduced=ReducedLinearizedSystem(load_problem(capture_dir/problem_file,device))
        m,n=reduced.A.shape
        gpu=canonical(connected_columns_gpu(reduced.A,reduced.C,reduced.information)[0].cpu().numpy())
        edges=[]
        for M,off in ((reduced.A,0),(reduced.C,m)):
            ix=M.coalesce().indices().cpu().numpy();keep=M.coalesce().values().cpu().numpy()!=0
            edges.append((n+off+ix[0,keep],ix[1,keep]))
        if reduced.information is not None:
            ix=reduced.information.coalesce().indices().cpu().numpy();edges.append((ix[0],ix[1]))
        cpu=cpu_union_find(n+m+reduced.C.shape[0],edges)[:n]
        _,sizes=np.unique(gpu,return_counts=True)
        node=column_nodes(reduced,record)
        actual,predicted=row_node_pairs_actual(reduced,node),row_node_pairs_predicted(record)
        w=np.concatenate([x['weights'].numpy() for x in record['warps'].values()])
        report[stage]={'scalar_dimension':reduced.problem.scalar_dimension,'reduced_free_dimension':n,'stochastic_rows':m,
            'remaining_constraints':int(reduced.C.shape[0]),'nnz_A_reduced':int(reduced.A._nnz()),'nnz_KKT':int(reduced.K._nnz()),
            'factor_counts':reduced.problem.factor_counts,'true_components':int(len(sizes)),'largest_component_scalars':int(sizes.max()),
            'component_size_quantiles':np.quantile(sizes,[.5,.9,.99,.999,1]).tolist(),
            'GPU_components_equal_independent_CPU_union_find':bool(np.array_equal(gpu,cpu)),
            'information_matrix_present':reduced.information is not None,
            'row_blocks':int(m//4),**trace_sets(actual,predicted,record['N']),
            'bilinear_weights':{'exact_zero':int((w==0).sum()),'nonzero_below_1e-12':int(((w!=0)&(np.abs(w)<1e-12)).sum()),'total':int(w.size)},
            'keys':key_integrity(record),'coupling':coupling_provenance(record)}
        if stage=='S0':K0=reduced.K
        else:K2=reduced.K;expected2=reduced.expected_inertia()
        del reduced
    w0,w2=records['S0']['warps'],records['S2']['warps'];change={}
    hybrid={}
    for k in w2:
        cell=(w0[k]['cell']!=w2[k]['cell']).any(1).numpy();nz0=w0[k]['weights'].numpy()!=0;nz2=w2[k]['weights'].numpy()!=0
        change[k]={'rows':int(len(cell)),'cell_changed':int(cell.sum()),'same_cell_corners_zero_to_nonzero':int(((~nz0)&nz2)[~cell].sum()),
                   'same_cell_corners_nonzero_to_zero':int((nz0&~nz2)[~cell].sum()),'same_points':bool(torch.equal(w0[k]['point'],w2[k]['point']))}
        hybrid[k]=np.where(cell[:,None],nz2,nz0)
    report['S0_to_S2']={'per_frame':change,'S2_with_S0_activation_on_unchanged_cells':coupling_provenance(records['S2'],hybrid)}
    if device=='cuda':
        def attempt(K,algorithm,**kw):
            torch.cuda.empty_cache()
            try:
                f=CudaSparseFactor(K,algorithm=algorithm,**kw);out={'status':'FACTORIZED',**f.profile};f.close()
            except SparseNumericalFailure as error:out={'status':'FAILED','error':str(error).split(';')[0],**getattr(error,'profile',{})}
            return {k:out.get(k) for k in ('status','error','rows','nnz','info','factor_nnz','max_LU_nnz_capacity','max_LU_nnz_capacity_source','inertia',
                                         'expected_inertia','native_memory_estimates_bytes','VRAM_free_before_analysis','analysis_seconds','factorization_seconds')}
        s0=attempt(K0,'general_btf');lu=s0.get('factor_nnz')
        report['INFO_controlled_reproduction']={
            'S0_general_btf_free_VRAM_bound':s0,
            'S0_general_btf_bound_half_of_its_measured_LU_nnz':attempt(K0,'general_btf',max_lu_nnz=lu//2) if lu else None,
            'S0_general_btf_bound_equal_measured_LU_nnz':attempt(K0,'general_btf',max_lu_nnz=lu) if lu else None,
            'S2_general_btf_free_VRAM_bound':attempt(K2,'general_btf'),
            'S2_symmetric_nd':attempt(K2,'symmetric_nd',expected_inertia=expected2),
            'memory_estimate_layout':'[0] permanent device, [1] peak device, [2] permanent host, [3] peak host (bytes), [6] estimate state'}
    atomic_json(output/'analysis.json',report);print(json.dumps(report,indent=1,default=str)[:6000])
    return 0


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);ap.add_argument('stage',choices=['capture','replay','analyze'])
    ap.add_argument('--cached-run',type=Path);ap.add_argument('--linearization',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--backend',choices=['general_btf','symmetric_nd'],default='general_btf')
    ap.add_argument('--benchmark-queries',action='store_true')
    a=ap.parse_args()
    if a.output.exists():ap.error('Output already exists; never overwrite evidence')
    a.output.mkdir(parents=True)
    if a.stage=='capture':
        if a.cached_run is None:ap.error('capture requires --cached-run')
        capture(a.cached_run,a.output);return 0
    if a.linearization is None:ap.error(f'{a.stage} requires --linearization')
    if a.stage=='analyze':return analyze(a.linearization,a.output)
    return replay(a.linearization,a.output,a.backend,a.benchmark_queries)


if __name__=='__main__':raise SystemExit(main())
