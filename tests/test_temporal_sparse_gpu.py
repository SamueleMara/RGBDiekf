import unittest
from types import SimpleNamespace
import numpy as np
import torch
from estimator.temporal_sparse_gpu import TemporalSoAGraph, UniquePixelTableGPU
from estimator.temporal_rgbd_terms import raw_values, sensor_covariance
from estimator.projective_sensor import SensorNoise
from estimator.point_kalman_cuda import transition, process_noise
from estimator.temporal_full_graph import TemporalWindowGraph
from tests.test_temporal_warp import rig, frame

SPARSE_WITNESSES = []


def fixture_graph(device='cpu', ideal=True):
    rr=rig(device);f0=frame(device);f1=frame(device);f1.k=1
    c=torch.tensor([0,0],device=device);pix=torch.tensor([27,28],device=device)
    uv=torch.stack((pix%8,pix//8),-1).double()
    noise=SensorNoise(torch.tensor([.01,.015,.02],device=device,dtype=torch.float64),.005,0.,0.,0.,'sum_w2')
    y=raw_values(f0,c,pix);R=sensor_covariance(y,noise)
    xyz=rr.backproject(c,uv,y[:,3])
    g=TemporalSoAGraph(rr,4,2,c,pix,xyz,y,R,torch.eye(2,device=device,dtype=torch.float64)/12,
                      torch.eye(3,device=device,dtype=torch.float64)*.7,0)
    g.layout();ref=g.source_state_and_blocks_gpu()[0]
    initial=ref.clone();initial[:,0]+=.007;initial[:,1]-=.003
    points=torch.tensor([0,1,0,1],device=device)
    cams=torch.tensor([0,0,1,1],device=device)
    pixels=torch.tensor([27,28,27,28],device=device)
    v=torch.stack((pixels%8,pixels//8),-1).double()
    q=v-rr.project_rows(initial[points,:3],cams)[0]+rr.project_rows(ref[points,:3],cams)[0]
    cell=q.floor().long();xy=cell[:,None,:]+torch.tensor([[0,0],[1,0],[0,1],[1,1]],device=device)
    rp=xy[...,1]*8+xy[...,0]
    rows=SimpleNamespace(point=points,camera=cams,pixel=pixels,reference_frame=torch.zeros(4,device=device,dtype=torch.long),
                         reference_pixels=rp,cell=cell,y=raw_values(f1,cams,pixels))
    F=transition(.04,device=device);Q=process_noise(.04,torch.eye(3,device=device,dtype=torch.float64)*.2)
    warp=torch.zeros((4,4),device=device,dtype=torch.float64) if ideal else torch.tensor(
        [[.02,.002,0.,0.],[.002,.03,.003,0.],[0.,.003,.04,.001],[0.,0.,.001,.01]],device=device,dtype=torch.float64)
    g.pack_source_process_warp_factors_gpu(0,1,F,Q,initial,rows,{0:f0,1:f1},noise,warp)
    g.layout()
    # Nonzero chart and latent perturbations exercise the actual nonlinear derivatives.
    g.anchor[:,0]=torch.tensor([.012,-.009],device=device)
    g.anchor[:,1]=torch.tensor([-.008,.011],device=device)
    g.anchor[:,2:]=.013
    g.pixels.clean += torch.tensor([.0001,-.0003,.0002,.0004],device=device,dtype=torch.float64)
    return g


def dense_oracle_for(graph):
    """Small TEST-ONLY graph built through the independent original equations."""
    dense=TemporalWindowGraph('cpu')
    t=dense.tensor
    # CameraRig is rebuilt on CPU; no production Jacobian or sparse matrix is copied.
    rr=rig('cpu')
    ids=graph.pixels.keys.cpu().numpy()
    for index,key in enumerate(ids):
        pixel=int(key%graph.pixels.pixels);a=int(key//graph.pixels.pixels)
        dense.pixels.add_raw_sensor_factors_once((a%graph.pixels.cameras,a//graph.pixels.cameras,pixel),graph.pixels.mean[index].cpu(),graph.pixels.covariance[index].cpu())
    for i in range(graph.N):
        c,p=int(graph.source_camera[i]),int(graph.source_pixel[i]);source=dense.values[('pixel',c,graph.anchor_frame,p)]
        dense.add_coupled_anchor_factors(i,graph.anchor_frame,(c,graph.anchor_frame,p),source,
            graph.pixels.covariance[graph.pixels.lookup(graph.source_keys[i:i+1])[0]].cpu(),graph.source_uv[i].cpu(),rr,c,
            graph.anchor_covariance[:2,:2].cpu(),graph.anchor_covariance[2:,2:].cpu(),graph.base[i].cpu())
    for current,(prev,F,Q) in graph.process.items():
        for i in range(graph.N):dense.add_process_factors(i,prev,current,F.cpu(),Q.cpu(),graph.states[current][i].cpu())
    for current,w in graph.warps.items():
        for i in range(len(w['point'])):
            p,c,ref=int(w['point'][i]),int(w['camera'][i]),int(w['reference_frame'][i])
            refs=[('pixel',c,ref,int(key%graph.pixels.pixels)) for key in w['reference_keys'][i]]
            target=('pixel',c,current,int(w['current_keys'][i]%graph.pixels.pixels))
            dense.add_warp_factors(p,current,ref,c,w['uv'][i].cpu(),refs,target,rr,w['covariance'].cpu(),w['cell'][i].cpu())
    for i in range(graph.N):dense.values[('anchor',i,graph.anchor_frame)]=graph.anchor[i].cpu().clone()
    for k,states in graph.states.items():
        for i in range(graph.N):dense.values[('state',i,k)]=states[i].cpu().clone()
    for i,key in enumerate(ids):
        pixel=int(key%graph.pixels.pixels);a=int(key//graph.pixels.pixels)
        dense.values[('pixel',a%graph.pixels.cameras,a//graph.pixels.cameras,pixel)]=graph.pixels.clean[i].cpu().clone()
    J,r,layout,blocks=dense.linearize();hard=dense.constraint_rows(blocks)
    perm=torch.empty(J.shape[1],dtype=torch.long)
    for key,sl in layout.items():
        if key[0]=='anchor':start=5*key[1]
        elif key[0]=='state':start=graph.state_offsets[key[2]]+6*key[1]
        else:
            encoded=(key[2]*graph.pixels.cameras+key[1])*graph.pixels.pixels+key[3]
            start=graph.pixel_offset+4*int(np.searchsorted(ids,encoded))
        perm[sl]=torch.arange(start,start+sl.stop-sl.start)
    A=torch.zeros_like(J.to_dense());A[:,perm]=J.to_dense()
    return dense,A[~hard],r[~hard],A[hard],r[hard],layout,perm


class SparseGPUModelTests(unittest.TestCase):
    def devices(self):
        return ['cpu']+(['cuda'] if torch.cuda.is_available() else [])

    def test_unique_registry_reuse_retirement_and_overflow(self):
        for device in self.devices():
            table=UniquePixelTableGPU(5,2,64,device)
            c=torch.tensor([1,0,1],device=device);k=torch.tensor([0,1,0],device=device);p=torch.tensor([7,8,7],device=device)
            key=table.encode(c,k,p);value=torch.tensor([[.1,.2,.3,2.],[.2,.1,.3,3.],[.1,.2,.3,2.]],device=device,dtype=torch.float64)
            cov=torch.eye(4,device=device,dtype=torch.float64).expand(3,-1,-1).clone()
            index=table.build_or_update_unique_pixel_table_gpu(key,value,cov)
            self.assertEqual(len(table.keys),2);self.assertEqual(int(index[0]),int(index[2]))
            table.clean[0,0]+=.3;before=table.clean.clone()
            table.build_or_update_unique_pixel_table_gpu(key,value,cov)
            torch.testing.assert_close(before,table.clean,atol=0,rtol=0)
            table.retired_frames=torch.tensor([0],device=device)
            with self.assertRaisesRegex(ValueError,'EXPIRED_PIXEL'):table.build_or_update_unique_pixel_table_gpu(key,value,cov)
            with self.assertRaises(ValueError):table.encode(c,k,torch.tensor([-1,8,7],device=device))
        with self.assertRaisesRegex(ValueError,'overflow'):UniquePixelTableGPU(2**50,2**20,2**20,'cpu')

    def test_source_state_and_depth_cross_covariance_matches_oracle(self):
        for device in self.devices():
            graph=fixture_graph(device);dense,*_=dense_oracle_for(graph)
            state,blocks=graph.source_state_and_blocks_gpu()
            for i in range(2):
                ds,db=dense.state(i,0)
                torch.testing.assert_close(state[i].cpu(),ds,atol=1e-15,rtol=1e-14)
                torch.testing.assert_close(blocks[0][1][i].cpu(),db[('anchor',i,0)],atol=1e-15,rtol=1e-14)
                source=('pixel',0,0,int(graph.source_pixel[i]))
                cross=blocks[1][1][i].cpu()@graph.pixels.covariance[graph.pixels.lookup(graph.source_keys[i:i+1])[0]].cpu()
                self.assertGreater(float(cross[:3,3].norm()),0.)
                torch.testing.assert_close(cross,db[source]@graph.pixels.covariance[graph.pixels.lookup(graph.source_keys[i:i+1])[0]].cpu(),atol=1e-15,rtol=1e-14)

    def test_whitened_joint_operator_matches_independent_dense_oracle(self):
        for device in self.devices():
            for ideal in [False,True]:
                g=fixture_graph(device,ideal);problem=g.assemble_sparse_linearized_problem_gpu()
                dense,A,r,C,c,_,_=dense_oracle_for(g)
                AA=problem.A.to_dense().cpu();rr=problem.residual.cpu()
                Herror=float((AA.T@AA-A.T@A).abs().max());gerror=float((AA.T@rr-A.T@r).abs().max())
                Hrelative=Herror/max(1.,float((A.T@A).abs().max()))
                self.assertLess(Hrelative,2e-14)
                torch.testing.assert_close(AA.T@rr,A.T@r,atol=2e-10,rtol=3e-13)
                self.assertAlmostEqual(float(rr@rr),float(r@r),delta=2e-10)
                torch.testing.assert_close(problem.C.to_dense().cpu(),C,atol=2e-14,rtol=2e-14)
                torch.testing.assert_close(problem.constraint.cpu(),c,atol=2e-14,rtol=2e-14)
                v=torch.linspace(-1.,1.,problem.scalar_dimension,device=device,dtype=torch.float64)
                z=torch.linspace(.3,.8,len(rr),device=device,dtype=torch.float64)
                adjoint=abs(float(problem.apply_J(v)@z-v@problem.apply_JT(z)))
                self.assertLess(adjoint,2e-10)
                self.assertEqual(problem.factor_counts['RAW_PIXEL'],dense.pixels.total_sensor_factors)
                SPARSE_WITNESSES.append({'device':device,'ideal':ideal,'H_absolute':Herror,'H_relative':Hrelative,
                    'gradient_absolute':gerror,'adjoint_absolute':adjoint,'dimension':problem.scalar_dimension,
                    'nnz_A':problem.A._nnz(),'nnz_C':problem.C._nnz(),'factor_counts':problem.factor_counts})

    def test_incremental_dirty_owner_matches_brute_force_with_competitors(self):
        from estimator.temporal_zbuffer import first_hit_pixel_owner,incremental_first_hit_pixel_owner,compare_owner_maps,owners_at_gpu
        for device in self.devices():
            rr=rig(device);ff=frame(device)
            old=torch.tensor([[0.,0.,2.],[0.,0.,2.],[-.5,0.,2.],[.5,0.,2.]],device=device,dtype=torch.float64)
            cache=first_hit_pixel_owner(old,rr,ff,dense=False)
            for delta in [torch.tensor([[0.,0.,-.1],[0.,0.,0.],[.4,0.,0.],[0.,0.,0.]],device=device),torch.zeros_like(old)]:
                new=old+delta
                inc=incremental_first_hit_pixel_owner(new,old,cache,rr,ff)
                brute=first_hit_pixel_owner(new,rr,ff,dense=False)
                self.assertEqual(compare_owner_maps(inc,brute)['switched_pixels'],0)
                torch.testing.assert_close(inc.reason,brute.reason,atol=0,rtol=0)
                cameras=torch.tensor([0,1,0],device=device);pixels=torch.tensor([27,27,0],device=device)
                torch.testing.assert_close(owners_at_gpu(inc,cameras,pixels),owners_at_gpu(brute,cameras,pixels),atol=0,rtol=0)

    def test_nonlinear_GPU_cost_visibility_and_constrained_acceptance(self):
        if not torch.cuda.is_available():raise RuntimeError('Nonlinear sparse acceptance must run on actual CUDA')
        from estimator.temporal_sparse_nonlinear import solve_nonlinear_sparse_iekf_gpu,nonlinear_objective_gpu
        import json
        from pathlib import Path
        g=fixture_graph('cuda',True);frames={0:frame('cuda'),1:frame('cuda')};frames[1].k=1
        noise=SensorNoise(torch.tensor([.01,.015,.02],device='cuda',dtype=torch.float64),.005,0.,0.,0.,'sum_w2')
        policy=json.loads(Path('configs/projective_rgbd.json').read_text())
        factor,result=solve_nonlinear_sparse_iekf_gpu(g,frames,noise,policy,4,1e-8)
        try:
            self.assertGreater(result['accepted_iterations'],0)
            for row in result['history']:
                if row['accepted']:self.assertLessEqual(row['proposal_objective'],row['objective']+1e-12)
            self.assertLess(float(nonlinear_objective_gpu(g)[1]),2e-11)
        finally:factor.close()

    def test_line_search_components_keep_full_point_and_shared_pixel_blocks(self):
        from estimator.temporal_sparse_nonlinear import true_variable_block_components_gpu
        for device in self.devices():
            graph=fixture_graph(device,True);problem=graph.assemble_sparse_linearized_problem_gpu()
            labels,count=true_variable_block_components_gpu(graph,problem)
            for offset,width,number in [(0,5,graph.N),(graph.state_offsets[1],6,graph.N),(graph.pixel_offset,4,len(graph.pixels.keys))]:
                block=labels[offset:offset+width*number].reshape(number,width)
                self.assertTrue(bool((block==block[:,:1]).all()))
            for matrix in [problem.A,problem.C]:
                ix=matrix.indices();rowmin=torch.full((matrix.shape[0],),len(labels),device=device,dtype=torch.long)
                rowmax=torch.zeros_like(rowmin)
                rowmin.scatter_reduce_(0,ix[0],labels[ix[1]],reduce='amin',include_self=True)
                rowmax.scatter_reduce_(0,ix[0],labels[ix[1]],reduce='amax',include_self=True)
                used=rowmin<len(labels)
                self.assertTrue(bool((rowmin[used]==rowmax[used]).all()))

    def test_mask_rejection_includes_competitor_that_exposes_stationary_point(self):
        from estimator.temporal_sparse_nonlinear import outside_mask,evaluate_nonlinear_objective_and_visibility_gpu
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        for device in self.devices():
            rr=rig(device);ff=frame(device);ff.mask[:,3,3]=False
            before=torch.tensor([[0.,0.,2.,0.,0.,0.],[0.,0.,3.,0.,0.,0.]],device=device,dtype=torch.float64)
            after=before.clone();after[0,0]=.6
            owners=first_hit_pixel_owner(before[:,:3],rr,ff,dense=False)
            graph=SimpleNamespace(N=2,rig=rr,warps={},state_and_blocks_gpu=lambda k:(after,None))
            ok,reason,_=evaluate_nonlinear_objective_and_visibility_gpu(graph,{0:ff},None,None,
                {0:before},{0:owners},{0:outside_mask(owners,ff,2)})
            self.assertFalse(ok);self.assertEqual(reason,'NEW_FOREGROUND_VIOLATION')
            self.assertEqual(graph.last_rejection['points'].cpu().tolist(),[0,1])

    def test_sparse_no_measurement_joint_marginal_is_exact_prediction(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA is mandatory for sparse covariance acceptance')
        from estimator.temporal_sparse_solve import SparseJointFactorization
        g=fixture_graph('cuda',False);g.anchor.zero_();g.pixels.clean.copy_(g.pixels.mean)
        g.warps.clear();source,blocks=g.source_state_and_blocks_gpu()
        F,Q=g.process[1][1:];g.states[1]=source@F.T
        source_cov=blocks[0][1]@g.anchor_covariance@blocks[0][1].transpose(1,2)
        pixel_cov=g.pixels.covariance[g.pixels.lookup(g.source_keys)]
        source_cov+=blocks[1][1]@pixel_cov@blocks[1][1].transpose(1,2)
        with SparseJointFactorization(g.assemble_sparse_linearized_problem_gpu()) as factor:
            step,_=factor.solve_constrained_sparse_joint_gpu()
            covariance=factor.extract_joint_selected_marginals_gpu(g.state_query_gpu(1))
        torch.testing.assert_close(step,torch.zeros_like(step),atol=2e-12,rtol=0)
        torch.testing.assert_close(covariance,F@source_cov@F.T+Q,atol=2e-12,rtol=2e-11)

    def test_actual_GPU_motion_refreshes_interpolation_cells_without_GT_input(self):
        if not torch.cuda.is_available():raise RuntimeError('CUDA is mandatory for sparse motion acceptance')
        from estimator.temporal_reference import ReferenceTable
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        from estimator.temporal_rgbd_terms import build_temporal_rows
        from estimator.temporal_sparse_nonlinear import solve_nonlinear_sparse_iekf_gpu
        for displacement in [.1,-.1]:
            rr=rig('cuda');f0=frame('cuda');f1=frame('cuda');f1.k=1
            f0.mask[1]=False;f1.mask[1]=False;f0.depth[:]=2.;f1.depth[:]=2.
            f1.rgb[0,:,:,0]-=.02*2*displacement;f1.rgb[0,:,:,1]-=.03*2*displacement
            noise=SensorNoise(torch.ones(3,device='cuda',dtype=torch.float64)*1e-8,1e-8,0.)
            camera=torch.tensor([0],device='cuda');pixel=torch.tensor([27],device='cuda')
            values=raw_values(f0,camera,pixel);xyz=rr.backproject(camera,torch.tensor([[3.,3.]],device='cuda'),values[:,3])
            graph=TemporalSoAGraph(rr,3,1,camera,pixel,xyz,values,sensor_covariance(values,noise),
                torch.eye(2,device='cuda',dtype=torch.float64)/12,torch.eye(3,device='cuda',dtype=torch.float64),0)
            graph.layout();mean=graph.source_state_and_blocks_gpu()[0]
            refs=ReferenceTable(2,1,'cuda',torch.float64);refs.initialize_from_source_catalogue(np.array([[0,0,27]]),0,mean)
            policy={'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.}
            warp=torch.eye(4,device='cuda',dtype=torch.float64)*1e-8
            owners=first_hit_pixel_owner(mean[:,:3],rr,f1)
            rows,_,_=build_temporal_rows(mean,owners,f1,refs,{0:f0},rr,noise,policy,warp,1,3)
            count=len(graph.pixels.keys)
            graph.pack_source_process_warp_factors_gpu(0,1,transition(.04,device='cuda'),process_noise(.04,torch.eye(3,device='cuda',dtype=torch.float64)*.01),
                mean,rows,{0:f0,1:f1},noise,warp)
            initial_keys=len(graph.pixels.keys)
            factor,result=solve_nonlinear_sparse_iekf_gpu(graph,{0:f0,1:f1},noise,policy,10,1e-8)
            try:
                st=graph.state_and_blocks_gpu(1)[0];sr=graph.source_state_and_blocks_gpu()[0]
                recovered=float(st[0,0]-sr[0,0])
                self.assertLess(abs(recovered-displacement),.002)
                self.assertLess(abs(float(st[0,3])-displacement/.04),.08)
                if displacement>0:self.assertGreater(len(graph.pixels.keys),initial_keys)
                SPARSE_WITNESSES.append({'actual_sparse_motion':True,'true_displacement_posthoc':displacement,
                    'estimated_relative_displacement':recovered,'estimated_velocity':float(st[0,3]),'new_pixels':len(graph.pixels.keys)-initial_keys})
            finally:factor.close()


if __name__=='__main__':unittest.main()
