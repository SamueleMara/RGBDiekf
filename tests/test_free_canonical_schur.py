import unittest
import numpy as np
import torch
from pointwise_rgbd import canonical_schur as cs
from pointwise_rgbd.config import load_config
from tests.test_free_joint_rgbd import fixture
from rgbd_common.backends.torch_cuda import Backend


class CanonicalSchurTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'Host CUDA required')
    def test_cuda_shared_motion_map_factors_match_cpu(self):
        model,rows,prior=self.data();results=[]
        for device in ('cpu','cuda'):
            backend=Backend(device);result=cs.fit(model,rows,prior,load_config(),backend)
            self.assertEqual(result['status'],'FIT');results.append(result)
        for key in ('P','J','U','Sinv','mu'):
            np.testing.assert_allclose(results[0]['map_state'][key],results[1]['map_state'][key],rtol=2e-8,atol=1e-9)

    def data(self):
        model,roles,_,p=fixture('PRISMATIC');chosen=np.array([0,6,12,26,35,50]);prior={'keys':np.c_[np.zeros(len(chosen),int),chosen],'xyz':p[chosen].copy(),'covariance':np.tile(np.eye(3)*.0001,(len(chosen),1,1))}
        rows=[]
        for old in roles[0]:
            row={**old,'p':old['p'][chosen],'anchor':old['anchor'][chosen],'z':old['z'][chosen],'R':old['R'][chosen],'pixel':old['pixel'][chosen],'map_key':prior['keys']};rows.append(row)
        return model,rows,prior

    def test_schur_solution_and_pose_point_correlation_match_dense(self):
        from pointwise_rgbd.group_evidence_audit import projection_numpy
        model,rows,prior=self.data();backend=Backend('cpu')
        # A second, independent camera at the conditioned source endpoint
        # informs canonical coordinates, never an extra pose gauge variable.
        endpoint={**rows[0],'k':0,'camera_index':1,'z':projection_numpy(prior['xyz'],rows[0]['camera'])[0]}
        rows=[endpoint]+rows;data=cs.prepare(rows,prior,backend)
        terms=cs.normal(model,backend.tensor(prior['xyz']),data,prior,backend);r=cs.eliminate(terms)
        A=terms['A'].numpy();C=terms['C'].numpy();S=terms['S'].numpy();N=len(S);D=len(A)
        full=np.zeros((D+3*N,D+3*N));full[:D,:D]=A
        for i in range(N):
            sl=slice(D+3*i,D+3*i+3);full[:D,sl]=C[i];full[sl,:D]=C[i].T;full[sl,sl]=S[i]
        covariance=np.linalg.inv(full);np.testing.assert_allclose(r['P'],covariance[:D,:D],rtol=1e-8,atol=1e-10)
        U=r['U'].numpy();Sinv=r['Sinv'].numpy();P=r['P'].numpy()
        self.assertGreater(abs(U).max(),0)
        for i in range(N):
            np.testing.assert_allclose(-P@U[i].T,covariance[:D,D+3*i:D+3*i+3],rtol=1e-8,atol=1e-10)
            np.testing.assert_allclose(Sinv[i]+U[i]@P@U[i].T,covariance[D+3*i:D+3*i+3,D+3*i:D+3*i+3],rtol=1e-8,atol=1e-10)

    def test_nonzero_endpoint_uncertainty_and_static_alternative_load(self):
        model,rows,prior=self.data();backend=Backend('cpu');cfg=load_config()
        result=cs.fit(model,rows,prior,cfg,backend)
        self.assertEqual(result['status'],'FIT');self.assertTrue(cs.score(result,rows,backend)['p_value']>.01)
        static=cs.fit({'type':'STATIC'},rows,prior,cfg,backend)
        self.assertEqual(static['status'],'FIT');self.assertEqual(static['covariance'].shape,(0,0))
        self.assertLess(cs.score(result,rows,backend)['nll'],cs.score(static,rows,backend)['nll'])

    def test_shared_predictive_quadratic_and_logdet_match_dense(self):
        from pointwise_rgbd.projection import project
        model,rows,prior=self.data();backend=Backend('cpu')
        result=cs.fit(model,rows,prior,load_config(),backend)
        state=result['map_state'];D=len(state['P']);N=len(prior['xyz']);information=np.zeros((D+3*N,D+3*N))
        information[:D,:D]=state['A']
        for i in range(N):
            sl=slice(D+3*i,D+3*i+3);information[:D,sl]=state['C'][i];information[sl,:D]=state['C'][i].T;information[sl,sl]=state['S'][i]
        held=[{**r,'z':r['z']+np.array([.05,-.08,.00002])} for r in rows]
        matrices=[];errors=[];variances=[]
        for row in cs.prepare(held,prior,backend):
            x,B,R,columns=cs.world(model,row,backend.tensor(state['mu']),backend);h,j=project(x,row['camera']);j=j[:,:,:3]
            H=(j@B).numpy();Hp=(j@R).numpy();full=np.zeros((len(x),3,D+3*N));full[:,:,columns]=H
            for n,mi in enumerate(row['mi'].numpy()):full[n,:,D+3*mi:D+3*mi+3]=Hp[n]
            matrices.append(full.reshape(-1,D+3*N));errors.append((row['zz']-h).numpy().ravel());variances.extend(row['R'])
            predicted,covariance=cs.prediction(result,row,backend)
            whole=full@np.linalg.inv(information)@full.transpose(0,2,1)
            np.testing.assert_allclose(covariance,whole,rtol=1e-8,atol=1e-10)
        from scipy.linalg import block_diag
        H=np.concatenate(matrices);e=np.concatenate(errors);covariance=block_diag(*variances)+H@np.linalg.solve(information,H.T)
        chi=float(e@np.linalg.solve(covariance,e));nll=.5*(chi+np.linalg.slogdet(covariance)[1]+len(e)*np.log(2*np.pi))
        actual=cs.score(result,held,backend)
        self.assertAlmostEqual(actual['chi2'],chi,places=8);self.assertAlmostEqual(actual['nll'],nll,places=7)

    def test_shared_rigid_map_uses_nonzero_source_covariance(self):
        model,rows,prior=self.data();from pointwise_rgbd.joint_rgbd import poses
        rigid={'type':'RIGID','poses':poses(model),'anchor':0};backend=Backend('cpu')
        result=cs.fit(rigid,rows,prior,load_config(),backend)
        self.assertEqual(result['status'],'FIT');self.assertEqual(result['covariance'].shape,(36,36))
        self.assertTrue(np.linalg.eigvalsh(result['covariance']).min()>0)
        self.assertGreater(abs(result['map_state']['U']).max(),0)
        self.assertGreater(cs.score(result,rows,backend)['p_value'],.01)


if __name__=='__main__':unittest.main()
