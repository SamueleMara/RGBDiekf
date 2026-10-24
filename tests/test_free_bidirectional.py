import tempfile,unittest
from pathlib import Path
import numpy as np
from pointwise_rgbd.config import load_config
from pointwise_rgbd.joint_rgbd import fit,predict,poses,initialize
from pointwise_rgbd.joint_bidirectional import run,compare,unique_pool
from pointwise_rgbd.group_evidence_audit import projection_numpy
from tests.test_free_joint_rgbd import fixture
from rgbd_common.backends.torch_cuda import Backend


class BidirectionalTests(unittest.TestCase):
    def directions(self,kind='PRISMATIC'):
        model,_,_,allp=fixture(kind);p=allp[[0,6,12,26,35,50]];K=len(model['q']);config=load_config();camera=fixture(kind)[1][0][0]['camera'];result=[]
        for anchor in (0,K-1):
            source=predict(model,p,anchor,np.zeros(len(p),int))
            prior={'keys':np.c_[np.full(len(p),anchor),np.arange(len(p))], 'xyz':source,'covariance':np.tile(np.eye(3)*.0001,(len(p),1,1)), 'source_refs':np.c_[np.full(len(p),anchor),np.zeros(len(p),int),np.arange(len(p))]}
            roles=[[],[],[]]
            for role in range(3):
                for k in range(K):
                    if k==anchor:continue
                    x=predict(model,source,k,np.full(len(p),anchor));z=projection_numpy(x,camera)[0]
                    roles[role].append({'k':k,'camera_index':role,'camera':camera,'p':source,'anchor':np.full(len(p),anchor),'pixel':np.arange(len(p))+100*role,'z':z,'R':config.measurement_covariance(z[:,2]),'map_key':prior['keys'],'source_covariance':prior['covariance'],'phase1_residual':np.zeros_like(z)})
            rigid={'poses':poses(model)@np.linalg.inv(poses(model)[anchor])}
            candidates={m['type']:fit(m,[r for group in roles for r in group],config,map_prior=prior,backend=Backend('cpu')) for m in initialize(rigid,source)}
            self.assertEqual(candidates[kind]['status'],'FIT')
            result.append({'selected':candidates[kind],'fits':candidates,'rows':[r for group in roles for r in group],'map_prior':prior})
        return result,config

    def test_unique_raw_pool_refits_with_both_nonzero_endpoint_priors(self):
        (forward,backward),config=self.directions()
        with tempfile.TemporaryDirectory() as tmp:
            result=run(forward,backward,config,tmp,backend=Backend('cpu'))
            self.assertEqual(result['status'],'PRISMATIC');self.assertEqual(result['camera_policy'],'all_available')
            self.assertEqual(len(result['selected']['map_state']['prior']['keys']),12)
            self.assertTrue((Path(tmp)/'canonical_map_and_joint_uncertainty.npz').exists())
            self.assertGreater(result['pool']['duplicate_direction_factors_removed'],0)

    def test_disagreeing_world_axes_are_not_forced_to_agree(self):
        (forward,backward),config=self.directions()
        backward['selected']['model']={**backward['selected']['model'],'axis':np.array([1.,0.,0.])}
        self.assertFalse(compare(forward['selected'],backward['selected'],config)['compatible'])

    def test_revolute_pool_has_finite_gauged_axis_and_chronological_q(self):
        (forward,backward),config=self.directions('REVOLUTE')
        with tempfile.TemporaryDirectory() as tmp:
            result=run(forward,backward,config,tmp,backend=Backend('cpu'))
            self.assertEqual(result['status'],'REVOLUTE');model=result['selected']['model']
            self.assertAlmostEqual(float(model['axis']@model['axis_point']),0.,places=10)
            self.assertEqual(model['q'][0],0.)


if __name__=='__main__':unittest.main()
