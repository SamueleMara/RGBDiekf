import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal

from rgbd_common.array_store import ArrayStore
from rgbd_common.config import load_config
from rgbd_common.model_contract import require_model_selection,UnspecifiedModelSelection


class UpdatedRequestTests(unittest.TestCase):
    def test_compact_cycle_exports_no_images_copies_or_checkpoint_generations(self):
        from parallel_fixtures import DrawerScene,configuration
        from rgbd_common.backends.torch_cuda import Backend
        from rgbd_common.outer_loop import run_outer
        from rgbd_common.outputs import export_result
        from iekf.filter import IEKF
        scene=DrawerScene();config=configuration();config.numerical['inner_iterations']=4;backend=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            result=run_outer(IEKF(config,backend),scene,config,backend,root,max_outer=1)
            report=export_result(result,scene,config,backend,root,'IEKF')
            files=[p for p in Path(root).rglob('*') if p.is_file()]
            self.assertEqual(report['output_mode'],'compact')
            self.assertTrue(report['complete_real_cycle'])
            self.assertEqual(report['source_foreground_violations'],0)
            self.assertFalse(any(p.suffix in ('.png','.html','.zip') for p in files))
            self.assertFalse((Path(root)/'raw_cache').exists())
            self.assertEqual(len(list(Path(root).glob('checkpoint*.npz'))),1)
            self.assertLess(len(files),35)

    def test_approved_noise_does_not_invent_a_group_selection_prior(self):
        config=load_config()
        self.assertIsNone(config.provenance['structural_priors']['initial_value'])
        with self.assertRaisesRegex(UnspecifiedModelSelection,'UNSPECIFIED_MODEL_SELECTION'):
            require_model_selection(config)

    def test_working_arrays_replace_rows_in_one_file_without_copying_input(self):
        with tempfile.TemporaryDirectory() as root:
            store=ArrayStore(Path(root)/'working.sqlite')
            for k in range(300):store.write(str(k),pixel=np.arange(k,dtype=np.int64))
            store.write('12',pixel=np.array([99]))
            assert_array_equal(store.read('12')['pixel'],[99])
            self.assertEqual(len(list(Path(root).iterdir())),1)
            store.close()
            reopened=ArrayStore(Path(root)/'working.sqlite')
            assert_array_equal(reopened.read('299')['pixel'],np.arange(299))
            reopened.close()

    def test_both_inner_filters_recover_measured_nonzero_six_dof_motion(self):
        import torch
        from rgbd_common.backends.torch_cuda import Backend
        from rgbd_common.geometry import PointMap
        from rgbd_common.projection_cuda import Camera
        from rgbd_common.dataloader import RGBDFrame
        from rgbd_common.lie import State,exp
        from rgbd_common.observation_model import Matches,covariance
        from rgbd_common.measurement_problem import MeasurementProblem
        from iekf.filter import IEKF
        from ukf.filter import UKF
        config=load_config();backend=Backend('cpu')
        rng=np.random.default_rng(42)
        xyz=rng.uniform([-.4,-.3,1.4],[.4,.3,2.2],(120,3));rgb=rng.uniform(.1,.9,(len(xyz),3))
        camera=Camera('measured',320,240,np.array([250.,250.,159.5,119.5]),np.eye(4),0,np.inf)
        truth=exp(np.array([.025,-.018,.012,.014,-.012,.017]))
        world=xyz@truth[:3,:3].T+truth[:3,3]
        uv=np.floor(world[:,:2]/world[:,2,None]*250+[159.5,119.5]+.5).astype(int)
        pixel=uv[:,1]*camera.width+uv[:,0];_,unique=np.unique(pixel,return_index=True)
        xyz=xyz[unique];rgb=rgb[unique];world=world[unique];uv=uv[unique];pixel=pixel[unique]
        depth=np.zeros((240,320));image=np.zeros((240,320,3))
        depth.ravel()[pixel]=world[:,2];image.reshape(-1,3)[pixel]=rgb
        frame=RGBDFrame(image,depth,np.ones_like(depth,dtype=bool),camera,1,0)
        model=PointMap();model.append(xyz,rgb,np.zeros(len(xyz),int),np.tile(np.eye(6),(len(xyz),1,1)),np.arange(len(xyz)))
        observed=backend.tensor(np.c_[uv,world[:,2],rgb])
        match=Matches(backend.tensor(np.arange(len(xyz)),torch.long),backend.tensor(pixel,torch.long),observed,covariance(config,backend.tensor(world[:,2]),backend),camera,1,0,backend.tensor(np.zeros(len(xyz))),backend.tensor(np.zeros(len(xyz))),None,(frame,config))
        problem=MeasurementProblem(model,[match],backend)
        initial=State(np.eye(4)[None],np.zeros((1,6)))
        for kind in (IEKF,UKF):
            with self.subTest(filter=kind.name):
                estimator=kind(config,backend);pred,pminus,_=estimator.predict(initial,config.p0(),.04)
                state,posterior,report=estimator.correct(pred,pminus,problem)
                error=State(truth[None],np.zeros((1,6))).minus(state)[:6]
                self.assertLess(np.linalg.norm(error[:3]),.008)
                self.assertLess(np.linalg.norm(error[3:]),.008)
                self.assertGreater(np.linalg.norm(state.poses[0,:3,3]),.02)
                self.assertGreaterEqual(np.linalg.eigvalsh(posterior).min(),-1e-12)
                self.assertLess(np.trace(posterior),np.trace(pminus))
                if kind is UKF:self.assertEqual(report['active_factors'],len(xyz))
