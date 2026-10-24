import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from grouped_fixtures import GroupScene
from pointwise_rgbd.group_pose_only import load_pose_noise,initialize,predict
from pointwise_rgbd.phase1_grouped_controller import run_phase1
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_pose import transport
from pointwise_rgbd.group_iekf import correct
from rgbd_common.backends.torch_cuda import Backend


class PoseOnlyTests(unittest.TestCase):
    def test_signed_pose_walk_covariance_and_no_velocity_state(self):
        b=Backend('cpu');noise=load_pose_noise();pose,velocity,p=initialize(2,noise,b)
        self.assertIsNone(velocity)
        for dt in (.01,-.03,.07):
            t,v,c=predict(pose,None,p,dt,noise);assert_array_equal(t,pose);self.assertIsNone(v)
            assert_allclose(c.numpy(),np.repeat((abs(dt)*noise.pose_spectral_density)[None],2,axis=0));self.assertEqual(c.shape,(2,6,6))

    def test_pose_only_static_variable_time_occlusion_and_checkpoint_dimension(self):
        b=Backend('cpu');c=load_config();noise=load_pose_noise();scene=GroupScene(frames=8)
        scene.times=np.array([0,.01,.03,.08,.1,.2,.22,.4]);scene.observation(4,0).mask[:]=False
        with tempfile.TemporaryDirectory() as root:
            p,a=run_phase1(scene,c,noise,b,root,motion_model='pose_only_6d');self.assertNotIn('twists',a)
            self.assertFalse((Path(root)/'twists.npy').exists());self.assertEqual(a['covariance'].shape[-2:],(6,6))
            assert_allclose(a['group_poses'],np.broadcast_to(np.eye(4),a['group_poses'].shape),atol=1e-14)
            self.assertTrue((a['group_measurements'][4]==0).all());self.assertTrue((np.trace(a['covariance'][4],axis1=1,axis2=2)>np.trace(a['covariance'][3],axis1=1,axis2=2)).all())
            snapshots=np.array(a['covariance']);run_phase1(scene,c,noise,b,root,motion_model='pose_only_6d',resume=True);assert_array_equal(a['covariance'],snapshots)
            with self.assertRaisesRegex(ValueError,'different motion model'):run_phase1(scene,c,noise,b,root,resume=True)

    def test_pose_only_translation_and_rotation_preserve_exact_distances(self):
        b=Backend('cpu');c=load_config();noise=load_pose_noise()
        for motion in ('translation','rotation'):
            scene=GroupScene(frames=6,motion=motion)
            with self.subTest(motion=motion),tempfile.TemporaryDirectory() as root:
                p,a=run_phase1(scene,c,noise,b,root,motion_model='pose_only_6d')
                for k in range(6):
                    for g in range(p.count):
                        ids=np.flatnonzero(p.membership==g);assert_allclose(np.linalg.norm(a['xyz'][k,ids]-a['xyz'][k,ids[0]],axis=1),np.linalg.norm(p.xyz[ids]-p.xyz[ids[0]],axis=1),atol=1e-13)
                self.assertTrue((np.linalg.eigvalsh(a['covariance'][1:])>0).all())


if __name__=='__main__':unittest.main()
