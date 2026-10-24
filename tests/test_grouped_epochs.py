import json,tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal
from grouped_fixtures import GroupScene
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_process import load_group_noise
from pointwise_rgbd.group_partition_epochs import run_partition_epochs
from rgbd_common.backends.torch_cuda import Backend


class PartitionEpochTests(unittest.TestCase):
    def test_wrong_boundary_replays_both_whole_trajectories_with_fresh_covariance(self):
        s=GroupScene(frames=10,motion='boundary');c=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);p,a,ledger=run_partition_epochs(s,c,load_group_noise(),b,root,target_occupancy=1000)
            self.assertEqual(p.count,2);self.assertEqual(p.version,1);self.assertEqual(len(ledger),2)
            old=np.load(root/'partition_v000/xyz.npy',mmap_mode='r');self.assertEqual(old.shape,a['xyz'].shape)
            with np.load(root/'partition_v000/partition.npz') as original:assert_array_equal(p.source,original['source'])
            assert_array_equal(a['group_poses'][0],np.repeat(np.eye(4)[None],2,axis=0))
            assert_array_equal(a['covariance'][0,:,:6,:],0)
            for g in range(2):assert_array_equal(a['covariance'][0,g,6:,6:],np.eye(6))
            truth=p.source[:,2]%96>=44
            for g in range(2):self.assertEqual(len(np.unique(truth[p.membership==g])),1)
            self.assertTrue((root/'independent_raw/complete.json').exists())

    def test_static_connected_patches_merge_on_independent_evidence_not_target_count(self):
        s=GroupScene(frames=6);c=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as temporary:
            p,a,ledger=run_partition_epochs(s,c,load_group_noise(),b,temporary,target_occupancy=100)
            self.assertEqual(ledger[0]['groups'],2);self.assertEqual(p.count,1)
            self.assertEqual(ledger[0]['accepted_operation'],'merge')
            self.assertTrue(all(not row['prior_covariance_reused'] for row in ledger))


if __name__=='__main__':unittest.main()
