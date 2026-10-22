import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal,assert_allclose
from rgbd_common.provenance import ProvenanceCSR
from rgbd_common.geometry import PointMap
from rgbd_common.fusion import fuse_many
from parallel_fixtures import Scene,configuration
from rgbd_common.fusion import fuse_frame_zero
from rgbd_common.backends.torch_cuda import Backend


class ProvenanceCSRTests(unittest.TestCase):
    def test_ragged_empty_rows_batch_add_replace_take_and_ownership(self):
        rows=[[1,2],[],[3],[4,5,6]];csr=ProvenanceCSR.from_rows(rows)
        csr.add([2,0,2],[11,12,13]);rows[2]+=[11,13];rows[0]+=[12]
        csr.replace({1:[21,22],3:[]});rows[1]=[21,22];rows[3]=[]
        for actual,expected in zip(csr,rows):assert_array_equal(actual,expected)
        selected=csr.take([3,0,2]);self.assertEqual(len(selected),3)
        for actual,index in zip(selected,[3,0,2]):assert_array_equal(actual,rows[index])
        clone=csr.copy();clone.add([0],[100]);assert_array_equal(csr[0],rows[0])
        raw=np.array([1,2]);source=ProvenanceCSR([0,2],raw);raw[0]=99;assert_array_equal(source[0],[1,2])
        with self.assertRaises(ValueError):source.raw_ids[0]=99
    def test_map_roundtrip_preserves_source_order_and_immutable_point_ids(self):
        model,_=fuse_frame_zero(Scene(3,2),configuration(),Backend('cpu'))
        self.assertIsInstance(model.provenance,ProvenanceCSR)
        copied=model.copy();copied.add_sources([1,1],[1000,1001]);self.assertEqual(model.source_owners([1000])[0],-1)
        selected=copied.take([10,1,3]);assert_array_equal(selected.source_owners([1000,1001]),[1,1])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'map.npz';selected.save(path);loaded=PointMap.load(path)
            assert_array_equal(selected.provenance.offsets,loaded.provenance.offsets);assert_array_equal(selected.provenance.raw_ids,loaded.provenance.raw_ids)
            assert_array_equal(selected.point_id,loaded.point_id)
        self.assertEqual(model.provenance.nbytes,8*(model.count+1+len(model.provenance.raw_ids)))
    def test_repeated_raw_factor_is_rejected_before_covariance_changes(self):
        model,_=fuse_frame_zero(Scene(3,1),configuration(),Backend('cpu'));before=model.covariance.copy()
        with self.assertRaisesRegex(ValueError,'Duplicate raw measurement'):
            fuse_many(model,[0,1],model.xyz[:2],model.rgb[:2],model.covariance[:2],[1000,1000],Backend('cpu'))
        assert_allclose(model.covariance,before)
