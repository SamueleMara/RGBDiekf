import tempfile,unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from rgbd_common.frame_cache import CachedDataset,RGBSamples,MaskSamples
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.config import load_config
from rgbd_common.observation_model import unassigned_log_probability
from rgbd_common.dataloader import load_dataset
from test_data_loading import real_scene


class FrameCacheTests(unittest.TestCase):
    def test_changed_mask_at_the_same_path_cannot_use_an_old_raw_cache(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            scene=Path(root)/'scene';scene.mkdir();real_scene(scene)
            first=CachedDataset(load_dataset(scene),Path(root)/'cache');first.observation(0,0)
            Image.fromarray(np.array([[0,255],[255,255]],np.uint8)).save(scene/'masks/cam_000/frame_000000.png')
            changed=CachedDataset(load_dataset(scene),Path(root)/'cache')
            with self.assertRaisesRegex(ValueError,'STALE_RAW_CACHE'):changed.observation(0,0)
            with self.assertRaisesRegex(ValueError,'STALE_RAW_CACHE'):changed.background_total(0,0,load_config(),Backend('cpu'))
    def test_sparse_normalization_and_packed_mask_are_exact(self):
        raw=np.arange(45,dtype=np.uint8).reshape(3,5,3);rgb=RGBSamples(raw)
        assert_allclose(np.asarray(rgb),raw.astype(float)/255,rtol=0,atol=0)
        pick=np.array([0,3,7,14]);assert_allclose(rgb.reshape(-1,3)[pick],raw.reshape(-1,3)[pick]/255,rtol=0,atol=0)
        assert_array_equal(np.uint8(rgb*255),raw)
        mask=(np.arange(15).reshape(3,5)%3)!=0;bits=MaskSamples(np.packbits(mask.ravel()),mask.shape)
        assert_array_equal(np.asarray(bits),mask);assert_array_equal(bits.ravel()[pick],mask.ravel()[pick])
        assert_array_equal(bits[np.array([-1,0]),np.array([-1,0])],mask[np.array([-1,0]),np.array([-1,0])])
        assert_array_equal(bits[:,2],mask[:,2]);self.assertEqual(bits.ravel()[-1],mask.ravel()[-1])
    def test_real_format_cache_and_background_term_preserve_original_values(self):
        with tempfile.TemporaryDirectory() as root:
            scene=Path(root)/'scene';scene.mkdir();real_scene(scene)
            data=load_dataset(scene);cached=CachedDataset(data,Path(root)/'cache');c=load_config();b=Backend('cpu')
            for k in range(len(data.times)):
                for ci in range(len(data.cameras)):
                    original=data.observation(k,ci);frame=cached.observation(k,ci)
                    assert_allclose(np.asarray(frame.rgb),original.rgb,rtol=0,atol=0);assert_allclose(frame.depth,original.depth)
                    assert_array_equal(np.asarray(frame.mask),original.mask);assert_array_equal(frame.valid,original.valid)
                    expected=unassigned_log_probability(original,c,b)+np.log1p(-c.match_prior)
                    expected=float(expected[b.tensor(original.valid.ravel(),__import__('torch').bool)].sum())
                    self.assertEqual(cached.background_total(k,ci,c,b),expected)
            old=cached.background_total(0,0,c,b);c.match_prior=.4
            new=cached.background_total(0,0,c,b)
            assert_allclose(new-old,int(data.observation(0,0).valid.sum())*np.log(.6/.5),atol=1e-12)
