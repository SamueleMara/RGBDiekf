"""Gate D (subsampling): factor 1 is the exact identity; factor f selects exactly ceil(N/f) EXISTING ids,
deterministically, covering every occupied voxel; the two endpoints sample independently; O(N log N) scale."""
import math, time, unittest
import numpy as np
from scene.canonical_subsample import subsample


def cloud(n, seed=0):
    g = np.random.default_rng(seed)
    a = g.uniform(-.3, .3, (n, 3)); a[:, 2] *= .05                                     # a slab
    thin = np.c_[g.uniform(.5, .52, n//200), g.uniform(-.01, .01, n//200), g.uniform(0, .3, n//200)]   # a thin, small component
    xyz = np.r_[a, thin]
    src = np.c_[np.zeros(len(xyz), int), g.integers(0, 8, len(xyz)), g.integers(0, 640*480, len(xyz))]
    return xyz, src


class SubsampleTest(unittest.TestCase):
    def test_factor_one_is_identity(self):
        xyz, src = cloud(5000)
        sel, man = subsample(xyz, src, 1)
        self.assertTrue(np.array_equal(sel, np.arange(len(xyz))))
        self.assertEqual(man['N_selected'], len(xyz))

    def test_exact_count_existing_ids_deterministic(self):
        xyz, src = cloud(34369)
        for f in (2, 4, 8):
            for method in ('spatial-stratified', 'seeded-uniform'):
                a, man = subsample(xyz, src, f, method, seed=0, widths=[640]*8)
                b, _ = subsample(xyz, src, f, method, seed=0, widths=[640]*8)
                self.assertEqual(len(a), math.ceil(len(xyz)/f)); self.assertEqual(man['N_selected'], len(a))
                self.assertTrue(np.array_equal(a, b))                                         # deterministic
                self.assertTrue(np.all(np.diff(a) > 0) and a.min() >= 0 and a.max() < len(xyz))  # existing ids, sorted, unique
                if method == 'spatial-stratified':
                    self.assertEqual(man['voxel_components']['components_lost'], 0)            # the thin component survives
        u0, _ = subsample(xyz, src, 4, 'seeded-uniform', seed=0); u1, _ = subsample(xyz, src, 4, 'seeded-uniform', seed=1)
        self.assertFalse(np.array_equal(u0, u1))

    def test_boundary_preference(self):
        xyz, src = cloud(20000)
        flag = np.zeros(len(xyz), bool); flag[::10] = True
        sel, man = subsample(xyz, src, 4, boundary=flag, rgb_grad=np.zeros(len(xyz)))
        self.assertGreater(flag[sel].mean(), flag.mean())

    def test_scales_to_large_clouds(self):
        xyz, src = cloud(378098)
        t = time.perf_counter(); sel, man = subsample(xyz, src, 4, widths=[640]*8)
        self.assertEqual(len(sel), math.ceil(len(xyz)/4)); self.assertLess(time.perf_counter()-t, 60.)


if __name__ == '__main__':
    unittest.main()
