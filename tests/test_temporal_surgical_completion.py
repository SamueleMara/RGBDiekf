"""CODEX_RGBDiekf_SURGICAL_EXACT_TEMPORAL_RGBD_IEKF_COMPLETION_2026-11-01.md, sections 3, 4, 7 and G1.

Connectivity is algebraic dependence only: every off-point coefficient of the reduced operator must be traced to
a reused raw pixel or a source chart, never to a motion/neighbour graph or a symbolic line-search tie.
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from estimator.temporal_sparse_gpu import TemporalSoAGraph, sparse_coo
from estimator.temporal_sparse_solve import ReducedLinearizedSystem, connected_columns_gpu
from estimator.temporal_rgbd_terms import raw_values, sensor_covariance
from estimator.projective_sensor import SensorNoise
from estimator.point_kalman_cuda import transition, process_noise
from tests.test_temporal_warp import rig, frame
from tests.test_temporal_sparse_gpu import dense_oracle_for
from tools.replay_temporal_sparse_failure import (graph_record, column_nodes, row_node_pairs_actual, row_node_pairs_predicted,
                                                  cpu_union_find, canonical, coupling_provenance, key_integrity)

REPO = Path(__file__).resolve().parents[1]


def two_point_graph(device, pixels, ideal=True):
    """The fixture_graph construction with selectable source/current pixels (8x8 images, 2 cameras)."""
    rr = rig(device); f0 = frame(device); f1 = frame(device); f1.k = 1
    c = torch.tensor([0, 0], device=device); pix = torch.tensor(pixels, device=device)
    uv = torch.stack((pix % 8, pix//8), -1).double()
    noise = SensorNoise(torch.tensor([.01, .015, .02], device=device, dtype=torch.float64), .005, 0., 0., 0., 'sum_w2')
    y = raw_values(f0, c, pix); R = sensor_covariance(y, noise)
    g = TemporalSoAGraph(rr, 4, 2, c, pix, rr.backproject(c, uv, y[:, 3]), y, R, torch.eye(2, device=device, dtype=torch.float64)/12,
                         torch.eye(3, device=device, dtype=torch.float64)*.7, 0)
    g.layout(); ref = g.source_state_and_blocks_gpu()[0]
    initial = ref.clone(); initial[:, 0] += .007; initial[:, 1] -= .003
    points = torch.tensor([0, 1, 0, 1], device=device); cams = torch.tensor([0, 0, 1, 1], device=device)
    current = torch.tensor(pixels*2, device=device); v = torch.stack((current % 8, current//8), -1).double()
    q = v-rr.project_rows(initial[points, :3], cams)[0]+rr.project_rows(ref[points, :3], cams)[0]
    cell = q.floor().long(); xy = cell[:, None, :]+torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=device)
    rows = SimpleNamespace(point=points, camera=cams, pixel=current, reference_frame=torch.zeros(4, device=device, dtype=torch.long),
                           reference_pixels=xy[..., 1]*8+xy[..., 0], cell=cell, y=raw_values(f1, cams, current))
    warp = torch.zeros((4, 4), device=device, dtype=torch.float64) if ideal else torch.diag(torch.tensor([.02, .03, .04, .01], device=device, dtype=torch.float64))
    g.pack_source_process_warp_factors_gpu(0, 1, transition(.04, device=device), process_noise(.04, torch.eye(3, device=device, dtype=torch.float64)*.2),
                                           initial, rows, {0: f0, 1: f1}, noise, warp)
    g.layout()
    return g


def dense_constrained_covariance(graph, query):
    """Independent dense oracle: original-coordinate whitened operator from the original equations, constrained KKT inverse."""
    _, A, _, C, _, _, _ = dense_oracle_for(graph)
    A, C = A.numpy(), C.numpy(); n = A.shape[1]; p = C.shape[0]
    K = np.block([[A.T@A, C.T], [C, np.zeros((p, p))]])
    return query@np.linalg.inv(K)[:n, :n]@query.T


class Connectivity(unittest.TestCase):
    def test_gpu_components_equal_independent_union_find_on_actual_nonzeros(self):
        gen = torch.Generator().manual_seed(11)
        for trial in range(5):
            m, n, p = 40, 30, 4
            A = torch.zeros((m, n), dtype=torch.float64)
            for r in range(m):
                cols = torch.randperm(n, generator=gen)[:2]; A[r, cols] = torch.randn(2, generator=gen, dtype=torch.float64)
            A[A.abs() < .3] = 0.                     # exact zeros must never connect columns
            C = torch.zeros((p, n), dtype=torch.float64); C[torch.arange(p), torch.randperm(n, generator=gen)[:p]] = 1.
            labels, _ = connected_columns_gpu(A.to_sparse().coalesce(), C.to_sparse().coalesce())
            ai, ci = A.nonzero().T.numpy(), C.nonzero().T.numpy()
            oracle = cpu_union_find(n+m+p, [(n+ai[0], ai[1]), (n+m+ci[0], ci[1])])[:n]
            np.testing.assert_array_equal(canonical(labels.numpy()), oracle)

    def test_symbolic_line_search_ties_never_enter_the_information_operator(self):
        from estimator.temporal_sparse_nonlinear import true_variable_block_components_gpu
        g = two_point_graph('cpu', [27, 28]); problem = g.assemble_sparse_linearized_problem_gpu()
        A0, info0 = problem.A.clone(), problem.information
        labels, count = true_variable_block_components_gpu(g, problem)
        self.assertIs(problem.information, info0); self.assertTrue(torch.equal(problem.A.indices(), A0.indices()))
        self.assertTrue(torch.equal(problem.A.values(), A0.values()))
        self.assertEqual(len(labels), problem.scalar_dimension)


class PixelReuseProvenance(unittest.TestCase):
    """G1: two points reuse one raw reference pixel (coupled) versus none (uncoupled)."""
    def provenance(self, pixels):
        g = two_point_graph('cpu', pixels); record = graph_record(g)
        reduced = ReducedLinearizedSystem(g.assemble_sparse_linearized_problem_gpu())
        node = column_nodes(reduced, record)
        return g, record, reduced, row_node_pairs_actual(reduced, node), row_node_pairs_predicted(record)

    def test_every_nonzero_is_traced_to_a_predicted_factor_and_coupling_follows_reuse(self):
        g, record, _, actual, predicted = self.provenance([27, 28])
        np.testing.assert_array_equal(actual, predicted)     # moved fixture state: no exact cancellation here
        shared = coupling_provenance(record)
        self.assertEqual(shared['largest_point_component'], 2); self.assertGreater(shared['keys_shared_as_nonzero_corner_by_2plus_points'], 0)
        g, record, _, actual, predicted = self.provenance([9, 54])
        np.testing.assert_array_equal(actual, predicted)
        self.assertEqual(coupling_provenance(record)['largest_point_component'], 1)

    def test_exact_zero_weight_adds_no_coefficient(self):
        g, record, reduced, actual, predicted = self.provenance([27, 28])
        w = record['warps'][1]
        w['weights'][0, 1:] = 0.                         # pretend corners 1..3 were exactly zero: prediction must now DIFFER
        self.assertFalse(np.array_equal(row_node_pairs_predicted(record), actual))

    def test_key_encoding_injective_and_no_duplicate_sensor_factor(self):
        g, record, reduced, _, _ = self.provenance([27, 28])
        k = key_integrity(record)
        self.assertTrue(k['encoding_injective_roundtrip']); self.assertEqual(k['unique_keys'], k['registered_keys'])
        self.assertEqual(k['frame1_cross_camera_key_collision'], 0); self.assertTrue(k['frame1_current_keys_unique'])
        self.assertEqual(reduced.problem.factor_counts['RAW_PIXEL'], int(g.pixels.active_likelihood.sum()))
        self.assertEqual(reduced.problem.factor_counts['RAW_PIXEL'], dense_oracle_for(g)[0].pixels.total_sensor_factors)

    @unittest.skipUnless(torch.cuda.is_available(), 'actual CUDA required')
    def test_cross_point_covariance_appears_and_vanishes_with_pixel_reuse(self):
        from estimator.temporal_sparse_solve import SparseJointFactorization
        for pixels, coupled in (([27, 28], True), ([9, 54], False)):
            g = two_point_graph('cuda', pixels)
            q = g.state_query_gpu(1); qdense = q.to_dense().cpu().numpy()
            with SparseJointFactorization(g.assemble_sparse_linearized_problem_gpu()) as f:
                P = f.all_point_marginals_selected_inverse_gpu(q, 12, certificate_blocks=4)[0].cpu().numpy()
                colored = f.extract_joint_selected_marginals_gpu(q, 12)[0].cpu().numpy()
                self.assertEqual(tuple(f.profile['inertia']), f.expected_inertia())
            oracle = dense_constrained_covariance(g, qdense)
            np.testing.assert_allclose(P, oracle, atol=1e-12*np.abs(oracle).max(), rtol=0)
            np.testing.assert_allclose(colored, oracle, atol=1e-12*np.abs(oracle).max(), rtol=0)
            cross = np.abs(P[:6, 6:]).max()
            if coupled:self.assertGreater(cross, 1e-9*np.abs(P).max())
            else:self.assertEqual(cross, 0.)                # disjoint true components: exactly zero, not thresholded


@unittest.skipUnless(torch.cuda.is_available(), 'actual CUDA required')
class Certificates(unittest.TestCase):
    def test_cudss_inertia_and_selected_inverse_inertia_refuse_wrong_structure(self):
        from estimator.temporal_cudss import CudaSparseFactor, SparseNumericalFailure
        from estimator.temporal_selected_inverse_gpu import selected_inverse_blocks_gpu
        A = torch.tensor([[1., 2., .2], [.3, -.5, 1.]], dtype=torch.float64)
        K = torch.cat((torch.cat((torch.eye(2, dtype=torch.float64), A), 1), torch.cat((A.T, -torch.eye(3, dtype=torch.float64)), 1)), 0).cuda()
        with CudaSparseFactor(K.to_sparse_csr(), expected_inertia=(2, 3)) as f:self.assertEqual(f.profile['inertia'], [2, 3])
        with self.assertRaisesRegex(SparseNumericalFailure, 'inertia'):CudaSparseFactor(K.to_sparse_csr(), expected_inertia=(3, 2))
        Q = sparse_coo(torch.tensor([[0, 1, 2], [2, 3, 4]]).cuda(), torch.ones(3, dtype=torch.float64).cuda(), (3, 5))
        selected_inverse_blocks_gpu(K.to_sparse_csr(), Q, 3, (2, 3))
        with self.assertRaisesRegex(SparseNumericalFailure, 'inertia'):selected_inverse_blocks_gpu(K.to_sparse_csr(), Q, 3, (3, 2))

    def test_btf_capacity_bound_below_fill_reports_device_info_never_a_result(self):
        """Controlled reproduction of the S2 failure mechanism: BTF/COLAMD fill is unknown at analysis; a
        MAX_LU_NNZ bound below the measured fill must fail with a device INFO, and the symmetric ND factorization
        of the SAME operator (whose analysis knows the fill) must succeed with the exact inertia."""
        from estimator.temporal_cudss import CudaSparseFactor, SparseNumericalFailure
        g = torch.Generator().manual_seed(5); n = 400
        B = torch.zeros((n, n), dtype=torch.float64)
        for i in range(n):B[i, torch.randperm(n, generator=g)[:6]] = torch.randn(6, generator=g, dtype=torch.float64)
        A = B[:n//2]; K = torch.cat((torch.cat((torch.eye(n//2, dtype=torch.float64), A), 1),
                                    torch.cat((A.T, -torch.eye(n, dtype=torch.float64)), 1)), 0).cuda().to_sparse_csr()
        with CudaSparseFactor(K, 'general_btf') as f:fill = f.profile['factor_nnz']
        with self.assertRaises(SparseNumericalFailure) as caught:CudaSparseFactor(K, 'general_btf', max_lu_nnz=fill//4)
        self.assertNotEqual(caught.exception.profile['info'], 0)
        with CudaSparseFactor(K, 'symmetric_nd', expected_inertia=(n//2, n)) as f:self.assertEqual(f.profile['perturbed_pivots'], 0)


@unittest.skipUnless(torch.cuda.is_available(), 'actual CUDA required')
class DoubleDoubleAccuracy(unittest.TestCase):
    def test_strongly_scaled_quasi_definite_selected_inverse_beats_unpivoted_float64(self):
        """Whitened Jacobians up to 1e6 (as in the real scenes): unpivoted float64 LDL loses ~1e-9 here; the
        double-double recurrences must match a refined 80-bit-residual truth to rounding level for every block and
        both orderings. Guards against FMA contraction silently cancelling the error-free transformations."""
        import scipy.linalg as sl
        from estimator.temporal_selected_inverse_gpu import selected_inverse_blocks_gpu
        m, n = 600, 300; g = np.random.default_rng(7); A = np.zeros((m, n))
        for i in range(m):
            c = g.choice(n, 3, replace=False); A[i, c] = g.normal(size=3)*10**g.uniform(0, 6)
        K = np.block([[np.eye(m), A], [A.T, -np.eye(n)]]); blocks = n//6; Q = np.zeros((6*blocks, m+n)); Q[np.arange(6*blocks), m+np.arange(6*blocks)] = 1
        lu = sl.lu_factor(K); X = sl.lu_solve(lu, Q.T).astype(np.longdouble); Kl = K.astype(np.longdouble)
        for _ in range(8):X += sl.lu_solve(lu, (Q.T.astype(np.longdouble)-Kl@X).astype(np.float64)).astype(np.longdouble)
        exact = -(Q@X.astype(np.float64)).reshape(blocks, 6, blocks, 6)[np.arange(blocks), :, np.arange(blocks)]
        sd = np.sqrt(np.einsum('bii->bi', exact))
        import estimator.temporal_selected_inverse_gpu as module
        chunk = module.CHUNK_MAX
        try:
            for module.CHUNK_MAX in (chunk, 8):                 # 8: every long row/column is walked in many chunks
                for first_block in (0, m):
                    P, profile = selected_inverse_blocks_gpu(torch.tensor(K, device='cuda').to_sparse_csr(), torch.tensor(Q, device='cuda').to_sparse(), 6, (m, n), first_block=first_block)
                    error = float((np.abs(P.cpu().numpy()-exact)/(sd[:, :, None]*sd[:, None, :])).max())
                    self.assertLess(error, 1e-14, (module.CHUNK_MAX, first_block, error)); self.assertEqual(profile['LDL_inertia'], [m, n])
        finally:module.CHUNK_MAX = chunk


@unittest.skipUnless(torch.cuda.is_available(), 'actual CUDA required')
class MemorySchedule(unittest.TestCase):
    def test_released_factor_and_host_offload_are_bit_identical(self):
        from estimator.temporal_sparse_solve import SparseJointFactorization, offload_device_tensors
        g = two_point_graph('cuda', [27, 28]); q = g.state_query_gpu(1)
        with SparseJointFactorization(g.assemble_sparse_linearized_problem_gpu()) as f:
            resident = f.all_point_marginals_selected_inverse_gpu(q, 6, certificate_blocks=2, release=False)
            A, T, problem_A = f.A.clone(), f.T.clone(), f.problem.A.clone()
            f.park_caller_state = lambda: offload_device_tensors([g])           # the caller's graph is parked too
            states_before = g.states[1].clone(); keys_before = g.pixels.keys.clone()
            released = f.all_point_marginals_selected_inverse_gpu(q, 6, certificate_blocks=2, release=True)
            self.assertGreater(f.profile['selected_marginals']['caller_tensors_parked'], 0)
            self.assertTrue(g.states[1].is_cuda and torch.equal(g.states[1], states_before) and torch.equal(g.pixels.keys, keys_before))
            self.assertTrue(f.profile['selected_marginals']['cuDSS_factor_released_during_inversion'])
            self.assertGreater(f.profile['selected_marginals']['host_offloaded_bytes'], 0)
            torch.testing.assert_close(released, resident, atol=0, rtol=0)
            for x, y in ((f.A, A), (f.T, T), (f.problem.A, problem_A)):
                self.assertTrue(x.is_cuda); self.assertTrue(torch.equal(x.indices(), y.indices())); self.assertTrue(torch.equal(x.values(), y.values()))
            step, certificate = f.solve_constrained_sparse_joint_gpu()            # re-created factor is usable and certified
            self.assertLessEqual(certificate['equality_absolute'], 2e-11)


class BatchForwarding(unittest.TestCase):
    def test_runner_forwards_factor_method_seed_to_every_child_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('sceneA',):(Path(tmp)/name/'rgb').mkdir(parents=True); (Path(tmp)/name/'metadata').mkdir()
            out = subprocess.run([sys.executable, '-B', str(REPO/'tools/run_temporal_rgbd_critical.py'), '--data-root', tmp, '--output-root', tmp,
                                  '--mode', 'all', '--direction', 'forward', '--frame-limit', '3', '--pcd-subsample-factor', '8',
                                  '--pcd-subsample-method', 'seeded-uniform', '--pcd-subsample-seed', '5', '--temporal-warp-noise', 'ideal', '--dry-run'],
                                 cwd=REPO, capture_output=True, text=True)
            commands = [l for l in out.stdout.splitlines() if 'train.py' in l]
            self.assertEqual(len(commands), 3, out.stdout+out.stderr)
            for c in commands:
                for flag in ('--pcd-subsample-factor 8', '--pcd-subsample-method seeded-uniform', '--pcd-subsample-seed 5'):self.assertIn(flag, c)


class UniformDownsampling(unittest.TestCase):
    def test_seeded_uniform_identity_determinism_seed_dependence_and_original_ids(self):
        from scene.canonical_subsample import subsample, ids_hash
        g = np.random.default_rng(3); N = 1001
        xyz = g.normal(size=(N, 3)); source = np.stack((np.zeros(N, int), g.integers(0, 8, N), g.integers(0, 64*64, N)), 1)
        full, _ = subsample(xyz, source, 1, 'seeded-uniform', 0, widths=[64]*8)
        np.testing.assert_array_equal(full, np.arange(N))
        for factor in (2, 4, 8):
            a, man = subsample(xyz, source, factor, 'seeded-uniform', 7, widths=[64]*8)
            b, _ = subsample(xyz, source, factor, 'seeded-uniform', 7, widths=[64]*8)
            c, _ = subsample(xyz, source, factor, 'seeded-uniform', 8, widths=[64]*8)
            np.testing.assert_array_equal(a, b); self.assertFalse(np.array_equal(a, c))
            self.assertEqual(len(a), math.ceil(N/factor)); self.assertEqual(len(np.unique(a)), len(a))
            self.assertTrue((np.diff(a) > 0).all()); self.assertTrue(set(a.tolist()) <= set(range(N)))
            self.assertEqual(ids_hash(a), ids_hash(b))


if __name__ == '__main__':
    unittest.main()
