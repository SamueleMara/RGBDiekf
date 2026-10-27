"""Phase-1 covariance models behind one named interface (contract section 5).

All three carry the per-point marginal blocks used as the block prior information Lambda- of the
reference solve, and differ only in what they know about cross-point covariance.

'restricted_joseph_structured' (validated, NOT production): the EXACT error covariance of the implemented estimator under the
    linearized model. With e the 6N error of [x, v], the restricted update moves e only inside range(G), and
    F G = G F~ (F~ = [[I, h I], [0, I]] on [pose, velocity] group coordinates, h the signed step). Hence
        e = d + G xi,   Cov(d) = blockdiag(D),  Cov(xi) = C [12S,12S],  Cov(d, xi) = X [6N,12S] (stored [6,N,12S])
    is closed under both steps:
        predict:  D <- F D F^T + Q,   C <- F~ C F~^T,   X <- F X F~^T
        update:   d unchanged;  xi <- T xi - B^-1 (U d + G^T H^T W n),   T = I - B^-1 K0,  K0 = G^T C~ G
                  C <- T C T^T + B^-1 (U D U^T + V) B^-1 - T (U X)^T B^-1 - B^-1 (U X) T^T
                  X <- X - (X K0 + D U^T) B^-1
    with U = G^T C~ (C~ = H^T W H on positions), V = G^T H^T W R W H G and the restricted gain's
    B = G^T (Lambda-_block + H^T W H) G at the final linearization (the reference solve's prior information).
    Marginals P_ii = D_i + G_i C_gg G_i^T + X_i,g G_i^T + G_i X_i,g^T. The 6N x 12S matrix X is held in
    CPU float64 and updated in point chunks; K0 and U are sparse.
    Measured on blade_103706: the restricted gain corrects only the 12 group coordinates, so every point's deviation
    from its group's rigid motion is never observed; under the independent-point prior (velocity sigma 1 m/s) its
    error grows like t sigma_v (median position variance 3.7e-4 -> 3.8e-2 m^2 over 10 frames, matching t^2 sigma_v^2),
    the group coordinates of small/near-collinear groups absorb huge cancelling values, and B lost positive
    definiteness at frame 11 (posterior blocks already indefinite at frame 10).
'restricted_joseph_marginals' (validated, NOT production): the same update applied to a block-diagonal (assumed-density)
    prior each frame (C = X = 0 at every frame start). Exact for one frame; 19-39% below the exact covariance after 5
    frames on the dense fixtures. On blade_103706 it DIVERGES exponentially (max position variance 1.3e-2 m^2 at frame 1,
    6.5e2 at frame 4, 1e18 at frame 14; Cholesky failure at frame 20): measured intra-group deviations that the restricted
    gain cannot correct individually are absorbed by the group corrections, whose variance inflates every member, which
    feeds the next frame's group corrections.
'rigeuler_block_information' (production for the complete runs): pinned approximate_posterior_information,
    Lambda+_i = Lambda-_i + blockdiag(F_i, 0); bounded, optimistic (20-1000x below the exact one-frame covariance on the
    dense fixtures). The only bounded method of the three at blade scale.
"""
import os
import time
import numpy as np
import torch
from .process6 import transition, process_covariance
from .restricted_posterior import restricted_system, restricted_joseph_marginals, rigeuler_block_information, _point_pairs_sum, project_psd

METHODS = ('restricted_joseph_structured', 'restricted_joseph_marginals', 'rigeuler_block_information')


def _fq(dt, reverse, config, like):
    return like.new_tensor(transition(dt, reverse)), like.new_tensor(process_covariance(dt, config.qc, config.qx, reverse))


class BlockCovariance:
    """Marginal blocks only ('restricted_joseph_marginals' or 'rigeuler_block_information')."""
    def __init__(self, P, method):
        if method not in METHODS[1:]: raise ValueError(method)
        self.P, self.method = P, method

    def marginals(self):
        return self.P

    def predict(self, dt, reverse, config):
        F, Q = _fq(dt, reverse, config, self.P)
        P = F@self.P@F.T+Q
        self.P = (P+P.transpose(1, 2))/2

    def update(self, lin, obs, terms, basis):
        if self.method == METHODS[1]: self.P, rep = restricted_joseph_marginals(self.P, lin, obs, terms, basis)
        else: self.P, rep = rigeuler_block_information(self.P, lin, obs, terms)
        return rep


class StructuredCovariance:
    """Exact d + G xi representation. X is stored component-major [6, N, 12S] (CPU float64) so that its position
    rows X[:3] form one contiguous (3N x 12S) matrix for the sparse product U X."""
    def __init__(self, D, basis, chunk=1024, threads=None, gpu_share=.5):
        self.D, self.basis, self.chunk, self.gpu_share = D, basis, chunk, gpu_share
        self.threads = threads or os.cpu_count() or 1
        N, S = D.shape[0], basis.count
        self.C = torch.zeros((12*S, 12*S), dtype=torch.float64)
        self.X = torch.zeros((6, N, 12*S), dtype=torch.float64)
        self.member = basis.member.cpu()
        self.method = METHODS[0]
        self._marg = None

    @property
    def nbytes(self):
        return self.X.numel()*8+self.C.numel()*8

    def marginals(self):
        if self._marg is None:
            S = self.basis.count
            Gi = self.basis.matrix().cpu()                                                    # [N,6,12]
            cols = self.member[:, None]*12+torch.arange(12)[None]                             # [N,12]
            Cgg = self.C.reshape(S, 12, S, 12)[torch.arange(S), :, torch.arange(S)]           # [S,12,12]
            Xo = torch.gather(self.X, 2, cols[None].expand(6, -1, -1)).permute(1, 0, 2)       # [N,6,12]
            cross = Xo@Gi.transpose(1, 2)
            P = self.D.cpu()+Gi@Cgg[self.member]@Gi.transpose(1, 2)+cross+cross.transpose(1, 2)
            self._marg = ((P+P.transpose(1, 2))/2).to(self.D.device)
        return self._marg

    def predict(self, dt, reverse, config):
        F, Q = _fq(dt, reverse, config, self.D)
        D = F@self.D@F.T+Q
        self.D = (D+D.transpose(1, 2))/2
        h = -dt if reverse else dt
        S = self.basis.count
        C = self.C.view(S, 12, S, 12)
        C[:, :6] += h*C[:, 6:]; C[..., :6] += h*C[..., 6:]                                    # F~ C F~^T
        prev = torch.get_num_threads(); torch.set_num_threads(self.threads)
        try:
            for a in range(0, self.X.shape[1], self.chunk):
                X = self.X[:, a:a+self.chunk].view(6, -1, S, 12)
                X[:3].add_(X[3:], alpha=h); X[..., :6].add_(X[..., 6:], alpha=h)             # F X F~^T
        finally:
            torch.set_num_threads(prev)
        self._marg = None

    @torch.no_grad()
    def update(self, lin, obs, terms, basis):
        clock = time.monotonic()
        S, N = basis.count, self.X.shape[1]
        Pm = self.marginals()
        sys = restricted_system(Pm, lin, obs, terms, basis, self.threads)
        tm = dict(sys['seconds']); tick = time.monotonic()
        UDU = _point_pairs_sum(sys, S, self.D).cpu()
        tm['UDU'] = time.monotonic()-tick; tick = time.monotonic()
        cpu = lambda t: t.detach().to('cpu', torch.float64)
        g, b, Uw = cpu(sys['g']).long(), cpu(sys['b']).long(), cpu(sys['Uw'])
        K0, V, Binv, pose = cpu(sys['GCG']), cpu(sys['V']), sys['Binv'], sys['pose']
        prev = torch.get_num_threads(); torch.set_num_threads(self.threads)
        try:
            Bp = Binv[:, pose].contiguous()                                                   # [12S,6S] = B^-1 E_pose
            BpT = Bp.T.contiguous()                                                           # [6S,12S]
            # U X: CSR (6S x 3N) times the contiguous position rows X[:3] (3N x 12S)
            K = len(b)
            rows = (6*g[:, None, None]+torch.arange(6)[None, :, None]).expand(K, 6, 3).reshape(-1)
            cols = (torch.arange(3)[None, None, :]*N+b[:, None, None]).expand(K, 6, 3).reshape(-1)
            Us = torch.sparse_coo_tensor(torch.stack([rows, cols]), Uw.reshape(-1), (6*S, 3*N)).coalesce().to_sparse_csr()
            UX = torch.sparse.mm(Us, self.X[:3].reshape(3*N, 12*S))                           # [6S,12S]
            tm['UX'] = time.monotonic()-tick; tick = time.monotonic()
            # T = I - Bp K0 E_pose^T (a rank-6S update of I): expand instead of forming dense 12S^3 products
            BK = Bp@K0                                                                        # [12S,6S]
            Cp = self.C[pose]                                                                 # [6S,12S]
            BKC = BK@Cp                                                                       # Bp K0 C_pose,:
            TCT = self.C-BKC-BKC.T+BK@Cp[:, pose]@BK.T
            Z = UX.T@BpT                                                                      # (U X)^T B^-1 E^T [12S,12S]
            M = Z-BK@Z[pose]
            C = TCT+Bp@(UDU+V)@BpT-M-M.T
            self.C = (C+C.T)/2
            tm['C'] = time.monotonic()-tick; tick = time.monotonic()
            # X <- X - (X_pose K0 + D U^T) B^-1 E_pose^T, in point chunks; K0 sparse (symmetric)
            K0s = K0.to_sparse_csr()
            DU = torch.einsum('ers,eqs->erq', self.D.cpu()[b, :, :3], Uw)                      # [K,6,6]: D_b[:, :3] U_e^T
            order = torch.argsort(b, stable=True); bo = b[order]
            starts = list(range(0, N, self.chunk))
            bounds = torch.searchsorted(bo, torch.tensor(starts+[N]))

            def chunk_rhs(ci):
                """D U^T rows of chunk ci, dense [6,c,6S] (deterministic CPU accumulation)."""
                a = starts[ci]; c = min(self.chunk, N-a)
                out = torch.zeros((6, c, S, 6), dtype=torch.float64)
                e = order[bounds[ci]:bounds[ci+1]]
                for r in range(6):
                    out[r].index_put_((b[e]-a, g[e]), DU[e][:, r, :], accumulate=True)
                return out.view(6*c, 6*S)

            def work(ids, device):
                Kd, Bd = (K0s, BpT) if device == 'cpu' else (K0s.to(device), BpT.to(device))
                for ci in ids:
                    a = starts[ci]; Xc = self.X[:, a:a+self.chunk]; c = Xc.shape[1]
                    Xd = Xc.to(device) if device != 'cpu' else Xc
                    Wd = torch.sparse.mm(Kd, Xd[..., pose].reshape(6*c, 6*S).T.contiguous()).T+chunk_rhs(ci).to(device)
                    delta = (Wd@Bd).view(6, c, 12*S)
                    Xc.sub_(delta if device == 'cpu' else delta.cpu())
            # Fixed, deterministic split of point chunks between the GPU (float64) and the CPU, run concurrently.
            ids = list(range(len(starts)))
            n_gpu = int(round(self.gpu_share*len(ids))) if torch.cuda.is_available() else 0
            if n_gpu:
                import threading
                err = []
                def run_gpu():
                    try: work(ids[len(ids)-n_gpu:], 'cuda')
                    except BaseException as ex: err.append(ex)
                th = threading.Thread(target=run_gpu); th.start()
                work(ids[:len(ids)-n_gpu], 'cpu'); th.join()
                if err: raise err[0]
            else:
                work(ids, 'cpu')
            tm['X'] = time.monotonic()-tick; tick = time.monotonic()
        finally:
            torch.set_num_threads(prev)
        self._marg = None
        P, floor = project_psd(self.marginals())
        self._marg = P
        tm['marginals'] = time.monotonic()-tick
        measured = torch.zeros(N, dtype=torch.bool); measured[b] = True
        return {'method': self.method, 'group_point_blocks': int(len(b)), 'restricted_dimension': 12*S,
                'null_directions_of_G': sys['null_directions'], 'psd_repair': floor, 'measured_points': int(measured.sum()),
                'cross_covariance_bytes': self.nbytes, 'seconds': time.monotonic()-clock, 'section_seconds': tm}


def make_covariance(method, P0, basis):
    """Covariance model at the anchor: blockdiag(P0), no cross-point covariance."""
    if method == METHODS[0]: return StructuredCovariance(P0.clone(), basis)
    return BlockCovariance(P0.clone(), method)
