"""Typed, device-resident temporal factors; one GPU batch per factor type/frame.

No per-point Python state, registry dictionary, callable factor or dense component
is used here. Frame dictionaries contain at most the active temporal window.
"""
from dataclasses import dataclass, field
import torch
from estimator.temporal_rgbd_terms import raw_values, sensor_covariance, temporal_prediction


def sparse_coo(indices, values, shape):
    nonzero = values != 0
    return torch.sparse_coo_tensor(indices[:, nonzero], values[nonzero], shape,
                                  device=values.device, dtype=values.dtype, check_invariants=True).coalesce()


class UniquePixelTableGPU:
    def __init__(self, frames, cameras, pixels, device):
        if min(frames, cameras, pixels) <= 0 or frames * cameras * pixels > torch.iinfo(torch.int64).max:
            raise ValueError('Raw pixel int64 encoding overflow or invalid dimensions')
        self.frames, self.cameras, self.pixels = frames, cameras, pixels
        self.device = torch.device(device)
        self.keys = torch.empty(0, dtype=torch.long, device=device)
        self.mean = torch.empty((0, 4), dtype=torch.float64, device=device)
        self.clean = self.mean.clone()
        self.covariance = torch.empty((0, 4, 4), dtype=torch.float64, device=device)
        self.active_likelihood = torch.empty(0,dtype=torch.bool,device=device)
        self.retired_keys = self.keys.clone()
        self.retired_frames = self.keys.clone()

    def encode(self, camera, frame, pixel):
        camera, frame, pixel = torch.broadcast_tensors(camera.long(), frame.long(), pixel.long())
        valid = (camera >= 0) & (camera < self.cameras) & (frame >= 0) & (frame < self.frames) & (pixel >= 0) & (pixel < self.pixels)
        if not bool(valid.all()):
            raise ValueError('Raw pixel key outside injective encoding domain')
        return (frame * self.cameras + camera) * self.pixels + pixel

    def lookup(self, keys):
        if not keys.numel():return torch.empty_like(keys)
        index = torch.searchsorted(self.keys, keys)
        if not len(self.keys) or bool((index >= len(self.keys)).any()) or not torch.equal(self.keys[index], keys):
            raise ValueError('Unregistered raw sensor key')
        return index

    def build_or_update_unique_pixel_table_gpu(self, keys, values, covariance):
        keys, values, covariance = keys.flatten(), values.reshape(-1, 4).double(), covariance.reshape(-1, 4, 4).double()
        if len(keys) != len(values) or len(keys) != len(covariance):
            raise ValueError('Pixel values/covariance count mismatch')
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(covariance).all()) or bool((values[:, 3] <= 0).any()):
            raise ValueError('Nonfinite or invalid raw sensor observation')
        if bool(torch.isin(keys, self.retired_keys).any()) or bool(torch.isin(keys // (self.cameras * self.pixels), self.retired_frames).any()):
            raise ValueError('EXPIRED_PIXEL_REINTRODUCED')
        all_keys = torch.cat((self.keys, keys))
        all_mean = torch.cat((self.mean, values))
        all_cov = torch.cat((self.covariance, covariance))
        order = torch.argsort(all_keys, stable=True)
        sorted_keys = all_keys[order]
        first = torch.ones(len(order), device=self.device, dtype=torch.bool)
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        chosen = order[first]
        unique = sorted_keys[first]
        positions = torch.searchsorted(unique, all_keys)
        mean, cov = all_mean[chosen], all_cov[chosen]
        if not torch.equal(mean[positions], all_mean) or not torch.equal(cov[positions], all_cov):
            raise ValueError('Inconsistent duplicate raw sensor likelihood')
        torch.linalg.cholesky(cov)  # one batched SPD check; no per-pixel synchronization
        clean = mean.clone()
        old_positions=torch.searchsorted(unique,self.keys)
        clean[old_positions] = self.clean
        active=torch.ones(len(unique),device=self.device,dtype=torch.bool)
        active[old_positions]=self.active_likelihood
        self.keys, self.mean, self.covariance, self.clean = unique, mean, cov, clean
        self.active_likelihood=active
        return torch.searchsorted(unique, keys)


@dataclass
class SparseLinearizedProblem:
    A: torch.Tensor
    residual: torch.Tensor
    C: torch.Tensor
    constraint: torch.Tensor
    scalar_dimension: int
    factor_counts: dict
    pivot_groups: list = field(default_factory=list)
    information: torch.Tensor | None = None
    information_gradient: torch.Tensor | None = None
    constant_cost: float = 0.
    prior_transform: torch.Tensor | None = None
    prior_identity_rows: torch.Tensor | None = None
    prior_identity_columns: torch.Tensor | None = None

    def apply_J(self, vector):
        return torch.sparse.mm(self.A, vector[:, None]).flatten()

    def apply_JT(self, vector):
        return torch.sparse.mm(self.A.T, vector[:, None]).flatten()


class SparseBlockRows:
    def __init__(self, device):
        self.device, self.offset = device, 0
        self.indices, self.values, self.residual = [], [], []

    def append(self, residual, blocks, covariance=None):
        count, rows = residual.shape
        L = torch.linalg.cholesky(covariance) if covariance is not None else None
        if L is not None:
            residual = torch.linalg.solve_triangular(L, residual[..., None], upper=False).squeeze(-1)
        starts = self.offset + torch.arange(count, device=self.device) * rows
        local_row = torch.arange(rows, device=self.device)
        for columns, block in blocks:  # a fixed number of typed blocks, never a factor loop
            if L is not None:
                block = torch.linalg.solve_triangular(L, block, upper=False)
            ri = (starts[:, None, None] + local_row[None, :, None]).expand_as(block)
            ci = columns[:, None, :].expand_as(block)
            self.indices.append(torch.stack((ri.reshape(-1), ci.reshape(-1))))
            self.values.append(block.reshape(-1))
        self.residual.append(residual.reshape(-1))
        self.offset += count * rows

    def finish(self, dimension):
        indices = torch.cat(self.indices, 1) if self.indices else torch.empty((2, 0), dtype=torch.long, device=self.device)
        values = torch.cat(self.values) if self.values else torch.empty(0, dtype=torch.float64, device=self.device)
        residual = torch.cat(self.residual) if self.residual else torch.empty(0, dtype=torch.float64, device=self.device)
        return sparse_coo(indices, values, (self.offset, dimension)), residual


class TemporalSoAGraph:
    def __init__(self, rig, frames_count, points, source_camera, source_pixel, source_world,
                 source_values, source_covariance, pixel_covariance, velocity_covariance, anchor_frame):
        self.rig, self.N, self.device = rig, points, rig.device
        self.anchor_frame = int(anchor_frame)
        self.anchor_active=True
        self.source_camera = source_camera.long()
        self.source_pixel = source_pixel.long()
        self.base = source_world.double().clone()
        self.source_depth = source_values[:, 3].double().clone()
        self.source_uv = torch.stack((source_pixel % max(rig.width), source_pixel // max(rig.width)), -1).double()
        if len(set(rig.width)) != 1 or len(set(rig.height)) != 1:
            raise ValueError('The current image stack requires equal camera resolutions')
        self.pixels = UniquePixelTableGPU(frames_count, rig.count, max(rig.width) * max(rig.height), self.device)
        self.source_keys = self.pixels.encode(source_camera, torch.full_like(source_camera, anchor_frame), source_pixel)
        self.pixels.build_or_update_unique_pixel_table_gpu(self.source_keys, source_values, source_covariance)
        self.anchor = torch.zeros((points, 5), device=self.device, dtype=torch.float64)
        self.anchor_covariance = torch.zeros((5, 5), device=self.device, dtype=torch.float64)
        self.anchor_covariance[:2, :2] = pixel_covariance
        self.anchor_covariance[2:, 2:] = velocity_covariance
        self.states, self.process, self.warps = {}, {}, {}
        self.fej=None
        self.fej_scalar_ids=None
        self.marginalization_log=[]

    def layout(self):
        offset = 5 * self.N if self.anchor_active else 0
        self.state_offsets = {}
        for frame in self.states:
            self.state_offsets[frame] = offset; offset += 6 * self.N
        self.pixel_offset = offset
        self.scalar_dimension = offset + 4 * len(self.pixels.keys)
        return self.scalar_dimension

    def pixel_columns(self, keys):
        index = self.pixels.lookup(keys)
        return self.pixel_offset + 4 * index[..., None] + torch.arange(4, device=self.device)

    def source_state_and_blocks_gpu(self, point=None):
        if not self.anchor_active:raise ValueError('Expired source state is no longer in the lag')
        if point is None:
            point = torch.arange(self.N, device=self.device)
        a = self.anchor[point]
        c = self.source_camera[point]
        ell = self.pixels.clean[self.pixels.lookup(self.source_keys[point])]
        z, D0 = ell[:, 3], self.source_depth[point]
        fx, fy, cx, cy = self.rig.K[c].unbind(-1)
        pixel = self.source_uv[point]
        ray0 = torch.stack(((pixel[:, 0]-cx)/fx, (pixel[:, 1]-cy)/fy, torch.ones_like(z)), -1)
        shift = torch.stack((a[:, 0]/fx, a[:, 1]/fy, torch.zeros_like(z)), -1)
        x = self.base[point] + torch.einsum('mji,mj->mi', self.rig.R[c], (z-D0)[:, None]*ray0+z[:, None]*shift)
        state = torch.cat((x, a[:, 2:]), -1)
        ja = torch.zeros((len(point), 6, 5), device=self.device, dtype=torch.float64)
        ja[:, :3, :2] = self.rig.R[c].transpose(1, 2)[:, :, :2] * torch.stack((z/fx, z/fy), -1)[:, None, :]
        ja[:, 3:, 2:] = torch.eye(3, device=self.device, dtype=torch.float64)
        jl = torch.zeros((len(point), 6, 4), device=self.device, dtype=torch.float64)
        jl[:, :3, 3] = torch.einsum('mji,mj->mi', self.rig.R[c], ray0+shift)
        columns = point[:, None]*5 + torch.arange(5, device=self.device)
        return state, [(columns, ja), (self.pixel_columns(self.source_keys[point]), jl)]

    def state_and_blocks_gpu(self, frame, point=None):
        if point is None:
            point = torch.arange(self.N, device=self.device)
        if frame == self.anchor_frame:
            return self.source_state_and_blocks_gpu(point)
        columns = self.state_offsets[frame] + 6*point[:, None] + torch.arange(6, device=self.device)
        return self.states[frame][point], [(columns, torch.eye(6, device=self.device, dtype=torch.float64).expand(len(point), -1, -1))]

    def pack_source_process_warp_factors_gpu(self, previous, current, F, Q, initial, rows, frames, noise, warp_covariance):
        if current in self.states or current == self.anchor_frame:
            raise ValueError('Duplicate processing frame')
        self.states[current] = initial.double().clone()
        self.process[current] = (previous, F.double(), Q.double())
        current_keys = self.pixels.encode(rows.camera, torch.full_like(rows.camera, current), rows.pixel)
        ref_keys = self.pixels.encode(rows.camera[:, None], rows.reference_frame[:, None], rows.reference_pixels)
        keys = [current_keys]; values = [rows.y.double()]
        for ref in frames:  # bounded by lag, no per-point or per-factor CPU loop
            selected = rows.reference_frame == ref
            if not bool(selected.any()):
                continue
            cc = rows.camera[selected, None].expand(-1, 4)
            keys.append(ref_keys[selected].reshape(-1))
            values.append(raw_values(frames[ref], cc, rows.reference_pixels[selected]).reshape(-1, 4).double())
        raw = torch.cat(values)
        self.pixels.build_or_update_unique_pixel_table_gpu(torch.cat(keys), raw, sensor_covariance(raw, noise))
        self.warps[current] = {'point': rows.point, 'camera': rows.camera, 'reference_frame': rows.reference_frame,
            'reference_keys': ref_keys, 'current_keys': current_keys, 'cell': rows.cell,
            'uv': torch.stack((rows.pixel % frames[current].W, rows.pixel // frames[current].W), -1).double(),
            'covariance': warp_covariance.double()}

    def evaluate_factor_residuals_and_blocks_gpu(self):
        dimension = self.layout()
        soft, hard = SparseBlockRows(self.device), SparseBlockRows(self.device)
        points = torch.arange(self.N, device=self.device)
        acols = points[:, None]*5 + torch.arange(5, device=self.device)
        if self.anchor_active:
            soft.append(self.anchor, [(acols, torch.eye(5, device=self.device, dtype=torch.float64).expand(self.N, -1, -1))], self.anchor_covariance)
        active=self.pixels.active_likelihood
        pcols = self.pixel_columns(self.pixels.keys[active])
        soft.append(self.pixels.clean[active]-self.pixels.mean[active], [(pcols, torch.eye(4, device=self.device, dtype=torch.float64).expand(len(pcols), -1, -1))], self.pixels.covariance[active])
        counts = {'SOURCE_PRIOR': self.N if self.anchor_active else 0, 'RAW_PIXEL': len(pcols), 'PROCESS': 0, 'TEMPORAL_WARP': 0}
        pivots = []
        for current, (previous, F, Q) in self.process.items():
            now, nb = self.state_and_blocks_gpu(current)
            old, ob = self.state_and_blocks_gpu(previous)
            soft.append(now-old@F.T, nb+[(col, -F@block) for col, block in ob], Q)
            counts['PROCESS'] += self.N
        for current, w in self.warps.items():
            for ref in [self.anchor_frame, *self.states]:
                ix = torch.where(w['reference_frame'] == ref)[0]
                if not len(ix):
                    continue
                p, cam = w['point'][ix], w['camera'][ix]
                now, nb = self.state_and_blocks_gpu(current, p)
                old, ob = self.state_and_blocks_gpu(ref, p)
                refkeys = w['reference_keys'][ix]
                values = self.pixels.clean[self.pixels.lookup(refkeys)]
                h, ht, hr, q, weights, _, _ = temporal_prediction(self.rig, now, old, cam, w['uv'][ix], values, w['cell'][ix])
                clean = self.pixels.clean[self.pixels.lookup(w['current_keys'][ix])]
                blocks = [(col, -ht@block) for col, block in nb] + [(col, -hr@block) for col, block in ob]
                eye = torch.eye(4, device=self.device, dtype=torch.float64).expand(len(ix), -1, -1)
                blocks.append((self.pixel_columns(w['current_keys'][ix]), eye))
                for corner in range(4):
                    blocks.append((self.pixel_columns(refkeys[:, corner]), -weights[:, corner, None, None]*eye))
                ideal = not bool(torch.count_nonzero(w['covariance']))
                if ideal:
                    pivots.append((torch.arange(hard.offset,hard.offset+4*len(ix),device=self.device),
                                   self.pixel_columns(w['current_keys'][ix]).reshape(-1)))
                (hard if ideal else soft).append(clean-h, blocks, None if ideal else w['covariance'])
                counts['TEMPORAL_WARP'] += len(ix)
        A, residual = soft.finish(dimension)
        C, constraint = hard.finish(dimension)
        problem=SparseLinearizedProblem(A, residual, C, constraint, dimension, counts, pivots)
        if self.fej is not None:
            ids=self.scalar_ids();order=torch.argsort(ids);where=torch.searchsorted(ids[order],self.fej_scalar_ids)
            if bool((where>=len(ids)).any()) or not torch.equal(ids[order][where],self.fej_scalar_ids):
                raise ValueError('FEJ boundary variable disappeared without Schur elimination')
            mapping=order[where]
            delta=self.flat_values()[mapping]-self.fej.linearization
            hi=self.fej.information.indices()
            problem.information=sparse_coo(mapping[hi],self.fej.information.values(),(dimension,dimension))
            gradient=torch.sparse.mm(self.fej.information,delta[:,None]).flatten()+self.fej.gradient
            problem.information_gradient=torch.zeros(dimension,device=self.device,dtype=torch.float64)
            problem.information_gradient[mapping]=gradient
            problem.constant_cost=self.fej.constant+float(delta@(gradient+self.fej.gradient)/2)
            ci=self.fej.constraints.indices()
            appended=sparse_coo(torch.stack((ci[0]+C.shape[0],mapping[ci[1]])),self.fej.constraints.values(),
                                  (C.shape[0]+self.fej.constraints.shape[0],dimension))
            original=sparse_coo(C.indices(),C.values(),appended.shape)
            problem.C=(original+appended).coalesce()
            problem.constraint=torch.cat((constraint,self.fej.constraint_residual+torch.sparse.mm(self.fej.constraints,delta[:,None]).flatten()))
            counts.update(FEJ_PRIOR=1,FEJ_CONSTRAINT=self.fej.constraints.shape[0])
        problem.prior_transform,problem.prior_identity_rows,problem.prior_identity_columns=self.prior_innovation_coordinates_gpu(problem)
        return problem

    assemble_sparse_linearized_problem_gpu = evaluate_factor_residuals_and_blocks_gpu

    def prior_innovation_coordinates_gpu(self,problem):
        """Exact Gaussian source/process/sensor change of coordinates.

        World state arrays remain the physical [x,v] variables. This triangular
        right preconditioner uses the supplied prior square roots, including
        source-depth coupling. It introduces NO independent position likelihood.
        The corresponding prior rows are algebraic identities, not near-zero
        cancellations thresholded out of a sparse matrix product.
        """
        from estimator.temporal_sparse_solve import sparse_rows
        n=problem.scalar_dimension
        pivots=torch.cat([p for _,p in problem.pivot_groups]) if problem.pivot_groups else torch.empty(0,device=self.device,dtype=torch.long)
        free=torch.ones(n,device=self.device,dtype=torch.bool);free[pivots]=False
        original=torch.where(free)[0]
        mapping=torch.full((n,),-1,device=self.device,dtype=torch.long);mapping[original]=torch.arange(len(original),device=self.device)
        G=sparse_coo(torch.empty((2,0),device=self.device,dtype=torch.long),torch.empty(0,device=self.device,dtype=torch.float64),(n,len(original)))
        identity_rows=[];identity_columns=[]
        def blocks(rows,columns,value):
            ri=rows[:,:,None].expand_as(value);ci=columns[:,None,:].expand_as(value)
            return sparse_coo(torch.stack((ri.reshape(-1),ci.reshape(-1))),value.reshape(-1),G.shape)
        points=torch.arange(self.N,device=self.device)
        row_offset=0
        if self.anchor_active:
            rows=points[:,None]*5+torch.arange(5,device=self.device)
            columns=mapping[rows]
            G=(G+blocks(rows,columns,torch.linalg.cholesky(self.anchor_covariance).expand(self.N,-1,-1))).coalesce()
            identity_rows.append(rows.reshape(-1));identity_columns.append(columns.reshape(-1));row_offset+=5*self.N
        pixels=self.pixel_columns(self.pixels.keys)
        independent=(mapping[pixels]>=0).all(1)
        pixelrows=pixels[independent]
        G=(G+blocks(pixelrows,mapping[pixelrows],torch.linalg.cholesky(self.pixels.covariance[independent]))).coalesce()
        active_indices=torch.where(self.pixels.active_likelihood)[0]
        active_independent=independent[active_indices]
        factor_rows=row_offset+4*torch.arange(len(active_indices),device=self.device)[:,None]+torch.arange(4,device=self.device)
        identity_rows.append(factor_rows[active_independent].reshape(-1))
        identity_columns.append(mapping[pixels[active_indices[active_independent]]].reshape(-1))
        row_offset+=4*len(active_indices)
        for current in self.states:
            rows=self.state_offsets[current]+points[:,None]*6+torch.arange(6,device=self.device)
            if current not in self.process:
                eye=torch.eye(6,device=self.device,dtype=torch.float64).expand(self.N,-1,-1)
                G=(G+blocks(rows,mapping[rows],eye)).coalesce()
                continue
            previous,F,Q=self.process[current]
            previous_query=self.state_query_gpu(previous)
            inherited=torch.sparse.mm(previous_query,G).coalesce()
            local=points[:,None]*6+torch.arange(6,device=self.device)
            ri=local[:,:,None].expand(self.N,6,6);ci=local[:,None,:].expand(self.N,6,6)
            blockF=sparse_coo(torch.stack((ri.reshape(-1),ci.reshape(-1))),F.expand(self.N,-1,-1).reshape(-1),(6*self.N,6*self.N))
            inherited=torch.sparse.mm(blockF,inherited).coalesce()
            ix=inherited.indices()
            G=(G+sparse_coo(torch.stack((ix[0]+self.state_offsets[current],ix[1])),inherited.values(),G.shape)).coalesce()
            G=(G+blocks(rows,mapping[rows],torch.linalg.cholesky(Q).expand(self.N,-1,-1))).coalesce()
            identity_rows.append(row_offset+local.reshape(-1));identity_columns.append(mapping[rows].reshape(-1))
            row_offset+=6*self.N
        return sparse_rows(G,original),torch.cat(identity_rows),torch.cat(identity_columns)

    def scalar_ids(self):
        self.layout()
        base=5*self.N
        ids=[torch.arange(base,device=self.device)] if self.anchor_active else []
        for k in self.states:
            ids.append(base+k*6*self.N+torch.arange(6*self.N,device=self.device))
        pixel_base=base+self.pixels.frames*6*self.N
        if pixel_base+self.pixels.frames*self.pixels.cameras*self.pixels.pixels*4>torch.iinfo(torch.int64).max:
            raise OverflowError('Semantic variable coordinate encoding overflow')
        ids.append((pixel_base+4*self.pixels.keys[:,None]+torch.arange(4,device=self.device)).reshape(-1))
        return torch.cat(ids)

    def flat_values(self):
        parts=[self.anchor.reshape(-1)] if self.anchor_active else []
        return torch.cat([*parts,*(s.reshape(-1) for s in self.states.values()),self.pixels.clean.reshape(-1)])

    def set_flat_values(self,value):
        offset=0
        if self.anchor_active:
            self.anchor.copy_(value[:5*self.N].reshape(self.N,5));offset=5*self.N
        for k in self.states:
            self.states[k].copy_(value[offset:offset+6*self.N].reshape(self.N,6));offset+=6*self.N
        self.pixels.clean.copy_(value[offset:].reshape(-1,4))

    def state_query_gpu(self,frame,points=None):
        self.layout()
        state,blocks=self.state_and_blocks_gpu(frame,points)
        builder=SparseBlockRows(self.device)
        builder.append(torch.zeros_like(state),blocks)
        return builder.finish(self.scalar_dimension)[0]

    def marginalize_lag_boundary_sparse_gpu(self,expired_frames,workspace_bytes):
        from estimator.temporal_sparse_marginalization import marginalize_lag_boundary_sparse_gpu
        expired_frames=set(expired_frames)
        if not expired_frames:return
        original=self.warps;affected={};unaffected={}
        for k,w in original.items():
            hit=torch.zeros(len(w['point']),device=self.device,dtype=torch.bool)
            if k in expired_frames:hit[:]=True
            for old in expired_frames:hit|=w['reference_frame']==old
            def subset(mask):
                return {name:value if name=='covariance' else value[mask] for name,value in w.items()}
            if bool(hit.any()):affected[k]=subset(hit)
            if bool((~hit).any()):unaffected[k]=subset(~hit)
        self.warps=affected
        try:
            problem=self.assemble_sparse_linearized_problem_gpu()
            ids=self.scalar_ids();values=self.flat_values()
            expired=torch.zeros(len(values),device=self.device,dtype=torch.bool)
            if self.anchor_active and self.anchor_frame in expired_frames:expired[:5*self.N]=True
            for k,offset in self.state_offsets.items():
                if k in expired_frames:expired[offset:offset+6*self.N]=True
            frames=self.pixels.keys//(self.pixels.cameras*self.pixels.pixels)
            removed=torch.zeros_like(frames,dtype=torch.bool)
            for k in expired_frames:removed|=frames==k
            expired[self.pixel_columns(self.pixels.keys[removed]).reshape(-1)]=True
            boundary=marginalize_lag_boundary_sparse_gpu(problem,torch.where(expired)[0],values,workspace_bytes)
        except BaseException:
            self.warps=original
            raise
        self.fej=boundary;self.fej_scalar_ids=ids[boundary.retained_columns]
        self.marginalization_log.append(boundary.profile)
        self.warps=unaffected;self.process={}
        if self.anchor_frame in expired_frames:self.anchor_active=False
        self.states={k:s for k,s in self.states.items() if k not in expired_frames}
        keep=~removed
        for name in ['keys','mean','clean','covariance']:
            setattr(self.pixels,name,getattr(self.pixels,name)[keep])
        self.pixels.active_likelihood=torch.zeros(int(keep.sum()),device=self.device,dtype=torch.bool)
        self.pixels.retired_frames=torch.unique(torch.cat((self.pixels.retired_frames,torch.tensor(sorted(expired_frames),device=self.device,dtype=torch.long))))
        self.layout()
