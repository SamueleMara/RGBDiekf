"""Sparse equality substitution, augmented GPU KKT solve and joint block queries.

The augmented system avoids forming normal equations. All covariance queries use
the same sparse factorization. Sparse component labels only compress independent
RHS columns: they never remove information edges or approximate covariance.
"""
import time
import torch
from estimator.temporal_sparse_gpu import sparse_coo
from estimator.temporal_cudss import CudaSparseFactor, SparseNumericalFailure


def sparse_rows(matrix, rows):
    matrix = matrix.coalesce()
    mapping = torch.full((matrix.shape[0],), -1, device=rows.device, dtype=torch.long)
    mapping[rows] = torch.arange(len(rows), device=rows.device)
    ix, value = matrix.indices(), matrix.values()
    take = mapping[ix[0]] >= 0
    return sparse_coo(torch.stack((mapping[ix[0, take]], ix[1, take])), value[take], (len(rows), matrix.shape[1]))


def eliminate_current_pixel_constraints_gpu(problem):
    n, device = problem.scalar_dimension, problem.A.device
    groups = problem.pivot_groups
    pivots = torch.cat([columns for rows, columns in groups]) if groups else torch.empty(0, device=device, dtype=torch.long)
    if len(torch.unique(pivots)) != len(pivots):
        raise SparseNumericalFailure('CONFLICTING_PIXEL_PIVOTS: a clean pixel cannot be substituted twice')
    free = torch.ones(n, device=device, dtype=torch.bool); free[pivots] = False
    free_columns = torch.where(free)[0]
    T = sparse_coo(torch.stack((free_columns, torch.arange(len(free_columns), device=device))),
                   torch.ones(len(free_columns), device=device, dtype=torch.float64), (n, len(free_columns)))
    offset = torch.zeros(n, device=device, dtype=torch.float64)
    processed = free.clone()
    eliminated_rows = torch.zeros(problem.C.shape[0], device=device, dtype=torch.bool)
    for rows, columns in groups:  # bounded processing-time groups, never one loop per factor
        C = sparse_rows(problem.C, rows)
        ix, values = C.indices(), C.values()
        pivot = ix[1] == columns[ix[0]]
        counts = torch.bincount(ix[0, pivot], minlength=len(rows))
        if not bool((counts == 1).all()) or not bool((values[pivot] == 1).all()):
            raise SparseNumericalFailure('Expected one exact +I clean-current-pixel pivot per constraint row')
        rest = ~pivot
        if not bool(processed[ix[1, rest]].all()):
            raise SparseNumericalFailure('PIVOT_DEPENDENCY_ORDER: cyclic or same-time clean pixel constraints')
        B = sparse_coo(ix[:, rest], values[rest], C.shape)
        update = -torch.sparse.mm(B, T).coalesce()
        ui = update.indices()
        inserted = sparse_coo(torch.stack((columns[ui[0]], ui[1])), update.values(), T.shape)
        T = (T + inserted).coalesce()
        offset[columns] = -problem.constraint[rows] - torch.sparse.mm(B, offset[:, None]).flatten()
        eliminated_rows[rows] = True
        processed[columns] = True
    remaining = sparse_rows(problem.C, torch.where(~eliminated_rows)[0])
    A = torch.sparse.mm(problem.A, T).coalesce()
    r = problem.residual + torch.sparse.mm(problem.A, offset[:, None]).flatten()
    C = torch.sparse.mm(remaining, T).coalesce()
    c = problem.constraint[~eliminated_rows] + torch.sparse.mm(remaining, offset[:, None]).flatten()
    return A, r, C, c, T, offset


def assemble_augmented_kkt_gpu(A, C, information=None):
    m, n = A.shape; p = C.shape[0]; device = A.device
    a, av = A.indices(), A.values()
    c, cv = C.indices(), C.values()
    d = torch.arange(m, device=device)
    index = torch.cat((torch.stack((d, d)), torch.stack((a[0], m+a[1])), torch.stack((m+a[1], a[0])),
                       torch.stack((m+n+c[0], m+c[1])), torch.stack((m+c[1], m+n+c[0]))), 1)
    values = torch.cat((torch.ones(m, device=device, dtype=torch.float64), av, av, cv, cv))
    if information is not None:
        information=information.coalesce()
        index=torch.cat((index,m+information.indices()),1)
        values=torch.cat((values,-information.values()))
    return sparse_coo(index, values, (m+n+p, m+n+p))


def connected_columns_gpu(A, C, information=None, iteration_limit=1024):
    n = A.shape[1]; device = A.device
    indices = torch.cat((A.indices(), C.indices()+torch.tensor([[A.shape[0]], [0]], device=device)), 1)
    if information is not None:
        d=torch.arange(n,device=device)
        indices=torch.cat((indices,information.indices()+torch.tensor([[A.shape[0]+C.shape[0]],[0]],device=device),
                           torch.stack((d+A.shape[0]+C.shape[0],d))),1)
    row, col = indices
    labels = torch.arange(n, device=device)
    for iteration in range(iteration_limit):
        row_min = torch.full((A.shape[0]+C.shape[0]+(n if information is not None else 0),), n, device=device, dtype=torch.long)
        row_min.scatter_reduce_(0, row, labels[col], reduce='amin', include_self=True)
        new = labels.clone(); new.scatter_reduce_(0, col, row_min[row], reduce='amin', include_self=True)
        if torch.equal(new, labels):
            return labels, iteration+1
        labels = new
    raise SparseNumericalFailure('CONNECTIVITY_ITERATION_LIMIT: no component edges were dropped')


def offload_device_tensors(objects, max_depth=3):
    """Move every CUDA tensor reachable through attributes/dict values of `objects` to host memory, unchanged.
    Returns a restore() that puts each one back where it was. Pure data movement: no value changes."""
    moved, seen = [], set()
    def visit(owner, depth):
        if id(owner) in seen or depth > max_depth:return
        seen.add(id(owner))
        items = owner.items() if isinstance(owner, dict) else (vars(owner).items() if hasattr(owner, '__dict__') and not isinstance(owner, (torch.nn.Module, type)) else ())
        for key, value in list(items):
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    moved.append((owner, key, value.device))
                    if isinstance(owner, dict):owner[key] = value.cpu()
                    else:setattr(owner, key, value.cpu())
            elif isinstance(value, dict) or (hasattr(value, '__dict__') and not callable(value)):
                visit(value, depth+1)
    for obj in objects:visit(obj, 0)
    def restore():
        for owner, key, device in moved:
            if isinstance(owner, dict):owner[key] = owner[key].to(device)
            else:setattr(owner, key, getattr(owner, key).to(device))
    restore.count = len(moved)
    return restore


class ReducedLinearizedSystem:
    """Exact reduced coordinates of one linearized problem, without any factorization.

    Pivot substitution, the innovation chart and unit scaling exactly as solved;
    shared by the production factorization and the read-only S0/S2 diagnostics.
    """
    def __init__(self, problem):
        self.problem = problem
        self.A, self.r, self.C, self.c, self.T, self.offset = eliminate_current_pixel_constraints_gpu(problem)
        if problem.prior_transform is not None:
            G=problem.prior_transform
            self.A=torch.sparse.mm(self.A,G).coalesce()
            self.C=torch.sparse.mm(self.C,G).coalesce()
            self.T=torch.sparse.mm(self.T,G).coalesce()
            known=torch.zeros(self.A.shape[0],device=self.A.device,dtype=torch.bool)
            known[problem.prior_identity_rows]=True
            ix=self.A.indices();take=~known[ix[0]]
            index=torch.cat((ix[:,take],torch.stack((problem.prior_identity_rows,problem.prior_identity_columns))),1)
            values=torch.cat((self.A.values()[take],torch.ones(len(problem.prior_identity_rows),device=self.A.device,dtype=torch.float64)))
            self.A=sparse_coo(index,values,self.A.shape)
        self.information=None
        self.information_gradient=torch.zeros(self.A.shape[1],device=self.A.device,dtype=torch.float64)
        if problem.information is not None:
            HT=torch.sparse.mm(problem.information,self.T).coalesce()
            self.information=torch.sparse.mm(self.T.T,HT).coalesce()
            local_gradient=torch.sparse.mm(problem.information,self.offset[:,None]).flatten()+problem.information_gradient
            self.information_gradient=torch.sparse.mm(self.T.T,local_gradient[:,None]).flatten()
            del HT,local_gradient
        if problem.prior_transform is not None:
            rows=problem.prior_identity_rows;columns=problem.prior_identity_columns
            if len(torch.unique(columns))!=len(columns):raise SparseNumericalFailure('Duplicate innovation prior coordinate')
            diagonal=sparse_coo(torch.stack((columns,columns)),torch.ones(len(columns),device=self.A.device,dtype=torch.float64),
                                (self.A.shape[1],self.A.shape[1]))
            self.information=diagonal if self.information is None else (self.information+diagonal).coalesce()
            self.information_gradient[columns]+=self.r[rows]
            remaining=torch.ones(len(self.r),device=self.A.device,dtype=torch.bool);remaining[rows]=False
            self.A=sparse_rows(self.A,torch.where(remaining)[0]);self.r=self.r[remaining]
            del known,index,values,take,rows,columns,remaining
        # Exact change of units, not an information prior or damping term.
        norm=torch.zeros(self.A.shape[1],device=self.A.device,dtype=torch.float64)
        norm.scatter_add_(0,self.A.indices()[1],self.A.values().square())
        if self.information is not None:
            ix=self.information.indices();diagonal=ix[0]==ix[1]
            norm.scatter_add_(0,ix[0,diagonal],self.information.values()[diagonal])
        if bool((norm<=0).any()):raise SparseNumericalFailure('Unconstrained stochastic coordinates after pivot substitution')
        scaling=norm.rsqrt() if problem.prior_transform is None else torch.ones_like(norm)
        self.A=sparse_coo(self.A.indices(),self.A.values()*scaling[self.A.indices()[1]],self.A.shape)
        self.C=sparse_coo(self.C.indices(),self.C.values()*scaling[self.C.indices()[1]],self.C.shape)
        self.T=sparse_coo(self.T.indices(),self.T.values()*scaling[self.T.indices()[1]],self.T.shape)
        if self.information is not None:
            ix=self.information.indices()
            self.information=sparse_coo(ix,self.information.values()*scaling[ix[0]]*scaling[ix[1]],self.information.shape)
        self.information_gradient*=scaling
        del norm,scaling
        if problem.prior_transform is not None or problem.information is not None:del ix
        K=assemble_augmented_kkt_gpu(self.A,self.C,self.information).to_sparse_csr()
        if K._nnz()<2**31:
            K=torch.sparse_csr_tensor(K.crow_indices().int(),K.col_indices().int(),K.values(),K.shape,device=K.device,dtype=K.dtype)
        self.K=K

    def expected_inertia(self):
        """Sylvester inertia of [[I,A,0],[A^T,-H,C^T],[0,C,0]] for SPD H restricted to null(C), full-row-rank C:
        congruent to diag(I_m, -(H+A^T A), Schur) = (m + p positive, n negative, 0 zero)."""
        m, n = self.A.shape
        return m+self.C.shape[0], n


class SparseJointFactorization(ReducedLinearizedSystem):
    def __init__(self, problem, algorithm='symmetric_nd'):
        if problem.A.device.type != 'cuda':
            raise ValueError('The sparse direct production solver requires CUDA; CPU dense is test-only')
        start = time.perf_counter()
        super().__init__(problem)
        try:self.factor = CudaSparseFactor(self.K,algorithm=algorithm,expected_inertia=self.expected_inertia())
        except SparseNumericalFailure as error:
            error.profile={**getattr(error,'profile',{}),'original_scalar_dimension':problem.scalar_dimension,
                'reduced_scalar_dimension':self.A.shape[1],'stochastic_rows':self.A.shape[0],
                'remaining_constraints':self.C.shape[0],'nnz_A':self.A._nnz(),'nnz_C':self.C._nnz(),
                'factor_counts':problem.factor_counts,'registered_information_edges_dropped':0}
            try:
                labels,iterations=connected_columns_gpu(self.A,self.C,self.information)
                _,counts=torch.unique(labels,return_counts=True)
                error.profile.update(failed_system_true_components=len(counts),
                    failed_system_largest_component_scalars=int(counts.max()),connectivity_iterations=iterations)
            except SparseNumericalFailure as secondary:error.profile['connectivity_diagnostic_failure']=str(secondary)
            raise
        self.profile = {'eliminated_pixel_scalars': problem.scalar_dimension-self.T.shape[1],
                        'free_dimension': self.T.shape[1], 'remaining_constraints': len(self.c),
                        'nnz_A_reduced': self.A._nnz(), 'nnz_C_reduced': self.C._nnz(),
                        'nnz_substitution': self.T._nnz(), 'nnz_KKT': self.K._nnz(),
                        'setup_seconds': time.perf_counter()-start,
                        'source_process_sensor_innovation_coordinates':problem.prior_transform is not None,
                        'exact_prior_residuals_eliminated':len(problem.prior_identity_rows) if problem.prior_transform is not None else 0,
                        **self.factor.profile}

    def solve_constrained_sparse_joint_gpu(self):
        m, n = self.A.shape
        rhs = torch.cat((-self.r, self.information_gradient, -self.c))
        solution = self.factor.solve(rhs).flatten()
        self.step = self.offset + torch.sparse.mm(self.T, solution[m:m+n, None]).flatten()
        equality = torch.sparse.mm(self.problem.C, self.step[:, None]).flatten()+self.problem.constraint
        residual = self.problem.residual+self.problem.apply_J(self.step)
        equality_abs = float(equality.abs().max()) if len(equality) else 0.
        scale = float(self.problem.constraint.abs().max()) if len(equality) else 0.
        if equality_abs > 2e-11*max(1., scale):
            raise SparseNumericalFailure(f'Uncertified ideal equality: absolute residual {equality_abs}')
        objective=float(residual@residual/2)+self.problem.constant_cost
        if self.problem.information is not None:
            objective+=float(self.step@(torch.sparse.mm(self.problem.information,self.step[:,None]).flatten()/2+self.problem.information_gradient))
        self.profile['solve_certificate'] = {**self.factor.profile['last_solve'], 'equality_absolute': equality_abs,
            'objective': objective, 'normal_equations_formed': False,'FEJ_information_present':self.problem.information is not None}
        return self.step, self.profile['solve_certificate']

    def extract_joint_selected_marginals_gpu(self, query, block_width=6, rhs_batch=8):
        """Return Q_i P_joint Q_i^T; a block can include requested cross variables.

        Color RHS by actual connected component and local query-row index. This
        combines independent solves without dropping a coupling or storing N².
        A query row spanning components is split and recombined exactly.
        """
        if query.shape[1] != self.problem.scalar_dimension or query.shape[0] % block_width:
            raise ValueError('Query dimensions must define equal-width output blocks')
        start = time.perf_counter()
        Q = torch.sparse.mm(query, self.T).coalesce()
        labels, rounds = connected_columns_gpu(self.A, self.C,self.information)
        n = self.A.shape[1]; m = self.A.shape[0]; device = Q.device
        if Q.shape[0] * n > torch.iinfo(torch.int64).max:
            raise OverflowError('Selected-query encoding overflow')
        qrow, qcol = Q.indices(); values = Q.values()
        pair, inverse = torch.unique(qrow*n+labels[qcol], sorted=True, return_inverse=True)
        piece_row, component = pair//n, pair%n
        order = torch.argsort(component, stable=True)
        starts = torch.ones(len(order), device=device, dtype=torch.bool)
        starts[1:] = component[order[1:]] != component[order[:-1]]
        positions = torch.arange(len(order), device=device)
        local_rank = positions-torch.cummax(torch.where(starts,positions,torch.zeros_like(positions)),0).values
        colors = torch.empty_like(local_rank); colors[order] = local_rank
        count = int(colors.max())+1 if len(colors) else 0
        pieces = sparse_coo(torch.stack((inverse,qcol)), values, (len(pair), n))
        result = torch.zeros((query.shape[0]//block_width,block_width,block_width), device=device,dtype=torch.float64)
        for begin in range(0,count,rhs_batch):
            end = min(count,begin+rhs_batch)
            take = (colors[inverse]>=begin)&(colors[inverse]<end)
            rhs = sparse_coo(torch.stack((m+qcol[take],colors[inverse[take]]-begin)),values[take],(self.K.shape[0],end-begin)).to_dense()
            solved = self.factor.solve(rhs)[m:m+n]
            projection = -torch.sparse.mm(pieces,solved)
            for j in range(block_width):
                target_query = (piece_row//block_width)*block_width+j
                key = target_query*n+component
                index = torch.searchsorted(pair,key)
                within = index<len(pair)
                safe = index.clamp_max(max(0,len(pair)-1))
                match = within&(pair[safe]==key)
                color = colors[safe]
                take = match&(color>=begin)&(color<end)
                ids = torch.where(take)[0]
                linear = (piece_row[ids]//block_width)*block_width**2+(piece_row[ids]%block_width)*block_width+j
                result.view(-1).index_add_(0,linear,projection[ids,color[ids]-begin])
        asymmetry = float((result-result.transpose(1,2)).abs().max()) if result.numel() else 0.
        _, sizes = torch.unique(labels,return_counts=True)
        self.profile['selected_marginals'] = {'seconds':time.perf_counter()-start,'output_blocks':len(result),
            'query_width':block_width,'rhs_colors':count,'rhs_batch':rhs_batch,'connectivity_iterations':rounds,
            'component_count':len(sizes),'largest_component_scalars':int(sizes.max()) if len(sizes) else 0,
            'symmetry_absolute':asymmetry,'method':'same augmented sparse factorization; exact component-colored RHS'}
        if not bool(torch.isfinite(result).all()):
            raise SparseNumericalFailure('Nonfinite selected joint marginal')
        return (result+result.transpose(1,2))/2

    def all_point_marginals_selected_inverse_gpu(self, query, block_width=6, certificate_blocks=32, seed=0, release=None):
        """Q_i P_joint Q_i^T for EVERY output block from one selected inversion (Takahashi) of the same KKT.

        Same operator, same constraints, no dropped coupling: the inverse entries on the symbolic factor pattern
        (plus zero-valued query cliques) are exact functions of the full joint KKT. Certificate: each sampled
        block is recomputed from refined cuDSS solves of the ORIGINAL operator K; blocks from the largest true
        component are always included. Agreement is measured in correlation units
        |dP_ab|/sqrt(P_aa P_bb) against sqrt(float64 eps), the agreement attainable by two backward-stable
        double-precision computations of the same quantity; the measured value is always reported.
        """
        from estimator.temporal_selected_inverse_gpu import selected_inverse_blocks_gpu, symbolic, required_bytes, stage_entries, OPERATIONAL_HEADROOM_BYTES
        if query.shape[1] != self.problem.scalar_dimension or query.shape[0] % block_width:
            raise ValueError('Query dimensions must define equal-width output blocks')
        start = time.perf_counter()
        Q = torch.sparse.mm(query, self.T).coalesce()
        m, n = self.A.shape; device = Q.device
        qi = Q.indices()
        Qk = sparse_coo(torch.stack((qi[0], qi[1]+m)), Q.values(), (Q.shape[0], self.K.shape[0]))
        blocks = Q.shape[0]//block_width
        # 1) certificate references from refined cuDSS solves of the original operator (before any release)
        labels, _ = connected_columns_gpu(self.A, self.C, self.information)
        unique, counts = torch.unique(labels, return_counts=True)
        largest = unique[torch.argmax(counts)]
        block_of = qi[0]//block_width
        in_largest = torch.unique(block_of[labels[qi[1]] == largest])
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.unique(torch.cat((in_largest[torch.randperm(len(in_largest), generator=generator)[:certificate_blocks//4].to(device)],
                                         torch.randperm(blocks, generator=generator)[:certificate_blocks-certificate_blocks//4].to(device))))
        references = {}
        for part in chosen.split(1):   # one 6-column block per refined solve: bounded device workspace
            rows = (part[:, None]*block_width+torch.arange(block_width, device=device)).flatten()
            dense = sparse_rows(Q, rows).to_dense()
            rhs = torch.zeros((self.K.shape[0], len(rows)), device=device, dtype=torch.float64); rhs[m:m+n] = dense.T
            solved = self.factor.solve(rhs)[m:m+n]
            for j, b in enumerate(part.tolist()):
                sl = slice(j*block_width, (j+1)*block_width)
                exact = -(dense[sl]@solved[:, sl]); references[b] = (exact+exact.T)/2
            del rhs, solved, dense
        del labels
        # 2) pattern-only symbolic analysis, then the measured device workspace decides whether the cuDSS factor
        #    must be released during the inversion (it is re-created from the same K afterwards; same operator)
        symbolic_start = time.perf_counter(); S = symbolic(self.K, Qk, block_width); symbolic_seconds = time.perf_counter()-symbolic_start
        del Q, qi, block_of
        print(f"[selected inverse] KKT n={len(S['perm'])} nnz_L={len(S['ri'])} levels={len(S['lp'])-1} max_row={int(S['sizes'][0::2].max())} "
              f"max_column={int(S['sizes'][1::2].max())} stage={stage_entries(S)} symbolic_seconds={symbolic_seconds:.1f}", flush=True)
        need = required_bytes(len(S['ri']), len(S['perm']), stage_entries(S)); torch.cuda.empty_cache(); free = torch.cuda.mem_get_info(device)[0]
        released = need+OPERATIONAL_HEADROOM_BYTES > free if release is None else bool(release); offloaded = []; restore_caller = None   # release=True: test of the schedule
        if released:
            # Exact memory schedule: the native factor is closed and every device tensor of this system, K and the
            # query included, moves to host memory unchanged. Restored bit-identically below.
            self.factor.close(); offloaded = self._offload(); self.K = self.K.cpu(); Qk = Qk.cpu()
            park = getattr(self, 'park_caller_state', None)   # the caller's own device state (graph, frames, ...)
            restore_caller = park() if park is not None else None
            torch.cuda.empty_cache()
        free_for_inversion = torch.cuda.mem_get_info(device)[0]
        torch_allocated_for_inversion = torch.cuda.memory_allocated(device)
        try:P, profile = selected_inverse_blocks_gpu(self.K, Qk, block_width, self.expected_inertia(), symbolic_arrays=S, device=device)
        finally:
            del S, Qk
            if released:
                if restore_caller is not None:restore_caller()
                self._restore(offloaded, device); self.K = self.K.to(device)
                self.factor = CudaSparseFactor(self.K, expected_inertia=self.expected_inertia())
        profile.update(symbolic_seconds=symbolic_seconds, cuDSS_factor_released_during_inversion=released, workspace_free_bytes_with_factor=free,
                       workspace_free_bytes_for_inversion=free_for_inversion, torch_allocated_bytes_during_inversion=torch_allocated_for_inversion,
                       caller_tensors_parked=getattr(restore_caller, 'count', 0) if released else 0, host_offloaded_bytes=sum(b for _, _, b in offloaded))
        worst = {'correlation_units': 0., 'absolute': 0., 'block': None}
        for b, exact in references.items():
            scale = exact.diagonal().clamp_min(0).sqrt()
            difference = (P[b]-exact).abs()
            correlation = float((difference/(scale[:, None]*scale[None]).clamp_min(torch.finfo(torch.float64).tiny)).max())
            if correlation > worst['correlation_units']:worst.update(correlation_units=correlation, absolute=float(difference.max()), block=b)
        tolerance = float(torch.finfo(torch.float64).eps)**.5
        profile.update(certificate={'blocks_checked': len(chosen), 'largest_component_scalars': int(counts.max()),
                                    'largest_component_blocks_checked': int(torch.isin(chosen, in_largest).sum()),
                                    'reference': 'refined cuDSS solves of the original KKT (componentwise backward error certified)',
                                    'max_error': worst, 'tolerance_correlation_units': tolerance},
                       seconds=time.perf_counter()-start, method='selected inversion of the same joint KKT', output_blocks=blocks)
        self.profile['selected_marginals'] = profile
        if not bool(torch.isfinite(P).all()):raise SparseNumericalFailure('Nonfinite selected joint marginal')
        if worst['correlation_units'] > tolerance:
            archive = getattr(self, 'failure_archive', None)
            if archive is not None:   # concise sparse evidence of the uncertified system (never a published posterior)
                K = self.K.to_sparse_csr(); chosen_list = sorted(references)
                torch.save({'K_crow': K.crow_indices().cpu(), 'K_col': K.col_indices().cpu(), 'K_values': K.values().cpu(), 'K_shape': tuple(K.shape),
                            'Qk': Qk.cpu(), 'block_width': block_width, 'blocks': chosen_list, 'selected_inverse': P[chosen_list].cpu(),
                            'cuDSS_refined_reference': torch.stack([references[b] for b in chosen_list]).cpu(),
                            'expected_inertia': self.expected_inertia(), 'profile': profile}, archive)
                profile['failure_archive'] = str(archive)
            raise SparseNumericalFailure(f"Selected-inverse covariance certificate failed: {worst} > {tolerance} correlation units")
        return P

    def _offload(self):
        moved = []
        for owner in (self, self.problem):
            for name, value in list(vars(owner).items()):
                if owner is self and name in ('K', 'factor'):continue
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    nbytes = (value._values().numel()*value.element_size()+value._indices().numel()*8) if value.layout == torch.sparse_coo else value.numel()*value.element_size()
                    setattr(owner, name, value.cpu()); moved.append((owner, name, nbytes))
                elif name == 'pivot_groups':
                    setattr(owner, name, [(r.cpu(), c.cpu()) for r, c in value]); moved.append((owner, name, 0))
        return moved

    def _restore(self, moved, device):
        for owner, name, _ in moved:
            value = getattr(owner, name)
            setattr(owner, name, [(r.to(device), c.to(device)) for r, c in value] if name == 'pivot_groups' else value.to(device))

    def close(self):
        self.factor.close()

    def __enter__(self):
        return self

    def __exit__(self,*unused):
        self.close()
