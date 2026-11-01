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


class SparseJointFactorization:
    def __init__(self, problem):
        if problem.A.device.type != 'cuda':
            raise ValueError('The sparse direct production solver requires CUDA; CPU dense is test-only')
        self.problem = problem
        start = time.perf_counter()
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
        try:self.factor = CudaSparseFactor(self.K)
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

    def close(self):
        self.factor.close()

    def __enter__(self):
        return self

    def __exit__(self,*unused):
        self.close()
