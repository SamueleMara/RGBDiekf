"""Exact local Gaussian component solver with unique pixel variables and FEJ Schur.

No point partition may cut a factor. CUDA sparse Jv/JT use the same COO
operator as the CPU oracle. Dense work is ONLY within independently connected
components and is checked against an explicit workspace budget before allocation.
Large connected surfaces fail FULL_RESOURCE_LIMIT, never a diagonal fallback.
"""
from dataclasses import dataclass
import torch
import time


class FullResourceLimit(RuntimeError):
    status='FULL_RESOURCE_LIMIT'

class FullRuntimeLimit(RuntimeError):
    status='FULL_RUNTIME_LIMIT'


@dataclass
class Factor:
    keys: tuple
    evaluate: object
    covariance: torch.Tensor | None
    tag: str
    domain: object = None
    context: object = None
    bind_reference: object = None


class UniquePixelRegistry:
    def __init__(self, graph):
        self.graph = graph
        self.sensor_counts = {}
        self.retired_frames=set();self.retired_pixels=set();self.total_sensor_factors=0

    def add_raw_sensor_factors_once(self, key, value, covariance):
        key = ('pixel',*key)
        if key in self.graph.values: return key
        if key in self.sensor_counts or key in self.retired_pixels or key[2] in self.retired_frames:
            raise ValueError('EXPIRED_PIXEL_REINTRODUCED: reference must remain inside lag')
        self.graph.add_variable(key,value)
        eye = torch.eye(4,device=value.device,dtype=value.dtype)
        def evaluate(v): return v[key]-value, {key:eye}
        self.graph.factors.append(Factor((key,),evaluate,covariance,'raw_sensor_once'))
        self.sensor_counts[key] = 1
        self.total_sensor_factors+=1
        return key


class TemporalWindowGraph:
    # shortcut: generic Python factor assembly is not full-scene validated;
    # use batched sparse assembly before removing the explicit resource guards.
    def __init__(self, device='cpu', workspace_bytes=128*1024**2):
        self.device = torch.device(device)
        self.values = {}; self.factors = []; self.pixels = UniquePixelRegistry(self)
        self.workspace_bytes = int(workspace_bytes)
        if self.device.type=='cuda':self.workspace_bytes=min(self.workspace_bytes,int(torch.cuda.mem_get_info(self.device)[0]*.8))
        self.posterior = []; self.last = None; self.marginalization_log = []
        self.anchors = {}
        self.constant_cost=0.
        self.publication_frame=None;self.publication_points=0;self.retain_keys=None
        self.current_covariances=None;self._one_component_cache=None
        self.profile=[]
        self.deadline=None

    def check_budget(self):
        if self.deadline is not None and time.perf_counter()>self.deadline:
            raise FullRuntimeLimit(f'FULL_RUNTIME_LIMIT: explicit diagnostic runtime budget exhausted; variables={len(self.values)}, factors={len(self.factors)}; no component skipped or truncated')

    @staticmethod
    def is_constraint(factor):
        if not hasattr(factor,'_exact_constraint'):
            factor._exact_constraint=factor.covariance is not None and not bool(torch.count_nonzero(factor.covariance))
        return factor._exact_constraint

    def constraint_rows(self,blocks,factors=None):
        factors=self.factors if factors is None else factors
        mask=torch.zeros(blocks[-1][1] if blocks else 0,dtype=torch.bool,device=self.device)
        for f,(lo,hi,_) in zip(factors,blocks):
            if self.is_constraint(f):mask[lo:hi]=True
        return mask

    @staticmethod
    def nullspace(A):
        """Numerical rank of exact constraints, never a covariance regularizer."""
        if not A.shape[0]:
            return A.T,torch.eye(A.shape[1],dtype=A.dtype,device=A.device),A,0
        U,s,Vh=torch.linalg.svd(A,full_matrices=True)
        tol=torch.finfo(A.dtype).eps*max(A.shape)*float(s.max()) if len(s) else 0.
        rank=int((s>tol).sum())
        inverse=(Vh[:rank].T/s[:rank])@U[:,:rank].T
        return inverse,Vh[rank:].T,U[:,rank:].T,rank

    def constrained_system(self,A,r,hard,need_covariance=True):
        """Exact equality-constrained least squares; stochastic covariance on null(C)."""
        soft=~hard;C=A[hard];c=r[hard]
        if bool(hard.any()):
            inverse,Z,left,rank=self.nullspace(C);d0=-inverse@c
            if left.numel() and float((left@c).norm())>1e-10*max(1.,float(c.norm())):
                raise ArithmeticError('INCONSISTENT_IDEAL_WARP_CONSTRAINTS')
        else:
            Q,R=self.qr_full_column_rank(A)
            step=torch.linalg.solve_triangular(R,(-Q.T@r)[:,None],upper=True).flatten()
            cov=None
            if need_covariance:
                inverseR=torch.linalg.solve_triangular(R,torch.eye(R.shape[0],dtype=A.dtype,device=A.device),upper=True);cov=inverseR@inverseR.T
            return step,cov,{'constraint_rank':0,'free_scalars':A.shape[1],'qr_condition':float(torch.linalg.cond(R)),
                             'constraint_residual_norm':0.,'covariance_policy':'EXACT_WITHIN_COMPONENT'}
        B=A[soft]@Z
        if Z.shape[1]:
            Q,R=self.qr_full_column_rank(B)
            step=d0+Z@torch.linalg.solve_triangular(R,(-Q.T@(r[soft]+A[soft]@d0))[:,None],upper=True).flatten()
            cov=None
            if need_covariance:
                inverseR=torch.linalg.solve_triangular(R,torch.eye(R.shape[0],dtype=A.dtype,device=A.device),upper=True)
                V=Z@inverseR;cov=V@V.T
            conditioning=float(torch.linalg.cond(R))
        else:step=d0;cov=torch.zeros((len(d0),len(d0)),dtype=A.dtype,device=A.device);conditioning=1.
        return step,cov,{'constraint_rank':rank,'free_scalars':Z.shape[1],'qr_condition':conditioning,
                         'constraint_residual_norm':float(c.norm()),'covariance_policy':'EXACT_WITHIN_COMPONENT'}

    def project_ideal_pixels(self):
        # References precede current time in factor insertion order, including B1.
        # Enforce the nonlinear equality, rather than adding jitter or a penalty.
        from estimator.temporal_rgbd_terms import temporal_prediction
        groups={}
        for f in self.factors:
            if f.tag=='temporal_warp' and f.context is not None and self.is_constraint(f):groups.setdefault(f.context['frame'],[]).append(f)
        state_cache={}
        def state(i,k):
            if (i,k) not in state_cache:state_cache[i,k]=self.state(i,k)[0]
            return state_cache[i,k]
        for factors in groups.values():
            contexts=[f.context for f in factors];rig=contexts[0]['rig']
            st=torch.stack([state(c['point'],c['frame']) for c in contexts]);sr=torch.stack([state(c['point'],c['reference']) for c in contexts])
            cameras=torch.tensor([c['camera'] for c in contexts],device=self.device);uv=self.tensor([c['pixel_uv'] for c in contexts])
            pixels=torch.stack([torch.stack([self.values[k] for k in c['reference_pixels']]) for c in contexts])
            cells=torch.stack([c['cell'] for c in contexts])
            h=temporal_prediction(rig,st,sr,cameras,uv,pixels,cells)[0]
            for c,value in zip(contexts,h):self.values[c['current_pixel']]=value

    def tensor(self,x): return torch.as_tensor(x,dtype=torch.float64,device=self.device)

    @staticmethod
    def qr_full_column_rank(A):
        Q,R=torch.linalg.qr(A,mode='reduced')
        tolerance=torch.finfo(A.dtype).eps*max(A.shape)*float(A.norm())
        if R.shape!=(A.shape[1],A.shape[1]) or bool((R.diag().abs()<=tolerance).any()):
            raise ArithmeticError(f'FULL_NUMERICAL_RANK_LIMIT: eliminated/solved block shape={tuple(A.shape)}, '
                                  f'QR rank tolerance={tolerance}; condition exact constraints/gauges explicitly, no diagonal jitter')
        return Q,R

    def add_variable(self,key,value):
        if len(self.values)%256==0:self.check_budget()
        if key in self.values: raise ValueError(f'duplicate variable {key}')
        self.values[key] = self.tensor(value).clone()
        return key

    def add_gaussian(self,key,mean,covariance,tag='prior'):
        eye = torch.eye(len(mean),device=self.device,dtype=torch.float64)
        mean = self.tensor(mean)
        self.factors.append(Factor((key,),lambda v:(v[key]-mean,{key:eye}),self.tensor(covariance),tag))

    def add_coupled_anchor_factors(self,point,frame,source_key,source_value,sensor_cov,pixel_uv,rig,cam,pixel_cov,velocity_cov,source_world=None):
        """Eliminate the exact x = backproject(pixel+epsilon_pixel, ell.depth)
        constraint. Free anchor coordinates are [epsilon_u,epsilon_v,v]. The
        source raw depth has exactly one sensor factor, never an independent Pxx.
        """
        ell = self.pixels.add_raw_sensor_factors_once(source_key,source_value,sensor_cov)
        key = self.add_variable(('anchor',point,frame),torch.zeros(5,device=self.device))
        cov = torch.zeros((5,5),dtype=torch.float64,device=self.device)
        cov[:2,:2] = self.tensor(pixel_cov); cov[2:,2:] = self.tensor(velocity_cov)
        self.add_gaussian(key,self.values[key].clone(),cov,'pixel_coordinate_and_velocity_prior')
        pixel=self.tensor(pixel_uv);D0=self.tensor(source_value[3]).clone();fx,fy,cx,cy=rig.K[cam]
        ray0=torch.stack(((pixel[0]-cx)/fx,(pixel[1]-cy)/fy,torch.ones_like(D0)))
        base=rig.R[cam].T@(D0*ray0-rig.t[cam]) if source_world is None else self.tensor(source_world).clone()
        self.anchors[(point,frame)] = (key,ell,pixel,rig,cam,base,D0)

    def state(self,point,frame,values=None):
        v = self.values if values is None else values
        if (point,frame) not in self.anchors:
            key = ('state',point,frame)
            return v[key],{key:torch.eye(6,dtype=torch.float64,device=self.device)}
        key,ell,pixel,rig,c,base,D0 = self.anchors[(point,frame)]
        a,z = v[key],v[ell][3]
        fx,fy,cx,cy = rig.K[c]
        uv = pixel+a[:2]
        ray = torch.stack(((uv[0]-cx)/fx,(uv[1]-cy)/fy,torch.ones_like(z)))
        ray0=torch.stack(((pixel[0]-cx)/fx,(pixel[1]-cy)/fy,torch.ones_like(z)))
        shift=torch.stack((a[0]/fx,a[1]/fy,torch.zeros_like(z)))
        # Algebraically identical coupled backprojection, centered so its zero
        # innovation is EXACTLY the recorded canonical mean in finite precision.
        x = base+rig.R[c].T@((z-D0)*ray0+z*shift)
        s = torch.cat((x,a[2:]))
        ja = torch.zeros((6,5),dtype=z.dtype,device=z.device)
        ja[:3,:2] = rig.R[c].T[:,:2]@torch.diag(torch.stack((z/fx,z/fy)))
        ja[3:,2:] = torch.eye(3,dtype=z.dtype,device=z.device)
        jl = torch.zeros((6,4),dtype=z.dtype,device=z.device); jl[:3,3] = rig.R[c].T@ray
        return s,{key:ja,ell:jl}

    def add_process_factors(self,point,prev,current,F,Q,initial):
        key = self.add_variable(('state',point,current),initial)
        _, jp = self.state(point,prev)
        keys = (key,*jp)
        def evaluate(v):
            sp,j = self.state(point,prev,v)
            return v[key]-F@sp, {key:torch.eye(6,dtype=F.dtype,device=F.device),**{k:-F@jj for k,jj in j.items()}}
        self.factors.append(Factor(keys,evaluate,Q,'process'))

    def add_warp_factors(self,point,current,reference,cam,pixel_uv,reference_pixels,current_pixel,rig,warp_cov,cell):
        _,jt = self.state(point,current); _,jr = self.state(point,reference)
        keys = tuple(dict.fromkeys((*jt,*jr,current_pixel,*reference_pixels)))
        c = torch.tensor([cam],device=self.device)
        binding={'pixels':tuple(reference_pixels),'cell':cell.clone()}
        from estimator.temporal_rgbd_terms import temporal_prediction
        def evaluate(v):
            st,dt = self.state(point,current,v); sr,dr = self.state(point,reference,v)
            vals = torch.stack([v[k] for k in binding['pixels']])[None]
            h,ht,hr,_,w,_,_ = temporal_prediction(rig,st[None],sr[None],c,pixel_uv[None],vals,binding['cell'][None])
            jac = {}
            def add(k,j): jac[k] = jac.get(k,0)+j
            for k,j in dt.items(): add(k,-ht[0]@j)
            for k,j in dr.items(): add(k,-hr[0]@j)
            eye = torch.eye(4,dtype=st.dtype,device=st.device)
            add(current_pixel,eye)
            for a,k in enumerate(binding['pixels']): add(k,-w[0,a]*eye)
            return v[current_pixel]-h[0],jac
        def domain(v):
            st,_=self.state(point,current,v);sr,_=self.state(point,reference,v)
            ut=rig.project_rows(st[None,:3],c)[0][0];ur=rig.project_rows(sr[None,:3],c)[0][0]
            q=pixel_uv-ut+ur
            return bool(torch.isfinite(q).all() and torch.equal(torch.floor(q).long(),binding['cell'].long()))
        def rebind(pixels,new_cell):
            binding['pixels']=tuple(pixels);binding['cell']=new_cell.clone()
            factor.keys=tuple(dict.fromkeys((*jt,*jr,current_pixel,*pixels)))
            factor.context.update(reference_pixels=tuple(pixels),cell=new_cell.clone())
        factor=Factor(keys,evaluate,warp_cov,'temporal_warp',domain,
                      {'point':point,'frame':current,'reference':reference,'camera':cam,'pixel_uv':pixel_uv.tolist(),'width':int(rig.width[cam]),
                       'reference_pixels':tuple(reference_pixels),'cell':cell.clone(),'current_pixel':current_pixel,'rig':rig},rebind)
        self.factors.append(factor)

    def valid_warp_domains(self):
        factors=[f for f in self.factors if f.context is not None]
        if factors:
            contexts=[f.context for f in factors];rig=contexts[0]['rig'];cache={}
            def state(i,k):
                if (i,k) not in cache:cache[i,k]=self.state(i,k)[0]
                return cache[i,k]
            st=torch.stack([state(c['point'],c['frame']) for c in contexts]);sr=torch.stack([state(c['point'],c['reference']) for c in contexts])
            cameras=torch.tensor([c['camera'] for c in contexts],device=self.device);uv=self.tensor([c['pixel_uv'] for c in contexts])
            q=uv-rig.project_rows(st[:,:3],cameras)[0]+rig.project_rows(sr[:,:3],cameras)[0]
            if not bool(torch.isfinite(q).all()) or not torch.equal(q.floor().long(),torch.stack([c['cell'] for c in contexts]).long()):return False
        return all(f.domain is None or f.domain(self.values) for f in self.factors if f.context is None)

    def correlated_warp_nis(self,new_factor_start):
        """Predictive compatibility of the NEW whitened warp residuals.
        Prior includes previous warp factors and each raw sensor once. Eliminating
        latent current pixels yields the same sensor innovation covariance. S is
        built per true connected component, never as independent 4m row tests.
        """
        J,r,layout,blocks=self.linearize()
        hard=self.constraint_rows(blocks)
        current_ranges=[(lo,hi) for i,(f,(lo,hi,_)) in enumerate(zip(self.factors,blocks))
                        if f.tag=='temporal_warp' and i>=new_factor_start]
        isnew=torch.zeros(J.shape[0],dtype=torch.bool,device=self.device)
        for lo,hi in current_ranges:isnew[lo:hi]=True
        total=0.;dof=0
        for keys,cols,rows,A,required in self.local_linearizations(J,layout,blocks):
            ww=rows[isnew[rows]]
            if not len(ww):continue
            required+=8*4*len(ww)**2
            if required>self.workspace_bytes:raise FullResourceLimit(f'FULL_RESOURCE_LIMIT: correlated innovation workspace={required}, budget={self.workspace_bytes}')
            w=isnew[rows];Ap,Aw=A[~w],A[w];rp,rw=r[rows[~w]],r[rows[w]]
            step,P,_=self.constrained_system(Ap,rp,hard[rows[~w]])
            nu=rw+Aw@step
            S=torch.diag((~hard[ww]).double())+Aw@P@Aw.T
            torch.linalg.cholesky(S) # ideal discrepancy is zero, total innovation must remain SPD
            total+=float(nu@torch.linalg.solve(S,nu));dof+=len(nu)
        return {'NIS':total,'dof':dof,'model':'joint prior-predictive whitened warp compatibility; sensor pixels marginalized; all cross blocks retained'}

    def linearize(self,factors=None):
        factors = self.factors if factors is None else factors
        keys = list(dict.fromkeys(k for f in factors for k in f.keys))
        layout = {}; off=0
        for k in keys: layout[k]=slice(off,off+self.values[k].numel()); off=layout[k].stop
        ri=[];ci=[];vv=[];res=[]; row=0; blocks=[]
        for fi,f in enumerate(factors):
            if fi%256==0:self.check_budget()
            r,jac = f.evaluate(self.values)
            L = torch.linalg.cholesky(f.covariance) if f.covariance is not None and not self.is_constraint(f) else None
            if L is not None:r = torch.linalg.solve_triangular(L,r[:,None],upper=False).flatten()
            active_keys=[]
            for k,j in jac.items():
                if L is not None:j = torch.linalg.solve_triangular(L,j,upper=False)
                if not bool((j!=0).any()):continue
                active_keys.append(k)
                rr,cc = torch.meshgrid(torch.arange(len(r),device=self.device)+row,
                                        torch.arange(layout[k].start,layout[k].stop,device=self.device),indexing='ij')
                ri.append(rr.flatten());ci.append(cc.flatten());vv.append(j.flatten())
            blocks.append((row,row+len(r),tuple(active_keys)));row+=len(r);res.append(r)
        idx = torch.stack((torch.cat(ri),torch.cat(ci))) if ri else torch.empty((2,0),dtype=torch.long,device=self.device)
        val = torch.cat(vv) if vv else self.tensor([])
        nonzero=val!=0
        J = torch.sparse_coo_tensor(idx[:,nonzero],val[nonzero],(row,off),device=self.device,check_invariants=True).coalesce()
        return J,torch.cat(res) if res else self.tensor([]),layout,blocks

    @staticmethod
    def apply_J(J,v): return torch.sparse.mm(J,v[:,None]).flatten()

    @staticmethod
    def apply_JT(J,z): return torch.sparse.mm(J.transpose(0,1),z[:,None]).flatten()

    def components(self,layout,blocks):
        parent = {k:k for k in layout}
        def root(k):
            while parent[k]!=k:
                parent[k]=parent[parent[k]];k=parent[k]
            return k
        for _,_,keys in blocks:
            if not keys:continue
            a=root(keys[0])
            for k in keys[1:]: parent[root(k)]=a
        groups={}
        for k in layout: groups.setdefault(root(k),[]).append(k)
        membership={k:a for a,ks in groups.items() for k in ks}
        grouped_rows={a:[] for a in groups}
        for lo,hi,fks in blocks:
            if not fks:continue
            assert all(membership[k]==membership[fks[0]] for k in fks), 'component cut'
            grouped_rows[membership[fks[0]]].append((lo,hi))
        out=[]
        for a,ks in groups.items():
            rows=grouped_rows[a]
            if not rows:raise ArithmeticError(f'FULL_UNCONSTRAINED_VARIABLE: {ks}; no nonzero linearized prior/factor')
            cols=torch.cat([torch.arange(layout[k].start,layout[k].stop,device=self.device) for k in ks])
            rr=torch.cat([torch.arange(lo,hi,device=self.device) for lo,hi in rows])
            required = 8*(3*len(cols)*len(rr)+8*len(cols)**2)
            if required>self.workspace_bytes:
                raise FullResourceLimit(f'FULL_RESOURCE_LIMIT: component variables={len(ks)}, scalars={len(cols)}, rows={len(rr)}, '
                                        f'estimated_workspace_bytes={required}, budget_bytes={self.workspace_bytes}; no component edge may be cut')
            out.append((ks,cols,rr,required))
        return out

    def local_linearizations(self,J,layout,blocks):
        components=self.components(layout,blocks)
        idx,val=J.indices(),J.values()
        row_component=torch.empty(J.shape[0],device=self.device,dtype=torch.long)
        for ci,(_,_,rows,_) in enumerate(components):row_component[rows]=ci
        membership=row_component[idx[0]]
        order=torch.argsort(membership,stable=True)
        sizes=torch.bincount(membership,minlength=len(components)).cpu().tolist()
        start=0
        for (keys,cols,rows,required),size in zip(components,sizes):
            ix=order[start:start+size];start+=size
            # Components preserve global layout/row order. Map only their own
            # nonzeros, never rescan the entire global Jacobian per component.
            lr=torch.searchsorted(rows,idx[0,ix]);lc=torch.searchsorted(cols,idx[1,ix])
            j=torch.zeros((len(rows),len(cols)),dtype=torch.float64,device=self.device)
            j.index_put_((lr,lc),val[ix],accumulate=True)
            yield keys,cols,rows,j,required

    def component_systems(self,J,r,layout,blocks):
        for keys,cols,rows,j,required in self.local_linearizations(J,layout,blocks):
            yield keys,cols,j.T@j,-j.T@r[rows],required

    def solve_sparse_map(self,max_iters=10,tol=1e-8,accept=None,active_keys=None):
        history=[];converged=False;accepted_keys=set()
        self.project_ideal_pixels()
        for it in range(max_iters):
            J,r,layout,blocks=self.linearize();hard=self.constraint_rows(blocks);cost=float(r[~hard]@r[~hard]/2)+self.constant_cost
            step=torch.zeros(J.shape[1],dtype=torch.float64,device=self.device)
            grad=0.;component_columns=[];component_keys=[]
            for keys,cols,rows,A,required in self.local_linearizations(J,layout,blocks):
                if active_keys is not None and not any(key in active_keys for key in keys):continue
                component_columns.append(cols)
                component_keys.append(keys)
                step[cols],_,stats=self.constrained_system(A,r[rows],hard[rows],False)
                b=-A.T@r[rows];grad+=float(b@b)
            old={k:v.clone() for k,v in self.values.items()}
            accepted=False
            for ls in range(25):
                rate=2.**-ls
                for k,sl in layout.items():self.values[k]=old[k]+rate*step[sl]
                self.project_ideal_pixels()
                feasible=(True if accept is None else accept(self)) and self.valid_warp_domains()
                if feasible:self.project_ideal_pixels()
                new=self.cost() if feasible else float('inf')
                if new<=cost+1e-12*max(1.,abs(cost)):accepted=True;break
            accepted_components=None
            if ls>0 and len(component_columns)>1:
                # Nonlinear visibility can block one independent component.
                # Try each TRUE linearized component without cutting any edge;
                # validate the whole ownership domain after every proposal.
                self.values.update(old)
                if accept is not None:accept(self)
                accepted_components=0;actual_step=torch.zeros_like(step);new=self.cost()
                for keys,cols in zip(component_keys,component_columns):
                    self.check_budget();base={key:value.clone() for key,value in self.values.items()};baseline=self.cost()
                    for local_ls in range(25):
                        local_rate=2.**-local_ls
                        increment=torch.zeros_like(step);increment[cols]=local_rate*step[cols]
                        for key,sl in layout.items():self.values[key]=base[key]+increment[sl]
                        self.project_ideal_pixels()
                        feasible=(True if accept is None else accept(self)) and self.valid_warp_domains()
                        if feasible:self.project_ideal_pixels()
                        candidate=self.cost() if feasible else float('inf')
                        if candidate<=baseline+1e-12*max(1.,abs(baseline)):
                            accepted_components+=1;accepted_keys.update(keys);actual_step[cols]=increment[cols];new=candidate;break
                    else:
                        self.values.update(base)
                        if accept is not None:accept(self)
                accepted=accepted_components>0
                metric=float(self.apply_J(J,actual_step).square().sum())
            else:
                metric=float(self.apply_J(J,rate*step).square().sum())
                if accepted:
                    for keys in component_keys:accepted_keys.update(keys)
            history.append({'iteration':it,'objective':cost,'proposal_objective':new,'gradient_norm':grad**.5,
                            'accepted':accepted,'backtracks':ls,'step_information_norm2':metric,'component_backtracking_accepted':accepted_components})
            if not accepted:
                # Candidate support expansion may have registered new raw sensor
                # pixels. Keep those factors/variables, restore the old state,
                # and restore its contributor cache; never delete observations.
                self.values.update(old)
                if accept is not None:accept(self)
                break
            if history[-1]['step_information_norm2']<tol and (accepted_components is None or accepted_components==len(component_columns)):converged=True;break
        self.factor_posterior()
        return {'history':history,'converged':converged,'objective':self.cost(),
                '_accepted_state_keys':accepted_keys,
                'components':[{'scalars':len(c['cols']),'workspace_bytes':c['required']} for c in self.posterior]}

    def cost(self):
        total=self.constant_cost
        for f in self.factors:
            if self.is_constraint(f):continue
            r,_=f.evaluate(self.values)
            total+=float(r@(torch.linalg.solve(f.covariance,r) if f.covariance is not None else r)/2)
        return total

    def factor_posterior(self):
        J,r,layout,blocks=self.linearize();self.posterior=[];self.last=(J,r,layout,blocks)
        hard=self.constraint_rows(blocks);self.profile=[]
        self.component_for_key={}
        self._one_component_cache=None
        self.current_covariances=(torch.empty((self.publication_points,6,6),dtype=torch.float64,device=self.device)
                                  if self.publication_frame is not None else None)
        for keys,cols,rows,A,required in self.local_linearizations(J,layout,blocks):
            _,cov,stats=self.constrained_system(A,r[rows],hard[rows])
            self.profile.append({'scalars':len(cols),'rows':len(rows),'workspace_bytes':required,
                                 'linear_J_nonzeros':int(torch.count_nonzero(A)),'dense_QR_square_storage_scalars':stats['free_scalars']**2,
                                 'constraint_nullspace_storage_scalars':len(cols)*stats['free_scalars'] if stats['constraint_rank'] else 0,**stats})
            local={};off=0
            for k in keys:local[k]=slice(off,off+len(self.values[k]));off=local[k].stop
            comp={'keys':keys,'cols':cols,'rows':rows,'layout':local,'required':required,**stats}
            if self.current_covariances is not None:
                for key in keys:
                    if key[0]=='state' and key[2]==self.publication_frame:
                        self.current_covariances[key[1]]=cov[local[key],local[key]]
            # Production streams genuinely independent components, retaining full
            # cross blocks only for requested traces. The oracle retains all.
            if self.retain_keys is None or any(key in self.retain_keys for key in keys):comp.update(cov=cov,H=A[~hard[rows]].T@A[~hard[rows]],b=-A.T@r[rows],constraint_matrix=A[hard[rows]],
                                                                                               linearized_J=A,linearized_residual=r[rows],constraint_rows=hard[rows])
            self.posterior.append(comp)
            for k in keys:self.component_for_key[k]=len(self.posterior)-1

    def component_covariance(self,ci):
        comp=self.posterior[ci]
        if 'cov' in comp:return comp
        if self._one_component_cache is not None and self._one_component_cache[0]==ci:return self._one_component_cache[1]
        J,r,_,_=self.last;idx,val=J.indices(),J.values();cols,rows=comp['cols'],comp['rows']
        cm=torch.full((J.shape[1],),-1,dtype=torch.long,device=self.device);cm[cols]=torch.arange(len(cols),device=self.device)
        rm=torch.full((J.shape[0],),-1,dtype=torch.long,device=self.device);rm[rows]=torch.arange(len(rows),device=self.device)
        ok=(cm[idx[1]]>=0)&(rm[idx[0]]>=0)
        A=torch.zeros((len(rows),len(cols)),device=self.device,dtype=torch.float64)
        A.index_put_((rm[idx[0,ok]],cm[idx[1,ok]]),val[ok],accumulate=True)
        hard=self.constraint_rows(self.last[3])[rows]
        _,cov,_=self.constrained_system(A,r[rows],hard)
        out={**comp,'cov':cov,'H':A[~hard].T@A[~hard],'b':-A.T@r[rows],'constraint_matrix':A[hard]}
        self._one_component_cache=(ci,out)
        return out

    def covariance(self,ja,jb):
        missing={key for key in (*ja,*jb) if key not in self.component_for_key}
        if missing:raise KeyError(f'COVARIANCE_VARIABLE_OUTSIDE_ACTIVE_WINDOW: {missing}')
        da=next(iter(ja.values())).shape[0];db=next(iter(jb.values())).shape[0]
        out=torch.zeros((da,db),dtype=torch.float64,device=self.device)
        if len(ja)==len(jb)==1:
            a,b=next(iter(ja)),next(iter(jb))
            if a==b and self.current_covariances is not None and a[0]=='state' and a[2]==self.publication_frame:
                return ja[a]@self.current_covariances[a[1]]@jb[b].T
        for ci in {self.component_for_key[k] for k in ja if k in self.component_for_key}:
            comp=self.component_covariance(ci)
            lay=comp['layout']; aa=[k for k in ja if k in lay];bb=[k for k in jb if k in lay]
            for a in aa:
                for b in bb:out+=ja[a]@comp['cov'][lay[a],lay[b]]@jb[b].T
        return out

    def extract_current_marginals(self,points,frame):
        if self.current_covariances is not None and frame==self.publication_frame:
            points=list(points)
            return torch.stack([self.state(int(i),frame)[0] for i in points]),self.current_covariances[points].clone()
        states=[];covs=[]
        for i in points:
            s,j=self.state(int(i),frame);states.append(s);covs.append(self.covariance(j,j))
        return torch.stack(states),torch.stack(covs)

    def marginalize_expired_exactly(self,expired):
        expired=set(expired)&set(self.values)
        if not expired:return
        affected=[f for f in self.factors if any(k in expired for k in f.keys)]
        untouched=[f for f in self.factors if not any(k in expired for k in f.keys)]
        J,r,layout,blocks=self.linearize(affected)
        hard=self.constraint_rows(blocks,affected)
        self.constant_cost+=sum(float(r[lo:hi]@r[lo:hi]/2) for lo,hi,keys in blocks if not keys)
        priors=[]
        for keys,cols,rows,Jlocal,required in self.local_linearizations(J,layout,blocks):
            sizes=[len(self.values[k]) for k in keys];start=0;old=[];keep=[]
            retained=[k for k in keys if k not in expired]
            for k,d in zip(keys,sizes):
                (old if k in expired else keep).extend(range(start,start+d));start+=d
            v=torch.tensor(old,device=self.device);u=torch.tensor(keep,device=self.device)
            if bool(hard[rows].any()):
                # Eliminate old coordinates under C_v dv + C_u du + c = 0.
                # Remaining left-null constraints are retained, not jittered.
                h=hard[rows];C=Jlocal[h];c=r[rows[h]];A=Jlocal[~h];b=r[rows[~h]]
                inverse,Z,left,rank=self.nullspace(C[:,v])
                E=A[:,u]-A[:,v]@inverse@C[:,u];e=b-A[:,v]@inverse@c
                D=A[:,v]@Z
                if Z.shape[1]:
                    Qd,Rd=self.qr_full_column_rank(D);E=E-Qd@(Qd.T@E);e=e-Qd@(Qd.T@e)
                    eliminated_logdet=float(2*Rd.diag().abs().log().sum())
                else:eliminated_logdet=0.
                origin=torch.cat([self.values[k].clone() for k in retained]) if retained else self.tensor([])
                splits=torch.tensor([0,*[len(self.values[k]) for k in retained]]).cumsum(0).tolist()
                for matrix,residual,is_hard in [(E,e,False),(left@C[:,u],left@c,True)]:
                    if not len(residual):continue
                    if is_hard:
                        # Scale by the ORIGINAL constraint, not the roundoff
                        # residue of a nullspace projection. Otherwise a zero
                        # equation becomes a spurious exact constraint.
                        U,s,Vh=torch.linalg.svd(matrix,full_matrices=False)
                        threshold=torch.finfo(matrix.dtype).eps*max(C.shape)*float(C.norm())
                        rr=int((s>threshold).sum());B=s[:rr,None]*Vh[:rr];d=U[:,:rr].T@residual
                        remainder=residual-U[:,:rr]@d
                    else:
                        # Retain every stochastic QR row, including weak modes.
                        # No covariance/information eigenvalue truncation.
                        Qb,B=torch.linalg.qr(matrix,mode='reduced');d=Qb.T@residual;rr=len(d)
                        remainder=residual-Qb@d
                    if is_hard and float(remainder.norm())>1e-10*max(1.,float(residual.norm())):raise ArithmeticError('INCONSISTENT_MARGINAL_CONSTRAINT')
                    if not is_hard:self.constant_cost+=float(remainder@remainder/2)
                    if not rr:continue
                    def evaluate(values,B=B,d=d,origin=origin,retained=tuple(retained),splits=splits):
                        delta=torch.cat([values[k] for k in retained])-origin
                        return B@delta+d,{k:B[:,splits[i]:splits[i+1]] for i,k in enumerate(retained)}
                    priors.append(Factor(tuple(retained),evaluate,torch.zeros((rr,rr),device=self.device,dtype=torch.float64) if is_hard else None,
                                         'FEJ_exact_constraint' if is_hard else 'FEJ_exact_Schur'))
                self.marginalization_log.append({'expired_scalars':len(old),'retained_scalars':len(keep),'constraint_rank_eliminated':rank,
                    'eliminated_precision_logdet':eliminated_logdet,'policy':'FEJ exact equality nullspace + QR elimination; no covariance jitter'})
                continue
            Qv,Rv=self.qr_full_column_rank(Jlocal[:,v])
            residual=r[rows]-Qv@(Qv.T@r[rows])
            eliminated_logdet=float(2*Rv.diag().abs().log().sum())
            if not keep:
                self.constant_cost+=float(residual@residual/2)
                continue
            B=Jlocal[:,u]-Qv@(Qv.T@Jlocal[:,u])
            Qb,A=torch.linalg.qr(B,mode='reduced')
            d=Qb.T@residual
            constant=max(0.,float((residual@residual-d@d)/2))
            self.constant_cost+=constant
            origin=torch.cat([self.values[k].clone() for k in retained])
            splits=torch.tensor([0,*[len(self.values[k]) for k in retained]]).cumsum(0).tolist()
            def evaluate(values,retained=tuple(retained),A=A,d=d,origin=origin,splits=splits):
                x=torch.cat([values[k] for k in retained]);
                return A@(x-origin)+d,{k:A[:,splits[i]:splits[i+1]] for i,k in enumerate(retained)}
            if len(d):priors.append(Factor(tuple(retained),evaluate,None,'FEJ_exact_Schur')) # already whitened; no dense identity
            self.marginalization_log.append({'expired_scalars':len(old),'retained_scalars':len(keep),
                                             'prior_linearization':origin.cpu().tolist(),'eliminated_precision_logdet':eliminated_logdet,
                                             'constant_residual_cost':constant,
                                             'policy':'FEJ; square-root QR projection equivalent to complete Schur; no eigenvalue truncation'})
        self.factors=untouched+priors
        for k in expired:del self.values[k]
        retired=[k for k in expired if k[0]=='pixel']
        for k in retired:
            self.pixels.sensor_counts.pop(k,None);self.pixels.retired_pixels.add(k)
        active_frames={k[2] for k in self.values if k[0]=='pixel'}
        for frame in {k[2] for k in retired}-active_frames:
            self.pixels.retired_frames.add(frame)
        self.pixels.retired_pixels={k for k in self.pixels.retired_pixels if k[2] not in self.pixels.retired_frames}
        self.anchors={k:v for k,v in self.anchors.items() if v[0] not in expired}
        self.posterior=[]
