import unittest
import numpy as np
import torch
from estimator.temporal_full_graph import TemporalWindowGraph,Factor,FullResourceLimit
from tests.test_temporal_warp import rig

IDEAL_ORACLE_WITNESSES=[]


def shared_graph(device='cpu'):
    g=TemporalWindowGraph(device)
    t=g.tensor
    for i in range(2):
        g.add_variable(('state',i,0),t([0.]));g.add_gaussian(('state',i,0),t([0.]),t([[2.]]))
    ell=g.pixels.add_raw_sensor_factors_once((0,0,4),t([.1,.2,.3,.4]),torch.eye(4,device=device,dtype=torch.float64)*.3)
    g.pixels.add_raw_sensor_factors_once((0,0,4),t([.1,.2,.3,.4]),torch.eye(4,device=device,dtype=torch.float64)*.3)
    for i,y in enumerate((1.,1.2)):
        key=('state',i,0)
        def ev(v,key=key,y=y):
            j=t([[0.,0.,0.,1.]])
            return v[key]+v[ell][3:4]-y,{key:t([[1.]]),ell:j}
        g.factors.append(Factor((key,ell),ev,t([[.2]]),'shared_reference_test'))
    return g


class FullGraphTests(unittest.TestCase):
    def test_repeated_reference_has_a_correlated_uncertainty_floor(self):
        variances=[]
        for count in (2,4,8):
            g=TemporalWindowGraph();t=g.tensor;x=g.add_variable(('state',0,0),t([0.]));g.add_gaussian(x,t([0.]),t([[1.]]))
            ref=g.pixels.add_raw_sensor_factors_once((0,0,7),t([0.,0.,0.,0.]),torch.eye(4,dtype=torch.float64)*.3)
            for k in range(1,count+1):
                y=g.add_variable(('sample',0,k),t([0.]));g.add_gaussian(y,t([0.]),t([[.2]]))
                def ev(v,y=y):return v[y]-v[x]-v[ref][:1],{y:t([[1.]]),x:t([[-1.]]),ref:t([[-1.,0,0,0]])}
                g.factors.append(Factor((y,x,ref),ev,t([[0.]]),'ideal_reused_reference'))
            g.factor_posterior();P=float(g.covariance({x:t([[1.]])},{x:t([[1.]])}));expected=1/(1+count/(.2+count*.3))
            self.assertAlmostEqual(P,expected,places=12);self.assertGreater(P,.3/1.3)
            self.assertEqual(g.pixels.total_sensor_factors,1);variances.append(P)
        IDEAL_ORACLE_WITNESSES.append({'repeated_reference_counts':[2,4,8],'posterior_variances':variances,'infinite_reuse_variance_floor':.3/1.3,'raw_reference_likelihood_count':1})

    def test_ideal_three_frame_kkt_oracle_and_schur_cross_covariance(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            g=TemporalWindowGraph(device);t=g.tensor;eye=torch.eye(4,dtype=torch.float64,device=device)
            ref=g.pixels.add_raw_sensor_factors_once((0,0,3),t([.1,.2,.3,2.]),eye*.02)
            x0=g.add_variable(('state',0,0),t([2.]))
            g.factors.append(Factor((x0,ref),lambda v:(v[x0]-v[ref][3:4],{x0:t([[1.]]),ref:t([[0.,0.,0.,-1.]])}),t([[0.]]),'coupled_source_exact'))
            for k in (1,2,3):
                x=g.add_variable(('state',0,k),t([2.]));previous=('state',0,k-1)
                def process(v,x=x,previous=previous):return v[x]-v[previous],{x:t([[1.]]),previous:t([[-1.]])}
                g.factors.append(Factor((x,previous),process,t([[.01]]),'process'))
                obs=g.pixels.add_raw_sensor_factors_once((0,k,3),t([.1,.2,.3,2.]),eye*.03)
                def warp(v,x=x,obs=obs):return v[obs]-v[ref]-t([0,0,0,1.])*(v[x]-v[x0]),{obs:eye,ref:-eye,x:-t([[0.],[0.],[0.],[1.]]),x0:t([[0.],[0.],[0.],[1.]])}
                g.factors.append(Factor((obs,ref,x,x0),warp,eye*0,'zero_warp'))
            J,r,layout,blocks=g.linearize();hard=g.constraint_rows(blocks);A=J.to_dense();soft=A[~hard];C=A[hard]
            H=soft.T@soft;K=torch.cat((torch.cat((H,C.T),1),torch.cat((C,torch.zeros((len(C),len(C)),device=device,dtype=torch.float64)),1)),0)
            dense=torch.linalg.inv(K)[:len(H),:len(H)]
            _,P,_=g.constrained_system(A,r,hard)
            discrepancy=float((P-dense).abs().max());self.assertLess(discrepancy,2e-12)
            g.factor_posterior();key=('state',0,3);other=('pixel',0,2,3);ja={key:t([[1.]])};jb={other:eye}
            before=g.covariance(ja,jb).clone();variance=g.covariance(ja,ja).clone()
            g.marginalize_expired_exactly([x0,ref]);g.factor_posterior()
            torch.testing.assert_close(g.covariance(ja,ja),variance,atol=2e-12,rtol=2e-12)
            torch.testing.assert_close(g.covariance(ja,jb),before,atol=2e-12,rtol=2e-12)
            IDEAL_ORACLE_WITNESSES.append({'device':device,'ideal_3_frame_KKT_covariance_max_abs':discrepancy,
              'Schur_state_pixel_cross_covariance_max_abs':float((g.covariance(ja,jb)-before).abs().max()),'sensor_noise_active':True})

    def test_zero_discrepancy_constrained_qr_matches_dense_conditioning(self):
        g=TemporalWindowGraph();t=g.tensor
        a=g.add_variable(('state',0,0),t([0.]));b=g.add_variable(('pixel',0,1,1),t([1.]))
        g.add_gaussian(a,t([0.]),t([[2.]]));g.add_gaussian(b,t([1.]),t([[.3]]))
        g.factors.append(Factor((a,b),lambda v:(v[b]-v[a],{a:t([[-1.]]),b:t([[1.]])}),t([[0.]]),'exact_test'))
        J,r,layout,blocks=g.linearize();hard=g.constraint_rows(blocks)
        delta,P,_=g.constrained_system(J.to_dense(),r,hard)
        prior=t([[2.,0.],[0.,.3]]);C=t([[-1.,1.]])
        oracle=prior-prior@C.T@torch.linalg.solve(C@prior@C.T,C@prior)
        torch.testing.assert_close(P,oracle,atol=2e-14,rtol=2e-14)
        torch.testing.assert_close(C@(t([0.,1.])+delta),t([0.]),atol=1e-14,rtol=0)
        self.assertGreater(float(P[0,0]),0.)
        # Marginalizing the old point retains the noisy posterior on the pixel.
        g.values[a]+=delta[layout[a]];g.values[b]+=delta[layout[b]]
        g.factor_posterior();before=g.covariance({b:t([[1.]])},{b:t([[1.]])})
        g.marginalize_expired_exactly([a]);g.factor_posterior()
        torch.testing.assert_close(g.covariance({b:t([[1.]])},{b:t([[1.]])}),before,atol=2e-14,rtol=2e-14)

    def test_conditioned_reference_limit_matches_simplified(self):
        from tests.test_temporal_warp import frame
        from estimator.temporal_rgbd_terms import temporal_prediction,TemporalRows
        from estimator.temporal_simplified_iekf import update_points_temporal_simplified
        from estimator.temporal_zbuffer import first_hit_pixel_owner
        from estimator.point_kalman_cuda import transition,process_noise
        r=rig();f=frame();c=torch.tensor([0]);uv=torch.tensor([[3.,3.]],dtype=torch.float64)
        ref=torch.tensor([[.03,.08,2.,.16,.12,.2]],dtype=torch.float64)
        mean=ref@transition(.5).T;prior=process_noise(.5,torch.eye(3,dtype=torch.float64))[None]
        vals=torch.tensor([[[.1,.2,.3,2.],[.2,.3,.4,2.01],[.15,.25,.35,2.02],[.3,.4,.5,2.03]]],dtype=torch.float64)
        y=temporal_prediction(r,mean,ref,c,uv,vals)[0]+torch.tensor([[.001,-.002,.003,.0005]],dtype=torch.float64)
        Rc=torch.diag(torch.tensor([.01,.012,.02,.03],dtype=torch.float64));Rw=torch.diag(torch.tensor([.02,.024,.025,.04],dtype=torch.float64))
        f.rgb[0,3,3]=y[0,:3];f.depth[0,3,3]=y[0,3]
        # Ref state and clean image are exactly conditioned OUT of both systems,
        # rather than approximated with invented tiny independent priors.
        def build(s):
            h,ht,hr,q,w,*_=temporal_prediction(r,s,ref,c,uv,vals)
            rows=TemporalRows(torch.tensor([0]),c,torch.tensor([27]),torch.tensor([0]),ref,torch.tensor([[18,19,26,27]]),
                              vals,y,h,ht,hr,q,w,q.floor().long(),(Rc+Rw)[None])
            own=first_hit_pixel_owner(s[:,:3],r,f)
            return rows,own,own.reason,{}
        simple=update_points_temporal_simplified(mean,prior,build,f,max_iters=20,tol=1e-24)
        g=TemporalWindowGraph();state=g.add_variable(('state',0,1),mean[0]);g.add_gaussian(state,mean[0],prior[0])
        ell=g.pixels.add_raw_sensor_factors_once((0,1,27),y[0],Rc)
        def evaluate(v):
            h,ht,*_=temporal_prediction(r,v[state][None],ref,c,uv,vals)
            return v[ell]-h[0],{state:-ht[0],ell:torch.eye(4,dtype=torch.float64)}
        g.factors.append(Factor((state,ell),evaluate,Rw,'conditional_warp'))
        g.solve_sparse_map(max_iters=20,tol=1e-24)
        torch.testing.assert_close(g.values[state],simple[0][0],atol=2e-12,rtol=2e-12)
        P=g.covariance({state:torch.eye(6,dtype=torch.float64)},{state:torch.eye(6,dtype=torch.float64)})
        torch.testing.assert_close(P,simple[1][0],atol=2e-12,rtol=2e-12)

    def test_shared_latents_mean_cov_dense_and_cross(self):
        g=shared_graph();g.solve_sparse_map()
        J,r,layout,_=g.linearize();dense=J.to_dense();cov=torch.linalg.inv(dense.T@dense)
        for comp in g.posterior:
            torch.testing.assert_close(comp['cov'],cov[comp['cols'][:,None],comp['cols']],atol=2e-13,rtol=2e-13)
        a=('state',0,0);b=('state',1,0)
        cross=g.covariance({a:g.tensor([[1.]])},{b:g.tensor([[1.]])})
        self.assertGreater(float(cross),.1)
        self.assertEqual(list(g.pixels.sensor_counts.values()),[1])
        self.assertEqual(sum(f.tag=='raw_sensor_once' for f in g.factors),1)
        self.assertLess(float(g.apply_JT(J,r).norm()),1e-12)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_cpu_mean_covariance(self):
        a,b=shared_graph(),shared_graph('cuda');a.solve_sparse_map();b.solve_sparse_map()
        for k in a.values:torch.testing.assert_close(a.values[k],b.values[k].cpu(),atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(a.posterior[0]['cov'],b.posterior[0]['cov'].cpu(),atol=1e-11,rtol=1e-11)

    def test_jvp_vjp(self):
        g=shared_graph();J,r,_,_=g.linearize();gen=torch.Generator().manual_seed(1)
        x=torch.randn(J.shape[1],generator=gen,dtype=torch.float64);y=torch.randn(J.shape[0],generator=gen,dtype=torch.float64)
        self.assertAlmostEqual(float(g.apply_J(J,x)@y),float(x@g.apply_JT(J,y)),places=13)

    def test_correlated_predictive_nis_against_dense(self):
        g=shared_graph()
        for f in g.factors[3:]:f.tag='temporal_warp'
        got=g.correlated_warp_nis(3)
        # Two unit state coefficients with independent prior variance 2 and
        # one shared reference-depth error variance .3, plus warp variance .2.
        nu=np.array([.4-1.,.4-1.2]);S=np.eye(2)*2.2+np.ones((2,2))*.3
        self.assertAlmostEqual(got['NIS'],float(nu@np.linalg.solve(S,nu)),places=13)
        self.assertEqual(got['dof'],2)

    def test_actual_nonlinear_warp_factor_all_columns(self):
        from tests.test_temporal_warp import frame
        from estimator.temporal_rgbd_terms import raw_values,sensor_covariance,validate_reference_patch
        from estimator.projective_sensor import SensorNoise
        from estimator.point_kalman_cuda import transition,process_noise
        g=TemporalWindowGraph();r=rig();f=frame();noise=SensorNoise(g.tensor([.01]*3),.001,0.)
        c=torch.tensor(0);pix=torch.tensor(27);source=raw_values(f,c,pix)
        g.add_coupled_anchor_factors(0,0,(0,0,27),source,sensor_covariance(source,noise),g.tensor([3.,3.]),r,0,np.eye(2)/12,np.eye(3))
        state,_=g.state(0,0);initial=state.clone();initial[0]+=.05
        F=transition(.04);Q=process_noise(.04,torch.eye(3,dtype=torch.float64)*.01)
        g.add_process_factors(0,0,1,F,Q,initial)
        q=g.tensor([[2.9,3.]])
        pixs,vals,reason,cell=validate_reference_patch(f,torch.tensor([0]),q,r,noise,{'same_surface_n_sigma':3.,'same_surface_max_incidence_deg':80.})
        refs=[]
        for pp in pixs[0].tolist():
            val=raw_values(f,c,torch.tensor(pp));refs.append(g.pixels.add_raw_sensor_factors_once((0,0,pp),val,sensor_covariance(val,noise)))
        current=g.pixels.add_raw_sensor_factors_once((0,1,27),source,sensor_covariance(source,noise))
        g.add_warp_factors(0,1,0,0,g.tensor([3.,3.]),refs,current,r,torch.eye(4,dtype=torch.float64)*.01,cell[0])
        factor=g.factors[-1];res,jac=factor.evaluate(g.values)
        eps=1e-6
        for key,J in jac.items():
            for col in range(J.shape[1]):
                old=g.values[key].clone();g.values[key][col]+=eps;plus=factor.evaluate(g.values)[0]
                g.values[key]=old.clone();g.values[key][col]-=eps;minus=factor.evaluate(g.values)[0];g.values[key]=old
                torch.testing.assert_close((plus-minus)/(2*eps),J[:,col],atol=2e-9,rtol=2e-6)
        # The source depth belongs to the same latent even when it is in a patch.
        self.assertEqual(g.pixels.sensor_counts[('pixel',0,0,27)],1)

    def test_source_coupling_single_depth(self):
        g=TemporalWindowGraph();r=rig();pix=g.tensor([3.,3.]);value=g.tensor([.1,.2,.3,2.])
        sensor=torch.diag(g.tensor([.01,.01,.01,.04]));px=torch.eye(2,dtype=torch.float64)/12;vv=torch.eye(3,dtype=torch.float64)
        g.add_coupled_anchor_factors(0,0,(0,0,27),value,sensor,pix,r,0,px,vv)
        g.factor_posterior();s,j=g.state(0,0);P=g.covariance(j,j)
        expected=torch.diag(g.tensor([4/16/12,4/16/12,.04,1.,1.,1.]))
        torch.testing.assert_close(P,expected,atol=1e-14,rtol=1e-13)
        ell=('pixel',0,0,27);link=g.covariance(j,{ell:torch.eye(4,dtype=torch.float64)})
        self.assertAlmostEqual(float(link[2,3]),.04)
        # Deliberately duplicated independent anchor is detectably overconfident.
        bad=1/(1/.04+1/.04)
        self.assertGreater(float(P[2,2]),bad)
        self.assertEqual(sum(f.tag=='raw_sensor_once' for f in g.factors),1)

    def test_exact_schur_and_lag_slide(self):
        g=TemporalWindowGraph();t=g.tensor
        for k in range(4):
            key=('state',0,k);g.add_variable(key,t([0.]))
            if k==0:g.add_gaussian(key,t([0.]),t([[1.]]))
            else:
                prev=('state',0,k-1)
                def ev(v,key=key,prev=prev):return v[key]-v[prev]-.3,{key:t([[1.]]),prev:t([[-1.]])}
                g.factors.append(Factor((key,prev),ev,t([[.1]]),'process'))
            g.add_gaussian(key,t([.2*k]),t([[.4]]),'observation')
        g.solve_sparse_map();key=('state',0,3);one={key:t([[1.]])}
        fullJ,_,layout,_=g.linearize();H=fullJ.to_dense().T@fullJ.to_dense()
        v=layout[('state',0,0)];logdet_full=float(torch.linalg.slogdet(H)[1]);logdet_v=float(torch.linalg.slogdet(H[v,v])[1])
        cov=g.covariance(one,one).clone();mean=g.values[key].clone()
        g.marginalize_expired_exactly([('state',0,0)]);g.solve_sparse_map()
        newJ,_,_,_=g.linearize();newH=newJ.to_dense().T@newJ.to_dense()
        self.assertAlmostEqual(logdet_full-logdet_v,float(torch.linalg.slogdet(newH)[1]),places=11)
        self.assertAlmostEqual(g.marginalization_log[0]['eliminated_precision_logdet'],logdet_v,places=11)
        torch.testing.assert_close(cov,g.covariance(one,one),atol=1e-13,rtol=1e-12)
        torch.testing.assert_close(mean,g.values[key],atol=1e-13,rtol=1e-12)
        self.assertTrue(any(f.tag=='FEJ_exact_Schur' for f in g.factors))
        g.marginalize_expired_exactly([('state',0,1)]);g.solve_sparse_map()
        torch.testing.assert_close(cov,g.covariance(one,one),atol=1e-13,rtol=1e-12)

    def test_conditional_is_not_marginal_and_simplified_limit(self):
        g=shared_graph();g.solve_sparse_map();comp=g.posterior[0];H=comp['H'];P=comp['cov']
        self.assertGreater(float(P[0,0]),float(1/H[0,0]))
        # Correlated measurement repeated twice vs falsely independent noise.
        V=np.array([[.5,.3],[.3,.5]]);h=np.ones(2)
        full=1/(.5+h@np.linalg.solve(V,h));simplified=1/(.5+2/.5)
        self.assertGreater(full,simplified)
        # Deterministic reference: the latent Schur equals independent sensor sums.
        V=np.eye(2)*.2
        self.assertAlmostEqual(1/(.5+h@np.linalg.solve(V,h)),1/(.5+2/.2))

    def test_resource_limit_never_cuts_component(self):
        g=shared_graph();g.workspace_bytes=64
        with self.assertRaisesRegex(FullResourceLimit,'FULL_RESOURCE_LIMIT'):g.solve_sparse_map()

    def test_expired_covariance_is_not_reported_as_zero_uncertainty(self):
        g=shared_graph();g.solve_sparse_map()
        g.marginalize_expired_exactly([('state',0,0)]);g.factor_posterior()
        with self.assertRaisesRegex(KeyError,'OUTSIDE_ACTIVE_WINDOW'):
            g.covariance({('state',0,0):g.tensor([[1.]])},{('state',1,0):g.tensor([[1.]])})

    def test_linear_gaussian_nees_nis_monte_carlo(self):
        from scipy.stats import chi2
        rng=np.random.default_rng(3);M=5000
        A=np.array([[1.,0.],[0.,1.],[1.,1.]])
        P=np.array([[1.,.2],[.2,2.]]);R=np.array([[.4,.1,.1],[.1,.3,.05],[.1,.05,.5]])
        S=A@P@A.T+R;post=np.linalg.inv(np.linalg.inv(P)+A.T@np.linalg.solve(R,A))
        K=P@A.T@np.linalg.inv(S)
        x=rng.multivariate_normal(np.zeros(2),P,M);y=x@A.T+rng.multivariate_normal(np.zeros(3),R,M)
        error=x-y@K.T
        nis=np.einsum('ni,ij,nj->n',y,np.linalg.inv(S),y).sum()
        nees=np.einsum('ni,ij,nj->n',error,np.linalg.inv(post),error).sum()
        for total,dof in ((nis,3*M),(nees,2*M)):
            self.assertGreater(total,chi2.ppf(.005,dof));self.assertLess(total,chi2.ppf(.995,dof))
