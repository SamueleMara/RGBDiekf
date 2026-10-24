import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pointwise_rgbd.joint_rgbd import predict,retract,initialize,fit,select
from pointwise_rgbd.config import load_config
from pointwise_rgbd.group_evidence_audit import projection_numpy
from tests.grouped_fixtures import GroupScene


def fixture(kind,angle=.1):
    scene=GroupScene(frames=7);camera=scene.cameras[0];p=scene.points[::3];K=len(scene.times)
    model={'type':kind,'axis':np.array([0.,1.,0.]),'q':np.arange(K)*(angle if kind=='REVOLUTE' else .01)}
    if kind=='REVOLUTE':model['axis_point']=np.array([-.2,0.,1.])
    rows=[[],[],[]];poses=[]
    for k in range(K):
        if kind=='PRISMATIC':R=np.eye(3);t=model['q'][k]*model['axis']
        else:R=Rotation.from_rotvec(model['q'][k]*model['axis']).as_matrix();t=model['axis_point']-R@model['axis_point']
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;poses.append(T)
        if k==0:continue
        x=predict(model,p,k,np.zeros(len(p),int));z=projection_numpy(x,camera)[0];r=load_config().measurement_covariance(z[:,2])
        for role in range(3):rows[role].append({'k':k,'p':p.copy(),'anchor':np.zeros(len(p),int),'z':z.copy(),'R':r.copy(),'camera':camera,'pixel':np.arange(len(p))+role*len(p),'camera_index':0})
    return model,rows,{'poses':np.asarray(poses)},p


class JointRGBD(unittest.TestCase):
    def test_observable_consistent_translation_has_no_inherited_bic_veto(self):
        model,groups,_,_=fixture('PRISMATIC');cfg=load_config();rows=groups[0]
        original=fit(model,rows,cfg)
        model={**model,'q':model['q']*np.sqrt(1.05*original['motion_gate']/original['motion_statistic'])}
        for row in rows:row['z']=projection_numpy(predict(model,row['p'],row['k'],row['anchor']),row['camera'])[0]
        result=fit(model,rows,cfg)
        self.assertEqual(result['status'],'FIT')
        self.assertGreater(result['motion_statistic'],result['motion_gate'])
        self.assertLess(result['motion_statistic'],len(result['information'])*np.log(3*sum(len(r['z']) for r in rows)))
        self.assertEqual(select({'PRISMATIC':result},rows,cfg)['status'],'PRISMATIC')

    def test_analytical_joint_position_derivatives(self):
        for kind in ('PRISMATIC','REVOLUTE'):
            model,_,_,p=fixture(kind);anchors=np.arange(len(p))%2
            x,g,v=predict(model,p,3,anchors,True);d=g.shape[-1]
            for j in range(d):
                step=np.zeros(d+len(model['q'])-1);step[j]=1e-6
                derivative=(predict(retract(model,step),p,3,anchors)-predict(retract(model,-step),p,3,anchors))/2e-6
                self.assertTrue(np.allclose(derivative,g[:,:,j],atol=1e-8))
            qplus={**model,'q':model['q'].copy()};qminus={**model,'q':model['q'].copy()};qplus['q'][3]+=1e-6;qminus['q'][3]-=1e-6
            self.assertTrue(np.allclose((predict(qplus,p,3,anchors)-predict(qminus,p,3,anchors))/2e-6,v,atol=1e-8))

    def test_prismatic_wins_degenerate_revolute(self):
        truth,rows,rigid,p=fixture('PRISMATIC');cfg=load_config()
        result={m['type']:fit(m,rows[0],cfg) for m in initialize(rigid,p)}
        self.assertEqual(result['PRISMATIC']['status'],'FIT')
        self.assertEqual(result['REVOLUTE']['status'],'JOINT_UNIDENTIFIABLE')
        self.assertEqual(select(result,[r for group in rows for r in group],cfg)['status'],'PRISMATIC')

    def test_revolute_finite_axis(self):
        truth,rows,rigid,p=fixture('REVOLUTE');cfg=load_config()
        result={m['type']:fit(m,rows[0],cfg) for m in initialize(rigid,p)}
        self.assertEqual(select(result,[r for group in rows for r in group],cfg)['status'],'REVOLUTE')
        actual=result['REVOLUTE']['model'];self.assertAlmostEqual(abs(actual['axis']@truth['axis']),1.,places=8)
        delta=actual['axis_point']-truth['axis_point'];self.assertLess(np.linalg.norm(np.cross(delta,truth['axis'])),1e-8)

    def test_static_unresolved(self):
        _,rows,_,p=fixture('PRISMATIC');cfg=load_config();rigid={'poses':np.tile(np.eye(4),(7,1,1))}
        for role in rows:
            for row in role:row['z']=projection_numpy(row['p'],row['camera'])[0]
        results={m['type']:fit(m,rows[0],cfg) for m in initialize(rigid,p)}
        self.assertEqual(select(results,[r for group in rows for r in group],cfg)['status'],'JOINT_TYPE_UNRESOLVED')

    def test_endpoint_relative_prediction(self):
        m,_,_,p=fixture('REVOLUTE');end=predict(m,p,6,np.zeros(len(p),int))
        for k in range(7):
            self.assertTrue(np.allclose(predict(m,end,k,np.full(len(p),6)),predict(m,p,k,np.zeros(len(p),int)),atol=1e-12))


if __name__=='__main__':unittest.main()
