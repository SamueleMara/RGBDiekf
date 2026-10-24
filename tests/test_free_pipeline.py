import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from pointwise_rgbd.config import load_config
from pointwise_rgbd.free_phase1 import run
from pointwise_rgbd.joint_rgbd import fit
from pointwise_rgbd.joint_support import decide
from pointwise_rgbd.joint_bidirectional import unique_pool
from pointwise_rgbd.reference_coherence import extract_moving_component
from pointwise_rgbd.reference_motion import edge_strain
from tests.grouped_fixtures import GroupScene
from tests.test_free_joint_rgbd import fixture
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.array_store import ArrayStore


class PipelineTests(unittest.TestCase):
    def test_static_last_endpoint_gauge_is_explicit_not_inferred_from_positions(self):
        from pointwise_rgbd.free_coherence import run as coherence
        from pointwise_rgbd.shared_rigid_rgbd import run as rigid
        scene=GroupScene(frames=4,cameras=8);cfg=load_config();backend=Backend('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);partition,arrays=run(scene,cfg,backend,root/'phase1','last',64)
            result=coherence(partition,arrays,scene.times,root/'coherence',32,48,anchor_frame=3)
            self.assertEqual(result['anchor_frame'],3)
            result=rigid(partition,arrays,np.ones(len(partition.xyz),bool),scene,cfg,backend,root/'phase1',root/'rigid',anchor_frame=3)
            self.assertEqual(result['anchor_frame'],3);np.testing.assert_array_equal(result['poses'][3],np.eye(4))
            np.testing.assert_array_equal(result['covariance'][3],np.zeros((6,6)))
            self.assertGreater(abs(result['map_fit']['map_state']['Sinv']).max(),0)
            self.assertEqual(result['production_camera_indices'],list(range(8)))
            self.assertEqual(set(result['complete_multiview']['per_camera']),set(map(str,range(8))))

    def test_rigid_endpoint_uses_other_camera_pixels_without_reusing_priors(self):
        from pointwise_rgbd.shared_rigid_rgbd import run as rigid
        scene=GroupScene(frames=3,cameras=8);cfg=load_config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);part,arrays=run(scene,cfg,b,root/'phase','first',64)
            support=part.source[:,1]==0
            result=rigid(part,arrays,support,scene,cfg,b,root/'phase',root/'rigid',anchor_frame=0)
            self.assertEqual(result['status'],'FIT')
            store=ArrayStore(root/'rigid/factors.sqlite');row=store.read('0');store.close()
            self.assertEqual(set(row['camera']),set(range(1,8)))
            source={(int(ci),int(pixel)) for _,ci,pixel in part.source[support]}
            self.assertFalse(source & {(int(ci),int(pixel)) for ci,pixel in zip(row['camera'],row['pixel'])})
            self.assertGreater(len(row['z']),0)

    def test_reference_hierarchy_finds_core_without_displacement_labeling(self):
        from scipy.spatial import cKDTree
        rng=np.random.default_rng(43);x=rng.normal(size=(40,3))*.03;x[20:,0]+=.08
        trajectory=np.stack([x.copy() for k in range(12)])
        for k in range(12):trajectory[k,20:,1]+=.006*k
        _,near=cKDTree(x).query(x,k=13)
        edges=np.unique(np.sort(np.c_[np.repeat(np.arange(40),12),near[:,1:].ravel()],axis=1),axis=0)
        traj=torch.tensor(trajectory);a=torch.tensor(edges[:,0]);b=torch.tensor(edges[:,1]);strain=edge_strain(traj,a,b)
        result=extract_moving_component(traj,a,b,strain['C'],40,refine=False,verbose=False)
        self.assertEqual(result['status'],'RESOLVED');moving=result['moving_nodes'].numpy()
        self.assertGreater(len(moving),2);self.assertTrue((moving>=20).all())

    def test_measured_prismatic_fit_completes_all_duplicate_endpoint_samples(self):
        from pointwise_rgbd.observations import samples
        from pointwise_rgbd.surface_supernodes import build_partition
        from pointwise_rgbd.group_partition_validation import weighted_kabsch
        from pointwise_rgbd.joint_rgbd import initialize,select
        from pointwise_rgbd.joint_support import run as complete
        scene=GroupScene(frames=5,motion='translation',cameras=8);cfg=load_config()
        source=samples(scene.observation(0,0),cfg);lookup={tuple(rgb):i for i,rgb in enumerate(source.rgb)}
        rows=[[] for _ in scene.cameras];poses=[np.eye(4)]
        for k in range(1,5):
            for ci in range(len(scene.cameras)):
                measured=samples(scene.observation(k,ci),cfg);ids=np.array([lookup[tuple(rgb)] for rgb in measured.rgb])
                rows[ci].append({'k':k,'p':source.xyz[ids],'anchor':np.zeros(len(ids),int),'z':measured.z,'R':measured.R,'camera':scene.cameras[ci],'camera_index':ci,'pixel':measured.pixel})
                if ci==0:poses.append(weighted_kabsch(source.xyz[ids],measured.xyz))
        fits={m['type']:fit(m,[r for group in rows for r in group],cfg) for m in initialize({'poses':np.stack(poses)},source.xyz)}
        decision=select(fits,[r for group in rows for r in group],cfg);self.assertEqual(decision['status'],'PRISMATIC')
        self.assertEqual(decision['measurement_camera_indices'],list(range(8)))
        for candidate in decision['candidates'].values():
            if 'complete_multiview' in candidate:self.assertEqual(set(candidate['complete_multiview']['per_camera']),set(map(str,range(8))))
        with tempfile.TemporaryDirectory() as tmp:
            _,partition=build_partition(scene,'first',cfg)
            result=complete(partition,scene,cfg,Backend('cpu'),fits['PRISMATIC'],0,tmp)
            self.assertEqual(result['labels']['MOVING_JOINT_COMPATIBLE'],len(partition.xyz))
            self.assertEqual(result['new_points_inserted'],0)
            self.assertEqual(result['production_camera_indices'],list(range(8)))
            self.assertTrue(all(f['processed_camera_indices']==list(range(8)) for f in result['frames']))

    def test_both_endpoint_passes_signed_time_spd_foreground_and_all_eight_cameras(self):
        scene=GroupScene(frames=4,motion='translation',cameras=8);config=load_config();backend=Backend('cpu')
        with tempfile.TemporaryDirectory() as tmp:
            for name,anchor in [('first',0),('last',3)]:
                root=Path(tmp)/name;p,a=run(scene,config,backend,root,name,64)
                self.assertEqual(a['state'].shape,(4,len(p.xyz),6))
                self.assertTrue(np.allclose(a['xyz'][anchor],p.xyz,atol=0,rtol=0))
                self.assertTrue((np.linalg.eigvalsh(a['covariance'])>0).all())
                import json
                metadata=json.loads((root/'run.json').read_text())
                self.assertEqual(metadata['camera_policy'],'all_available')
                self.assertEqual(metadata['production_camera_indices'],list(range(8)))
                store=ArrayStore(root/'factors.sqlite')
                for k in range(4):
                    if k==anchor:continue
                    report=json.loads((root/f'filter/{k:06d}.json').read_text())
                    self.assertEqual([v['camera_index'] for v in report['association']],list(range(8)))
                    row=store.read(str(k));self.assertEqual(set(row['camera']),set(range(8)))
                    identities=set()
                    for ci,pixel in zip(row['camera'],row['pixel']):
                        self.assertTrue(scene.observation(k,int(ci)).valid.ravel()[pixel])
                        identities.add((int(ci),int(pixel)))
                    self.assertEqual(len(identities),len(row['pixel']))
                store.close()
                self.assertGreater(np.max(abs(a['xyz'][3-anchor]-p.xyz)),0)
                self.assertFalse((root/'group_poses.npy').exists())

    def test_camera_split_cache_cannot_be_reused(self):
        import json
        from types import SimpleNamespace
        from tools.run_free_grouped_rgbd_joint import load_phase1
        scene=SimpleNamespace(manifest=lambda:{'scene':'same'})
        config=load_config()
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);phase=source/'first/phase1';phase.mkdir(parents=True)
            (source/'manifest.json').write_text(json.dumps({'dataset':scene.manifest(),'noise':config.manifest()}))
            old={'state':'individual XYZ + physical-time velocity, no group pose state','anchor':'first','occupancy':64,
                 'covariance':'No block assumed-density projection','folds':'camera index modulo 3'}
            (phase/'run.json').write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,'camera partition'):load_phase1(source,'first',scene,config,64)

    def test_support_near_axis_occluded_and_mask_infeasible_stay_honest(self):
        labels=decide(np.array([20.,1.,1.,1.,1.]),np.array([1.,20.,.9,20.,.1]),np.array([5,5,5,0,5]),np.array([1,1,0,1,1],bool),np.zeros(5,bool),np.array([0,0,0,0,1],bool))
        np.testing.assert_array_equal(labels,[2,1,0,0,1])

    def test_raw_direction_pool_never_double_counts(self):
        _,roles,_,_=fixture('REVOLUTE');a=roles[0][0]
        a.update(phase1_residual=np.zeros_like(a['z']))
        b={**a,'phase1_residual':np.ones_like(a['z']),'anchor':np.full(len(a['p']),6)}
        pool,report=unique_pool([[a],[b]],{0,6})
        self.assertEqual(report['unique_raw_returns'],len(a['z']));self.assertEqual(report['duplicate_direction_factors_removed'],len(a['z']))
        self.assertEqual(sum(len(r['z']) for r in pool),len(a['z']))
        self.assertTrue((pool[0]['anchor']==0).all())

    def test_static_and_shared_global_drift_have_no_articulated_core(self):
        rng=np.random.default_rng(310);x=rng.normal(size=(15,3));edges=np.array([(i,j) for i in range(15) for j in range(i+1,15)])
        for drift in (0.,.01):
            traj=torch.tensor(np.stack([x+[drift*k,0,0] for k in range(6)]),dtype=torch.float64)
            a=torch.tensor(edges[:,0]);b=torch.tensor(edges[:,1]);strain=edge_strain(traj,a,b)
            result=extract_moving_component(traj,a,b,strain['C'],15,verbose=False)
            self.assertEqual(result['status'],'NO_COHERENT_MOVING_COMPONENT')

    def test_negligible_motion_not_promoted_to_a_joint(self):
        model,rows,_,_=fixture('REVOLUTE',angle=1e-10)
        result=fit(model,rows[0],load_config())
        self.assertEqual(result['status'],'JOINT_UNIDENTIFIABLE')


if __name__=='__main__':unittest.main()
