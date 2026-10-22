import tempfile,unittest,json,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal,assert_allclose
from rgbd_common.backends.torch_cuda import Backend
from pointwise_rgbd.controller import run_exact_pointwise_discovery,match_all_frozen
from pointwise_rgbd.discovery import track_one_point
from pointwise_rgbd.state6 import ObservationRef
from pointwise_rgbd.observations import initialize_seed
from pointwise_fixtures import PointScene,config


class PointwiseDiscoveryTests(unittest.TestCase):
    def test_late_invisible_rigid_part_is_discovered_as_individual_moving_points(self):
        scene=PointScene(frames=7,hidden=False)
        offsets=np.array([[-.15,0,0],[0,.15,0],[.15,-.10,0]])
        late_pixels=set()
        for k in range(2,7):
            theta=.13*(k-2);rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
            points=offsets@rotation.T+[-.45+.004*k,-.3,2.5]
            for ci,camera in enumerate(scene.cameras):
                image=scene.observation(k,ci)
                for i,point in enumerate(points):
                    q=camera.world_to_camera[:3,:3]@point+camera.world_to_camera[:3,3]
                    x,y=np.floor(q[:2]/q[2]*camera.intrinsics[:2]+camera.intrinsics[2:]+.5).astype(int)
                    image.depth[y,x]=q[2];image.mask[y,x]=True;image.rgb[y,x]=np.eye(3)[i]
                    if k==6:late_pixels.add((ci,y*camera.width+x))
        with tempfile.TemporaryDirectory() as root:
            store,owner=run_exact_pointwise_discovery(scene,config(),Backend('cpu'),root,max_events=3)
            self.assertEqual(store.n0,2);self.assertEqual(len(store.added),3)
            for row in store.added:
                k,ci,pixel=row['source'];self.assertEqual(k,6);self.assertIn((ci,pixel),late_pixels)
                with np.load(row['path']) as track:
                    self.assertGreater(int(track['observed'].sum()),1)
                    self.assertEqual(track['state'].shape,(7,6));self.assertNotIn('group',track.files)
    def test_estimator_import_graph_has_no_rigid_or_Lie_state_or_smoother(self):
        import ast
        package=Path(__file__).resolve().parents[1]/'pointwise_rgbd'
        forbidden=('rgbd_iekf','rgbd_ukf','rgbd_common.lie','rgbd_common.group','estimator','gaussian','rasterizer')
        for file in package.glob('*.py'):
            tree=ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):names=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom):names=[node.module or '']
                else:continue
                self.assertFalse(any(name.startswith(forbidden) for name in names),str(file))
    def test_multiple_additions_run_initial_once_and_one_optimizer_per_new_point(self):
        from pointwise_rgbd.controller import initial_forward
        scene=PointScene()
        for ci in range(2):
            image=scene.observation(4,ci)
            for x,y,z in ((40,10,1.6),(50,35,2.)):
                image.mask[y,x]=True;image.depth[y,x]=z;image.rgb[y,x]=[.3,.4,.9]
        with tempfile.TemporaryDirectory() as root,patch('pointwise_rgbd.controller.initial_forward',wraps=initial_forward) as initial,patch('pointwise_rgbd.controller.track_one_point',wraps=track_one_point) as individual:
            store,owner=run_exact_pointwise_discovery(scene,config(),Backend('cpu'),root,max_events=3)
            self.assertEqual(len(store.added),3);self.assertEqual(initial.call_count,1);self.assertEqual(individual.call_count,3)
            starts=[json.loads(row) for row in (Path(root)/'discovery/events.jsonl').read_text().splitlines() if json.loads(row)['event']=='START']
            self.assertEqual([row['model_version'] for row in starts],[0,1,2]);self.assertEqual(len({row['point_id'] for row in starts}),3)
            self.assertEqual(owner.db.execute('SELECT count(*) FROM reservations').fetchone()[0],0)
    def test_initial_once_hidden_seed_anchor_once_frozen_old_tracks_and_restart(self):
        scene=PointScene();c=config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            store,owner=run_exact_pointwise_discovery(scene,c,b,root,max_events=1)
            self.assertEqual(owner.meta('initial_invocations'),1);self.assertEqual(store.n0,2)
            self.assertEqual(len(store.added),1)
            record=store.added[0];source=ObservationRef(*record['source'])
            self.assertEqual(source,ObservationRef(4,0,scene.hidden_pixels[4,0]))
            event=json.loads((Path(root)/f"discovery/events/{record['point_id']:012d}/seed.json").read_text())
            self.assertEqual(event['anchor_uses'],1);self.assertEqual(event['backward_frames'],[3,2,1,0]);self.assertEqual(event['forward_frames'],[])
            signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in store.directory.glob('*.npy')}
            before=np.array(store.arrays['state']);before_p=np.array(store.arrays['covariance'])
            again,owner2=run_exact_pointwise_discovery(scene,c,b,root,max_events=1,resume=True)
            assert_array_equal(again.arrays['state'],before);assert_array_equal(again.arrays['covariance'],before_p)
            self.assertEqual(signature,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in store.directory.glob('*.npy')})
            self.assertEqual(owner2.meta('initial_invocations'),1)
            self.assertTrue(all(not a.flags.writeable for a in again.arrays.values()))
    def test_anchor_two_independent_branches_no_velocity_sign_flip_or_RTS(self):
        scene=PointScene();c=config();b=Backend('cpu')
        from pointwise_rgbd.store import Ownership
        with tempfile.TemporaryDirectory() as root:
            owner=Ownership(scene,Path(root)/'owners.sqlite');ref=ObservationRef(2,0,scene.hidden_pixels[2,0])
            track,claims,color=track_one_point(0,ref,scene,c,b,owner)
            self.assertEqual(track['backward_frames'],[1,0]);self.assertEqual(track['forward_frames'],[3,4]);self.assertEqual(track['anchor_uses'],1)
            anchor,p,_=initialize_seed(ref,scene,c)
            assert_array_equal(track['state'][2],anchor);assert_array_equal(track['covariance'][2],p)
            refs=[(k,ci,int(u)) for k,ci,us,_ in claims for u in us]
            self.assertEqual(refs.count((2,0,ref.pixel)),1);self.assertEqual(len(set(refs)),len(refs))
            self.assertFalse(track['observed'][0]);self.assertGreater(np.trace(track['covariance'][0,:3,:3]),np.trace(p[:3,:3]))
    def test_failed_seed_registry_is_versioned_and_point_ids_never_reused(self):
        from pointwise_rgbd.store import Ownership
        scene=PointScene();ref=ObservationRef(4,0,scene.hidden_pixels[4,0])
        with tempfile.TemporaryDirectory() as root:
            owner=Ownership(scene,Path(root)/'owners.sqlite');owner.fail(ref,0,'numeric failure')
            self.assertTrue(owner.attempted(ref,0));self.assertFalse(owner.attempted(ref,1))
            a=owner.allocate_id(2);b=owner.allocate_id(2);self.assertEqual(b,a+1)
            self.assertNotEqual(owner.first_unclaimed(0),ref)
    def test_real_numeric_visual_debug_outputs_are_readable(self):
        scene=PointScene();c=config();b=Backend('cpu')
        with tempfile.TemporaryDirectory() as root:
            store,owner=run_exact_pointwise_discovery(scene,c,b,root,max_events=1)
            root=Path(root);point=store.added[0]['point_id'];event=root/f'discovery/events/{point:012d}'
            for path in ('initial_forward/summary.json','initial_forward/nis_by_frame.csv','initial_forward/matching_histogram.json','initial_forward/diagnostics.html','initial_forward/trajectory_viewer.html','discovery/events.jsonl','discovery/matching_by_frame.csv','discovery/nearest_residuals.csv','discovery/geometry_growth.html','final/summary.json','final/point_tracks.json','final/unexplained_observations.csv'):
                self.assertTrue((root/path).is_file(),path)
            for path in ('seed.json','state_by_frame.csv','associations.csv','optimizer_by_frame.json','trajectory.png','covariance_nis.png','local_3d.html'):
                self.assertTrue((event/path).is_file(),path)
            from PIL import Image
            for path in event.glob('*.png'):
                with Image.open(path) as image:image.verify()
            html=(event/'local_3d.html').read_text()
            self.assertIn('id="frame"',html);self.assertIn('id="event"',html);self.assertIn('PREDICTION_ONLY',html)
            self.assertIn('100',html);self.assertNotIn('placeholder',html.lower())
            self.assertTrue(list(event.glob('image_overlay_*.png')));self.assertTrue(list(event.glob('depth_residual_*.png')))
            rows=json.loads((event/'optimizer_by_frame.json').read_text());self.assertTrue(rows)
            self.assertTrue(all(r['energy_after']<=r['energy_before'] for r in rows))
            self.assertIn('invalid_foreground_pixels',(root/'discovery/matching_by_frame.csv').read_text().splitlines()[0])
    def test_single_observation_seed_is_immediately_solved_without_confirmation(self):
        scene=PointScene();scene.images[2][0].mask[:]=False;scene.images[3][0].mask[:]=False
        scene.images[2][1].mask[:]=False;scene.images[3][1].mask[:]=False
        with tempfile.TemporaryDirectory() as root:
            store,owner=run_exact_pointwise_discovery(scene,config(),Backend('cpu'),root,max_events=1)
            self.assertEqual(len(store.added),1)


if __name__=='__main__':unittest.main()
