import json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from test_data_loading import real_scene
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.observations import initialize_all_frame_zero
from pointwise_rgbd.process6 import process_covariance
from pointwise_rgbd.store import Ownership,TrackStore
from pointwise_rgbd.state6 import ObservationRef
from pointwise_fixtures import config


class PointwiseInputStorageTests(unittest.TestCase):
    def test_cli_metadata_pointer_cannot_write_inside_actual_input_scene(self):
        import contextlib,importlib.util,io
        script=Path(__file__).resolve().parents[1]/'tools/run_pointwise_rgbd_discovery.py'
        spec=importlib.util.spec_from_file_location('pointwise_cli_test',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);real_scene(root);output=root/'results'
            with contextlib.redirect_stderr(io.StringIO()) as error,self.assertRaises(SystemExit) as failure:
                module.main(['--scene',str(root/'metadata'),'--output',str(output),'--device','cpu'])
            self.assertEqual(failure.exception.code,2);self.assertIn('read-only source dataset',error.getvalue());self.assertFalse(output.exists())
    def test_directory_metadata_pointer_and_zip_load_same_raw_data_and_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'scene';root.mkdir();real_scene(root)
            a=Dataset(root);b=Dataset(root/'metadata')
            archive=Path(directory)/'scene.zip'
            with zipfile.ZipFile(archive,'w') as output:
                for path in root.rglob('*'):
                    if path.is_file():output.write(path,'scene/'+str(path.relative_to(root)))
            c=Dataset(archive)
            for dataset in (a,b,c):
                assert_allclose(dataset.times,[0,.04,.08]);self.assertEqual(len(dataset.cameras),2)
                x=initialize_all_frame_zero(dataset,config())
                y=initialize_all_frame_zero(a,config())
                for first,second in zip(x,y):assert_array_equal(first,second)
    def test_nonuniform_timestamp_Q_and_unrectified_distortion_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);real_scene(root)
            (root/'metadata/real_conversion.json').unlink()
            (root/'metadata/frame_timestamps.csv').write_text('frame_index,timestamp_s\n0,0\n1,.02\n2,.09\n')
            data=Dataset(root);assert_allclose(np.diff(data.times),[.02,.07])
            q1=process_covariance(.02,config().qc,config().qx)
            q2=process_covariance(.07,config().qc,config().qx)
            self.assertFalse(np.allclose(q1,q2))
            path=root/'metadata/cameras.json';doc=json.loads(path.read_text());doc['cameras'][0]['distortion']=[.1,0,0,0,0];path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'UNSUPPORTED_UNRECTIFIED_DISTORTION'):Dataset(root)
    def test_resume_checks_raw_content_and_measurement_references_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'scene';root.mkdir();real_scene(root);dataset=Dataset(root)
            state,p,color,source=initialize_all_frame_zero(dataset,config())
            store=TrackStore(Path(directory)/'output/initial_forward/trajectory_store',dataset,len(state),(color,source))
            store.write_frame(0,state,p,np.ones(len(state),bool),np.full(len(state),np.nan),np.ones(len(state),np.uint8))
            store.write_measurements(0,[{'point':np.flatnonzero(source[:,1]==ci),'pixel':source[source[:,1]==ci,2],'camera_index':ci} for ci in range(2)])
            store.validate_completed_inputs(1)
            file=next((root/'depth_npy').rglob('frame_000000.npy'));depth=np.load(file);depth.flat[0]+=.001;np.save(file,depth)
            dataset.observation.cache_clear()
            with self.assertRaisesRegex(ValueError,'RESUME_RAW_OBSERVATION_CHANGED'):store.validate_completed_inputs(1)
            store.freeze()
            with self.assertRaisesRegex(ValueError,'frozen'):store.write_measurements(0,[])
    def test_failed_and_committed_seed_reservations_are_released(self):
        from pointwise_fixtures import PointScene
        scene=PointScene();ref=ObservationRef(4,0,scene.hidden_pixels[4,0])
        with tempfile.TemporaryDirectory() as directory:
            owner=Ownership(scene,Path(directory)/'owners.sqlite');point=owner.allocate_id(2)
            owner.reserve(ref,point);self.assertEqual(owner.db.execute('SELECT count(*) FROM reservations').fetchone()[0],1)
            owner.release_reservations(point);self.assertEqual(owner.db.execute('SELECT count(*) FROM reservations').fetchone()[0],0)
            owner.reserve(ref,point)
            owner.claim([(ref.frame,ref.camera,np.array([ref.pixel]),np.array([point]))],point_record=(point,'unused.npz',{}))
            self.assertEqual(owner.db.execute('SELECT count(*) FROM reservations').fetchone()[0],0)
            with self.assertRaisesRegex(ValueError,'owned'):owner.reserve(ref,owner.allocate_id(2))


if __name__=='__main__':unittest.main()
