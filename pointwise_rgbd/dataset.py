from functools import lru_cache
import json
import time
import numpy as np
from rgbd_common.dataloader import load_dataset


class Dataset:
    def __init__(self,path,frame_limit=None,camera_limit=None,**timing):
        self.base=load_dataset(path,**timing)
        calibration=json.loads(self.base.files.read('metadata/cameras.json'))
        for camera in calibration['cameras']:
            distortion=camera.get('distortion_coefficients',camera.get('distortion'))
            if isinstance(distortion,dict):distortion=list(distortion.values())
            if distortion is not None and np.any(np.asarray(distortion,dtype=float)!=0) and not camera.get('rectified',False):
                raise ValueError('UNSUPPORTED_UNRECTIFIED_DISTORTION: supply rectified RGB/depth/masks and matching K; nonzero distortion must not be silently ignored')
        self.root=self.base.root
        self.frame_ids=self.base.frame_ids[:frame_limit];self.times=self.base.times[:frame_limit]
        self.cameras=self.base.cameras[:camera_limit]
        if len(self.times)<1 or len(self.cameras)<1:raise ValueError('Empty selected sequence')
        self.image_io_seconds=0.;self.image_loads=0
        self.observation=lru_cache(maxsize=len(self.cameras))(self._observation)
        self.test_limits={'frame_limit':frame_limit,'camera_limit':camera_limit}

    def _observation(self,k,c):
        start=time.monotonic();frame=self.base.observation(k,c)
        self.image_io_seconds+=time.monotonic()-start;self.image_loads+=1
        return frame

    def manifest(self):
        return {'input':str(self.root),'original_frame_ids':self.frame_ids,'timestamps_seconds':self.times.tolist(),
            'timing':self.base.timing_report,'test_limits':self.test_limits,'metadata':self.base.metadata_report,
            'cameras':[{'name':c.name,'size':[c.width,c.height],'intrinsics':c.intrinsics.tolist(),'world_to_camera':c.world_to_camera.tolist()} for c in self.cameras]}
