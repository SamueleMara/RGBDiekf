"""Verify preserved frame-0 initialization under the corrected common operator."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from rgbd_common.backends.torch_cuda import Backend
from rgbd_common.config import load_config
from rgbd_common.dataloader import load_dataset
from rgbd_common.frame_cache import CachedDataset
from rgbd_common.fusion import fuse_frame_zero
from rgbd_common.geometry import PointMap
from rgbd_common.outer_loop import model_digest
from rgbd_common.projection_cuda import PROJECTION_OPERATOR_VERSION


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--experiments',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    backend=Backend('cuda');dataset=CachedDataset(load_dataset(a.input),a.experiments/'iekf/raw_cache')
    current,details=fuse_frame_zero(dataset,load_config(),backend);digest=model_digest(current);rows=[]
    for name in ('iekf','ukf'):
        previous=PointMap.load(a.experiments/name/'initial_points.npz')
        rows.append({'pipeline':name,'initial_points':previous.count,'sha256':model_digest(previous),'identical_to_current_operator':model_digest(previous)==digest})
    result={'projection_operator_version':PROJECTION_OPERATOR_VERSION,'current_initial_points':current.count,'current_sha256':digest,'camera_fusion':details,'pipeline_checks':rows,'all_identical':all(r['identical_to_current_operator'] for r in rows),'gpu':backend.memory()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['all_identical'] else 1


if __name__=='__main__':sys.exit(main())
