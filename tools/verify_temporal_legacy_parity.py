#!/usr/bin/env python3
"""Compare the captured PRE-EDIT dirty train.py against the new legacy default.
The baseline is reconstructed in /tmp from the recorded commit and dirty patch;
no working-tree checkout/reset. Fixture-only, both directions, every output array.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.image_fixtures import two_plate_scene
from tests.projective_fixtures import write_scene
from utils.io import json_write
import train


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=Path('reports/temporal_rgbd_baseline/parity.json'));ap.add_argument('--device',default='cpu');a=ap.parse_args()
    record=Path('reports/temporal_rgbd_baseline');commit=(record/'commit.txt').read_text().strip()
    patch=(record/'preexisting.patch').read_bytes()
    with tempfile.TemporaryDirectory(prefix='rgbd_legacy_parity_') as tmp:
        root=Path(tmp);baseline=root/'source';baseline.mkdir()
        files=[s[6:] for s in patch.decode().splitlines() if s.startswith('+++ b/')]
        for f in files:
            p=baseline/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','show',f'{commit}:{f}']))
        subprocess.run(['patch','-p1','--directory',str(baseline)],input=patch,check=True,capture_output=True)
        preserved={f:hashlib.sha256((baseline/f).read_bytes()).hexdigest()==hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files if f!='train.py'}
        if not all(preserved.values()):raise AssertionError(preserved)
        spec=importlib.util.spec_from_file_location('preedit_train',baseline/'train.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.REPO=train.REPO
        ds=two_plate_scene('prismatic',frames=3,cameras=2,width=32,height=24,speed=.004)
        scene=write_scene(ds,root/'scene')
        args=['--scene',str(scene),'--fps','30','--device',a.device,'--stop-after','free','--direction','both','--sensor-noise','config',
              '--debug','none','--debug-frames','0,1,2','--save-full-covariance','--save-state-interval','1','--no-evaluate','--no-qa']
        old.main([*args,'--output',str(root/'before')]);train.main([*args,'--output',str(root/'after')])
        checks=[]
        for before in sorted((root/'before').rglob('*.npy')):
            relative=before.relative_to(root/'before');b=np.load(before);c=np.load(root/'after'/relative)
            equal=np.array_equal(b,c,equal_nan=True)
            checks.append({'file':str(relative),'shape':list(b.shape),'bitwise_equal':bool(equal)})
            if not equal:raise AssertionError(f'legacy changed: {relative}')
        # Full covariance snapshots, not just stored diagonal summaries.
        snapshots=0
        for before in sorted((root/'before').rglob('*.npz')):
            rel=before.relative_to(root/'before');p=np.load(before);q=np.load(root/'after'/rel)
            for key in p.files:
                if not np.array_equal(p[key],q[key],equal_nan=True):raise AssertionError(f'legacy snapshot changed: {rel}:{key}')
            snapshots+=1
        result={'baseline_commit':commit,'baseline':'captured pre-edit dirty working tree, not clean HEAD','device':a.device,
                'preserved_existing_sources':preserved,'checks':checks,'npz_snapshots_equal':snapshots,'status':'PASS','directions':['forward','backward'],'frames':3}
        json_write(a.output,result);print(a.output)

if __name__=='__main__':main()
