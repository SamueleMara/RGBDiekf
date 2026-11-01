#!/usr/bin/env python3
"""Reproduce the seven recorded failures on captured pre-edit and active sources.
No checkout/reset or edits of estimator mathematics. Baseline is an isolated copy.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
CASES=[
 'tests.test_f1_definitive_drift_forensics.DefinitiveArchivedDataTests.test_G0a_exact_sequential_required_points_and_covariance',
 'tests.test_image_rigid_joint.PipelineTest.test_prismatic_backward_chronological',
 'tests.test_image_rigid_joint.PipelineTest.test_prismatic_forward',
 'tests.test_image_rigid_joint.PipelineTest.test_revolute_forward',
 'tests.test_image_rigid_joint.RigidFromZeroInitTest.test_image_iekf_recovers_hinge_without_phase1_motion',
 'tests.test_image_support.SupportLabelsTest.test_labels_on_textured_scene',
 'tests.test_rendered_mask_four_pass.RenderedMaskFourPassTest.test_on_keeps_a_static_scene_static']

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 baseline=Path(tempfile.mkdtemp(prefix='rgbd_pre_temporal_regressions_'))
 shutil.copytree(REPO,baseline,dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__'))
 record=REPO/'reports/temporal_rgbd_baseline';patch=(record/'preexisting.patch').read_bytes();commit=(record/'commit.txt').read_text().strip()
 changed=[line[6:] for line in patch.decode().splitlines() if line.startswith('+++ b/')]
 for name in changed:
  (baseline/name).write_bytes(subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=REPO))
 subprocess.run(['patch','-p1','--directory',str(baseline)],input=patch,check=True,capture_output=True)
 provenance={'baseline':str(baseline),'commit':commit,'captured_pre_edit_patch_sha256':hashlib.sha256(patch).hexdigest(),
  'qualification':'Captured pre-edit dirty tracked sources, same untracked fixtures and import-only CUDA fixes. Not claimed pristine Git main.',
  'preserved_numerical_sources':{n:hashlib.sha256((baseline/n).read_bytes()).hexdigest()==hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in changed if n!='train.py'}}
 records=[]
 for index,case in enumerate(CASES):
  item={'case':case,'runs':{}}
  for label,root in [('baseline',baseline),('active',REPO)]:
   log=a.output/f'{index+1}_{label}.log'
   with log.open('w') as f:
    result=subprocess.run([sys.executable,'-B','-m','unittest',case,'-v'],cwd=root,env={**os.environ,'PYTHONPATH':str(root),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},stdout=f,stderr=subprocess.STDOUT)
   content=log.read_text();assertion=[line for line in content.splitlines() if line.startswith('AssertionError:')]
   item['runs'][label]={'returncode':result.returncode,'assertion':[line[:1500] for line in assertion],
     'full_assertion_sha256':hashlib.sha256('\n'.join(assertion).encode()).hexdigest(),'log':str(log)}
  b,c=item['runs']['baseline'],item['runs']['active']
  item['classification']='PRE_EXISTING' if b['returncode'] and c['returncode'] and b['full_assertion_sha256']==c['full_assertion_sha256'] else 'INTRODUCED' if not b['returncode'] and c['returncode'] else 'UNKNOWN'
  if not b['returncode'] and not c['returncode']:item['classification']='ENVIRONMENT_OR_SUITE_ORDER'
  records.append(item);(a.output/'regressions.json').write_text(json.dumps({'provenance':provenance,'cases':records},indent=2))
  print(case,item['classification'],flush=True)
 print(a.output/'regressions.json')
if __name__=='__main__':main()
