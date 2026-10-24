#!/usr/bin/env python3
"""Inventory local corrective sources and the exact reference revision."""
import argparse,ast,hashlib,json,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
REFERENCE=REPO.parent/'RigEuler3DGS'
REF='481b2cf2d5602932286948cf14114f9caee4de3c'
BASELINE=('controller','process6','filter_iekf','association','observations','projection','dataset','store')
REF_FILES=('estimator/supernodes.py','estimator/sparse_iekf.py','estimator/motion_coherence_support.py','estimator/coherence_extraction.py','estimator/rigid_part_runtime.py','estimator/rigid_part_se3.py','joint_reduction/rigid_trajectory_reduction.py','tools/kpart_pipeline.py')


def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    local={}
    files=set([*REPO.glob('pointwise_rgbd/*.py'),*REPO.glob('pointwise_rgbd/assets/*viewer.html'),*REPO.glob('tests/*grouped*py'),*REPO.glob('tests/test_free_*.py'),*REPO.glob('tools/*group*py'),*REPO.glob('tools/*free*py'),*REPO.glob('specs/free_grouped_rgbd/*'),REPO/'tools/audit_correction_sources.py'])
    for path in files:
        value={'sha256':sha(path.read_bytes()),'lines':len(path.read_text().splitlines())}
        if path.suffix=='.py':value['definitions']=[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno} for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
        local[str(path.relative_to(REPO))]=value
    baseline={}
    for name in BASELINE:
        relative=f'pointwise_rgbd/{name}.py';raw=(REPO/relative).read_bytes();committed=git(REPO,'show','HEAD:'+relative)
        baseline[relative]={'sha256':sha(raw),'identical_to_HEAD':raw==committed}
    if not all(v['identical_to_HEAD'] for v in baseline.values()):raise AssertionError('Baseline estimator changed')
    reference={}
    for relative in REF_FILES:
        old=git(REFERENCE,'show',REF+':'+relative);current=(REFERENCE/relative).read_bytes()
        reference[relative]={'requested_revision_sha256':sha(old),'actual_checkout_sha256':sha(current),'matches_requested_revision':old==current}
    status=git(REPO,'status','--short').decode();diff=git(REPO,'diff').decode()
    manifest={'branch':git(REPO,'branch','--show-current').decode().strip(),'HEAD':git(REPO,'rev-parse','HEAD').decode().strip(),'uncommitted':True,'baseline':baseline,'sources':local,'reference_requested_revision':REF,'reference_actual_HEAD':git(REFERENCE,'rev-parse','HEAD').decode().strip(),'reference_files':reference}
    (args.output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'git_status.txt').write_text(status);(args.output/'tracked_changes.patch').write_text(diff)
    print(json.dumps({'baseline_unchanged':True,'source_files':len(local),'reference_changed_files':[k for k,v in reference.items() if not v['matches_requested_revision']]}))


if __name__=='__main__':main()
