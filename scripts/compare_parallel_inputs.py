"""Verify shared input/code/initialization, without claiming controlled performance."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import yaml


def main():
    p=argparse.ArgumentParser();p.add_argument('--iekf',type=Path,required=True);p.add_argument('--ukf',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing comparison')
    runs=[args.iekf,args.ukf];configs=[yaml.safe_load((r/'run_config_resolved.yaml').read_text()) for r in runs]
    code=[];initial=[]
    for name,run in zip(('iekf','ukf'),runs):
        with zipfile.ZipFile(run.parent/(name+'_source.zip')) as archive:
            code.append({p:hashlib.sha256(archive.read(p)).hexdigest() for p in archive.namelist() if p.startswith('rgbd_common/') and p.endswith('.py')})
        records=[json.loads(s) for s in (run/'reports/stage_log.jsonl').read_text().splitlines()]
        initial.append(next(r['initialization_sha256'] for r in records if r['stage']=='initialization-complete'))
    report={'iekf_run':str(args.iekf),'ukf_run':str(args.ukf),'byte_identical_common_code':bool(code[0]) and code[0]==code[1],'common_module_count':len(code[0]),'identical_scene_calibration_timing_mask_validity_and_raw_fingerprints':configs[0]['scene']==configs[1]['scene'],'identical_noise_and_prior_provenance':configs[0]['provenance']==configs[1]['provenance'],'identical_frame_zero_xyz_rgb_group_sources_digest':initial[0]==initial[1],'initialization_digests':initial,'performance_comparison_controlled':False,'qualification':'Shared proposal implementation/initialization, not a claim that downstream hypotheses stay identical after different estimates. Concurrent workers do not provide an equal-budget controlled dominance comparison; full acceptance remains pending.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
