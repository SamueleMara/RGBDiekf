"""Run independent full-scene experiments; preserve commands, source and logs."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--pipelines',nargs='+',choices=['iekf','ukf'],default=['iekf','ukf']);p.add_argument('--max-outer',type=int)
    p.add_argument('--resume',action='store_true',help='Interrupt only this launcher\'s saved worker PID, preserve files and resume its accepted checkpoint')
    p.add_argument('--rebase-legacy-projection-score',action='store_true')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True);records=[]
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    for name in a.pipelines:
        output=a.output/name
        if output.exists() and not a.resume:raise FileExistsError(f'Preserving existing experiment: {output}')
        interrupted=None
        if a.resume:
            checkpoint=output/'checkpoint.json'
            if not checkpoint.exists():raise FileNotFoundError('No accepted checkpoint to resume')
            pid_file=a.output/f'{name}.pid'
            if pid_file.exists():
                pid=int(pid_file.read_text());proc=Path(f'/proc/{pid}/cmdline')
                if proc.exists():
                    tokens=proc.read_bytes().decode().split('\0')
                    if '--output' not in tokens or tokens[tokens.index('--output')+1]!=str(output):raise RuntimeError('PID belongs to another command; refusing to interrupt')
                    if not any(Path(token).name==f'main_{name}.py' for token in tokens):raise RuntimeError('Worker entry does not match selected independent estimator')
                    raise RuntimeError('A reconstruction worker is still running. It has not been interrupted; resume requires it to finish or separate user authorization to stop it.')
            archive_dir=output/'reports'/f'before_resume_{time.time_ns()}';archive_dir.mkdir(parents=True)
            for artifact in ('validation_report.json','error.log'):
                existing=output/artifact
                if existing.exists():(archive_dir/artifact).write_bytes(existing.read_bytes())
            meta=json.loads(checkpoint.read_text())
            for key,fallback in [('point_file','checkpoint_points.npz'),('trajectory_file','checkpoint_trajectory.npz')]:
                with zipfile.ZipFile(output/meta.get(key,fallback)) as archive:
                    if archive.testzip() is not None:raise RuntimeError('Incomplete checkpoint: preserving files, resume refused')
        command=[sys.executable,'-B',str(repo/f'main_{name}.py'),'--input',str(a.input),'--output',str(output)]
        if a.resume:command+=['--resume']
        if a.rebase_legacy_projection_score:command+=['--rebase-legacy-projection-score']
        if a.max_outer is not None:command+=['--max-outer',str(a.max_outer)]
        files=sorted([*repo.glob('main*.py'),*repo.glob('rgbd_common/**/*.py'),*repo.glob('iekf/**/*.py'),*repo.glob('ukf/**/*.py'),*repo.glob('configs/*')])
        digest=hashlib.sha256()
        tag=f'{name}_resume_{time.time_ns()}' if a.resume else name
        source_hashes={}
        for file in files:
            if not file.is_file():continue
            relative=str(file.relative_to(repo));data=file.read_bytes();digest.update(relative.encode());digest.update(data);source_hashes[relative]=hashlib.sha256(data).hexdigest()
        manifest=a.output/'source_hashes.json'
        if manifest.exists() and json.loads(manifest.read_text())!=source_hashes:
            raise RuntimeError('Source has changed; preserve this experiment and use a fresh output directory')
        manifest.write_text(json.dumps(source_hashes,indent=2)+'\n')
        specifications=['CODEX_IMPLEMENT_PARALLEL_IEKF_UKF_RGBD_SLAM.md','Parallel_RGBD_IEKF_UKF_Theory_CUDA.pdf','00_COMMON_SOLVED_UNSOLVED_SPEC.md','01_IEKF_THEORY.md','02_UKF_THEORY.md','03_CUDA_AND_IMPLEMENTATION_THEORY.md','CODEX_PARALLEL_IEKF_UKF_HARD_FOREGROUND_MASKS.md','Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md']
        record={'pipeline':name,'command':command,'source_sha256':digest.hexdigest(),'specification_sha256':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in specifications},'started_unix':time.time(),'resume_from_accepted_checkpoint':a.resume,'interrupted_own_worker_pid':interrupted}
        (a.output/f'{tag}_command.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
        with (a.output/f'{tag}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT)
            (a.output/f'{name}.pid').write_text(str(process.pid)+'\n');record['exit_code']=process.wait()
        record['elapsed_seconds']=time.time()-record['started_unix']
        report=output/'validation_report.json'
        if report.exists():
            validation=json.loads(report.read_text())
            record['validation_file']=str(report)
            record['validation']={key:validation.get(key) for key in ('pipeline','status','complete_real_cycle','frames','cameras','runtime_seconds')}
        (a.output/f'{name}_experiment.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record);print(json.dumps(record),flush=True)
        (a.output/f'{tag}_experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    current=[json.loads(path.read_text()) for path in (a.output/f'{name}_experiment.json' for name in ('iekf','ukf')) if path.exists()]
    (a.output/'experiments.json').write_text(json.dumps(current,indent=2)+'\n')
    return int(any(row['exit_code'] not in (0,3) for row in records))


if __name__=='__main__':sys.exit(main())
