"""Interrupt only the exact two workers recorded by our experiment launcher."""
import argparse,json,os,signal,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiments',required=True,type=Path);parser.add_argument('--reason',required=True);parser.add_argument('--pipelines',nargs='+',choices=['iekf','ukf'],default=['iekf','ukf']);args=parser.parse_args();records=[]
    for name in args.pipelines:
        pid_file=args.experiments/f'{name}.pid'
        if not pid_file.exists():continue
        pid=int(pid_file.read_text());proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists():continue
        tokens=proc.read_bytes().decode().split('\0');output=args.experiments/name
        if '--output' not in tokens or tokens[tokens.index('--output')+1]!=str(output) or not any(Path(t).name==f'main_{name}.py' for t in tokens):raise RuntimeError('Worker identity differs; refusing to interrupt')
        checkpoint=output/'checkpoint.json'
        record={'pipeline':name,'pid':pid,'command':tokens,'reason':args.reason,'checkpoint':json.loads(checkpoint.read_text()) if checkpoint.exists() else None}
        os.kill(pid,signal.SIGTERM);records.append(record)
    target=args.experiments/f'preserved_workers_{time.time_ns()}.json';target.write_text(json.dumps(records,indent=2)+'\n');print(target);print(json.dumps(records,indent=2))


if __name__=='__main__':main()
