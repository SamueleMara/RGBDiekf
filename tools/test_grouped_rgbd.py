#!/usr/bin/env python3
"""Execute grouped contract tests, retaining individual results and source hashes."""
import argparse,hashlib,json,sys,time,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'tests'))
from tools.test_pointwise_rgbd import RecordedResult


class GroupRecordedResult(RecordedResult):
    def addSuccess(self,test):
        super().addSuccess(test)
        if hasattr(test,'metrics'):self.records[-1]['metrics']=test.metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern='test_grouped_*.py');start=time.monotonic()
    with (args.output/'tests.log').open('w') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=GroupRecordedResult).run(suite)
    import torch
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'successful':result.wasSuccessful(),'seconds':time.monotonic()-start,'cuda_available':torch.cuda.is_available(),'records':result.records,'source_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(REPO/'pointwise_rgbd').glob('*.py'),*(REPO/'tests').glob('test_grouped_*.py')]}}
    (args.output/'tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','records')}));return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
