#!/usr/bin/env python3
"""Run only the new pointwise contract tests and retain machine-readable evidence."""
import argparse,hashlib,json,platform,sys,time,unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'tests'))


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.records=[]
    def startTest(self,test):self.start=time.monotonic();super().startTest(test)
    def record(self,test,status,detail=None):
        self.records.append({'test':test.id(),'status':status,'seconds':time.monotonic()-self.start,'detail':detail})
    def addSuccess(self,test):super().addSuccess(test);self.record(test,'PASS')
    def addError(self,test,error):super().addError(test,error);self.record(test,'ERROR',self._exc_info_to_string(error,test))
    def addFailure(self,test,error):super().addFailure(test,error);self.record(test,'FAIL',self._exc_info_to_string(error,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.record(test,'SKIP',reason)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(REPO/'tests'),pattern='test_pointwise*.py')
    start=time.monotonic()
    with (args.output/'tests.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordedResult).run(suite)
    import torch
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'successful':result.wasSuccessful(),'seconds':time.monotonic()-start,'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'records':result.records,'source_hashes':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*(REPO/'pointwise_rgbd').glob('*.py'),*(REPO/'tests').glob('*pointwise*.py'),REPO/'tools/run_pointwise_rgbd_discovery.py'])},'scope':'22 October 2026 pointwise contract; no optional 6D UKF implemented; no legacy rigid-framework pass claim'}
    (args.output/'tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('tests_run','failures','errors','skips','successful','seconds','cuda_available')}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
