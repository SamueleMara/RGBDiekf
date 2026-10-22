"""Run the installed standard-library tests; no pytest installation required."""
import argparse,contextlib,hashlib,json,os,sys,time,unittest,zipfile
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,ElementTree


class Result(unittest.TextTestResult):
    def startTest(self,test):
        super().startTest(test);self.started=time.monotonic()
        self.case=SubElement(self.xml,'testcase',name=test.id())
    def stopTest(self,test):
        self.case.set('time',str(time.monotonic()-self.started));super().stopTest(test)
    def addFailure(self,test,error):
        super().addFailure(test,error);SubElement(self.case,'failure').text=self.failures[-1][1]
    def addError(self,test,error):
        super().addError(test,error);SubElement(self.case,'error').text=self.errors[-1][1]
    def addSkip(self,test,reason):
        super().addSkip(test,reason);SubElement(self.case,'skipped',message=reason)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--input',type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];sys.path[:0]=[str(repo),str(repo/'tests')]
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'test_results.json').exists():raise FileExistsError('Preserving prior results; choose a fresh test output directory')
    digest=hashlib.sha256()
    files=sorted([*repo.glob('main*.py'),*repo.glob('rgbd_common/**/*.py'),*repo.glob('iekf/**/*.py'),*repo.glob('ukf/**/*.py'),*repo.glob('tests/**/*.py'),*repo.glob('scripts/*.py'),*repo.glob('configs/*')])
    manifest={}
    for file in files:
        if not file.is_file():continue
        relative=str(file.relative_to(repo));data=file.read_bytes();digest.update(relative.encode());digest.update(data);manifest[relative]=hashlib.sha256(data).hexdigest()
    (a.output/'source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if a.input:os.environ['RGBD_REAL_SCENE']=str(a.input)
    os.environ['RGBD_TEST_REPORTS']=str(a.output)
    import torch
    Result.xml=Element('testsuite');start=time.monotonic()
    with (a.output/'tests.log').open('w') as log,contextlib.redirect_stderr(log),contextlib.redirect_stdout(log):
        suite=unittest.defaultTestLoader.discover(str(repo/'tests'))
        result=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=Result).run(suite)
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),'runtime_seconds':time.monotonic()-start,
            'cuda_available':torch.cuda.is_available(),'real_scene':str(a.input) if a.input else None,'success':result.wasSuccessful() and not result.skipped,
            'runner':'Python standard-library unittest; pytest is not installed in the existing environment',
            'failed_tests':[test.id() for test,_ in result.failures+result.errors],'source_sha256':digest.hexdigest(),'source_manifest':'source_hashes.json'}
    for key in ('tests','failures','errors','skipped'):Result.xml.set(key,str(report[key]))
    ElementTree(Result.xml).write(a.output/'junit.xml',encoding='unicode',xml_declaration=True)
    (a.output/'test_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    cases=[json.loads(path.read_text()) for path in sorted((a.output/'mask_port_cases').glob('*.json'))]
    mask_report={'specification':'Parallel_Masked_RGBD_IEKF_UKF_Theory_CUDA_v2.md M1--M8; implementation contract Section15',
                 'source_sha256':digest.hexdigest(),'cuda_available':torch.cuda.is_available(),'assertion_cases':cases,
                 'unit_suite_success':report['success'],'failed_tests':report['failed_tests'],
                 'real_reconstruction_acceptance':'NOT_ESTABLISHED_BY_TEST_SUITE; see independent experiment stage logs',
                 'all_acceptance_criteria_passed':False}
    (a.output/'MASK_PORT_VALIDATION.json').write_text(json.dumps(mask_report,indent=2)+'\n')
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
