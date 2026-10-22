"""Refresh measured acceptance while the explicitly launched real workers run."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    reporter=Path(__file__).with_name('report_masked_v2_status.py')
    log=args.root/'reports/status_monitor.log';log.parent.mkdir(exist_ok=True)
    while True:
        result=subprocess.run([sys.executable,'-B',str(reporter),'--root',str(args.root)],text=True,capture_output=True)
        with log.open('a') as stream:stream.write(result.stdout+result.stderr)
        if result.returncode:return result.returncode
        report=json.loads((args.root/'reports/v2_current_acceptance.json').read_text())
        if not any(branch['worker_active_at_report'] for branch in report['independent_experiments']):
            report['workers_finished_unix']=time.time()
            (args.root/'reports/final_measured_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'workers_finished':True,'all_acceptance_criteria_passed':report['all_acceptance_criteria_passed']}),flush=True)
            return 0 if report['all_acceptance_criteria_passed'] else 1
        time.sleep(60)


if __name__=='__main__':raise SystemExit(main())
