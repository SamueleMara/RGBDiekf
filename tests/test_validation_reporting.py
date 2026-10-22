import json,os,tempfile,unittest
from pathlib import Path
from scripts.validate_parallel_runs import inspect


class ValidationReportingTests(unittest.TestCase):
    def test_resumed_cycles_use_latest_outer_and_final_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);(path/'reports').mkdir();rows=[]
            for outer in (0,3):
                rows.extend({'stage':'forward-frame','label':f'outer-{outer}','frame':k,'score_epoch':2} for k in range(3))
                rows.append({'stage':'outer-complete','outer':outer,'score_epoch':2})
                rows.extend({'stage':'forward-frame','label':'final-new-map','frame':k,'score_epoch':2} for k in range(3))
            log=path/'reports/stage_log.jsonl'
            def save():
                log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                (path/'validation_report.json').write_text(json.dumps({'status':'NON_CONVERGED','frames':3,'cameras':1,'projection_operator_version':2}))
            save();self.assertTrue(inspect(path,'IEKF')['acceptance_gates']['all_forward_frames'])
            rows.append({'stage':'forward-frame','label':'final-new-map','frame':0,'score_epoch':2})
            save();self.assertFalse(inspect(path,'IEKF')['acceptance_gates']['all_forward_frames'])

    def test_old_failed_report_does_not_describe_a_newer_incomplete_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);(path/'reports').mkdir()
            (path/'validation_report.json').write_text(json.dumps({'status':'FAILED','reason':'old projection guard'}))
            os.utime(path/'validation_report.json',(1,1))
            (path/'reports/stage_log.jsonl').write_text(json.dumps({'stage':'outer-start','outer':3,'score_epoch':2})+'\n')
            (path/'reports/dataset_audit.json').write_text(json.dumps({'frames':328,'cameras':3}))
            result=inspect(path,'IEKF')
            self.assertEqual(result['status'],'INCOMPLETE_LATEST_ATTEMPT')
            self.assertEqual(result['projection_operator_version'],2)
            self.assertEqual(result['frames'],328)
            self.assertEqual(result['previous_attempt_report']['reason'],'old projection guard')
            self.assertFalse(result['all_acceptance_gates_pass'])
