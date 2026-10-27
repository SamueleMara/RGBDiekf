#!/usr/bin/env python3
"""Run the contract gate suites and write reports/rigeuler_rgbd_test_matrix.csv (test, contract section, outcome, seconds)."""
import argparse, csv, io, sys, time, unittest
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SUITES = {'tests.test_image_renderer': '3 measurement / renderer', 'tests.test_image_iekf': '2, 4, 5 process / Phase-1 IEKF / covariance',
          'tests.test_image_coherence': '7 coherence', 'tests.test_image_rigid_joint': '8, 9 rigid / joint', 'tests.test_image_support': '10, 11 bidirectional / support'}


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.rows = []; self._t = {}

    def startTest(self, test):
        self._t[test.id()] = time.monotonic(); super().startTest(test)

    def _row(self, test, outcome, detail=''):
        self.rows.append({'test': test.id(), 'outcome': outcome, 'seconds': round(time.monotonic()-self._t.get(test.id(), time.monotonic()), 2),
                          'detail': detail.strip().splitlines()[-1][:300] if detail else ''})

    def addSuccess(self, test): super().addSuccess(test); self._row(test, 'PASS')
    def addFailure(self, test, err): super().addFailure(test, err); self._row(test, 'FAIL', self._exc_info_to_string(err, test))
    def addError(self, test, err): super().addError(test, err); self._row(test, 'ERROR', self._exc_info_to_string(err, test))
    def addSkip(self, test, reason): super().addSkip(test, reason); self._row(test, 'SKIP', reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suites', nargs='*', default=list(SUITES)); ap.add_argument('--out', type=Path, default=REPO/'reports/rigeuler_rgbd_test_matrix.csv')
    a = ap.parse_args()
    rows = []
    for name in a.suites:
        suite = unittest.defaultTestLoader.loadTestsFromName(name)
        runner = unittest.TextTestRunner(stream=io.StringIO(), resultclass=Recorder, verbosity=0)
        res = runner.run(suite)
        for r in res.rows: rows.append({'contract_section': SUITES.get(name, ''), **r})
        print(name, {o: sum(r['outcome'] == o for r in res.rows) for o in ('PASS', 'FAIL', 'ERROR', 'SKIP')}, flush=True)
    a.out.parent.mkdir(exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['contract_section', 'test', 'outcome', 'seconds', 'detail']); w.writeheader(); w.writerows(rows)
    return 0 if all(r['outcome'] in ('PASS', 'SKIP') for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
