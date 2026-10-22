"""Preserve audit/browser evidence separately from reconstruction acceptance."""
import argparse,json,shutil
from html.parser import HTMLParser
from pathlib import Path


class Body(HTMLParser):
    def handle_starttag(self,tag,attrs):
        if tag=='body':self.attributes=dict(attrs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiments',required=True,type=Path);a=p.parse_args()
    reports=a.experiments/'reports';reports.mkdir(exist_ok=True);repo=Path(__file__).resolve().parents[1]
    for name in ('AUDIT.md','COVARIANCE_AUDIT.md','VALIDATION.md','README.md'):shutil.copyfile(repo/name,reports/name)
    browser=reports/'browser_checks';browser.mkdir(exist_ok=True)
    for name in ('rgbdekf_parallel_motion.png','rgbdekf_parallel_motion_last_frame.png','rgbdekf_parallel_motion_test_dom.html','rgbdekf_parallel_motion_test_browser.log'):
        source=Path('/tmp')/name
        if source.is_file():shutil.copyfile(source,browser/name)
    dom=browser/'rgbdekf_parallel_motion_test_dom.html'
    if dom.exists():
        parser=Body();parser.feed(dom.read_text());attrs=parser.attributes
        result={'plotted_coordinates_match_saved_frame':attrs.get('data-plot-matches-frame')=='true','selected_frame':int(attrs['data-slider-frame']),
                'group_color_mode':attrs.get('data-group-color-mode'),'sample_max_estimated_displacement_m':float(attrs['data-max-point-displacement']),
                'qualification':'Offline renderer/control check on provisional actual estimates. Does not verify physical motion accuracy or full-scene reconstruction.'}
        if not result['plotted_coordinates_match_saved_frame']:raise AssertionError('Browser plot does not match selected saved frame')
        (browser/'browser_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(str(reports))


if __name__=='__main__':main()
