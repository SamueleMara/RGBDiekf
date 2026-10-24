#!/usr/bin/env python3
"""Headless browser QA against actual exported binary snapshots, no estimator."""
import argparse,functools,hashlib,http.server,json,re,shutil,subprocess,tempfile,threading,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--frame',type=int,default=0);args=parser.parse_args();view=args.run/'viewer'
    manifest=json.loads((view/'manifest.json').read_text())
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(view)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();start=time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='rgbd_grouped_chrome_') as profile:
            command=['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run','--no-default-browser-check','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader',f'--user-data-dir={profile}','--window-size=1600,1100','--virtual-time-budget=20000',f'--screenshot={view}/browser_frame_{args.frame:06d}.png','--dump-dom',f'http://127.0.0.1:{server.server_port}/index.html?qa=1&frame={args.frame}']
            result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (view/f'browser_frame_{args.frame:06d}.html').write_text(result.stdout);(view/f'browser_frame_{args.frame:06d}.log').write_text(result.stderr)
        match=re.search(r'<script type="application/json" id="qa-results">(.*?)</script>',result.stdout,re.S)
        if not match:raise RuntimeError('Browser did not complete source-frame QA; inspect browser.log and DOM')
        qa=json.loads(match.group(1));qa.update(process_seconds=time.monotonic()-start,chrome_exit_code=result.returncode,html_sha256=hashlib.sha256((view/'index.html').read_bytes()).hexdigest())
        assert qa['pass'] and result.returncode==0
        assert qa['source_frames']==len(manifest['frames']) and qa['metrics']['total']==manifest['points']
        (view/f'browser_qa_frame_{args.frame:06d}.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa,indent=2))
    finally:server.shutdown();server.server_close()


if __name__=='__main__':main()
