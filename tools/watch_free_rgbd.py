#!/usr/bin/env python3
"""Serve actual RGB-D/point trajectories and refresh completed checkpoints."""
import argparse,functools,http.server,json,os,sys,threading,time,traceback
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pointwise_rgbd.dataset import Dataset
from pointwise_rgbd.config import load_config
from pointwise_rgbd.free_live_visualization import refresh,read_json
from pointwise_rgbd.store import json_write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8765);p.add_argument('--once',action='store_true');args=p.parse_args();root=args.run.resolve()
    manifest=read_json(root/'manifest.json')
    if manifest is None:p.error('The run has not written manifest.json yet')
    o=manifest['arguments'];data=Dataset(o['scene'],o.get('frame_limit'),o.get('camera_limit'),fps=o.get('fps'),dt_seconds=o.get('dt_seconds'),timestamps_file=o.get('timestamps_file'));config=load_config(o.get('noise_config'))
    view=root/'viewer';view.mkdir(exist_ok=True)
    if args.once:
        refresh(root,data,config);return 0
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
        def end_headers(self):self.send_header('Cache-Control','no-store');super().end_headers()
    handler=functools.partial(Handler,directory=str(view))
    try:server=http.server.ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    except OSError:server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
    url=f'http://127.0.0.1:{server.server_port}/'
    json_write(view/'server.json',{'url':url,'pid':os.getpid(),'run':str(root),'read_only_estimation':True})
    print(f'Live visual debug: {url} (actual saved RGB-D and point estimates)',flush=True)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        while True:
            try:
                refresh(root,data,config)
                if (view/'refresh_error.json').exists():(view/'refresh_error.json').unlink()
            except (OSError,ValueError,KeyError,EOFError) as error:
                json_write(view/'refresh_error.json',{'error':str(error),'traceback':traceback.format_exc(),'note':'Checkpoint may be in transit; previous complete visualization remains available.'})
            time.sleep(2)
    except KeyboardInterrupt:return 0
    finally:server.shutdown();server.server_close()


if __name__=='__main__':raise SystemExit(main())
