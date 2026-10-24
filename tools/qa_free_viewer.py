#!/usr/bin/env python3
"""Check real WebGL buffers after actual asynchronous completion, using Chrome CDP."""
import argparse,base64,functools,http.server,json,os,select,subprocess,sys,tempfile,threading,time
from pathlib import Path
from urllib.parse import urlencode


class Pipe:
    def __init__(self,read,write):self.read=read;self.write=write;self.buffer=b'';self.sequence=0
    def call(self,method,params=None,session=None,timeout=60):
        self.sequence+=1;number=self.sequence;request={'id':number,'method':method,'params':params or {}}
        if session:request['sessionId']=session
        wire=json.dumps(request).encode()+b'\0'
        while wire:wire=wire[os.write(self.write,wire):]
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            while b'\0' in self.buffer:
                raw,self.buffer=self.buffer.split(b'\0',1)
                if not raw:continue
                answer=json.loads(raw)
                if answer.get('id')==number:
                    if 'error' in answer:raise RuntimeError(answer['error'])
                    return answer.get('result',{})
            if select.select([self.read],[],[],max(0,deadline-time.monotonic()))[0]:
                chunk=os.read(self.read,1024*1024)
                if not chunk:raise RuntimeError('Chrome CDP pipe closed')
                self.buffer+=chunk
        raise TimeoutError(f'Chrome did not finish {method}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--frame',type=int,default=0)
    p.add_argument('--direction',choices=['first','last'],default='first');p.add_argument('--mode',choices=['free','rigid','canonical','joint'],default='free');p.add_argument('--colors',choices=['rgb','groups','core','observed','labels'],default='rgb')
    args=p.parse_args();view=args.run/'viewer';tag=f'{args.frame:06d}_{args.direction}_{args.mode}_{args.colors}';query=urlencode(dict(qa=1,frame=args.frame,direction=args.direction,mode=args.mode,colors=args.colors))
    # Preserve failed harness evidence when retrying the same diagnostic.
    for suffix in ('html','log','png','json'):
        old=view/f'qa_{tag}.{suffix}'
        if old.exists():
            i=1
            while old.with_name(old.name+f'.attempt{i}').exists():i+=1
            old.rename(old.with_name(old.name+f'.attempt{i}'))
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(view)));threading.Thread(target=server.serve_forever,daemon=True).start()
    process=None;read_in=write_in=read_out=write_out=None;start=time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='rgbd_free_chrome_') as profile,(view/f'qa_{tag}.log').open('w') as log:
            read_in,write_in=os.pipe();read_out,write_out=os.pipe()
            chrome=['/usr/bin/google-chrome','--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run','--no-default-browser-check','--enable-unsafe-swiftshader','--use-gl=angle','--use-angle=swiftshader',f'--user-data-dir={profile}','--window-size=1600,1100','--remote-debugging-pipe','about:blank']
            # A fresh single-threaded child installs Chrome's required pipe
            # FDs. Avoid preexec_fn in the threaded HTTP-server process.
            wrapper='import os,sys;os.dup2(int(sys.argv[1]),3);os.dup2(int(sys.argv[2]),4);os.execv(sys.argv[3],sys.argv[3:])'
            process=subprocess.Popen([sys.executable,'-c',wrapper,str(read_in),str(write_out),*chrome],pass_fds=(read_in,write_out),stdout=subprocess.DEVNULL,stderr=log)
            os.close(read_in);read_in=None;os.close(write_out);write_out=None;pipe=Pipe(read_out,write_in)
            target=pipe.call('Target.createTarget',{'url':'about:blank'})['targetId'];session=pipe.call('Target.attachToTarget',{'targetId':target,'flatten':True})['sessionId']
            pipe.call('Page.enable',session=session);pipe.call('Page.navigate',{'url':f'http://127.0.0.1:{server.server_port}/index.html?{query}'},session)
            expression="""(async()=>{let deadline=Date.now()+45000;while(Date.now()<deadline){let q=document.getElementById('qa-results');if(q)return {qa:JSON.parse(q.textContent),audit:window.viewerAudit};let e=document.getElementById('error');if(e&&e.textContent)throw Error(e.textContent);await new Promise(r=>setTimeout(r,100));}throw Error('Actual viewer QA readiness timeout');})()"""
            answer=pipe.call('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True},session)
            if 'exceptionDetails' in answer:raise RuntimeError(answer['exceptionDetails'])
            ready=answer['result']['value'];qa=ready['qa'];qa.update(geometry=ready['audit']['geometry'],colors=args.colors,process_seconds=time.monotonic()-start)
            qa['pass']&=qa['frame']==args.frame and qa['direction']==args.direction and qa['geometry']==args.mode
            shot=pipe.call('Page.captureScreenshot',{'format':'png'},session)['data'];(view/f'qa_{tag}.png').write_bytes(base64.b64decode(shot))
            dom=pipe.call('Runtime.evaluate',{'expression':'document.documentElement.outerHTML','returnByValue':True},session)['result']['value'];(view/f'qa_{tag}.html').write_text(dom)
            (view/f'qa_{tag}.json').write_text(json.dumps(qa,indent=2)+'\n');print(json.dumps(qa,indent=2))
            try:pipe.call('Browser.close',timeout=5)
            except (RuntimeError,TimeoutError):pass
            process.wait(timeout=10)
            return 0 if qa['pass'] else 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for fd in (read_in,write_in,read_out,write_out):
            if fd is not None:os.close(fd)
        server.shutdown();server.server_close()


if __name__=='__main__':raise SystemExit(main())
