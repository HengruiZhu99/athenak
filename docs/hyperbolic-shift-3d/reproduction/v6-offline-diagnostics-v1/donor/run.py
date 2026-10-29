from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
donor=json.loads((root/'docs/hyperbolic-shift-3d/reproduction/larger-block-initial-data/n24/provenance.json').read_text())
sources=dict(donor['donor_source_sha256'])
sources['tst/hyperbolic_shift_3d/baseline.py']=sha(root/'tst/hyperbolic_shift_3d/baseline.py')
sources['tst/hyperbolic_shift_3d/initial_data.py']=sha(root/'tst/hyperbolic_shift_3d/initial_data.py')
for path,digest in sources.items():assert sha(root/path)==digest,path
lib=root/'build-hyperbolic-core-shell/tst/hyperbolic_shift_3d/libhyperbolic_shift_3d_native.dylib'
assert sha(lib)=='a25a42547c9d1f0dcb303114cb51b21e88d2d653fac38bab0ffba0c80a436c0e'
args=[sys.executable,str(root/'tst/hyperbolic_shift_3d/baseline.py'),'--n','513','--target','1','--kind','puncture','--library',str(lib),'--output',str(out/'n513')]
record=dict(command=args,sources=sources,library_sha256=sha(lib),native_limit_seconds=400,external_limit_seconds=420,controller_sha256=sha(Path(__file__)))
(out/'source.json').write_text(json.dumps(record,indent=2)+'\n');began=time.monotonic()
with (out/'run.log').open('xb') as log:
 p=subprocess.Popen(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record['pid']=p.pid;(out/'process.json').write_text(json.dumps(record,indent=2)+'\n')
 try:record['returncode']=p.wait(timeout=420);record['timed_out']=False
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM);record['returncode']=p.wait();record['timed_out']=True
record['seconds']=time.monotonic()-began
record['sources_unchanged']=all(sha(root/path)==digest for path,digest in sources.items())
record['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((out/'n513').glob('*')) if p.is_file()}
(out/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['pid','returncode','timed_out','seconds','sources_unchanged']}))
