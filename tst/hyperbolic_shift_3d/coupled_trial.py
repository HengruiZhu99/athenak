"""Immutable, fresh, <=410-second subprocesses for the coupled 3D prototype."""
import argparse, hashlib, json, os, shlex, shutil, subprocess, sys, tarfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,required=True)
    p.add_argument('--kind',choices=['minkowski','puncture','pulse'],required=True)
    p.add_argument('--target',type=float,required=True);p.add_argument('--dt',type=float,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--adaptive',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=(a.build/'tst/hyperbolic_shift_3d/hyperbolic_shift_3d_coupled').resolve()
    libraries=list((a.build/'tst/hyperbolic_shift_3d').glob('libhyperbolic_shift_3d_native.*'))
    lib=next(x.resolve() for x in libraries if x.suffix in ['.dylib','.so'])
    binding=json.loads((a.build/'coupled-source.json').read_text())
    assert digest(exe)==binding['binary_sha256'],'Executable changed after source binding'
    for rel,sha in binding['files'].items():assert digest(ROOT/rel)==sha,('Rebuild and freeze C++ source before running',rel)
    shutil.copy2(a.build/'coupled-source.tar.gz',out/'compiled-source.tar.gz')
    shutil.copy2(a.build/'coupled-source.json',out/'compiled-source.json')
    frozen=out/'hyperbolic_shift_3d_coupled';shutil.copy2(exe,frozen)
    files=sorted(p for directory in ['src/z4c/hyperboloidal','src/mesh/cubed_sphere','tst/hyperbolic_shift_3d','tst/hyperbolic_shift_1d','tst/elliptic_shift_1d'] for p in (ROOT/directory).glob('*') if p.is_file())
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:
        for path in files:archive.add(path,arcname=str(path.relative_to(ROOT)))
    listing=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    inventory=[]
    for line in listing.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        try:args=shlex.split(fields[1])
        except ValueError:continue
        if not args:continue
        executable=Path(args[0]).name.lower()
        scripts=[Path(x).name for x in args[1:] if x.endswith('.py')]
        if executable=='hyperbolic_shift_3d_coupled' or ('python' in executable and any(x in ['baseline.py','run_segment.py','frozen_continuation.py','continue_puncture.py','run_telegraph.py'] for x in scripts)):
            inventory.append(dict(pid=int(fields[0]),executable=executable,scripts=scripts))
    assert len(inventory)<2,'Two evolution processes are already running'
    manifest=dict(status='preparing',pid=os.getpid(),arguments=vars(a)|{'build':str(a.build),'output':str(a.output)},
        base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        compiled_source_archive_sha256=digest(out/'compiled-source.tar.gz'),binary_sha256=digest(frozen),library_sha256=digest(lib),source_archive_sha256=digest(out/'source.tar.gz'),
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in files},process_inventory=inventory,wall_seconds_limit=410)
    def record(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    record();start=time.monotonic()
    try:
        # Coordinate preparation is not a continuing evolution or a restart.
        with (out/'mesh.log').open('w') as log:
            subprocess.run([str(frozen),str(a.n),str(out/'mesh'),'-','0',str(a.dt),'0'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=410)
        initial='-'
        if a.kind=='puncture':
            initial=str(out/'initial.txt')
            with (out/'initial.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'tst/hyperbolic_shift_3d/initial_data.py'),'--coordinates',str(out/'mesh.coordinates.csv'),'--library',str(lib),'--output',initial],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=410)
        elif a.kind=='pulse':initial='pulse'
        command=[str(frozen),str(a.n),str(out/'evolution'),initial,str(a.target),str(a.dt),str(int(a.adaptive)),'.125',str(a.workers)]
        manifest.update(status='running',command=command);record()
        with (out/'evolution.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            manifest['child_pid']=child.pid;record()
            try:returncode=child.wait(timeout=410)
            except subprocess.TimeoutExpired:
                child.kill();child.wait();manifest['status']='external_timeout';record();raise
        result=json.loads((out/'evolution.json').read_text())
        manifest.update(status=result['status'],returncode=returncode,result=result)
    except Exception as ex:
        if manifest['status']!='external_timeout':manifest['status']='setup_or_run_error'
        manifest['error']=str(ex)
    finally:
        manifest['seconds']=time.monotonic()-start
        manifest['evidence_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json']}
        record()
    print(json.dumps({k:manifest[k] for k in ['status','seconds','result','error'] if k in manifest}))
    if manifest['status']!='completed':raise SystemExit(2)
if __name__=='__main__':main()
