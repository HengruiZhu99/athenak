#!/usr/bin/env python3
"""ARCHIVED campaign helper for frozen pre-cleanup binaries only.

One bounded science segment; no self-submission or production restart."""
import argparse,hashlib,json,os,re,struct,subprocess,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def header(path):
    with path.open('rb') as f:
        text=bytearray()
        while not text.endswith(b'<par_end>\n'):
            b=f.read(1)
            if not b:raise ValueError('truncated parameters')
            text+=b
        fixed=f.read(struct.calcsize('=ii9d38i2di'))
        nmb=struct.unpack_from('=i',fixed)[0]
        cycle=struct.unpack_from('=i',fixed,len(fixed)-4)[0]
        t,dt=struct.unpack_from('=2d',fixed,len(fixed)-20)
        atlas=f.read(struct.unpack('=Q',f.read(8))[0])
        f.seek(nmb*24,1);meta=struct.unpack('=8Q',f.read(64));assert meta[0]==0x4859504552525354
        identity=f.read(meta[2]).decode();stride=struct.unpack('=Q',f.read(8))[0]
        assert not meta[7] and stride==meta[3]*8
        assert path.stat().st_size==f.tell()+nmb*stride
    return {'time':t,'time_M0':3*t,'dt':dt,'cycle':cycle,'identity':identity,'blocks':nmb}

def main():
    p=argparse.ArgumentParser()
    for name in ['root','run']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wrapper',required=True);p.add_argument('--binding',required=True)
    p.add_argument('--n',type=int,choices=[12,16],required=True)
    p.add_argument('--mode',choices=['legacy','masked','agreement'],required=True)
    p.add_argument('--restart',type=Path)
    p.add_argument('--walltime-seconds',type=int,default=3120)
    a=p.parse_args();root=a.root.resolve();run=a.run.resolve();case='N'+str(a.n)+'-'+a.mode
    assert 300<=a.walltime_seconds<=35520
    assert run.parent==root/'science'/case and not (run/'provenance.json').exists()
    manifest=json.loads((root/'science-inputs/manifest.json').read_text())
    inp=run/'input.athinput';assert sha(inp)==manifest['files'][case+'.athinput']
    target={12:21300,16:28400}[a.n];dt={12:.00046944203735617445,16:.00035208152801713085}[a.n]
    assert 3*target*dt<30 and 3*(target+4)*dt>30
    env=[s.split() for s in (run/'rank-environment.txt').read_text().splitlines()]
    assert len(env)==24 and len({r[0] for r in env})==2
    for host in {r[0] for r in env}:
        rows=[r for r in env if r[0]==host]
        assert len(rows)==12 and len({r[2] for r in rows})==12 and {int(r[1]) for r in rows}==set(range(12))
    assert all(r[3:8]==['1','1','specialized','limited','128'] for r in env)
    cpus=[set(range(int(s.split('-')[0]),int(s.split('-')[1])+1)) for s in a.binding.split(':')]
    assert all(set(map(int,r[8].split(',')))==cpus[int(r[1])] for r in env)
    build=root/('build-baseline' if a.mode=='legacy' else 'qualified-candidate-v4')
    subprocess.run(['sha256sum','-c',str(build/'executable.sha256')],check=True)
    exe=build/'src/athena';args=['-i',str(inp)]
    initial=json.loads((root/'science-inputs'/('initial-data-N'+str(a.n)+'.json')).read_text())
    datadir=Path(re.search(r'(?m)^binary_directory\s*=\s*(.+)$',inp.read_text())[1].strip())
    assert sha(datadir/'manifest.json')==manifest['files']['initial-data-N'+str(a.n)+'.json']
    assert (root/('export-check-N'+str(a.n)+'.txt')).read_text().startswith('PASS ')
    provenance={'mode':a.mode,'n':a.n,'native_step_limit':target,'maximum_time_M0':3*target*dt,
        'executable_sha256':sha(exe),'input_sha256':sha(inp),'source_manifest_sha256':sha(build/'source-sha256.json'),
        'initial_data_identity':initial['identity'],'initial_data_manifest_sha256':sha(datadir/'manifest.json'),
        'jobid':os.environ['PBS_JOBID'],'files':{f.name:sha(f) for f in [run/'nodes.txt',run/'modules.txt',run/'rank-environment.txt',run/'hybrid-science-two.pbs',Path(__file__)]}}
    startcycle=0
    if a.restart:
        src=a.restart.resolve();assert src.is_file() and src.parent.name=='rst'
        assert src.parent.parent.parent==root/'science'/case, 'restart must belong to this isolated configuration'
        h=header(src);assert h['cycle']<target and h['dt']==dt and h['time_M0']<30
        previous=json.loads((src.parent.parent/'provenance.json').read_text())
        assert previous['executable_sha256']==provenance['executable_sha256'] and previous['input_sha256']==provenance['input_sha256']
        if a.mode=='legacy':assert 'activity-policy=' not in h['identity']
        else:assert ';overlap-mode='+{'masked':'1','agreement':'2'}[a.mode] in h['identity']
        args=['-r',str(src),*args];startcycle=h['cycle'];provenance['restart']={**h,'path':str(src),'sha256':sha(src)}
    walltime='%02d:%02d:%02d'%(a.walltime_seconds//3600,a.walltime_seconds//60%60,a.walltime_seconds%60)
    command=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+a.binding,a.wrapper,str(exe),*args,'-d',str(run),'-t',walltime]
    provenance['command']=command;(run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    begin=time.monotonic()
    # Trackers open during problem initialization, before AthenaK handles -d.
    # Start in the isolated segment directory so those early files stay local.
    with (run/'run.log').open('w') as f:result=subprocess.run(command,cwd=run,stdout=f,stderr=subprocess.STDOUT,timeout=a.walltime_seconds+330)
    (run/'exit-code.txt').write_text(str(result.returncode)+'\n');result.check_returncode()
    checkpoints=[(header(f),f) for f in (run/'rst').glob('*.rst')]
    assert checkpoints
    checkpoints.sort(key=lambda x:x[0]['cycle']);last,path=checkpoints[-1]
    assert startcycle<last['cycle']<=target and last['time_M0']<=30
    assert abs(last['time']-last['cycle']*dt)<1e-9
    summary={'wall_seconds':time.monotonic()-begin,'initial_cycle':startcycle,'last':last,
       'last_checkpoint':str(path),'last_checkpoint_sha256':sha(path),'complete':last['cycle']==target,
       'histories':{f.name:sha(f) for f in run.glob('*.hst')}}
    (run/'segment-result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
