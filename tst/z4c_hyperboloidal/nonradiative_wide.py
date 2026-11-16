#!/usr/bin/env python3
"""Bounded wide-pulse follow-up, reusing the unchanged evolution executable."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def require_completed_run(folder):
    """Accept explicit analysis recovery without erasing the original failure."""
    code=int((folder/'exit-code.txt').read_text())
    if code==0:return
    assert (folder/'evolution-exit.txt').read_text().strip()=='0'
    recovery=json.loads((folder/'analysis-recovery.json').read_text())
    receipt=json.loads((folder/'receipt.json').read_text())
    assert recovery['exit_status']==0 and recovery['original_exit_status']==code
    assert recovery['checkpoint_before_sha256']==recovery['checkpoint_after_sha256']==receipt['final_sha256']

def evolve(root,out,n):
    baseline=root.parent/'nonradiative-20261115'
    build=json.loads((baseline/'build/receipt.json').read_text())
    binary=baseline/'build/athena';assert sha(binary)==build['executables']['athena']
    seed=root/'seed/nonradiative_seed';seed_receipt=json.loads((root/'seed/receipt.json').read_text())
    assert sha(seed)==seed_receipt['executable_sha256']
    mpi=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+os.environ['NONRAD_BINDING'],os.environ['NONRAD_WRAPPER']]
    receipt={'baseline_executable':str(binary),'baseline_executable_sha256':sha(binary),
             'seed_receipt_sha256':sha(root/'seed/receipt.json'),'driver_sha256':sha(__file__),
             'pulse':{'center':.86,'half_width':.08,'shape':6,'amplitude':.001},'cases':[]}
    checkpoint=[]
    for family in ('control','gauge'):
        name=f'{family}-N{n}';case=out/name;case.mkdir();(case/'seed').mkdir()
        text=(baseline/'inputs'/f'{name}.athinput').read_text()
        cycles=1704*n//12
        text=re.sub(r'^nlim = \d+$',f'nlim = {cycles}',text,flags=re.M)
        # N12 gauge retains dense records to measure cadence effects directly.
        stride=1 if n==12 and family=='gauge' else n//4
        text=text.replace('dcycle=1',f'dcycle={stride}')
        inp=case/'input.athinput';inp.write_text(text)
        command=mpi+[str(seed),str(inp),'.001' if family=='gauge' else '0','.86','.08','6']
        with (case/'seed/run.log').open('w') as log:
            subprocess.run(command,cwd=case/'seed',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        initial=next((case/'seed/rst').glob('*.rst'));checkpoint.append(initial)
        command2=mpi+[str(binary),'-r',str(initial),'-i',str(inp),'-d',str(case),'-t','00:54:00']
        start=time.monotonic()
        with (case/'run.log').open('w') as log:
            subprocess.run(command2,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3000)
        match=re.findall(r'^time=([^ ]+) cycle=(\d+)$',(case/'run.log').read_text(),re.M)
        assert len(match)==1 and int(match[0][1])==cycles,match
        final=sorted((case/'rst').glob('*.rst'))[-1]
        receipt['cases'].append(dict(name=name,seconds=time.monotonic()-start,commands=[command,command2],
            input_sha256=sha(inp),seed=str(initial),seed_sha256=sha(initial),final=str(final),
            final_sha256=sha(final),cycles=cycles,time=float(match[0][0]),record_stride=stride))
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt['cases'][-1]),flush=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('nonradiative_seed_check.py')),
                    *map(str,checkpoint),str(out/'seed-check.json')],check=True)
    if n==16 and os.environ.get('REFERENCE_PROBE')=='1':
        # Separate control: corrected reference jets, identical initial values.
        import numpy as np
        from restart_roundtrip import parse,payload
        prior=list((root/'runs').glob('N12-*/control-N12'));assert len(prior)==1,prior
        prior=prior[0];case=out/'reference-control-N12';case.mkdir()
        inp=case/'input.athinput';inp.write_bytes((prior/'input.athinput').read_bytes())
        fixed=root/'reference-fix/build/athena'
        fixed_receipt=json.loads((root/'reference-fix/build/receipt.json').read_text())
        assert sha(fixed)==fixed_receipt['executables']['athena']
        command=mpi+[str(fixed),'-i',str(inp),'-d',str(case),'-t','00:54:00']
        start=time.monotonic()
        with (case/'run.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        rst=sorted((case/'rst').glob('*.rst'));assert len(rst)==2
        before=parse(next((prior/'seed/rst').glob('*.rst')));initial=parse(rst[0]);final=parse(rst[-1])
        assert initial['cycle']==0 and final['cycle']==1704
        assert np.array_equal(payload(before).view('u8'),payload(initial).view('u8'))
        receipt['reference_probe']={'seconds':time.monotonic()-start,'command':command,
            'executable_sha256':sha(fixed),'input_sha256':sha(inp),'initial_values_bitwise_equal':True,
            'initial_sha256':sha(rst[0]),'final_sha256':sha(rst[-1]),'cycles':final['cycle'],'time':final['time']}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

def analyze(root,out,n):
    env={k:os.environ[k] for k in ('PALS_RANKID','PMI_RANK','PMIX_RANK','OMPI_COMM_WORLD_RANK') if k in os.environ}
    assert env and len(set(env.values()))==1,env
    rank=int(next(iter(env.values())));assert rank in (0,1)
    name=('control','gauge')[rank]+f'-N{n}'
    dest=out/'analysis'/name
    from nonradiative_native_analysis import run
    from nonradiative_pointwise_summary import run as summarize
    start=time.monotonic();run(out/name,dest);summarize(dest)
    from nonradiative_tensor_attribution import run as attribute
    attribute(dest,peak_only=True)
    if n==12 and rank==1:
        from scri_news import analyze as extract
        import numpy as np
        from scri_news import interpolate_time
        # Dense versus same physical cadence used at N16/N20.
        extract(dest/'cache',dest/'cadence3',lmax=6,stride=3,mass=1/3)
        dense=np.load(dest/'news/news.npz');coarse=np.load(dest/'cadence3/news.npz')
        t=coarse['bondi_time'];assert np.array_equal(dense['ell_m'],coarse['ell_m'])
        values=np.column_stack([interpolate_time(dense['bondi_time'],dense['news_modes'][:,i],t) for i in range(len(dense['ell_m']))])
        comparison={'dense_vs_stride3_max_mode_difference':float(np.max(abs(values-coarse['news_modes']))),
                    'comparison_Bondi_interval_M':[float(3*t[0]),float(3*t[-1])]}
        (dest/'cadence-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    if n==16 and rank==0 and (out/'reference-control-N12').is_dir():
        probe=out/'reference-analysis/control-N12'
        run(out/'reference-control-N12',probe);summarize(probe);attribute(probe,peak_only=True)
    if n==16:
        prior=list((root/'runs').glob('N12-*/analysis/'+('control' if rank==0 else 'gauge')+'-N12'));assert len(prior)==1
        attribute(prior[0],output=out/f'wide-N12-attribution-rank-{rank}.json',peak_only=True)
    (out/f'analysis-rank-{rank}.json').write_text(json.dumps({'rank':rank,'case':name,'seconds':time.monotonic()-start})+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evolve','analyze']);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('N',type=int,choices=[12,16,20]);a=p.parse_args()
    globals()[a.action](a.root.resolve(),a.output.resolve(),a.N)
