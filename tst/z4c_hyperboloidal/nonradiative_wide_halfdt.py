#!/usr/bin/env python3
"""Separate timestep control and point-matched reference-jet comparison."""
import argparse,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
from nonradiative_wide import sha

def load_extractor():
    # Establish vis/python before importing: tst also has a CLI named scri_news.
    from nonradiative_alignment import aligned
    from scri_news import analyze as extract
    return extract,aligned

def evolve(root,out):
    prior=list((root/'runs').glob('N12-*/gauge-N12'));assert len(prior)==1
    prior=prior[0];case=out/'gauge-N12-halfdt';case.mkdir(parents=True)
    text=(prior/'input.athinput').read_text()
    matches=re.findall(r'^dt = ([^\n]+)$',text,re.M);assert len(matches)==1
    text=re.sub(r'^dt = [^\n]+$',f'dt = {float(matches[0])/2:.17g}',text,flags=re.M)
    text=re.sub(r'^nlim = \d+$','nlim = 3408',text,flags=re.M)
    text=text.replace('dcycle=1','dcycle=6')
    inp=case/'input.athinput';inp.write_text(text)
    binary=root.parent/'nonradiative-20261115/build/athena'
    receipt=json.loads((root.parent/'nonradiative-20261115/build/receipt.json').read_text())
    assert sha(binary)==receipt['executables']['athena']
    mpi=['mpiexec','-n','24','-ppn','12','--cpu-bind=list:'+os.environ['NONRAD_BINDING'],os.environ['NONRAD_WRAPPER']]
    seedexe=root/'seed/nonradiative_seed'
    assert sha(seedexe)==json.loads((root/'seed/receipt.json').read_text())['executable_sha256']
    (case/'seed').mkdir()
    seed_command=mpi+[str(seedexe),str(inp),'.001','.86','.08','6']
    with (case/'seed/run.log').open('w') as log:subprocess.run(seed_command,cwd=case/'seed',stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    seed=next((case/'seed/rst').glob('*.rst'))
    control=next((prior.parent/'control-N12/seed/rst').glob('*.rst'))
    full=next((prior/'seed/rst').glob('*.rst'))
    subprocess.run([sys.executable,str(Path(__file__).with_name('nonradiative_seed_check.py')),str(control),str(full),str(out/'seed-check.json'),'--halfdt',str(seed)],check=True)
    command=mpi+[str(binary),'-r',str(seed),'-i',str(inp),'-d',str(case),'-t','00:54:00']
    start=time.monotonic()
    with (case/'run.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3000)
    match=re.findall(r'^time=([^ ]+) cycle=(\d+)$',(case/'run.log').read_text(),re.M)
    assert len(match)==1 and int(match[0][1])==3408,match
    final=sorted((case/'rst').glob('*.rst'))[-1]
    (out/'receipt.json').write_text(json.dumps(dict(command=command,seed_command=seed_command,seconds=time.monotonic()-start,
        executable_sha256=sha(binary),input_sha256=sha(inp),seed=str(seed),seed_sha256=sha(seed),
        final_sha256=sha(final),driver_sha256=sha(__file__),cycle=3408,time=float(match[0][0])),indent=2)+'\n')

def compare(a,b,stride,output):
    left=np.load(a/'pointwise.npz');right=np.load(b/'pointwise.npz')
    assert np.array_equal(left['x'],right['x'])
    dt=float(np.max(abs(left['times'][::stride]-right['times'])))
    assert dt<1e-12,dt
    result={'native_time_mismatch':dt,'baseline':str(a),'comparison':str(b),'baseline_stride':stride}
    for key in ('T_rhs','T_fd','lapse_terms','penalty_rates','pole_over_h'):
        result[key+'_difference_max']=float(np.max(abs(left[key][::stride][3:-3]-right[key][3:-3])))
    output.write_text(json.dumps(result,indent=2)+'\n')

def analyze(root,out):
    ranks={os.environ[k] for k in ('PALS_RANKID','PMI_RANK','PMIX_RANK','OMPI_COMM_WORLD_RANK') if k in os.environ}
    assert len(ranks)==1,ranks
    rank=int(next(iter(ranks)));assert rank in (0,1)
    if rank==0:
        from nonradiative_native_analysis import run
        from nonradiative_pointwise_summary import run as summarize
        from nonradiative_tensor_attribution import run as attribute
        dest=out/'analysis/gauge-N12-halfdt';run(out/'gauge-N12-halfdt',dest);summarize(dest);attribute(dest,peak_only=True)
        prior=list((root/'runs').glob('N12-*/analysis/gauge-N12'));assert len(prior)==1
        compare(prior[0],dest,3,out/'halfdt-pointwise-comparison.json')
        from nonradiative_alignment import aligned
        a=np.load(prior[0]/'cadence3/news.npz');b=np.load(dest/'news/news.npz');t,v,labels=aligned([a,b])
        (out/'halfdt-news-comparison.json').write_text(json.dumps({'common_Bondi_interval_M':[float(3*t[0]),float(3*t[-1])],
            'news_mode_difference_max':float(np.max(abs(v[0]-v[1])))},indent=2)+'\n')
    else:
        baseline=list((root/'runs').glob('N12-*/analysis/control-N12'))
        probe=list((root/'runs').glob('N16-*/reference-analysis/control-N12'))
        assert len(baseline)==len(probe)==1
        compare(baseline[0],probe[0],1,out/'reference-pointwise-comparison.json')
        extract,aligned=load_extractor()
        gauge=baseline[0].parent/'gauge-N12';archives=[np.load(gauge/'cadence3/news.npz')]
        for lmax in (4,8):
            dest=out/'analysis'/f'frame-l{lmax}'/'news'
            extract(gauge/'cache',dest,lmax=lmax,stride=3,mass=1/3)
            archives.append(np.load(dest/'news.npz'))
        t,v,labels=aligned(archives,limit=4)
        (out/'frame-comparison.json').write_text(json.dumps({'common_Bondi_interval_M':[float(3*t[0]),float(3*t[-1])],
            'common_ell_max':4,'lmax6_vs4_difference':float(np.max(abs(v[0]-v[1]))),
            'lmax6_vs8_difference':float(np.max(abs(v[0]-v[2])))},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evolve','analyze']);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    globals()[a.action](a.root.resolve(),a.output.resolve())
