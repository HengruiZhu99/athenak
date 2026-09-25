#!/usr/bin/env python3
"""Actual vacuum AMR and restart cache equality; no matter continuity certificate."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
from z4c_kerr_trumpet import replace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--launcher',default='mpiexec')
    p.add_argument('--default',action='store_true',help='Omit flag in the cached fresh input')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];exe=a.exe.resolve()
    base=(repo/'tst/inputs/z4c_kerr_cache_amr.athinput').read_text()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');results={}
    for mode in ['zero','pulse']:
        for phase in ['fresh','restart']:
            runs=[]
            for flag in ['false','true']:
                d=out/(mode+'-'+phase+'-'+flag);d.mkdir();runs.append(d)
                if phase=='fresh':
                    s=replace(base,'cache_stationary_background',flag)
                    if a.default and flag=='true':
                        s=re.sub(r'^cache_stationary_background\s*=.*\n', '', s, flags=re.M)
                    s=replace(s,'vacuum_gauge_pulse_amplitude','0' if mode=='zero' else '1e-8')
                    (d/'input.athinput').write_text(s);args=['-i','input.athinput']
                else:
                    prior=out/(mode+'-fresh-'+flag)
                    rst=sorted((prior/'rst/rank_00000000').glob('*.rst'))[-1]
                    args=['-r',str(rst),'time/nlim=8']
                with (d/'run.log').open('w') as f:
                    subprocess.run([a.launcher,'-n','4',str(exe)]+args,cwd=d,env=env,
                                   stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
                cmd=[sys.executable,str(repo/'analysis/kerr_trumpet/check_checkpoint.py'),
                     str(d),'--ranks','4','--output',str(d/'audit.json')]
                if mode=='zero':cmd+=['--exact-zero']
                with (d/'audit.log').open('w') as f:
                    subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
                audit=json.loads((d/'audit.json').read_text())
                assert audit['passed'] and audit['cycle']==(4 if phase=='fresh' else 8)
                assert audit['residual_exactly_zero']==(mode=='zero')
                log=(d/'run.log').read_text()
                if flag=='true':
                    stats=re.findall(r'Z4C_BACKGROUND_CACHE rank=(\d+) fills=(\d+) hits=(\d+) invalidations=(\d+)',log)
                    assert sorted(int(x[0]) for x in stats)==list(range(4))
                    assert all(int(x[1])>0 and int(x[2])>0 for x in stats)
                if phase=='fresh':
                    created=re.search(r'(\d+) MeshBlocks created, (\d+) deleted by AMR',log)
                    migrated=re.search(r'(\d+) communicated for load balancing',log)
                    assert created and int(created[1])>0,log[-2000:]
                    assert migrated and int(migrated[1])>0,log[-2000:]
                results[d.name]={'passed':True,'cycle':audit['cycle'],'blocks':audit['blocks']}
            target=out/(mode+'-'+phase+'-comparison.json')
            subprocess.run([sys.executable,str(repo/'tst/regression/compare_stationary_cache.py'),
                            str(runs[0]),str(runs[1]),str(target),'4'],check=True)
            results[mode+'-'+phase+'-comparison']=json.loads(target.read_text())
            (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
