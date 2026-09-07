#!/usr/bin/env python3
"""Audit actual RK and exchange stage dumps on independently seeded periodic data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import re
import shlex
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart
from intrinsic_diagnostics import diagnostics, norms, reduction_injection

from intrinsic_stage import read_dump, global_fields, ghost_error, assemble_ranks

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--launcher',default='')
p.add_argument('--ranks',type=int,default=1)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[];runs=[]
for dim in [2,3]:
    for order in [2,4,6]:
        case=f'fd{order}-{dim}d';folders=[]
        source=a.fixtures/(case+'-seed-multi.rst')
        text=(a.fixtures/(case+'-seeded-multi')/'used.athinput').read_text().replace('nlim = 3','nlim = 1').replace('tlim = 0.0003','tlim = 0.0001')
        for enabled in [False,True]:
            d=a.output/(case+('-dump' if enabled else '-control'));d.mkdir();folders.append(d)
            inp=d/'used.athinput';inp.write_text(text.replace('formulation = intrinsic_clean',f'formulation = intrinsic_clean\nintrinsic_stage_dump = {str(enabled).lower()}'))
            command=shlex.split(a.launcher)+[str(a.binary.resolve()),'-i',str(inp.resolve()),'-r',str(source.resolve())]
            with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            runs.append(dict(command=command,returncode=r.returncode,input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),restart_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            (a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
            assert r.returncode==0,(d,(d/'run.log').read_text())
            assert re.findall(r'Number of parallel ranks = (\d+)',(d/'run.log').read_text())==[str(a.ranks)]
            assert len(list(d.glob('intrinsic-health-rank*.csv')))==a.ranks
        baseline=read_restart(sorted((folders[0]/'rst').glob('*.rst'))[-1])['state']
        observed=read_restart(sorted((folders[1]/'rst').glob('*.rst'))[-1])['state']
        assert np.array_equal(baseline,observed)
        paths=sorted(folders[1].glob('intrinsic-stage-*.dat'));assert len(paths)==9*a.ranks
        snapshots={}
        for v in map(read_dump,paths):
            snapshots.setdefault((v['header']['stage'],v['header']['operation']),[]).append(v)
        snapshots={k:assemble_ranks(v,a.ranks) for k,v in snapshots.items()}
        assert all(len(v['header']['blocks'])==2**dim for v in snapshots.values())
        stages=[]
        for stage in [1,2,3]:
            pre=snapshots[stage,'pre-rk'];post=snapshots[stage,'post-rk'];exchange=snapshots[stage,'post-exchange'];h=pre['header']
            assert h['ghosts_valid'] and not post['header']['ghosts_valid'] and exchange['header']['ghosts_valid']
            predicted=h['gam0']*pre['active']+h['gam1']*pre['register']+h['beta_dt']*pre['rhs']
            rk_error=float(np.max(abs(predicted-post['active'])/(1+abs(post['active']))))
            transfer_error=float(np.max(abs(post['active']-exchange['active'])))
            ghosts=max(ghost_error(pre),ghost_error(exchange))
            assert rk_error<=2e-12 and transfer_error==0 and ghosts<=2e-12
            before,spacing,_=global_fields(pre,pre['active']);after,_,_=global_fields(post,post['active'])
            rhs,_,_=global_fields(pre,pre['rhs'])
            rate=h['reduction_rate']*(before[0]*before[1] if h['reduction_profile']=='lapse_scaled' else np.ones_like(before[0]))
            injection,curl_injection=reduction_injection(before,rhs,spacing,order,rate,h['dissipation'])
            assert np.isfinite(injection).all() and np.isfinite(curl_injection).all()
            db=diagnostics(before,spacing,order);da=diagnostics(after,spacing,order)
            delta={n:da[n]-db[n] for n in da};volume=float(np.prod(spacing)*np.prod(before.shape[1:]))
            stages.append(dict(stage=stage,rk_error=rk_error,active_exchange_increment=transfer_error,valid_ghost_error=ghosts,diagnostic_norm_of_RK_difference=norms(delta,volume),semidiscrete_defect=norms(dict(reduction=injection,curl=curl_injection),volume)))
            np.savez(folders[1]/f'signed-stage{stage}.npz',state_increment=after-before,reduction_injection=injection,curl_injection=curl_injection,**delta)
        original_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with (folders[1]/'duplicate.log').open('w') as log:
            duplicate=subprocess.run(command,cwd=folders[1],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        assert duplicate.returncode!=0 and 'cannot exclusively create stage dump' in (folders[1]/'duplicate.log').read_text()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in original_hashes.items())
        record=dict(case=case,ranks=a.ranks,status='PASS',dump_neutral_bitwise=True,duplicate_rejected=True,stages=stages)
        records.append(record);(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
        print(case,'PASS',flush=True)
(a.output/'summary.json').write_text(json.dumps(dict(status='PASS',ranks=a.ranks,cases=6,stages=18,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),scope='actual rank-verified multi-block RK and periodic exchange; physical diagnostics assembled offline from global active cells; no stale-ghost derivatives'),indent=2)+'\n')
