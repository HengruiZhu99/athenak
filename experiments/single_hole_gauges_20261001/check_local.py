#!/usr/bin/env python3
"""Small CPU checks: pulse support, unchanged geometry, diagnostic vetoes, restart."""
import json
import math
from pathlib import Path
import subprocess
import tempfile
from generate import make, parse, dump

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT/'build/local-serial/src/athena'

def table(path):
    return [[float(x) for x in line.split()] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith('#')]


def main():
    with tempfile.TemporaryDirectory(prefix='single-hole-check-') as tmp:
        root = Path(tmp)
        _, text = make('pulse','tel',32,.5)
        base = parse(text)
        for key in list(base):
            if key.startswith('refined_region'):
                del base[key]
        for a in (1,2,3):
            base['mesh']['nx'+str(a)] = 32
            base['mesh']['x'+str(a)+'min'] = -8
            base['mesh']['x'+str(a)+'max'] = 8
            base['meshblock']['nx'+str(a)] = 16
        base['time'].update(nlim=0, tlim=.1)
        base['z4c'].update(horizon_finder='none',history_interior_radius=3,history_inner_radius=2)
        base['problem']['lapse_pulse_width'] = 1.5
        for n in (1,2,3,4): base['output'+str(n)]['dt'] = .00001
        def run(name, amp=.2, overrides=(), restart=None):
            d=root/name; d.mkdir()
            base['problem']['lapse_pulse_amplitude']=amp
            (d/'input').write_text(dump(base))
            args=[str(EXE),'-i',str(d/'input')]
            if restart: args=['%s'%EXE,'-r',str(restart)]
            result=subprocess.run(args+list(overrides),cwd=str(d),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
            (d/'run.log').write_text(result.stdout)
            assert result.returncode==0 and 'FATAL' not in result.stdout, result.stdout[-2000:]
            return d
        zero=run('zero',0); pulse=run('pulse')
        def fields(d,token):
            paths=list(d.rglob('*.'+token+'.*.tab'))
            if not paths:
                paths=[p for p in d.rglob('*.tab') if token in p.name]
            return table(sorted(paths)[-1])
        # Discover column names from header rather than assume a variable offset.
        zpath=next(pulse.rglob('*z4c*.tab'))
        header=zpath.read_text().splitlines()[:8]
        import re
        names=re.findall(r'\[\d+\]=([^\s]+)', '\n'.join(header))
        z0=fields(zero,'z4c');zp=fields(pulse,'z4c')
        assert len(z0)==len(zp) and len(z0)>0
        changed=set()
        for a,b in zip(z0,zp):
            for j,(x,y) in enumerate(zip(a,b)):
                if x!=y: changed.add(j)
        assert len(changed)==1, (changed,header)
        alpha_col=changed.pop()
        assert all(b[alpha_col]>=a[alpha_col] for a,b in zip(z0,zp))
        assert sum(b[alpha_col]>a[alpha_col] for a,b in zip(z0,zp))>0
        assert fields(zero,'con')==fields(pulse,'con'), 'pulse changed initial constraints'
        hz=table(next(zero.rglob('*.hst')))[0]
        # The sphere excludes cube corners: its sampled volume is below the old cubical shell.
        assert 0<hz[17]<(2*3)**3-(2*2)**3
        assert len(hz)==20 and hz[14]>0 and hz[15]==0 and hz[16]==0 and hz[18]==0 and hz[19]==0, hz
        bad=run('boundary',overrides=('z4c/history_boundary_buffer=8',))
        assert table(next(bad.rglob('*.hst')))[0][15]>0
        badspeed=run('speed',overrides=('z4c/history_boundary_speed=0.01',))
        assert table(next(badspeed.rglob('*.hst')))[0][19]>0
        base['output5']=dict(file_type='rst',dt=.00001)
        full=run('full',overrides=('time/nlim=2',))
        half=run('half',overrides=('time/nlim=1',))
        resumed=run('resumed',overrides=('time/nlim=2',),restart=sorted(half.rglob('*.rst'))[-1])
        assert 'OnePuncture initialized.' not in (resumed/'run.log').read_text()
        a=fields(full,'z4c');b=fields(resumed,'z4c')
        assert len(a)==len(b)
        assert all(math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-13) for r,s in zip(a,b) for x,y in zip(r,s)), 'restart changed evolved fields'
        print(json.dumps(dict(pulse_only_lapse=True,initial_constraints_unchanged=True,
              clean_shell=True,boundary_veto=True,speed_veto=True,restart_matches=True),indent=2))

if __name__=='__main__': main()
