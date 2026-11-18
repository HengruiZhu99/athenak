#!/usr/bin/env python3
"""Native pulse control across a Cartesian 2:1 boundary and the overlapping shell."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from temporal_convergence import parse, payload, seed_radial_probe


def run(exe,directory,uniform,factor):
    directory.mkdir()
    text=Path(__file__).with_name('small.athinput').read_text()
    start=text.index('<mesh>'); end=text.index('<meshblock>')
    text=text[:start]+text[start:end].replace('= 12','= 36')+text[end:]
    text+='''
<mesh_refinement>
refinement = static
<refined_region0>
x1min = -.012
x1max = .012
x2min = -.012
x2max = .012
x3min = -.012
x3max = .012
level = 1
'''
    fixture=directory/'fixture.athinput';fixture.write_text(text)
    launch=['mpiexec','-n','2',str(exe)]
    with (directory/'seed.log').open('w') as stream:
        subprocess.run([*launch,'-i',str(fixture),'time/nlim=0',f'time/dt={2**-11/factor}',
                        'z4c_hyperboloidal/chi_floor=0','multirate/uniform_reference='+str(uniform).lower()],
                       check=True,cwd=directory,stdout=stream,stderr=subprocess.STDOUT)
    seed=parse(next((directory/'rst').glob('*.rst')))
    seed_radial_probe(seed)
    with (directory/'evolve.log').open('w') as stream:
        subprocess.run([*launch,'-r',seed['path'],f'time/nlim={4*factor}','output1/last_time=-1'],
                       check=True,cwd=directory,stdout=stream,stderr=subprocess.STDOUT)
    final=parse(sorted((directory/'rst').glob('*.rst'))[-1])
    if final['time']!=4*2**-11 or final['cycle']!=4*factor:raise ValueError('incomplete evolution')
    data=payload(final)
    if not np.isfinite(data).all() or np.any(data[...,33:36]):raise ValueError('invalid evolved state')
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();directory=args.output.resolve();directory.mkdir(parents=True)
    exe=directory/'athena';shutil.copy2(args.binary,exe)
    mixed=run(exe,directory/'mixed',False,1)
    uniform=run(exe,directory/'uniform',True,4)
    if mixed['topology']!=uniform['topology']:raise ValueError('mesh mismatch')
    locations=np.frombuffer(mixed['topology'],dtype='=i4').reshape(-1,5)
    cart=locations[:,4]==0
    root=locations[:,3].min()
    groups={'cartesian_fine':cart&(locations[:,3]>root), 'cartesian_root':cart&(locations[:,3]==root),
            'overlapping_shell':(~cart)&(locations[:,0]==0)}
    a,b=payload(mixed),payload(uniform)
    result={}
    for name,blocks in groups.items():
        if not blocks.any():raise ValueError('fixture missing '+name)
        active=b[blocks,:,0]>0
        diff=(a[blocks,:,:31]-b[blocks,:,:31])[active]
        result[name]=dict(blocks=int(blocks.sum()),rms=float(np.sqrt(np.mean(diff**2))),max_abs=float(np.max(abs(diff))))
    (directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
