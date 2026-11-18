#!/usr/bin/env python3
"""Compare extracted Bondi news on a spherical gauge-pulse control at matched output times."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from temporal_convergence import seed_radial_probe, parse
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vis/python'))
from scri_collect import collect
from scri_news import analyze


def run(args, name, factor):
    directory=args.output/name
    directory.mkdir()
    seed=directory/'seed'
    seed.mkdir()
    case=directory/'evolution'
    case.mkdir()
    fixture=seed/'input.athinput'
    fixture.write_text(Path(__file__).with_name('small.athinput').read_text()+f"""
<output3>
file_type = scri
dcycle = {factor}
nodal = true
modes = true
lmax = 4
""")
    prefix=['mpiexec','-n',str(args.ranks),str(args.output/'athena')]
    def launch(where, log, parameters):
        with (where/log).open('w') as stream:
            subprocess.run([*prefix,*parameters],cwd=where,stdout=stream,stderr=subprocess.STDOUT,check=True)
    launch(seed,'initialize.log',['-i',str(fixture),'time/nlim=0',f'time/dt={2.0**-10 / factor}',
                                  'z4c_hyperboloidal/chi_floor=0',
                                  'multirate/uniform_reference='+str(factor>1).lower()])
    checkpoint=next((seed/'rst').glob('*.rst'))
    seed_radial_probe(parse(checkpoint))
    # Recompute the time-zero scri record after perturbing the radial closure support.
    launch(case,'initial-cut.log',['-r',str(checkpoint),'time/nlim=0','output3/file_number=0','output3/scri_last_cycle=-1'])
    launch(case,'evolution.log',['-r',str(checkpoint),f'time/nlim={12*factor}','output3/file_number=1'])
    return extract(directory)


def extract(directory):
    case=directory/'evolution'
    records=collect(case,directory/'cache',capacity=16,record_pattern='*.o3.*.r00000000.scri')
    if len(records)!=13: raise ValueError(f'Expected 13 common cuts, got {len(records)}')
    analyze(directory/'cache',directory/'news',lmax=4)
    return np.load(directory/'news/news.npz')


def compare(a,b):
    if not np.array_equal(a['ell_m'],b['ell_m']): raise ValueError('waveform modes differ')
    ta,tb=a['bondi_time'],b['bondi_time']
    same=(ta>=tb[0]) & (ta<=tb[-1])
    modes=a['news_modes'][same]
    ref=np.column_stack([np.interp(ta[same],tb,column) for column in b['news_modes'].T])
    if not modes.size or not np.isfinite(modes-ref).all(): raise ValueError('invalid common Bondi cuts')
    report=dict(native_duration=12*2**-10,common_bondi_samples=int(same.sum()),
                maximum_mode_difference=float(np.max(abs(modes-ref))),
                uniform_peak=float(np.max(abs(ref))),mixed_peak=float(np.max(abs(modes))),
                interpretation='Nonradiative spherical control at fixed spatial and output resolution.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ranks',type=int,default=2)
    args=p.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.binary,args.output/'athena')
    a,b=run(args,'mixed',1),run(args,'uniform',4)
    report=compare(a,b)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
