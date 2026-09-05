"""Separate the seeded reduction family from transfer-generated families.

This analyzes preserved full-volume dumps. It does not remove any component
from the primary error/norm measurement in verify_pulses.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from make_inputs import parse
from verify_pulses import table, prediction


def measure(run):
    params=parse((run/'used_input.athinput').read_text())
    problem=params['problem'];rate=float(params['pc_gh']['reduction_rate'])
    initial=table(run,'initial');final=table(run,'final')
    for data in [initial,final]:
        if not all(np.isfinite(data[name]).all() for name in data.dtype.names):
            raise ValueError('Nonfinite full-volume pulse dump')
    groups=dict(p=(0,3),Q=(3,21),L=(21,24),B=(24,33))
    expected=prediction(final,problem,rate)
    result=dict(run=run.name,time=float(final['time'][0]),family=problem['pulse_family'],
                rate=rate,groups={},scope='All cells retained; family decomposition supplements total error')
    for family,(lo,hi) in groups.items():
        e0=np.column_stack([initial[f'E{i}'] for i in range(lo,hi)])
        e1=np.column_stack([final[f'E{i}'] for i in range(lo,hi)])
        energy0=(e0**2).sum(axis=1)*initial['volume']
        energy1=(e1**2).sum(axis=1)*final['volume']
        n0=float(np.sqrt(energy0.sum()));n1=float(np.sqrt(energy1.sum()))
        error=e1-expected[:,lo:hi]
        centroid0=float((energy0*initial['x']).sum()/energy0.sum()) if n0>0 else None
        centroid1=float((energy1*final['x']).sum()/energy1.sum()) if n1>0 else None
        amp=float(problem['pulse_amplitude']);t=result['time']
        result['groups'][family]=dict(initial_norm=n0,final_norm=n1,
            error_l2_over_amplitude=float(np.sqrt((error**2*final['volume'][:,None]).sum())/amp),
            centroid_final=centroid1,
            speed_fit=(centroid1-centroid0)/t if n0>0 and n1>0 and t>0 else None,
            damping_fit=-float(np.log(n1/n0))/t if n0>0 and n1>0 and t>0 else None,
            interface_band_final_energy=float(energy1[np.abs(final['x'])<=.25].sum()))
    (run/'pulse-decomposition.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',nargs='+',type=Path)
    args=ap.parse_args()
    for run in args.runs:
        print(json.dumps(measure(run)),flush=True)
