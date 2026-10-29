#!/usr/bin/env python3
"""Read-only comparison of bounded local n6 radial-closure controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Output must be new')
    source=json.loads((args.raw/'source.json').read_text())
    for name,digest in source['sources'].items():
        if sha(args.raw/'source-tree'/name)!=digest:raise ValueError('Source snapshot mismatch: '+name)
    cases=[];histories=[]
    for variant,name in [(1,'v5'),(2,'donor-seven')]:
        execution=json.loads((args.raw/(name+'-execution.json')).read_text())
        for file,bound in execution['artifacts'].items():
            path=args.raw/file
            if sha(path)!=bound['sha256'] or path.stat().st_size!=bound['bytes']:raise ValueError('Artifact differs: '+file)
        report=json.loads((args.raw/(name+'.json')).read_text())
        if report['n']!=6 or report['variant']!=variant or report['checkpoint_admitted']:raise ValueError('Wrong test configuration')
        with (args.raw/(name+'.history.csv')).open() as stream:rows=list(csv.DictReader(stream))
        history={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
        if not np.all(np.diff(history['t'])>0) or history['t'][-1]!=report['diagnostic_time']:raise ValueError('Invalid history clock')
        arrays=[np.fromfile(args.raw/(name+suffix),dtype=np.float64).reshape(-1,31) for suffix in ['.state.bin','.surface.bin']]
        if [len(a) for a in arrays]!=[19*6**3,6*6**2] or not all(np.isfinite(a).all() for a in arrays):raise ValueError('Invalid final arrays')
        for a in arrays:a[:,0]-=1
        maximum=max(float(np.max(np.abs(a))) for a in arrays)
        if report['t']==report['diagnostic_time'] and maximum!=report['state']:raise ValueError('Final state/report mismatch')
        limits={'state':1e-8,'H_L2':1e-7,'M_L2':1e-7,'null_surface':1e-12,'null_ghost':1e-12}
        gates={k:bool(np.isfinite(history[k]).all() and np.max(history[k])<=limit) for k,limit in limits.items()}
        cases.append(dict(name=name,report=report,execution_returncode=execution.get('returncode'),
            recorded_gates=gates,completed_target=report['status']=='completed' and report['t']==report['target'],
            scientifically_admitted=False,artifact_hashes_verified=True,arrays_finite=True))
        histories.append(history)
    args.out.mkdir(parents=True)
    result=dict(scope='Local n6 CPU controls; no long-duration, GPU or convergence admission',
        cases=cases,source_sha256=sha(args.raw/'source.json'),analysis_sha256=sha(Path(__file__)),
        growth_fit='Not fitted: measured errors remain near floating-point truncation/cancellation scales')
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,field,title in zip(axes,['state','M_L2','null_ghost'],['State deviation','Momentum RMS','Ghost null closure']):
        for case,data in zip(cases,histories):
            ax.semilogy(data['t'],np.where(data[field]>0,data[field],np.nan),marker='o',ms=3,label=case['name'])
        ax.set_title(title);ax.set_xlabel('Coordinate time');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Bounded n6 CPU controls — target t=2 remains uncompleted')
    fig.tight_layout();fig.savefig(args.out/'comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
