#!/usr/bin/env python3
"""Reuse accepted records for cadence and angular-frame error measurements."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vis/python'))
from scri_news import analyze,stencil
from scri_geometry import spacetime_derivatives,ricci
from scri_frame import generator_geometry

def run(base):
    start=time.monotonic();cache=base/'cache';original=np.load(base/'pointwise.npz')
    times=original['times'];x=original['x'];archive=np.load(cache/'metric.npy',mmap_mode='r')
    report={'angular':{},'cadence':{}}
    for lmax in (4,8):
        report['angular'][str(lmax)]=analyze(cache,base/f'news-L{lmax}',lmax=lmax)
    for stride in (2,4):
        indices=np.arange(0,len(times),stride);subtimes=times[indices];tensors=[]
        for q in range(len(indices)):
            window,center=stencil(subtimes,q)
            g,dg,ddg=spacetime_derivatives(archive[indices[window]],subtimes[window],center)
            _,connection,inverse=ricci(g,dg,ddg)
            _,source,nt,H=generator_geometry(x,inverse,connection)
            tensors.append(H-g*(source*nt)[:,None,None])
        tensors=np.array(tensors);interior=slice(3,-3)
        np.savez(base/f'cadence-{stride}.npz',indices=indices,times=subtimes,T_fd=tensors)
        report['cadence'][str(stride)]={
            'T_fd_vs_dense':float(np.max(abs(tensors[interior]-original['T_fd'][indices[interior]]))),
            'T_fd_vs_rhs':float(np.max(abs(tensors[interior]-original['T_rhs'][indices[interior]]))),
            'news':analyze(cache,base/f'news-stride{stride}',lmax=6,stride=stride)}
    report['seconds']=time.monotonic()-start
    (base/'controls.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('analysis',type=Path);a=p.parse_args();run(a.analysis)
