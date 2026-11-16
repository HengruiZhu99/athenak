#!/usr/bin/env python3
"""Summarize the independent tests and plot the noninteger regularity result."""
import argparse
import json
from pathlib import Path
import numpy as np


def history(path):
    a=np.genfromtxt(path/'scri.csv',delimiter=',',names=True)
    assert all(np.isfinite(a[k]).all() for k in a.dtype.names)
    return a[a['time']<=.8]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('previous',type=Path);args=p.parse_args();root=args.output
    summary={}
    for f in root.glob('*-R*/receipt.json'):
        receipt=json.loads(f.read_text());assert receipt['returncode']==0
        a=history(f.parent)
        summary[f.parent.name]=dict(peak_T=float(max(a['T_max'])),peak_clock=float(max(abs(a['clock']))),seconds=receipt['seconds'])
    coarse=history(root/'wide-k0.5-R20-N16');fine=history(root/'wide-k0.5-R20-halfdt-N16')
    summary['halfdt']={k:float(max(abs(coarse[k]-np.interp(coarse['time'],fine['time'],fine[k])))) for k in ['clock','T00','T0x','Txx','Tyy','Tzz']}
    summary['default_parity']=(root/'control-k0.5-R4-N16/scri.csv').read_bytes()==(args.previous/'control-damping-N16/scri.csv').read_bytes()
    assert summary['default_parity']
    (root/'paired-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    mode=json.loads((root/'mode-test/summary.json').read_text());observed=[]
    for k,tag,ns in [(.5,'damping',(64,128)),(.75,'strong075',(32,64)),(1.,'strong100',(32,64))]:
        vals=[]
        for n in ns:
            d=args.previous/f'control-{tag}-N{n}'
            if not d.exists():d=args.previous/f'long-{tag}-N{n}'
            vals.append(float(max(history(d)['T_max'])))
        observed.append(dict(kappa=k,coarse_N=ns[0],fine_N=ns[1],order=float(np.log2(vals[0]/vals[1])),predicted=float((-7+np.sqrt(41+32/k))/2)))
    (root/'observed-orders.json').write_text(json.dumps(observed,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    k=np.linspace(.26,1.03,300)
    ax[0].plot(k,(-7+np.sqrt(41+32/k))/2,label='Noninteger branch: s−1',color='black')
    for kk in [.375,.5,.75,1.]:
        z=[a for a in mode if a['kappa']==kk]
        ax[0].plot(kk,z[-1]['order'],'o',color='C0',label='Manufactured closure' if kk==.375 else None)
        ax[1].loglog([a['h'] for a in z],[a['clock'] for a in z],'o-',label=f'κ={kk}')
    ax[0].plot([a['kappa'] for a in observed],[a['order'] for a in observed],'x',color='C3',markersize=9,label='Evolution: last tested doubling')
    ax[0].set(xlabel='Lapse damping κ, shift restoration R=4',ylabel='Decay order',title='Predicted and measured orders')
    ax[1].set(xlabel='Normal spacing h',ylabel='Manufactured-mode |clock|',title='Projection + normal solve + lift')
    for a in ax:a.grid(alpha=.25,which='both');a.legend(fontsize=8)
    fig.suptitle('Radial regularity mechanism — not a full 3D stability certificate')
    fig.savefig(root/'regularity.png',dpi=180);fig.savefig(root/'regularity.pdf')
    print(json.dumps({'paired':summary,'observed_orders':observed},indent=2))


if __name__=='__main__':main()
