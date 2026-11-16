#!/usr/bin/env python3
"""Feed the inferred power-law mode through the actual linearized closure."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze import sample
from strip import assemble


def amplitudes(s,fields):
    d=s*s+5*s-4
    q=[2*(s*s-3*s+20)/(3*d),4*(-s*s+3*s+4)/(3*d),
       2*s*(s*s+5*s+4)/(3*d),(s**3+2*s*s-11*s-12)/d,0,
       4*s*(s*s-3*s-4)/(3*d),1,0,2*s*(-s-1)/d,
       4*s*(s*s-3*s-4)/(3*d)]
    vals=np.zeros(26)
    for f,v in zip([0,1,7,13,14,15,18,19,22,25],q):vals[fields.index(f)]=v
    vals[fields.index(4)]=-q[1]/2;vals[fields.index(10)]=-q[2]/2
    return vals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);records=[]
    for kappa in [.25,.375,.5,.75,1.]:
        s=(-5+np.sqrt(41+32/kappa))/2;prior=None
        for h in [.005,.0025,.00125,.000625,.0003125,.00015625]:
            r=1-h*np.arange(12,-1,-1)
            meta,K=sample(args.binary.resolve(),out,f'mode-k{kappa}-h{h}',r,kappa,h)
            L,E,free,clock,residual,fields,r,response=assemble(meta,K,0)
            amp=amplitudes(s,fields)
            power=np.array([s if f<=6 or f in [13,14,18,22,23,24] else s-1 for f in fields])
            state=(1-r[:,None])**power[None,:]*amp[None,:]
            value=abs(np.einsum('i,i->',clock,state.ravel()[free]))
            record=dict(kappa=kappa,s=float(s),h=h,clock=float(value),
                        expected_noninteger_order=float(s-1),
                        order=None if prior is None else float(np.log2(prior/value)),
                        normal_compatibility_residual=residual)
            records.append(record);prior=value;print(json.dumps(record),flush=True)
    (out/'summary.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
