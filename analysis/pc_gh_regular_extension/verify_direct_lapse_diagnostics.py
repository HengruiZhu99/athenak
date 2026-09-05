"""Independently check appended direct-L diagnostics in zero-step projection oracles."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from make_inputs import parse


def check(run):
    p=parse((run/'used_input.athinput').read_text());order=int(p['pc_gh']['spatial_order'])
    dim=sum(int(p['mesh']['nx'+str(d)])>1 for d in (1,2,3))
    coeff={2:[-.5,0,.5],4:[1/12,-2/3,0,2/3,-1/12],6:[-1/60,3/20,-3/4,0,3/4,-3/20,1/60]}[order]
    maximum=0.;integral=0.
    for path in run.glob('projection-before-rank*.csv'):
        data=np.genfromtxt(path,delimiter=',',names=True)
        for block in np.unique(data['block']):
            b=data[data['block']==block];shape=tuple(int(b[k].max())+1 for k in ['k','j','i'])
            u=np.stack([b['u'+str(n)].reshape(shape) for n in range(55)])
            active=b['active'].reshape(shape).astype(bool);alpha=u[18]*u[0];r2=np.zeros(shape)
            for d in range(3):
                target=0 if d>=dim else 2*sum(c*np.roll(alpha,-q,axis=2-d) for c,q in zip(coeff,range(-order//2,order//2+1)))/b['d'+'xyz'[d]][0]
                r2+=(u[43+d]-target)**2
            val=np.sqrt(r2[active]);maximum=max(maximum,float(val.max()))
            dv=float(np.prod([b['d'+'xyz'[d]][0] for d in range(dim)]))
            integral+=float(val.sum())*dv
    with next(run.glob('*.pcgh-reduction.csv')).open() as f:
        rows=[r for r in csv.DictReader(f) if r['constraint']=='RL_direct' and r['operation']=='-2']
    with next(run.glob('*.hybrid.csv')).open() as f:
        regional=[r for r in csv.DictReader(f) if r['quantity']=='RL_direct' and r['operation']=='-2' and r['region']=='all']
    assert rows and regional
    errors=[abs(float(rows[-1]['max'])-maximum),abs(float(regional[-1]['max'])-maximum),abs(float(regional[-1]['coordinate_l1'])-integral)]
    assert max(errors)<2e-11*max(1,maximum,integral),(run,errors)
    result=dict(run=str(run),max_errors=errors,expected_max=maximum,expected_coordinate_l1=integral,decision='PASS')
    print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps([check(r) for r in a.runs],indent=2)+'\n')
