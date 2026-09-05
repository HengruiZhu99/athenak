"""Separate changes of unmasked maximum norms across RK/transfer brackets.

These are differences of maxima, not norms of vector changes. RK is sampled
before ghost refresh, so its change includes temporarily inconsistent ghosts.
Neither these statistics nor extreme locations alone prove causal attribution.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def analyze(run):
    paths=list(run.glob('*.pcgh-reduction.csv'));assert len(paths)==1,paths
    before={};rk={};results={}
    def record(row,old,operation):
        q=row['constraint'];change=float(row['max'])-float(old['max'])
        key=(math.floor(float(row['t_step'])/.25),operation,q)
        out=results.setdefault(key,dict(time_bin=key[0]*.25,operation=operation,quantity=q,
            count=0,sum_positive_delta_max=0.,sum_negative_delta_max=0.,
            sum_signed_delta_max=0.,largest_positive_delta_max=0.,
            largest_abs_delta_max=0.,event=None))
        out['count']+=1;out['sum_positive_delta_max']+=max(0,change)
        out['sum_negative_delta_max']+=min(0,change);out['sum_signed_delta_max']+=change
        out['largest_positive_delta_max']=max(out['largest_positive_delta_max'],change)
        if abs(change)>out['largest_abs_delta_max']:
            out['largest_abs_delta_max']=abs(change)
            out['event']=dict(before=old,after=row,delta_max=change)
    with paths[0].open() as f:
        for row in csv.DictReader(f):
            if row['constraint'] in ['alpha','alpha2_chi']:continue
            token=tuple(row[k] for k in ['cycle','stage','constraint'])
            op=row['operation']
            if op=='-1' and row['phase']=='before':rk[token]=row
            if op=='0' and row['phase']=='before' and token in rk:
                record(row,rk.pop(token),'RK_before_ghost_refresh')
            if int(op)<0: continue
            key=token+(op,)
            if row['phase']=='before':before[key]=row
            elif key in before:record(row,before.pop(key),op)
    output=run/'transfer-bracket-summary.json'
    output.write_text(json.dumps(dict(scope=__doc__,bins=list(results.values())),indent=2)+'\n')
    print(str(output),len(results),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('runs',type=Path,nargs='+')
    for run in ap.parse_args().runs:analyze(run)
