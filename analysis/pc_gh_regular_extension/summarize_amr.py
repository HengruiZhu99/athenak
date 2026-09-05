"""Compare AMR pulse error with uniform transport and bracketed transfer changes."""
import argparse
import json
from pathlib import Path

import numpy as np
from make_inputs import parse


def summarize(groups, output):
    rows=[];keys=set()
    for group in groups:
        for done in sorted(group.glob('*/completed.json')):
            run=done.parent
            if not (run/'pulse-metrics.json').exists():
                continue
            params=parse((run/'used_input.athinput').read_text())
            metrics=json.loads((run/'pulse-metrics.json').read_text())
            key=(metrics['family'],metrics['rate'],int(params['mesh']['nx1']),
                 int('mesh_refinement' in params))
            if key in keys:
                raise ValueError(f'Duplicate experiment {key}')
            keys.add(key)
            bounds=np.atleast_1d(np.genfromtxt(next(run.glob('*.pcgh-boundedness.dat')),names=True))
            if not all(np.isfinite(bounds[n]).all() for n in bounds.dtype.names):
                raise ValueError(f'Nonfinite boundedness data: {run}')
            if abs(float(bounds['time'][-1])-metrics['time'])>1e-12:
                raise ValueError(f'Incomplete final diagnostics: {run}')
            transfer={n:float(bounds[n].max()) for n in bounds.dtype.names
                      if n.startswith(('dRw_','dRQ_','dRalpha_','dRB_','dcurl'))}
            constraint={n:float(bounds[n][-1]) for n in bounds.dtype.names
                        if n.startswith(('pcgh_red_','pcgh_curl_'))}
            row=dict(run=str(run),family=key[0],rate=key[1],n=key[2],smr=key[3],
                metrics=metrics,transfer_max_norm_change=transfer,constraints_final=constraint,
                min_metric_eigenvalue=float(bounds['min_eigenvalue'].min()),
                min_w=float(bounds['min_w'].min()),min_rho=float(bounds['min_rho'].min()),
                provenance=json.loads((run/'provenance.json').read_text()))
            if (run/'pulse-decomposition.json').exists():
                row['decomposition']=json.loads((run/'pulse-decomposition.json').read_text())
            rows.append(row)
    report=dict(runs=rows,completed_runs=len(rows),
        scope='All-volume pulse errors and per-operation changes; no automatic qualification',
        transfer_definition='max_cell abs(norm_after - norm_before), maximized over each operation and time; this is not norm of the vector difference')
    output.write_text(json.dumps(report,indent=2)+'\n')
    for family in ['p','Q','L','B']:
        selected=sorted((r for r in rows if r['family']==family and r['smr']==1),key=lambda r:(r['rate'],r['n']))
        for row in selected:
            m=row['metrics'];t=row['transfer_max_norm_change']
            print(family,row['rate'],row['n'],m['error_l2_over_amplitude'],
                  m['damping_fit'],m['speed_fit'],max(v for n,v in t.items() if '_prolong_' in n))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('groups',type=Path,nargs='+');ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();summarize(args.groups,args.output)
