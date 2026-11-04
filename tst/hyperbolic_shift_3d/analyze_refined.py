#!/usr/bin/env python3
"""Compare RF1 histories without inventing missing horizon masses or convergence."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('runs',type=Path)
parser.add_argument('output',type=Path)
a=parser.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
keys=['H_L2','H_exterior_L2','M_L2','M_exterior_L2','M_physical_L2','Theta_exterior_L2',
      'min_chi','min_lapse','null_surface','null_ghost','pole_max','clock_max','cfl_max',
      'horizon_expansion_rms','puncture_x','puncture_y','puncture_z']

def finite(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None

summary={}
series={}
for name in ['legacy-n16','flat-n16','redistributed-n16']:
    path=a.runs/(name+'.history.csv')
    report_path=a.runs/(name+'.report.json')
    if not path.exists():
        summary[name]=dict(status='not_run_or_no_history')
        continue
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    rows=list(csv.DictReader(path.open()))
    mass=[]
    quality=[]
    for row in rows:
        existing=finite(row.get('horizon_mass'))
        rms=finite(row.get('horizon_expansion_rms'))
        accepted=(row.get('horizon_found')=='1' and row.get('horizon_complete_surface')=='1'
                  and existing is not None and rms is not None and rms<=1e-2)
        # A trial radius is not a mass measurement. Failed/missing masses remain gaps.
        mass.append(existing if accepted else None)
        quality.append(accepted)
    intervals=[(float(y['wall_seconds'])-float(x['wall_seconds']))/(int(y['steps'])-int(x['steps']))
               for x,y in zip(rows[2:-1],rows[3:]) if int(y['steps'])>int(x['steps'])]
    series[name]=dict(rows=rows,quality_mass=mass)
    summary[name]=dict(status=report.get('status','incomplete'),records=len(rows),
       final_time=finite(rows[-1]['t']) if rows else None,
       final_t_over_M=finite(rows[-1]['t_over_M']) if rows else None,
       initial={k:finite(rows[0].get(k)) for k in keys} if rows else {},
       final={k:finite(rows[-1].get(k)) for k in keys} if rows else {},
       maxima={k:max((v for r in rows if (v:=finite(r.get(k))) is not None),default=None) for k in keys},
       horizon_mass_gaps=sum(not x for x in quality),
       max_mass_error=max((abs(v-1/3) for v in mass if v is not None),default=None),
       final_quality_mass=mass[-1] if mass else None,
       step_plus_diagnostics_seconds_median=statistics.median(intervals) if intervals else None,
       timing_note='One bounded run per formulation; first two diagnostic intervals excluded from interval median. Not a repeated performance benchmark.',
       dt=report.get('dt'),seconds=report.get('elapsed_seconds',report.get('seconds')),
       runtime={k:v for k,v in report.items() if 'seconds' in k and not isinstance(v,list)},
       floors={k:v for k,v in report.items() if 'floor' in k and not isinstance(v,list)},
       identity=report.get('formulation_identity'),checkpoint_admitted=report.get('checkpoint_admitted'))
result=dict(cases=summary,online_horizon_tolerance=1e-3,postprocessing_quality_threshold=1e-2,
            missing_masses_preserved=True,convergence_demonstrated=False,
            norm_notes='H is physical; M_exterior_L2 uses the conformal metric and its normalization changes with Omega. M_physical_L2 is a physical norm over the full domain. All use normalized compact coordinate volume.',
            limit='Single-resolution bounded evolution comparison; no robustness or convergence claim follows from completion alone.')
(a.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
# Plot only supplied samples. NaNs explicitly break curves at mass gaps.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('JSON written; install matplotlib to render the figure')
fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
labels={'legacy-n16':'Legacy','flat-n16':'Flat','redistributed-n16':'Flat + radial placement'}
for name,data in series.items():
    rows=data['rows'];t=[finite(r['t_over_M']) for r in rows]
    for ax,key in [(axes[0,0],'H_exterior_L2'),(axes[0,1],'M_exterior_L2'),(axes[1,1],'puncture_x')]:
        ax.plot(t,[finite(r.get(key)) for r in rows],label=labels[name])
    axes[1,0].plot(t,[v-1/3 if v is not None else math.nan for v in data['quality_mass']],label=labels[name],marker='.',ms=3)
axes[0,0].set(ylabel='Exterior Hamiltonian L2',yscale='log')
axes[0,1].set(ylabel='Exterior conformal momentum L2',yscale='log')
axes[1,0].set(ylabel='Quality-accepted horizon mass − 1/3')
axes[1,1].set(ylabel='Puncture compact x')
for ax in axes.flat:
    ax.set_xlabel('t / M');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=8)
fig.suptitle('N16 staged comparisons: bounded evolution, no convergence claim')
fig.savefig(a.output/'comparison.png',dpi=180)
