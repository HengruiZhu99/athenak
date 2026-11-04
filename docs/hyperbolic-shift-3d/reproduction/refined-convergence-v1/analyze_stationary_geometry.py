"""Independent differentiation control for the full initial geometric RHS."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
RAW=ROOT/'output/refined-convergence/initial-stationary-geometry-refinement';OUT=HERE/'initial-stationary-geometry';OUT.mkdir(exist_ok=True)
a=np.genfromtxt(RAW/'central20.csv',delimiter=',',names=True);rows=[]
for mode in [0,1]:
 for h in [.0008,.0004,.0002,.0001,.00005]:
  q=a[(a['mode']==mode)&(a['h']==h)]
  rows.append(dict(mode=mode,h=h,**{f:float(max(abs(q[f]))) for f in ['max_geometric_rate','metric_rr_rate','chi_rate','K_rate','Theta_rate','max_A_rate','max_Lambda_rate']}))
r=[x for x in rows if x['mode']==1];orders=[float(np.log(r[i]['max_geometric_rate']/r[i+1]['max_geometric_rate'])/np.log(2)) for i in range(4)]
report=dict(samples=rows,stationary_full_geometry_effective_orders=orders,csv_sha256=hashlib.sha256((RAW/'central20.csv').read_bytes()).hexdigest(),qualification='Pointwise stationary geometry only: actual conformal geometry RHS with independently differenced first/second derivatives, no KO/upwind/RK/boundary evolution. Twelve radii .4-.95 outside the puncture. Fourth-order truncation trend followed by cancellation/roundoff; smallest h is not necessarily most accurate. Existing gauge RHS is not stationary and must be changed consistently before an evolution control.')
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['central20.csv','build.json']:shutil.copy2(RAW/name,OUT/name)
fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
for mode,label in [(0,'Current free initial gauge'),(1,'Compatible stationary outer gauge')]:
 q=[r for r in rows if r['mode']==mode];ax.loglog([x['h'] for x in q],[x['max_geometric_rate'] for x in q],'o-',label=label)
h=np.array([.0008,.0004,.0002]);ax.loglog(h,r[0]['max_geometric_rate']*(h/h[0])**4,'k:',label='Fourth-order reference')
ax.set(xlabel='Independent derivative spacing h',ylabel='Max absolute geometric RHS',title='Initial geometry: truncation trend and roundoff limit');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'geometry-rate-control.png',dpi=170);fig.savefig(OUT/'geometry-rate-control.pdf');print(json.dumps(report,indent=2))
