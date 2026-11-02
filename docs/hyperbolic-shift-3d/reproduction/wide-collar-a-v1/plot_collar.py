from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
P=Path(__file__).resolve().parent
r=np.linspace(.65,.95,2001)
def C(r,begin,width):
 t=np.clip((r-begin)/width,0,1);return t**8/(t**8+(1-t)**8)
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
x=.75+(np.arange(16)+.5)*.25/16
for begin,width,label in [(.8,.1,'A: original 0.8–0.9'),(.7,.2,'A-width: 0.7–0.9')]:
 line,=ax.plot(r,C(r,begin,width),label=label)
 ax.plot(x,C(x,begin,width),'o',color=line.get_color(),ms=4)
ax.set(xlim=(.65,.95),ylim=(-.04,1.04),xlabel='Compact radius r',ylabel='Restoring weight C(r)',title='Same C7 shape and strength; doubled collar width')
ax.legend();ax.grid(alpha=.2);fig.savefig(P/'collar-profile.png',dpi=180)
t0=1/(1+9**(1/8));delta=1-2*t0
(P/'collar-width.json').write_text(json.dumps({'maximum_slope_original':80,'maximum_slope_wide':40,'N16_outer_dr':.25/16,'ten_to_ninety_width_original':.1*delta,'ten_to_ninety_width_wide':.2*delta,'N16_ten_to_ninety_cells_original':.1*delta/(.25/16),'N16_ten_to_ninety_cells_wide':.2*delta/(.25/16),'scope':'Analytic profile and nominal N16 cell centers; no evolution accuracy claim.'},indent=2)+'\n')
