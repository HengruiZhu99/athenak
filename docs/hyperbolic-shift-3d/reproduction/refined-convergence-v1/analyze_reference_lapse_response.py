"""Frozen-geometry lower-order lapse response, not the coupled PDE spectrum."""
from pathlib import Path
import gzip,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;out=H/'stationary-reference-response';out.mkdir(exist_ok=True)
with gzip.open(H/'initial-gauge-shear/central20.csv.gz','rt') as f:a=np.genfromtxt(f,delimiter=',',names=True)
a=a[(a['mode']==3)&(a['h']==.000025)&(a['r']>=.78)]
r=a['r'];o=(1-r*r)/2;S=-a['beta_r']/r;K=-3*S+1.5*o*a['Ar'];alpha=a['alpha'];xi=1+2*(1-r*r)**4;A=(1+r*r)/2
v1=(-(2*alpha+4*o*o)*(K+3)-2*xi*alpha+16*o**4*r*r/A-S*r*r)/o
v2=(-2*xi*alpha+16*o**4*r*r/A-S*r*r)/o
np.savetxt(out/'response.csv',np.column_stack([r,K,alpha,v1,v2]),delimiter=',',header='r,K_reference,alpha_reference,v1_derivative,v2_derivative',comments='')
report=dict(v1_max=float(max(v1)),v1_min=float(min(v1)),v1_peak_r=float(r[np.argmax(v1)]),v2_max=float(max(v2)),v2_min=float(min(v2)),qualification='Analytic derivative of the lapse RHS with evolved geometry, shift and spatial jets held fixed, sampled on the same stationary geometry. Does not include coupled K/metric/shift/auxiliary modes, spatial propagation, boundary closure or RK; negative v2 response is not a stability proof.')
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True);ax.plot(r,v1,label='v1: fixed reference subtraction');ax.plot(r,v2,label='v2: consistent K - K* target');ax.axhline(0,color='gray',ls=':');ax.set(xlabel='Compact radius r',ylabel='Partial derivative of lapse RHS with respect to lapse',title='Local lapse response with geometry and spatial jets held fixed');ax.grid(alpha=.2);ax.legend();fig.savefig(out/'response.png',dpi=170);fig.savefig(out/'response.pdf')
print(json.dumps(report,indent=2))
