"""Bounded single-frequency TT-row comparison; no full spectral stability claim."""
from pathlib import Path
import json
import numpy as np
from scipy import optimize
from schur_check import ROOT,assess_schur,bn
out=[]
for lam in [.01,.1,.2,.6,1.,1.230711636682,bn*2*np.pi,1.6,2.,3.,5.]:
 a=assess_schur(lam,2*np.pi,'radiation_weyl',True);out.append({'lambda_real':lam,**a})
fit=optimize.minimize_scalar(lambda x:assess_schur(x,2*np.pi,'radiation_weyl',True)['sigma_min'],bounds=(.6,2.),method='bounded')
(ROOT/'weyl-row-check.json').write_text(json.dumps({'mode':'radiation_weyl','scope':'Only two TT rows replaced; original gauge + physical F constraints retained. Bounded real positive-frequency test at one k; not uniform stability proof.','k_y':2*np.pi,'damping':True,'scan':out,'minimum_0p6_to2':{'lambda_real':float(fit.x),'sigma_min':float(fit.fun)}},indent=2)+'\n')
