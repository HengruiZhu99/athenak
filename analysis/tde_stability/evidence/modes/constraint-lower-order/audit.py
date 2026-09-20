"""Independent algebra/radial scale audit; no PDE evolution or source edits."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vacuum-preservation-20260918/python-deps'))
import sympy as s
import mpmath as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mp.mp.dps=60
x=s.symbols('x y z',real=True)
f=s.Rational(1,10)*x[0]+s.Rational(1,20)*x[1]*x[2]
h=s.Rational(1,5)*x[1]+s.Rational(3,100)*x[0]*x[2]
gt=s.diag(s.exp(2*f),s.exp(2*h),s.exp(-2*(f+h)))
chi=1+s.Rational(1,5)*x[0]+s.Rational(1,10)*x[2]**2
g=gt/chi;gu=g.inv();Q=s.Matrix([s.Rational(3,10)+x[0]-x[1]/5,-s.Rational(1,5)+x[2]/10,s.Rational(3,20)+x[0]*x[1]])
Z=gt*Q/2
Gamma=[[[sum(gu[k,l]*(s.diff(g[l,j],x[i])+s.diff(g[l,i],x[j])-s.diff(g[i,j],x[l]))/2 for l in range(3)) for j in range(3)] for i in range(3)] for k in range(3)]
direct=s.zeros(3);formula=s.zeros(3)
for i in range(3):
 for j in range(3):
  C=sum(gt[k,i]*s.diff(Q[k],x[j])+gt[k,j]*s.diff(Q[k],x[i]) for k in range(3))/2
  direct[i,j]=s.diff(Z[j],x[i])+s.diff(Z[i],x[j])-2*sum(Gamma[k][i][j]*Z[k] for k in range(3))-C
  formula[i,j]=sum(Q[k]*s.diff(gt[i,j],x[k]) for k in range(3))/2
  formula[i,j]+=(sum(gt[i,k]*Q[k] for k in range(3))*s.diff(chi,x[j])+sum(gt[j,k]*Q[k] for k in range(3))*s.diff(chi,x[i])-gt[i,j]*sum(Q[k]*s.diff(chi,x[k]) for k in range(3)))/(2*chi)
trace_formula=-sum(Q[k]*s.diff(chi,x[k]) for k in range(3))/2
evalall=s.lambdify(x,list(direct)+list(formula)+[s.trace(gu*direct),trace_formula],modules='mpmath',cse=True)
checks=[]
for p in [('0.2','0.3','0.4'),('-0.3','0.5','0.1'),('0.7','-0.2','0.6')]:
 v=evalall(*map(mp.mpf,p));error=max(abs(v[k]-v[k+9]) for k in range(9));trace_error=abs(v[18]-v[19]);assert error<mp.mpf('1e-55') and trace_error<mp.mpf('1e-55')
 checks.append({'point':list(map(float,p)),'E_tensor_max_error':float(error),'trace_identity_error':float(trace_error)})

rad=np.geomspace(.01,8,301);alpha=rad/(1+rad);K=1/(1+rad)**2;aK=alpha*K
profiles={'r_M':rad.tolist(),'alpha':alpha.tolist(),'K_per_M':K.tolist(),'alphaK_per_M':aK.tolist(),'positive_Theta_state_diagonal_per_M':(4*aK/3).tolist(),'positive_H_propagation_diagonal_per_M':(2*aK).tolist()}
cases={'uniform_kappa0.1':alpha*.1,'uniform_kappa0.5':alpha*.5,'coordinate_sigma0.1':np.full_like(rad,.1),'coordinate_sigma0.3':np.full_like(rad,.3)}
fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for name,sigma in cases.items():
 theta=4*aK/3-2*sigma
 toy=aK-sigma+np.sqrt(aK*aK+sigma*sigma)
 profiles[name]={'coordinate_sigma_per_M':sigma.tolist(),'Theta_state_diagonal_per_M':theta.tolist(),'isolated_H_Theta_positive_eigenvalue_per_M':toy.tolist(),'CCZ4_radial_Q_algebraic_coefficient_per_M':(K-sigma).tolist(),'CCZ4_tangential_Q_algebraic_coefficient_per_M':((1-rad)/(1+rad)**3-sigma).tolist()}
 ax[0].plot(rad,theta,label=name.replace('_',' '));ax[1].plot(rad,toy,label=name.replace('_',' '))
for a in ax:
 a.set_xscale('log');a.set_xlabel('r / M');a.grid(alpha=.2);a.axvline(1,color='.4',lw=.8,ls=':');a.legend(fontsize=7)
ax[0].axhline(0,color='.4',lw=.8);ax[0].set_ylabel('Explicit Theta diagonal / M$^{-1}$')
ax[1].set_ylabel('Isolated H–Theta toy eigenvalue / M$^{-1}$')
fig.savefig(Path(__file__).with_name('lower-order-scales.png'),dpi=160)

sample=[]
for r0 in [np.sqrt(3)/16,np.sqrt(3)/8,.5,1,2]:
 aa=r0/(1+r0);kk=1/(1+r0)**2
 sample.append({'r_M':r0,'alpha':aa,'K_per_M':kk,'alphaK_per_M':aa*kk,'uniform_kappa0.1_Theta_damping':.2*aa,'coordinate_sigma0.1_Theta_damping':.2,'coordinate_sigma0.3_Theta_damping':.6,'CCZ4_radial_Q_before_damping':kk})
out={'scope':'Independent lower-order algebra and scale checks, not a closed global spectral or evolution test','E_identity_checks':checks,'radial_samples':sample,'profiles':profiles,'Theta_positive_diagonal_analytic_bound_per_M':16/81,'H_positive_diagonal_analytic_bound_per_M':8/27,'no_source_edits':True,'no_evolutions':True}
Path(__file__).with_name('audit-results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='profiles'},indent=2))
