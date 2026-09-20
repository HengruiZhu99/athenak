"""Off-grid and tensor-coefficient checks for independent radial spectrum."""
import radial_collocation as rc
import radial_modes as rm
from scipy.fft import dct
import numpy as np
from numpy.polynomial.chebyshev import chebval,chebder
import sympy as s
from pathlib import Path
import json

# Independent radial covariant-divergence identities.
r=s.symbols('r',positive=True)
a,ch,ss,th,q,m,kk=[s.Function(z)(r) for z in ('alpha','chi','sigma','Theta','q','m','K')]
A=a*ch*s.diff(q,r)-ss*th;B=a*ch*q/r-ss*th
cov_div_minus_trace=s.diff(A,r)+2*(A-B)/r-s.diff(ch,r)/ch*(A-B)-s.diff(A+2*B,r)
expected=-a*s.diff(ch,r)*s.diff(q,r)-(2*ch*s.diff(a,r)+a*s.diff(ch,r))*q/r+2*ss*s.diff(th,r)+2*s.diff(ss,r)*th
assert s.simplify(cov_div_minus_trace-expected)==0
volume=ch**s.Rational(-3,2)
D_M=s.diff(volume*ch*m*r*r,r)/(volume*r*r)
assert s.simplify(D_M-ch*s.diff(m,r)-(2*ch/r-s.diff(ch,r)/2)*m)==0

def verify(item):
  n=item['polynomial_degree'];inner=item['inner_M'];outer=item['outer_M'];rate=item['rate_per_M'];scaled=item['lapse_scaled'];top=item['leading_profiles'][0];p=top['profile'];lam=complex(top['eigenvalue_real_per_M'],top['eigenvalue_imag_per_M'])
  u=np.array([np.array(p[z+'_real_normalized'])+1j*np.array(p[z+'_imag_normalized']) for z in ('Theta','Q','H','M')])
  coefficients=dct(u,type=1,axis=1)/n;coefficients[:,[0,-1]]/=2
  # Uniform off-collocation samples excluding the inner/outer endpoints.
  x=np.linspace(-1+1e-5,1-1e-5,2003);rr=(inner+outer)/2+(outer-inner)*x/2
  un=np.array([chebval(x,c) for c in coefficients]);du=np.array([chebval(x,chebder(c))*2/(outer-inner) for c in coefficients])
  v=rm.values(rr,rate,scaled);a,ch,b,k,ss,ap,cp,bp,_=v[:9];sp=np.zeros_like(a) if scaled else rate*ap
  th,q,H,m=un;dt,dq,dH,dm=du
  pieces=[np.array([b*dt,a*H/2,a*ch*dq/2,a*ch*q/rr,-2*ss*th]),np.array([b*dq,2*a*m,2*a*dt,-2*ss*q]),np.array([b*dH,-2*a*ch*dm,(-4*a*ch/rr+a*cp-4*ch*ap)*m,2*a*k*H,4*a*ch*k*dq,-4*ss*k*th]),np.array([b*dm,(bp+a*k)*m,-a*dH/2,-ap*H,-a*cp*dq,-(2*ch*ap+a*cp)*q/rr,2*ss*dt,2*sp*th])]
  rhs=np.array([sum(v) for v in pieces]);err=lam*un-rhs
  return {'source':item.get('source_file'),'degree':n,'inner_M':inner,'outer_M':outer,'rate_per_M':rate,'lapse_scaled':scaled,'outer_wave_power':item.get('outer_wave_power',0.),'growth_rate_per_M':lam.real,'off_grid_samples':len(rr),'residual_relative_to_sum_of_term_L2_norms':[float(np.linalg.norm(err[i])/(sum(np.linalg.norm(v) for v in pieces[i])+np.linalg.norm(lam*un[i]))) for i in range(4)],'relative_Chebyshev_tail_last8_L2':float(np.linalg.norm(coefficients[:,-8:])/np.linalg.norm(coefficients))}

if __name__=='__main__':
  root=Path(__file__).parent;out=[]
  files=sorted(root.glob('collocation-*.json'))+sorted(root.glob('annulus-*.json'))
  for f in files:
    item=json.loads(f.read_text());item['source_file']=f.name;out.append(verify(item))
  (root/'radial-offgrid-validation.json').write_text(json.dumps({'coefficient_symbolic_checks':{'covariant_momentum_modification':'exact symbolic zero','physical_momentum_divergence':'exact symbolic zero','wave_principal_factorization':'exact symbolic zero on import'},'offgrid_checks':out},indent=2)+'\n')
  print(json.dumps(out,indent=2))
