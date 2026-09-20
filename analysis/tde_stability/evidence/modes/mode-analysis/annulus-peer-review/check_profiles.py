"""Independent read-only check of packaged original-equation eigenprofiles."""
from pathlib import Path
import json
import numpy as np
from scipy.fft import dct
from numpy.polynomial.chebyshev import chebval,chebder
root=Path(__file__).resolve().parent
source=root.parents[1]/'constraint-lower-order'
rows=[]
for f in sorted(source.glob('annulus-*-n128.json')):
 j=json.loads(f.read_text());item=j['leading_profiles'][0];p=item['profile'];n=j['polynomial_degree']
 lam=complex(item['eigenvalue_real_per_M'],item['eigenvalue_imag_per_M']);lo=j['inner_M'];hi=j['outer_M'];k=j['rate_per_M'];power=j['outer_wave_power']
 raw=np.array([np.array(p[name+'_real_normalized'])+1j*np.array(p[name+'_imag_normalized']) for name in ['Theta','Q','H','M']])
 coef=dct(raw,type=1,axis=1)/n;coef[:,[0,-1]]/=2
 x=np.r_[-1,np.linspace(-.99999,.99999,1001),1];r=(hi+lo)/2+(hi-lo)*x/2
 u=np.array([chebval(x,c) for c in coef]);du=np.array([chebval(x,chebder(c))*2/(hi-lo) for c in coef])
 t,q,h,m=u;tp,qp,hp,mp=du
 R=1+r;a=r/R;ch=a*a;b=r/R**2;K=1/R**2;ap=1/R**2;cp=2*r/R**3;bp=(1-r)/R**3
 sig=np.full_like(r,k) if j['lapse_scaled'] else k*a
 sp=np.zeros_like(r) if j['lapse_scaled'] else k*ap
 # Independent tensor expansion of radial C_rr=q', C_tangent=q/r.
 A=a*ch*qp-sig*t
 B=a*ch*q/r-sig*t
 # D_j T^j_r-D_r trT after symbolic differentiation/cancellation.
 divT=-a*cp*qp-(2*ch*ap+a*cp)*q/r+2*sig*tp+2*sp*t
 divM=ch*mp+(2*ch/r-cp/2)*m
 rhs=np.array([b*tp+a*(h+ch*(qp+2*q/r))/2-2*sig*t,
 b*qp+2*a*(m+tp)-2*sig*q,
 b*hp-2*a*divM-4*ch*m*ap+2*a*K*h+4*K*A,
 b*mp+(bp+a*K)*m-a*hp/2-ap*h+divT])
 err=lam*u-rhs
 norm=np.linalg.norm(lam*u,axis=1)+np.linalg.norm(rhs,axis=1)
 norm=np.maximum(norm,1e-100)
 endpoint=np.abs(err[:,[0,-1]])/np.maximum(np.max(abs(lam*u)+abs(rhs),axis=1)[:,None],1e-100)
 bc=lam*u[:2,-1]+(a[-1]**2-b[-1])*(du[:2,-1]+power*u[:2,-1]/r[-1])
 rec={'source':str(f),'gamma':lam.real,'sigma_constant':j['lapse_scaled'],'rate':k,'p':power,'original_equation_relative_l2':(np.linalg.norm(err,axis=1)/norm).tolist(),'endpoint_residual_relative_to_global_peak':endpoint.tolist(),'boundary_residual_relative_to_field_norm':float(np.linalg.norm(bc)/np.linalg.norm(u)),'all_finite_spectrum':j['all_finite_spectrum_summary']}
 rows.append(rec)
(root/'profile-checks.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
