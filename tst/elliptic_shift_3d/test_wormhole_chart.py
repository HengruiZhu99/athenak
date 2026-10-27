#!/usr/bin/env python3
"""Independent radial gauge/chart audit; no coupled evolution claimed here."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from wormhole_data import Jet
from gauge import lapse_coefficient,restoring_coefficient,collapsed_background_correction

data=json.loads(subprocess.check_output([sys.argv[1]],text=True))
errors=[];chart_error=0.;matched_error=0.;smooth_error=0.
for case in data['cases']:
 r=case['r'];a=case['alpha'];p=case['trace'];b=case['beta'];O=(1-r*r)/2;A=(1+r*r)/2
 xi=restoring_coefficient(r,1.,1.,profile='physical-log')
 # Original spherical total-lapse equation, using long double to avoid losing
 # its analytic AF cancellation before comparison with the factored 3D source.
 z=np.longdouble(r)**2;al=np.longdouble(a);om=(1-z)/2;ar=(1+z)/2
 xx=1+2*(1-z)**4
 ref=np.longdouble(b)*case['alpha_r']+(-lapse_coefficient(np.longdouble(r),al)*(np.longdouble(p)+3)
   -2*xx*al*(al-ar)+al*np.longdouble(b)*np.longdouble(r)+z
   +collapsed_background_correction(np.longdouble(r),al-ar))/om
 errors.append(abs(float(ref)-case['rate'])/max(abs(case['rate']),1e-28))
 j=Jet(r,1);omega=(1-j*j)/2
 radial=[j**4/(j**4+omega**4),j*j/(j*j+omega**4)]
 radial.append((1+j*j)/2*radial[1])
 n=np.array([1,2,2])/3
 for ref,actual in zip(radial,case['refs']):
  expected=np.r_[ref.v,ref.d*n,(ref.dd*np.outer(n,n)+ref.d/r*(np.eye(3)-np.outer(n,n))).ravel()]
  chart_error=max(chart_error,float(np.max(abs(expected-np.array(actual)))/max(1.,np.max(abs(expected)))))
 matched=[radial[0],radial[2]]
 if r<.3:
  k=.37754694621653295;rp=j+k/6
  inner=[k*k*j**4/(omega**2*rp**4),omega*j*j/rp**2]
  t=Jet(1)
  if r>.2:
   y=(.3-j)*10;t=y**8/(y**8+(1-y)**8)
  matched=[t*v+(1-t)*w for v,w in zip(inner,matched)]
 for ref,actual in zip(matched,case['isotropic_refs']):
  expected=np.r_[ref.v,ref.d*n,(ref.dd*np.outer(n,n)+ref.d/r*(np.eye(3)-np.outer(n,n))).ravel()]
  matched_error=max(matched_error,float(np.max(abs(expected-np.array(actual)))/max(1.,np.max(abs(expected)))))
 k=.37754694621653295;rp=j+k/6
 smooth=[k*k*j**4/(16*omega**6*rp**4+k*k*j**4),(1+j*j)/2*j*j/(j*j+16*omega**4*rp**2)]
 for ref,actual in zip(smooth,case['asymptotic_refs']):
  expected=np.r_[ref.v,ref.d*n,(ref.dd*np.outer(n,n)+ref.d/r*(np.eye(3)-np.outer(n,n))).ravel()]
  smooth_error=max(smooth_error,float(np.max(abs(expected-np.array(actual)))/max(1.,np.max(abs(expected)))))
assert max(errors)<2e-13,errors
assert chart_error<2e-14,chart_error
assert matched_error<2e-14,matched_error
assert smooth_error<2e-14,smooth_error
print(json.dumps(dict(cases=len(errors),lapse_relative_error=max(errors),chart_relative_error=chart_error,
 matched_chart_relative_error=matched_error,smooth_chart_relative_error=smooth_error,moving_reference_time_error=data['moving_reference_time_error'])))
