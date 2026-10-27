#!/usr/bin/env python3
"""Native-tensor collapsed-lapse limit with finite, independent Khat/Theta."""
import argparse,json
import numpy as np
from kernel import Kernel
p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args();kernel=Kernel(a.library);records=[]
r=np.array([.001,.0005,.00025]);gamma=1.3;A0=.7;X=2.;B=.03;Khat=-.8;b1=.2
O=(1-r*r)/2;A=(1+r*r)/2;alpha=A0*r**gamma;chi=X*r*r
q=np.zeros((len(r),8));d=q.copy();dd=q.copy()
q[:,0]=chi-1;d[:,0]=2*X*r;dd[:,0]=2*X
q[:,2]=B;q[:,3]=Khat+3;q[:,4]=.003;q[:,5]=.01
q[:,6]=alpha-A;d[:,6]=A0*gamma*r**(gamma-1)-r;dd[:,6]=A0*gamma*(gamma-1)*r**(gamma-2)-1
beta=b1*r+1.5*A0*B/gamma*r**(gamma+1)
db=b1+1.5*A0*B*(gamma+1)/gamma*r**gamma;ddb=1.5*A0*B*(gamma+1)*r**(gamma-1)
R,S,diag=kernel.parts(r,q,d,dd,beta,db,ddb,total_chi=chi)
# The old symmetric source cannot have a stationary zero lapse with bounded
# curvature: its positive constant tends to xi/2, independent of these errors.
symmetric=R[:,6]+(S[:,6]-(alpha+A)*q[:,6])/O
assert abs(symmetric[-1]-.5)<5e-4
assert all(max(abs(diag[:,i]))>0 for i in range(4))
for family in ['linear','rational','blended-log']:
 nu_inner=6. if family=='rational' else 1.
 oldF=alpha*alpha+4*O*O*alpha
 F=oldF if family=='linear' else (alpha*alpha+4*O*O*alpha*(A+.1)/(alpha+.1) if family=='rational' else (1-4*O*O)*alpha*alpha+8*O**3*alpha)
 strength=1+(3*nu_inner-1)*(1-r*r)**4
 source=S[:,6]-(F-oldF)*(Khat+3)-2*strength*alpha*q[:,6]+(1-r*r)**4*r*r*q[:,6]/A
 normalized=(R[:,6]+source/O)/alpha;expected=b1*gamma-2*nu_inner*Khat
 errors=abs(normalized-expected);orders=np.log2(errors[:-1]/errors[1:])
 assert min(orders)>1 and errors[-1]<5e-4*nu_inner,(family,errors,orders)
 records.append(dict(family=family,expected_limit=expected,values=normalized.tolist(),errors=errors.tolist(),orders=orders.tolist()))
print(json.dumps(dict(native_collapsed_trumpet_limit=True,independent_constraints_retained=True,
    symmetric_Minkowski_source_obstruction_values=symmetric.tolist(),records=records,
    scope='Leading physical1+log source with finite independent Khat; no nonlinear stability admission')))
