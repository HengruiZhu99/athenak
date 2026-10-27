"""Independent chain-rule oracle for metric-connection gauge transport."""
import argparse,json
from pathlib import Path
import numpy as np
from evolution import Evolution
from puncture import PunctureEvolution
from elliptic import connection
from metric_driver import transport_rate
from principal import symbol,speeds,eigenspace_audit
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def fields(r):return 1+.2*r+.05*r*r,.2+.1*r,(1+r*r)/2+.01*r*r,.03*r*r,-.2*r+.02*r*r
def metric_rate(r):
 g,dg,al,arr,be=fields(r);db=-.2+.04*r
 return be*dg-2*al*arr+4*g*(db-be/r)/3
r=np.array([.03,.1,.4,.8,1.]);g,dg,al,arr,be=fields(r);eps=1e-25
gt=metric_rate(r);gtp=np.imag(metric_rate(r+1j*eps))/eps
Vt=np.imag(connection(r,g+1j*eps*gt,dg+1j*eps*gtp))/eps
rc=r+1j*eps;gc,dgc,_,_,_=fields(rc);Vp=np.imag(connection(rc,gc,dgc))/eps
expected=Vt-be*Vp
actual=transport_rate(r,g,dg,al,1.02*r,arr,.06*r,be,-.2+.04*r,np.full(len(r),.04))
error=float(max(abs(expected-actual)));assert error<1e-12
# Verify that the metric rate in the oracle is the native tensor rate.
e=Evolution(65,a.library);q=np.zeros((len(r),8));d=q.copy();dd=q.copy();q[:,1]=g-1;d[:,1]=dg;dd[:,1]=.1;q[:,2]=arr;d[:,2]=.06*r;dd[:,2]=.06;q[:,6]=.01*r*r;d[:,6]=.02*r;dd[:,6]=.02
R,S,_=e.kernel.parts(r,q,d,dd,be,-.2+.04*r,np.full(len(r),.04))
assert np.max(abs(R[:,1]-gt))<1e-12 and np.all(S[:,1]==0)
rr=np.logspace(-3,-6,4);gg=1+.2*rr+.1*rr*rr
v=transport_rate(rr,gg,.2+.2*rr,.7*rr*rr,1.4*rr,.3*rr*rr,.6*rr,-.9*rr+.03*rr*rr,-.9+.06*rr,np.full(4,.06))
af=abs(v-(-.56));assert np.all(np.diff(af)<0) and af[-1]<1e-5
P=symbol(1,1,1,1,1,0,.005,.02,1.5,connection_driver='metric');audit=eigenspace_audit(P,-speeds(1,1,1,1,1,0,.005,.02,1.5));assert audit['complete']
x=PunctureEvolution(129,a.library,shift_source='lapse-inner',connection_driver='metric');y=PunctureEvolution(129,a.library,shift_source='lapse-inner');q=y.initial();q[40,5]+=.001
rx,s=x.rhs(q,return_state=True);ry=y.rhs(q)
assert np.array_equal(rx[:,:9],ry[:,:9]) and s[40,5]==q[40,5]
qq=q.copy();qq[40,5]+=.002;rr2=x.rhs(qq);assert np.array_equal(rr2[:,9],rx[:,9])
reg=Evolution(65,a.library,restoring=4,shift_source='lapse-inner',connection_driver='metric');eq=float(np.max(abs(reg.rhs(np.zeros((65,10))))));assert eq<2e-9
z=np.zeros((65,10));z[:,0]=.001*reg.r**2;z[:,1]=-.001*reg.r**2;z[:,6]=.002*reg.r**2;reg.step(z,1e-8);assert reg.stats['max_relative_null']<2e-15
record=dict(complex_step_connection_rate_error=error,native_metric_rate_error=float(np.max(abs(R[:,1]-gt))),AF_errors=af.tolist(),scri_eigenspaces=audit,independent_Lambda_retained=True,unchanged_first_nine_rates=True,driver_not_sourced_by_Lambda=True,Minkowski_max_RHS=eq,stage_null=reg.stats['max_relative_null'],scope='Gauge source uses metric-derived V; native Einstein/Z4 system unchanged. Scalar audit is not a 3D or uniform puncture proof.')
a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
