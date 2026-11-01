"""All-direction lapse/shift separation on a verified N16 GD1/IT1 snapshot."""
from pathlib import Path
import argparse,hashlib,ctypes as C
import numpy as np,struct,json
p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--coordinates',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
cp=args.snapshot
assert hashlib.sha256(cp.read_bytes()).hexdigest()==args.sha256
lib=C.CDLL(str(args.library.resolve()));fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');lib.inner_gauge_symbol.argtypes=[C.c_int]+[C.c_double]*3+[fp,fp,fp]
with cp.open('rb') as f:h=struct.unpack('<16s4i3Q2d',f.read(72))
assert h[0] in [b'HYP3D_MPI_GD1\0\0\0',b'HYP3D_MPI_IT1\0\0\0'] and h[1:5]==(16,31,2,1)
u=np.memmap(cp,dtype='<f8',offset=168,mode='r',shape=(h[5],31))
x=np.genfromtxt(args.coordinates,delimiter=',',names=True)
assert len(x)==h[5]
r=x['r'];o=(1-r*r)/2;A=(1+r*r)/2;alpha=u[:,18]+A;chi=u[:,0];F=alpha**2+4*o*o*alpha;eta=1.0001*alpha/(alpha+.0001*A)
g=np.empty((len(r),3,3));g[:,0,0]=1+u[:,1];g[:,0,1]=g[:,1,0]=u[:,2];g[:,0,2]=g[:,2,0]=u[:,3];g[:,1,1]=1+u[:,4];g[:,1,2]=g[:,2,1]=u[:,5];g[:,2,2]=1+u[:,6]
values,vectors=np.linalg.eigh(g);assert values.min()>0

def smooth(z):
 z=np.clip(z,0,1);t=np.minimum(z,1-z);v=t**5*(126+t*(-420+t*(540+t*(-315+70*t))));return np.where(z>.5,1-v,v)
w=smooth((.8-r)/.24);t=np.clip((r-.8)/.1,0,1);C=t**8/(t**8+(1-t)**8)
mu=(1-C)*4+C*1.5*chi*F
result={}
for mode in range(3):
 s=np.zeros(len(r)) if mode==0 else smooth((.56-r)/.36) if mode==1 else smooth((.9-r)/.7)
 outer_mu=mu if mode!=2 else 1.5*chi*F
 muk=s+(1-s)*outer_mu
 k=(1-s)*F+s*2*o*alpha
 eu=(1-s)*eta*chi*w if mode!=2 else np.zeros(len(r))
 met=s*o*np.maximum(o*o*chi,1e-5)
 b=(1-s)*.25*w+s*2/o if mode!=2 else s*2/o
 gap=4*muk/3-chi*k-(eu*values[:,-1]+met)*b
 idx=int(np.argmin(gap))
 result[str(mode)]=dict(min_gap_over_Q=float(gap[idx]),negative_cells=int(sum(gap<0)),minimum_cell=idx,r=float(r[idx]),alpha=float(alpha[idx]),chi=float(chi[idx]),metric=g[idx].tolist(),normal=vectors[idx,:,-1].tolist())
 # Check the analytic bound against the native symbol at its extremizing direction.
 normal=np.ascontiguousarray(vectors[idx,:,-1]);P=np.zeros((26,26))
 assert lib.inner_gauge_symbol(mode,float(r[idx]),float(alpha[idx]),float(chi[idx]),np.ascontiguousarray(g[idx]),normal,P)==0
 Q=normal@np.linalg.inv(g[idx])@normal
 actual=4*(P[17,13]+P[17,23])/3+chi[idx]*P[16,11]-(P[16,20:23]@P[20:23,16])/Q
 assert abs(actual-gap[idx])<1e-8
 result[str(mode)]['native_gap_crosscheck_error']=float(abs(actual-gap[idx]))
report=dict(snapshot_sha256=args.sha256,coordinates_sha256=hashlib.sha256(args.coordinates.read_bytes()).hexdigest(),schema=h[0].strip(b'\0').decode(),t_over_M=h[8]*3,cells=h[5],modes=result,scope='Worst direction from lambda_max(g); positive separation rules out only lapse/longitudinal-shift coincidence at these frozen cells, not every possible defect or uniform bounds.')
print(json.dumps(report,indent=2))
args.output.write_text(json.dumps(report,indent=2)+'\n')
