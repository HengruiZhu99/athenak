from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parents[2]/'review/stability-modes-20260919/mode-analysis';sys.path.insert(0,str(OLD))
from mode_operator import Operator,SHAPE,ACTIVE
op=Operator(str(ROOT/'physical_probe'),binary=OLD/'athena-constraint-probe',input_file=OLD/'input-constraints.athinput')
def constraints(v,label):
 out=[]
 for sign in [1,-1]:
  lab=f'{label}_{sign}';op.advance(sign*1e-4*v,0,label=lab);out.append(np.fromfile(op.directory/lab/'output.bin.constraints.bin').reshape(7,24,24,24))
 return (out[0]-out[1])/2e-4

def dx(v,axis):
 out=np.zeros((16,16,16))
 for o,c in [(-3,-1/60),(3,1/60),(-2,3/20),(2,-3/20),(-1,-3/4),(1,3/4)]:
  sl=[slice(4,20)]*3;sl[2-axis]=slice(4+o,20+o);out+=c*v[tuple(sl)]/.25
 return out

def Q(v):
 g=[[1,2,3],[2,4,5],[3,5,6]];tr=v[1]+v[4]+v[6]
 return np.stack([v[14+i,4:20,4:20,4:20]-sum(dx(v[g[i][j]],j) for j in range(3))+.5*dx(tr,i) for i in range(3)])
v=np.fromfile(ROOT/'mode0-real.bin').reshape(SHAPE)
p=np.fromfile(ROOT/'validation_calls/mode0_8_real_plus/output.bin').reshape(SHAPE);m=np.fromfile(ROOT/'validation_calls/mode0_8_real_minus/output.bin').reshape(SHAPE);r=(p-m)*np.max(abs(v))/2e-4
c=constraints(v,'initial');cr=constraints(r,'advanced');q=Q(v);qr=Q(r)
zz,yy,xx=np.meshgrid(np.arange(16),np.arange(16),np.arange(16),indexing='ij');codim=((xx==0)|(xx==15)).astype(int)+((yy==0)|(yy==15)).astype(int)+((zz==0)|(zz==15)).astype(int)
res={}
for name,a,b in [('physical_H',c[1,4:20,4:20,4:20],cr[1,4:20,4:20,4:20]),('physical_M_cov',c[4:7,4:20,4:20,4:20],cr[4:7,4:20,4:20,4:20]),('volume_D6_Q',q,qr)]:
 e=abs(a)**2
 if e.ndim==4:e=e.sum(axis=0)
 idx=np.unravel_index(np.argmax(abs(a)),a.shape)
 res[name]={'norm_gain_0p3M':float(np.linalg.norm(b)/np.linalg.norm(a)),'rms':float(np.sqrt(np.mean(abs(a)**2))),'energy_fraction_by_codimension':[float(e[codim==k].sum()/e.sum()) for k in range(4)],'peak_xyz':[(z+.5)*.25-2 for z in idx[-3:][::-1]]}
res['scope']='Linear physical-constraint increments about exact background. H/M central finite differences at peak1e−4. Q uses actual volume D6 conformal-background linearization, not helper D4 functional. Candidate is unconverged; norm gains are directional, not eigenvalues.'
(ROOT/'physical-constraints.json').write_text(json.dumps(res,indent=2)+'\n');np.savez_compressed(ROOT/'physical-constraints.npz',H=c[1,4:20,4:20,4:20],M=c[4:7,4:20,4:20,4:20],Q=q,H_advanced=cr[1,4:20,4:20,4:20],M_advanced=cr[4:7,4:20,4:20,4:20],Q_advanced=qr)
print(json.dumps(res,indent=2))
