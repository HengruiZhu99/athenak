from pathlib import Path
import numpy as np,json,hashlib
root=Path(__file__).resolve().parents[3];raw=root/'output/hyperbolic-core-shell/aurora/stability-spherical-overlap-puncture-v1/output'
n=16;o=3;total=189512;coords=np.genfromtxt(root/'output/hyperbolic-core-shell/spherical-overlap-local-v1/coordinates-n16.csv',delimiter=',',names=True)
xyz=np.c_[coords['x'],coords['y'],coords['z']];h=.18/n
ids={};cells={}
for m in range(26):
 for k in range(n):
  for j in range(n):
   for i in range(n):p=m*n**3+(k*n+j)*n+i;ids[m,i,j,k]=p;cells[p]=(m,i,j,k)
p=26*n**3
for m in range(14):
 for k in range(-o,n+o):
  for j in range(-o,n+o):
   for i in range(0 if m>=8 else -o,n+o):
    if 0<=i<n and 0<=j<n and 0<=k<n:continue
    ids[m,i,j,k]=p;cells[p]=(m,i,j,k);p+=1
assert p==total
coeff=np.array([1,-4,6,-4,1])
def ko(q,p):
 cell=list(cells[p]);value=0;terms=[]
 for axis in range(3):
  for target in range(5):
   donors=[]
   for a in range(5):
    d=cell.copy();d[axis+1]+=a-target
    if tuple(d) not in ids:break
    donors.append(ids[tuple(d)])
   if len(donors)!=5:continue
   w=np.maximum(0,1-coords['r'][donors]**2/.64)**6
   fourth=sum(coeff*(q[donors]-q[donors[0]]))
   contribution=-.5/(256*h)*coeff[target]*np.mean(w)*fourth
   value+=contribution;terms.append(contribution)
 return value,len(terms)
initial=np.loadtxt(root/'output/hyperbolic-core-shell/spherical-overlap-local-v1/puncture-n16.txt',skiprows=1)[:,0]
accepted=np.memmap(raw/'puncture.checkpoint',dtype='<f8',mode='r',offset=136,shape=(total,31))[:,0]
failed=np.memmap(raw/'puncture.failed-stage.bin',dtype='<f8',mode='r',shape=(total,31))[:,0]
near=np.flatnonzero(np.max(abs(xyz-xyz[4095]),axis=1)<1e-14)
result=[]
for p in near:
 d=dict(point=int(p),cell=cells[p],xyz=xyz[p].tolist(),initial_chi=float(initial[p]),accepted_chi=float(accepted[p]),failed_chi=float(failed[p]))
 d['initial_KO'],d['terms']=ko(initial,p);d['accepted_KO'],_=ko(accepted,p);result.append(d)
out=Path(__file__).with_name('ko-copy-analysis.json');out.write_text(json.dumps(dict(note='Read-only diagnostic: original V10 block-local KO at identical physical points, no trajectory modification',h=h,copies=result),indent=2)+'\n')
print(out.read_text())
