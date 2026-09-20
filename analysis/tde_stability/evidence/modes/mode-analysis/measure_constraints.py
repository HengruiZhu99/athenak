from mode_operator import *
from pathlib import Path
op=Operator('physical_constraints',binary=ROOT/'athena-constraint-probe',input_file=ROOT/'input-constraints.athinput')
def probe(v,label,eps=1e-3):
 cc=[]
 for sign in [1,-1]:
  lab=f'{label}_sign{sign}';out=op.advance(sign*eps*v,0,label=lab)
  assert np.array_equal(out.view(np.uint64), (sign*eps*v).view(np.uint64))
  cc.append(np.fromfile(op.directory/lab/'output.bin.constraints.bin').reshape(7,24,24,24))
 return (cc[0]-cc[1])/(2*eps)
def dx(v,axis):
 a=np.zeros((16,16,16))
 for offset,c in [(-3,-1/60),(3,1/60),(-2,3/20),(2,-3/20),(-1,-3/4),(1,3/4)]:
  sl=[slice(4,20)]*3;sl[2-axis]=slice(4+offset,20+offset)
  a+=c*v[tuple(sl)]/.25
 return a
def qconstraint(v):
 gm=[[1,2,3],[2,4,5],[3,5,6]];trace=v[1]+v[4]+v[6]
 return np.stack([v[14+i,4:20,4:20,4:20]-sum(dx(v[gm[i][j]],j) for j in range(3))+.5*dx(trace,i) for i in range(3)])
def normfield(a):
 idx=np.unravel_index(np.argmax(abs(a)),a.shape)
 return {'rms':float(np.sqrt(np.mean(a*a))),'max':float(abs(a[idx])),'xyz':[(q+.5)*.25-2 for q in idx[-3:][::-1]]}
res=[]
for n in [0,1]:
 v=np.fromfile(ROOT/f'arnoldi_s40_m32_eps0.001-mode{n}.bin').reshape(SHAPE)
 ev=json.loads((ROOT/'validated-modes.json').read_text())[n]['eigenvalue_40steps']
 c=probe(v,f'baseline{n}')
 plus=np.fromfile(ROOT/'mode_validation'/f'mode{n}_eps0.001_plus/output.bin').reshape(SHAPE)
 minus=np.fromfile(ROOT/'mode_validation'/f'mode{n}_eps0.001_minus/output.bin').reshape(SHAPE)
 rv=(plus-minus)/2e-3
 rc=probe(rv,f'baseline{n}_advanced')
 Q=qconstraint(v);Qr=qconstraint(rv)
 rec={'mode':n,'constraints':{}}
 for name,a,b in [('H',c[1,4:20,4:20,4:20],rc[1,4:20,4:20,4:20]),('M_cov',c[4:7,4:20,4:20,4:20],rc[4:7,4:20,4:20,4:20]),('Q_contrav',Q,Qr)]:
  rec['constraints'][name]=dict(initial=normfield(a),advanced=normfield(b),growth_multiplier=float(np.linalg.norm(b)/np.linalg.norm(a)),predicted=ev,relative_shape_residual=float(np.linalg.norm(b-ev*a)/np.linalg.norm(b)))
 res.append(rec);print(json.dumps(rec),flush=True)
 np.savez_compressed(ROOT/f'baseline-mode{n}-physical-constraints.npz',H=c[1,4:20,4:20,4:20],M=c[4:7,4:20,4:20,4:20],Q=Q,H_advanced=rc[1,4:20,4:20,4:20],M_advanced=rc[4:7,4:20,4:20,4:20],Q_advanced=Qr)
(ROOT/'physical-constraint-modes.json').write_text(json.dumps(res,indent=2)+'\n')
