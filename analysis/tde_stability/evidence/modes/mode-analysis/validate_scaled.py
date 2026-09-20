from mode_operator import *
p=next(ROOT.glob('*scaledk03-results.json'));j=json.loads(p.read_text())
op=Operator('scaled_mode_validation',binary=ROOT/'athena-lapse-scaled',input_file=ROOT/'input-lapse-scaled.athinput',dt=.0375,overrides=('time/cfl_number=.15','z4c/damp_lapse_scaled=true','z4c/damp_kappa1=.3'))
probe=Operator('scaled_physical_constraints',binary=ROOT/'athena-constraint-probe',input_file=ROOT/'input-constraints.athinput')
def con(v,label):
 arr=[]
 for sign in [1,-1]:
  lab=f'{label}_sign{sign}';probe.advance(sign*1e-3*v,0,label=lab)
  arr.append(np.fromfile(probe.directory/lab/'output.bin.constraints.bin').reshape(7,24,24,24))
 return (arr[0]-arr[1])/2e-3
def dx(v,axis):
 a=np.zeros((16,16,16))
 for o,c in [(-3,-1/60),(3,1/60),(-2,3/20),(2,-3/20),(-1,-3/4),(1,3/4)]:
  sl=[slice(4,20)]*3;sl[2-axis]=slice(4+o,20+o);a+=c*v[tuple(sl)]/.25
 return a
def qcon(v):
 gm=[[1,2,3],[2,4,5],[3,5,6]];tr=v[1]+v[4]+v[6]
 return np.stack([v[14+i,4:20,4:20,4:20]-sum(dx(v[gm[i][a]],a) for a in range(3))+.5*dx(tr,i) for i in range(3)])
res=[]
for n,m in enumerate(j['modes'][:2]):
 v=np.fromfile(m['file']).reshape(SHAPE);mu=m['real']
 r=op.response(v,80,1e-3,label=f'mode{n}_map80')
 rlow=op.response(v,80,3e-4,label=f'mode{n}_map80_epslow')
 r1=op.response(v,1,1e-3,label=f'mode{n}_map1')
 q=float(np.vdot(v[ACTIVE],r1[ACTIVE])/np.vdot(v[ACTIVE],v[ACTIVE]))
 c=con(v,f'mode{n}_initial');cr=con(r,f'mode{n}_advanced');Q=qcon(v);Qr=qcon(r)
 rec={'mode':n,'gamma':m['gamma'],'mu':mu,'eigen_residual':discrepancy(mu*v,r),'amplitude_convergence':discrepancy(r,rlow),'one_step_eigen_residual':discrepancy(mu**(1/80)*v,r1),'one_step_gamma_active':float(np.log(q)/.0375),'profile':profile(v),'physical_constraints':{}}
 for name,a,b in [('H',c[1,4:20,4:20,4:20],cr[1,4:20,4:20,4:20]),('M_cov',c[4:7,4:20,4:20,4:20],cr[4:7,4:20,4:20,4:20]),('Q_contrav',Q,Qr)]:
  rec['physical_constraints'][name]={'initial_rms':float(np.sqrt(np.mean(a*a))),'multiplier':float(np.linalg.norm(b)/np.linalg.norm(a)),'relative_eigen_residual':float(np.linalg.norm(b-mu*a)/np.linalg.norm(b))}
 res.append(rec);print(json.dumps({k:x for k,x in rec.items() if k!='profile'},indent=2),flush=True)
 (ROOT/'scaled-validated-modes.json').write_text(json.dumps(res,indent=2)+'\n')
