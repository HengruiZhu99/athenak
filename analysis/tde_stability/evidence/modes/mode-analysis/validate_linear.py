from mode_operator import *
j=json.loads((ROOT/'arnoldi_s40_m32_eps0.001_z4c_extrap_order2-results.json').read_text())
op=Operator('linear_mode_validation',overrides=('z4c/extrap_order=2',))
results=[]
for n,m in enumerate(j['modes'][:2]):
 v=np.fromfile(m['file']).reshape(SHAPE);mu=m['real']
 rs={e:op.response(v,40,e,label=f'mode{n}_eps{e:g}') for e in [1e-3,1e-4]}
 r1=op.response(v,1,1e-4,label=f'mode{n}_onestep')
 q=np.vdot(v[ACTIVE],r1[ACTIVE])/np.vdot(v[ACTIVE],v[ACTIVE])
 c=np.fromfile(ROOT/f'arnoldi_s40_m32_eps0.001-mode{n}.bin').reshape(SHAPE)
 rec=dict(index=n,gamma=m['gamma'],mu=mu,eigen_residual=discrepancy(mu*v,rs[1e-4]),amplitude_convergence=discrepancy(rs[1e-3],rs[1e-4]),one_step_residual=discrepancy(mu**(1/40)*v,r1),one_step_gamma_active=float(np.log(q)/DT),cosine_with_cubic_active=float(np.vdot(v[ACTIVE],c[ACTIVE])/(np.linalg.norm(v[ACTIVE])*np.linalg.norm(c[ACTIVE]))),profile=profile(v))
 results.append(rec);print(json.dumps({k:x for k,x in rec.items() if k!='profile'},indent=2),flush=True)
(ROOT/'linear-validated-modes.json').write_text(json.dumps(results,indent=2)+'\n')
