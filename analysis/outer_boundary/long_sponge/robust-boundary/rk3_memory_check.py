"""Explicitly form and apply the RK3 exterior-memory convolution."""
exec(open(__file__.replace('rk3_memory_check.py','exact_elimination_check.py')).read().split('B=prepare(1.);A=prepare(1.5)')[0])
dt=.6;I=np.eye(20*N);X=dt*L;R=I+X+X@X/2+X@X@X/6
II=R[np.ix_(inside,inside)];IE=R[np.ix_(inside,outside)];EI=R[np.ix_(outside,inside)];EE=R[np.ix_(outside,outside)]
rng=np.random.default_rng(674);u0=np.zeros(20*N,complex);u0[inside]=rng.normal(size=len(inside))*1e-8
u=u0.copy();hist=[u0[inside].copy()];kernels=[];power=EI.copy();checks=[]
for n in range(16):
 kernels.append(IE@power);power=EE@power
 got=II@hist[-1]
 for j in range(n):got+=kernels[n-1-j]@hist[j]
 u=R@u;checks.append({'step':n+1,'relative_error':float(np.linalg.norm(got-u[inside])/np.linalg.norm(u[inside]))});hist.append(got)
zero=II@np.zeros(len(inside),complex)
for K in kernels:zero+=K@np.zeros(len(inside),complex)
out={'scope':'Actual matrix-memory convolution, not merely exterior state carry. Identical original explicit RK3 and bulk stencil. Zero initial exterior is assumed in this check; nonzero exterior adds R_IE R_EE^n uE0.','N':N,'interior_cells':len(inside)//20,'dt':dt,'checks':checks,'max_relative_error':max(r['relative_error']for r in checks),'zero_max':float(abs(zero).max())};(p/'rk3-memory-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
