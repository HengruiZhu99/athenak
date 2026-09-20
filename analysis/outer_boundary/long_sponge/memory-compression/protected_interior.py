"""P-orthogonal reduction with physical interior coordinates retained exactly.

For stable complete R, R* P R-P=-I. Write T=-P_EE^{-1}P_EI and
Phi=[[I,0],[T,V]]. The first P-Galerkin test row is [I,0], so the
retained update is exactly the physical top row evaluated at the lifted
exterior. Orthogonal compression is P-contractive in exact arithmetic.
This is a different approximation to memory, not stable-pole fitting.
"""
exec(open(__file__.replace('protected_interior.py','balanced_memory.py')).read().split('P=la.solve_discrete_lyapunov')[0])

# Complete weighted R is block circulant, so its Lyapunov metric can be
# constructed from20x20 Fourier systems instead of a full dense solve.
RW=[];PW=[]
for v in V:
    x=dt*v*weights[:,None]/weights[None,:]
    rr=np.eye(20)+x+x@x/2+x@x@x/6
    pp=la.solve_discrete_lyapunov(rr.conj().T,np.eye(20),method='bilinear')
    RW.append(rr);PW.append((pp+pp.conj().T)/2)
PK=np.fft.ifft(np.array(PW),axis=0)
Pfull=np.block([[PK[(i-j)%N] for j in range(N)] for i in range(N)])
Pfull=(Pfull+Pfull.conj().T)/2
Pe=Pfull[np.ix_(outside,outside)];Pei=Pfull[np.ix_(outside,inside)]
T=-la.solve(Pe,Pei,assume_a='pos')
out={'scope':__doc__,'parameters':vars(args)|{'h':h,'dt':dt},'full_rho':rho(R),
 'metric_min_eigen':float(la.eigvalsh(Pfull)[0]),'metric_max_eigen':float(la.eigvalsh(Pfull)[-1]),
 'full_Lyapunov_relative_residual':float(la.norm(R.conj().T@Pfull@R-Pfull+np.eye(len(R)))/np.sqrt(len(R))),
 'runs':[]}
old,_=full_matrix(n=n,h=h,angle_y=angle,degree=1,damping=True,lapse_damping=.1)
wo,vo=la.eig(old);j=np.argmax(wo.real)
uold=(vo[:,j].reshape(20,n)*weights[:,None]).T.reshape(-1);uold/=la.norm(uold)
u0=np.zeros(20*N,complex);u0[inside]=uold
out['old_eigenvalue']=[float(wo[j].real),float(wo[j].imag)]

# Generate balanced exterior directions, then reproject in the COMPLETE
# Lyapunov metric. Include initial lift cancellation for the selected seed.
# No all-initial-data guarantee is implied by one protected direction.
F=la.cholesky(Pe,lower=False)
def sqroot(M):
    M=(M+M.conj().T)/2;w,U=la.eigh(M)
    return U*np.sqrt(np.maximum(w,0))
At=A+C@T
Et=E-T@C
Bt=B+E@T-T@At
out['lifted_exterior_rho']=rho(Et)
# Reduce the actual shifted realization; reducing the original E directions
# would waste modes on a different input/output problem.
PP=la.solve_discrete_lyapunov(Et,Bt@Bt.conj().T,method='bilinear')
QQ=la.solve_discrete_lyapunov(Et.conj().T,C.conj().T@C,method='bilinear')
PS=sqroot(PP);QS=sqroot(QQ)
UU,ss,Vh=la.svd(QS.conj().T@PS,full_matrices=False)
Vbase=PS@Vh.conj().T
times=[0,100,1000,5000,10000,50000]
def evolve(M,x):
    powers=[M]
    for _ in range(1,round(max(times)/dt).bit_length()):powers.append(powers[-1]@powers[-1])
    states=[]
    for t in times:
        m=round(t/dt);v=x.copy();i=0
        while m:
            if m&1:v=powers[i]@v
            i+=1;m>>=1
        states.append(v)
    return states
reference=evolve(R,u0)
out['reference']=[{'time':round(t/dt)*dt,'full_norm':float(la.norm(v)),'interior_norm':float(la.norm(v[inside]))} for t,v in zip(times,reference)]
for r in args.orders:
    if r>=len(E):continue
    # Orthogonalization in the exact exterior metric, with protected initial
    # direction as first basis vector. Realization order equals r.
    basis=np.column_stack([-T@uold,Vbase[:,:r-1]])
    Q,_=la.qr(F@basis,mode='economic')
    VV=la.solve_triangular(F,Q)
    Phi=np.zeros((20*N,len(inside)+r),complex)
    Phi[np.ix_(inside,np.arange(len(inside)))]=np.eye(len(inside))
    Phi[np.ix_(outside,np.arange(len(inside)))]=T
    Phi[np.ix_(outside,np.arange(len(inside),len(inside)+r))]=VV
    # Evaluate shifted coordinates directly to avoid subtracting enormous
    # complete-metric cross terms in a supposedly zero mass-matrix block.
    At=A+C@T;Ct=C@VV
    lower=Q.conj().T@F
    Bt=lower@(B+E@T-T@At)
    Et=lower@(E@VV-T@Ct)
    Rr=np.block([[At,Ct],[Bt,Et]])
    xr=np.r_[uold,-Q.conj().T@(F@T@uold)]
    eigs=la.eigvals(Rr);jj=np.argmax(abs(eigs))
    initerr=la.norm(Phi@xr-u0)
    rec={'order':r,'closed_rho':float(abs(eigs[jj])),
         'closed_growth':float(np.log(abs(eigs[jj]))/dt),
         'exterior_metric_orthogonality':float(la.norm(Q.conj().T@Q-np.eye(r))),
         'initial_full_relative_error':float(initerr),
         'interior_top_row_error':float(la.norm(Rr[:len(inside)]-(R@Phi)[inside])),
         'zero_max':float(abs(Rr@np.zeros(len(Rr),complex)).max()),'records':[]}
    if abs(eigs[jj])<1.0001:
        states=evolve(Rr,xr)
        for t,x,full in zip(times,states,reference):
            v=Phi@x
            rec['records'].append({'time':round(t/dt)*dt,'full_norm':float(la.norm(v)),
                'full_relative_error':float(la.norm(v-full)/la.norm(full)),
                'interior_relative_error':float(la.norm(x[:len(inside)]-full[inside])/la.norm(full[inside]))})
    out['runs'].append(rec)
    Path(__file__).with_name(f'protected-N{N}-I{n}-tangent{args.tangent_divisor:g}.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
print('elapsed',time.monotonic()-tick,flush=True)
