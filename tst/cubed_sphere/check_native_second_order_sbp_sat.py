#!/usr/bin/env python3
"""Compatible second-order frozen principal model, never native adoption."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
np.seterr(all="raise")
p=argparse.ArgumentParser();p.add_argument('--symbols',nargs='+',type=Path,required=True);p.add_argument('--sbp',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sbp=json.loads(a.sbp.read_text());weights=np.array(sbp['norm_boundary_weights']);rows=np.array(sbp['left_boundary_rows']);trace=np.array(sbp['left_trace'])
def mm(x,y):return np.einsum("ij,jk->ik",x,y)

def symbols(path):
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if not line.startswith('case '):
            continue
        _, family, radius, label, omega, lo, hi = line.split()
        inverse = np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3)
        normal = next(lines)
        assert normal.startswith('normal ')
        assert next(lines) == 'constraints'
        constraints = np.array([list(map(float,next(lines).split())) for _ in range(8)])
        assert next(lines) == 'matrix 1'
        matrix = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert next(lines) == 'matrix 0.5'
        other = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert np.linalg.norm(matrix-other)/max(1,np.linalg.norm(matrix)) < 1e-11
        tr = np.array([inverse[0,0],2*inverse[0,1],2*inverse[0,2],inverse[1,1],2*inverse[1,2],inverse[2,2]])
        cq = np.zeros((1,11));cq[0,1:7] = tr
        cp = np.zeros((1,11));cp[0,:6] = tr
        q = np.linalg.svd(cq,full_matrices=True)[2][1:].T
        v = np.linalg.svd(cp,full_matrices=True)[2][1:].T
        basis = np.zeros((22,20));basis[:11,:10] = q;basis[11:,10:] = v
        yield dict(source=str(path),family=int(family),radius=float(radius),sample=label,omega=float(omega),
                   lo=float(lo),hi=float(hi),matrix=mm(mm(basis.T,matrix),basis),constraints=mm(constraints,basis))

def spatial(n):
    m = len(weights);h = np.ones(n);h[:m] = weights;h[-m:] = weights[::-1]
    d = np.zeros((n,n));width=rows.shape[1];d[:m,:width] = rows;d[-m:,-width:] = -rows[::-1,::-1]
    for k in range(m,n-m):
        for off,c in ([(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)] if rows.shape[1]-m==3 else [(-2,1/12),(-1,-2/3),(1,2/3),(2,-1/12)]):d[k,k+off] = c
    left = np.zeros(n);left[:m] = trace;right = left[::-1]
    bl,br = np.outer(left,left),np.outer(right,right)
    residual = np.linalg.norm(h[:,None]*d+d.T*h[None,:]-(br-bl))
    assert min(h)>0 and residual<1e-10 and abs(sum(h)-n)<1e-10
    return h,d,bl,br,residual
records=[]
for path in a.symbols:
 for s in symbols(path):
  M=s.pop('matrix');s.pop('constraints');A,B,C,F=M[:10,:10],M[:10,10:],M[10:,:10],M[10:,10:];beta=-(s['lo']+s['hi'])/2;c=(s['hi']-s['lo'])/2;speed2=c*c-beta*beta;eye=np.eye(10)
  errors=[np.linalg.norm(mm(A,A)+mm(B,C)-2*beta*A-speed2*eye)/max(1,np.linalg.norm(M)**2),np.linalg.norm(mm(A,B)+mm(B,F)-2*beta*B)/max(1,np.linalg.norm(M)**2)]
  assert speed2>0 and max(errors)<1e-8 and np.linalg.matrix_rank(B)==10
  tests=[]
  for n in [24,48]:
   h0,d0,bl,br,sbperr=spatial(n);h=h0/n;d=d0*n;d2=mm(d,d);H=np.diag(h);zero=np.zeros((n,n));I=np.eye(n)
   tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];ql=mm(tl[None,:],d);qr=mm(tr[None,:],d)
   L=np.block([[zero,I],[speed2*d2,2*beta*d]])
   left=np.vstack([np.zeros((n,1)),((c-beta)*tl/h)[:,None]]);right=np.vstack([np.zeros((n,1)),((c+beta)*tr/h)[:,None]])
   leftrow=np.hstack([-(c+beta)*ql,tl[None,:]]);rightrow=np.hstack([(c-beta)*qr,tr[None,:]])
   L-=mm(left,leftrow)+mm(right,rightrow)
   W=np.block([[speed2*mm(mm(d.T,H),d),zero],[zero,H]])
   E=mm(W,L)+mm(L.T,W);expected=np.block([[zero,zero],[zero,-2*c*(bl+br)]]);scale=max(1,np.linalg.norm(expected));error=np.linalg.norm(E-expected)/scale;largest=np.linalg.eigvalsh(E).max()/scale
   unclosed=L+mm(right,rightrow);Eu=mm(W,unclosed)+mm(unclosed.T,W);positive=np.linalg.eigvalsh(Eu).max()/scale
   eig=np.linalg.eigvals(L);dt=.02/n/(c+abs(beta));z=dt*eig;rk=1+z+z*z/2+z*z*z/6+z*z*z*z/24;radius=float(max(abs(rk)))
   x=(np.arange(n)+.5)/n;moments=[float(max(abs(mm(d2,(x**degree)[:,None])[:,0]-(degree*(degree-1)*x**(degree-2) if degree>=2 else 0)))) for degree in range(5)]
   passed=error<=1e-10 and largest<=1e-10 and positive>1e-12 and radius<=1+1e-8 and max(moments[:3])<=1e-8
   tests.append(dict(nodes=n,energy_identity_relative_error=float(error),max_energy_eigenvalue=float(largest),missing_outer_penalty_positive_energy_direction=float(positive),rk4_spectral_radius=radius,rk4_dt=dt,semidiscrete_max_real_eigenvalue=float(max(eig.real)),near_zero_eigenvalues=int(sum(abs(eig)<1e-5)),D2_polynomial_max_errors_degree0_to4=moments,passed=bool(passed)))
  records.append(dict(**s,beta=beta,c=c,native_wave_identity_errors=list(map(float,errors)),B_condition=float(np.linalg.cond(B)),tests=tests,passed=all(x['passed'] for x in tests)))
r=dict(passed=all(x['passed'] for x in records),records=records,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),symbol_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.symbols},sbp_sha256=hashlib.sha256(a.sbp.read_bytes()).hexdigest(),scope='Frozen native second-order principal reduction with compatible D2=D1 squared and incoming SAT. Boundary-order loss recorded. No native nonlinear implementation, actual native integrator/KO/seam test or stability acceptance.')
a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(dict(passed=r['passed'],states=len(records),failed=[(x['sample'],t) for x in records for t in x['tests'] if not t['passed']]))
