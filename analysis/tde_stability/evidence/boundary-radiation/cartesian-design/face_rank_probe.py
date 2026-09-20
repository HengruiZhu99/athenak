"""Frozen one-face rank probe; NOT an implemented or validated boundary operator.

Unknowns are all independent q_t,p_t values at a boundary point. D_n q_t's
boundary contribution is eta*q_t. This probes whether retaining q_t freedom
removes the documented p-only obstruction. Tangential couplings and full
boundary derivatives are deliberately not approximated by this local matrix.
"""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
# q=(chi,h_nn,alpha,beta_n,h_n1,beta_1,h_n2,beta_2,h_TT+,h_TTx)
# p=(Khat,Theta,A_nn,Gamma_n,A_n1,Gamma_1,A_n2,Gamma_2,A_TT+,A_TTx)

def rows(alpha,chi,G,eta,beta=0.,f=1.):
    a=np.sqrt(chi);L=2*f*alpha
    cs=np.sqrt(4*G/3);cl=np.sqrt(chi*L)
    left=[];names=[]
    for sign in [1,-1]:
        pref='in' if sign==1 else 'out'
        mu=sign*cs;dg=mu
        ls=chi*L-mu*dg;lt=chi*alpha**2-mu**2
        C=np.zeros((4,20))
        C[0,10]=-sign*cl/chi;C[0,2]=eta
        C[1,10]=alpha*dg**2*lt
        C[1,11]=.5*alpha*(4*G/3)*ls
        C[1,13]=.25*dg*(4*chi*alpha**2-3*mu**2)*ls
        C[1,0]=.5*alpha**2*dg*ls*eta
        C[1,2]=-chi*alpha*dg*lt*eta
        C[1,3]=ls*lt*eta
        C[2,11]=sign*a;C[2,13]=.5*chi;C[2,0]=eta
        C[3,10]=sign*4/(3*a);C[3,11]=sign*2/(3*a)
        C[3,12]=-sign*2/a;C[3,13]=-1;C[3,1]=eta
        for n,name in enumerate(['lapse','shift_n','constraint1','constraint2']):
            left.append(C[n]);names.append(f'{pref}_{name}')
        for qg,qb,pA,pG,t in [(4,5,14,15,1),(6,7,16,17,2)]:
            c=np.zeros(20);c[pG]=sign*np.sqrt(G);c[qb]=eta
            left.append(c);names.append(f'{pref}_shift_{t}')
            c=np.zeros(20);c[pA]=-sign*2/a;c[pG]=-1;c[qg]=eta
            left.append(c);names.append(f'{pref}_constraint_{t}')
        for qg,pA,t in [(8,18,'+'),(9,19,'x')]:
            c=np.zeros(20);c[pA]=-sign*2/a;c[qg]=eta
            left.append(c);names.append(f'{pref}_radiation_{t}')
    return dict(zip(names,left))

def inspect(r,dx):
    alpha=r/(r+1);chi=alpha**2;eta=1.5/dx
    rr=rows(alpha,chi,2.,eta)
    kept=[(n,a) for n,a in rr.items() if n.startswith('out_') or
          (n.startswith('in_') and ('lapse' in n or 'shift' in n or 'radiation' in n))]
    F=np.zeros((4,20));F[0,11]=1
    for j,(pG,qh) in enumerate([(13,1),(15,4),(17,6)]):
        F[1+j,pG]=1;F[1+j,qh]=-eta
    A=np.array([a for n,a in kept]+list(F))
    # Row normalization makes the reported conditioning independent of arbitrary
    # characteristic eigenvector scaling in the current implementation.
    A/=np.linalg.norm(A,axis=1)[:,None]
    s=np.linalg.svd(A,compute_uv=False)
    return {'r':r,'dx':dx,'rank':int(np.linalg.matrix_rank(A)),
            'smallest_singular_value':float(s[-1]),
            'condition_number':float(s[0]/s[-1]),
            'scope':'Local frozen face rank only; no PDE consistency/stability claim.'}

if __name__=='__main__':
    result=[inspect(r,dx) for r in [1.5,2.,4.,8.] for dx in [.25,.125]]
    (ROOT/'face-rank-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
