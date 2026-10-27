"""Frozen normal scalar symbol. D_t u = P d_r u; speeds are -beta-eig(P).

Reduction variables: chi_r, g_r, A_rr, delta(Khat)/Omega, Theta/Omega,
Lambda, alpha_r, beta_r, (telegraph B)_r, shift-driver D.
Omega rescaling is invertible only in the interior; a finite limiting matrix
alone does not prove regularity or well posedness at actual scri.
"""
import numpy as np

NAMES = ['chi_r','g_r','Arr','p/Omega','Theta/Omega','Lambda',
         'alpha_r','beta_r','B_r','D']

def symbol(alpha, chi, g, F, eta, w, kappa, tau, mu):
    P=np.zeros((10,10))
    P[0,[3,4,7]]=[2*alpha*chi/3,4*alpha*chi/3,-2*chi/3]
    P[1,[2,7]]=[-2*alpha,4*g/3]
    P[2,[0,1,5,6]]=[alpha/3,-alpha*chi/(2*g),2*alpha*chi*g/3,-2*chi/3]
    P[3,6]=-chi/g
    P[4,[0,5]]=[alpha/g,alpha*chi/2]
    P[5,[3,4,7]]=[-4*alpha/(3*g),-2*alpha/(3*g),4/(3*g)]
    P[6,[3,8]]=[-F,eta*chi*w]
    P[7,9]=mu
    P[8,6]=kappa*w/tau
    P[9]=P[5]
    return P

def coefficient(alpha,chi,g,F,eta,w,kappa,tau,ratio=2.,kind='separated'):
    if kind=='light':return .75*ratio*alpha**2*chi
    if kind=='separated':return .75*ratio*chi*(F+g*eta*kappa*w*w/tau)
    raise ValueError(kind)

def speeds(alpha,chi,g,F,eta,w,kappa,tau,mu,beta=0.):
    light=alpha*np.sqrt(chi/g)
    lapse=np.sqrt(chi*F/g+eta*chi*kappa*w*w/tau)
    shift=np.sqrt(4*mu/(3*g))
    return -beta+np.array([0.,0.,-light,-light,light,light,-lapse,lapse,-shift,shift])

def eigenspace_audit(P, expected, tolerance=2e-8):
    """Use SVD of each distinct eigenspace; repeated eig() vectors are unreliable."""
    groups=[]
    for value in sorted(expected):
        if groups and abs(value-groups[-1][0])<tolerance:groups[-1][1]+=1
        else:groups.append([float(value),1])
    vectors=[];ranks=[]
    for value,multiplicity in groups:
        _,s,vh=np.linalg.svd(P-value*np.eye(len(P)))
        nullity=int(np.count_nonzero(s<tolerance))
        ranks.append(dict(eigenvalue=value,algebraic=multiplicity,geometric=nullity))
        vectors.extend(vh[-multiplicity:])
    complete=all(x['algebraic']==x['geometric'] for x in ranks)
    return dict(eigenspaces=ranks,complete=complete,
                condition=float(np.linalg.cond(np.array(vectors).T)) if complete else None)
