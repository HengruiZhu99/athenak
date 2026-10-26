#!/usr/bin/env python3
"""Independent Cartesian native endpoint identities. No evolved auxiliary state."""
import numpy as np
pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def Geometry(u, x, F):
    (chi, alpha) = (u[0], u[10])
    beta = u[[40, 60, 80]]
    dBeta = np.stack([u[41:44], u[61:64], u[81:84]], axis=1)
    g = np.zeros((3, 3))
    dg = np.zeros((3, 3, 3))
    A = np.zeros((3, 3))
    Fg = np.zeros((3, 3))
    for (m, (i, j)) in enumerate(pairs):
        b = 100 + 14 * m
        g[i, j] = g[j, i] = u[b]
        dg[:, i, j] = dg[:, j, i] = u[b + 1:b + 4]
        A[i, j] = A[j, i] = u[b + 10]
        Fg[i, j] = Fg[j, i] = F[1 + m]
    gi = np.linalg.inv(g)
    C = chi * gi
    dC = u[1:4, None, None] * gi[None, :, :] - chi * np.einsum('ik,dkl,lj->dij', gi, dg, gi)
    nu = -2 * x
    w = -np.einsum('i,i->', beta, nu) / alpha
    dw = -(np.einsum('di,i->d', dBeta, nu) - 2 * beta + w * u[11:14]) / alpha
    G = np.einsum('i,ij,j->', nu, C, nu)
    dG = np.einsum('dij,i,j->d', dC, nu, nu) - 4 * np.einsum('ij,j->i', C, nu)
    N1 = np.einsum('i,i->', nu, dG - 2 * w * dw) / 4
    Q = np.einsum('i,i->', nu, u[21:24] + 2 * u[31:34] - 3 * dw) / 4
    tau = np.einsum('i,i->', nu, u[31:34]) / 4
    b = g / chi
    db = dg / chi - g[None, :, :] * u[1:4, None, None] / chi ** 2
    gamma = np.einsum('ij,kl,ilj->k', C, C, db) - 0.5 * np.einsum('ij,kl,lij->k', C, C, db)
    lap = -2 * np.trace(C) - np.einsum('i,i->', gamma, nu)
    gt = np.einsum('ij,kl,ilj->k', gi, gi, dg) - 0.5 * np.einsum('ij,kl,lij->k', gi, gi, dg)
    Z = (u[[50, 70, 90]] - gt) / 2
    q = np.einsum('ij,j->i', gi, nu)
    zeta = chi * np.einsum('i,i->', Z, nu) / w ** 2
    kappa = 10 / alpha
    wt = -(np.einsum('i,i->', nu, F[19:22]) + w * F[18]) / alpha
    Ct = F[0] * gi - chi * np.einsum('ik,kl,lj->ij', gi, Fg, gi)
    Nt = np.einsum('i,ij,j->', nu, Ct, nu) - 2 * w * wt
    Dt = F[7] + 2 * F[17] - 3 * wt
    cross = np.einsum('i,ij,j->', nu, C, u[11:14])
    betaw = np.einsum('i,i->', beta, dw)
    M = -2 * (u[21:24] + 2 * u[31:34]) / 3 - 2 * np.einsum('ij,jk,k->i', A, gi, nu)
    H1 = 4 * w * Q + 4 * lap - 6 * N1
    predictions = {'Z_normal_vector': (Z, zeta * q), 'zeta_leading': (zeta, (2 * N1 / w - 4 * Q / 3 + 2 * tau - 4 * lap / (3 * w)) / (2 * kappa - 4 * w / 3)), 'Theta_raw': (F[17], alpha * (2 * lap + 2 * w * Q - 3 * N1 - (4 * w + 2 * kappa) * tau)), 'Theta_eliminated': (F[17], -alpha * ((w + 2 * kappa) * tau + w * (3 * kappa - 2 * w) * zeta)), 'w_time': (wt, betaw + cross + alpha * lap - 4 * alpha * N1 + alpha * w * (Q - 2 * tau) - 2 * alpha * zeta * w * w), 'D_time_raw': (Dt, alpha * (2 * lap + 2 * w * Q + 3 * N1 - 3 * kappa * tau + 6 * zeta * w * w)), 'D_time_eliminated': (Dt, alpha * (6 * N1 + 3 * (w - kappa) * tau + (8 * w * w - 3 * w * kappa) * zeta)), 'N_time_raw': (Nt, 2 * alpha * w * (3 * N1 - 2 * (lap + w * Q) / 3 + 2 * w * tau + 2 * zeta * w * w / 3)), 'N_time_eliminated': (Nt, 2 * alpha * w * (2 * N1 + w * tau + w * kappa * zeta)), 'momentum_leading': (M, nu * ((kappa + 2 * w) * zeta - tau)), 'H_normal': (H1, 6 * w * (tau - (kappa - 2 * w / 3) * zeta))}
    defects = {}
    for (name, (a, b)) in predictions.items():
        a = np.asarray(a)
        b = np.asarray(b)
        defects[name] = float(np.max(abs(a - b) / np.maximum(1, np.maximum(abs(a), abs(b)))))
    return locals()

def FirstRHS(u,n):
    z=Geometry(u,n,np.zeros(22));g=z['g'];dg=z['dg'];chi=z['chi'];alpha=z['alpha'];beta=z['beta'];dbeta=z['dBeta'];C=z['C'];dC=z['dC'];nu=z['nu'];A=z['A'];w=z['w'];Q=z['Q'];tau=z['tau'];N1=z['N1'];lap=z['lap'];Z=z['Z'];gamma=z['gamma'];kappa=z['kappa']
    F=np.zeros(22);div=np.trace(dbeta)
    F[0]=np.einsum('i,i->',beta,u[1:4])-2*chi*div/3+2*alpha*chi*Q/3
    Fg=-2*alpha*A-2*g*div/3+np.einsum('d,dij->ij',beta,dg)+np.einsum('dj,id->ij',g,dbeta)+np.einsum('id,jd->ij',g,dbeta)
    for m,(i,j)in enumerate(pairs):F[1+m]=Fg[i,j]
    L=alpha*alpha*C-np.einsum('i,j->ij',beta,beta);trL=np.trace(L);v=np.einsum('ij,j->i',C,n);bx=np.einsum('i,i->',beta,n)
    dtr=2*alpha*u[11:14]*np.trace(C)+alpha*alpha*np.einsum('dii->d',dC)-2*np.einsum('i,di->d',beta,dbeta)
    dsalpha=-alpha*(u[11:14]*u[20]+alpha*u[21:24]+np.einsum('di,i->d',dbeta,nu)-2*beta+dtr-n*trL)
    F[18]=np.einsum('i,i->',beta,u[11:14])+np.einsum('i,i->',nu,dsalpha)/4
    dsbeta=np.zeros((3,3))
    for d in range(3):
        dsbeta[d]=-16*alpha*u[11+d]*v-8*alpha*alpha*(np.einsum('ij,j->i',dC[d],n)+C[:,d])+4*dbeta[d]*bx+4*beta*(np.einsum('i,i->',dbeta[d],n)+beta[d])+(n[d]*beta+2*n[d]*n-dbeta[d])*trL-beta*dtr[d]
    F[19:22]=np.einsum('j,ji->i',beta,dbeta)+alpha*alpha*gamma-alpha*np.einsum('ij,j->i',C,u[11:14])+2*alpha*alpha*chi*Z+np.einsum('d,di->i',nu,dsbeta)/4
    F[17]=alpha*(2*lap+2*w*Q-3*N1-(4*w+2*kappa)*tau)
    F[7]=3*z['betaw']+3*z['cross']+alpha*lap+alpha*w*Q-3*alpha*N1+alpha*(2*w+kappa)*tau
    Ct=F[0]*z['gi']-chi*np.einsum('ik,kl,lj->ij',z['gi'],Fg,z['gi']);c=np.einsum('i,ij,j->',n,C,n);ct=np.einsum('i,ij,j->',n,Ct,n);vt=np.einsum('ij,j->i',Ct,n);v2=np.einsum('i,i->',v,v);T=np.trace(C)-v2/c;Tt=np.trace(Ct)-2*np.einsum('i,i->',v,vt)/c+v2*ct/(c*c)
    alpha0t=alpha*(ct/(2*c)-Tt/T);beta0t=-4*vt/T+4*v*Tt/(T*T);Ea=F[18]-alpha0t;Eb=F[19:22]-beta0t
    axis=np.eye(3)[np.argmin(abs(n))];e1=axis-n*np.einsum('i,i->',axis,n);e1/=np.linalg.norm(e1);e2=np.cross(n,e1)
    six=np.r_[tau,z['zeta'],N1,Ea,np.einsum('i,i->',e1,Eb),np.einsum('i,i->',e2,Eb)]
    wt=-(np.einsum('i,i->',nu,F[19:22])+w*F[18])/alpha;Dt=F[7]+2*F[17]-3*wt;Nt=np.einsum('i,ij,j->',nu,Ct,nu)-2*w*wt
    scalar=np.array([2*alpha*chi*Dt/3,alpha*(2*w*Dt-3*Nt+kappa*F[17]),alpha*(2*w*Dt-3*Nt-(3*w+2*kappa)*F[17])])
    Lt=2*alpha*F[18]*C+alpha*alpha*Ct-np.einsum('i,j->ij',F[19:22],beta)-np.einsum('i,j->ij',beta,F[19:22]);trLt=np.trace(Lt)
    Salphat=-alpha*(alpha*F[7]+u[20]*F[18]+np.einsum('i,i->',nu,F[19:22])+trLt)
    Sbetat=-16*alpha*F[18]*v-8*alpha*alpha*vt+F[19:22]*(4*bx-trL)+beta*(4*np.einsum('i,i->',n,F[19:22])-trLt)
    seven=np.r_[scalar,Salphat,Sbetat]
    null_time_identity=Nt-2*w*(np.einsum('i,i->',nu,Eb)+w*Ea)/alpha
    return {'F':F,'six':six,'seven':seven,'Ct':Ct,'Ealpha':Ea,'Ebeta':Eb,'null_time_identity':float(null_time_identity),'branch_gap':float(8*w*w+4*w*kappa-3*kappa*kappa),'w':float(w),'kappa':float(kappa)}

def MetricFlow(u,n,F):
    z=Geometry(u,n,F);g=z['g'];b=g/z['chi'];nu=z['nu'];C=z['C'];k=z['alpha']/z['w'];dk=u[11:14]/z['w']-z['alpha']*z['dw']/z['w']**2;V=np.einsum('ij,j->i',C,nu);dV=np.einsum('dij,j->di',z['dC'],nu)-2*C.T;df=z['dBeta']+np.einsum('d,i->di',dk,V)+k*dV;B=np.einsum('d,di->i',nu,df)/4;Bcov=np.einsum('ij,j->i',b,B);Zcov=np.einsum('ij,j->i',g,z['Z']);D=Bcov-dk;R=z['Q']+z['lap']/z['w']+2*z['chi']*np.einsum('i,i->',z['Z'],nu)/z['w'];generalD=D+2*k*Zcov;general=np.einsum('i,j->ij',nu,generalD)+np.einsum('i,j->ij',generalD,nu)-2*z['alpha']*b*R/3;regular=np.einsum('i,j->ij',nu,D)+np.einsum('i,j->ij',D,nu);actual=z['Fg']/z['chi']-g*F[0]/z['chi']**2;T=np.eye(3)-np.einsum('i,j->ij',n,n);transverse=np.einsum('ik,kl,lj->ij',T,actual,T);without_B=-np.einsum('i,j->ij',nu,dk)-np.einsum('i,j->ij',dk,nu)
    return {'general_flow_error':float(max(abs(actual-general).ravel())),'regular_flow_error':float(max(abs(actual-regular).ravel())),'transverse_leading_metric_time':float(max(abs(transverse).ravel())),'normal_B_identity':float(abs(np.einsum('i,i->',nu,B)-z['alpha']*z['N1']/z['w'])),'negative_omit_B_error':float(max(abs(actual-without_B).ravel())),'B':B.tolist(),'configuration_leading_metric_fixed_to_reference':False}
