"""Independent global-periodic snapshot diagnostics for the intrinsic state.

These are analysis operators, not the evolution's Rstar or a production halo
implementation. All derivatives apply the declared centered stencil to complete
materialized fields on a synchronized global periodic grid. Repeated derivatives
therefore use a wider stencil. Physical H/M use only primary w,s,Ahat,K.
"""
import numpy as np
import math


def ko(f,spacing,order,amplitude):
    radius=order//2+1;out=np.zeros_like(f)
    for d,h in enumerate(spacing):
        if f.shape[-1-d]==1:continue
        for j in range(-radius,radius+1):
            out-=amplitude*(-1.)**j*math.comb(2*radius,radius+j)*np.roll(f,-j,axis=-1-d)/(2**(2*radius)*h)
    return out


def reduction_injection(u,rhs,spacing,order,rate,dissipation):
    """Signed semidiscrete defects from synchronized global state and active RHS.

    The returned arrays have direction/family and curl-pair/family ordering.
    No RHS ghosts are read. The lapse uses the tangent rho*wdot+w*rhodot.
    """
    D=lambda f,d:derivative(f,d,spacing,order)
    def auxiliary(v):
        return np.array([np.concatenate([v[20+d:21+d],v[23+d:24+d],v[26+5*d:31+5*d],v[41+3*d:44+3*d]]) for d in range(3)])
    potential=np.concatenate([u[0:1],(u[0]*u[1])[None],u[2:7],u[7:10]])
    tangent=np.concatenate([rhs[0:1],(u[1]*rhs[0]+u[0]*rhs[1])[None],rhs[2:7],rhs[7:10]])
    g=auxiliary(u);gdot=auxiliary(rhs)
    e=g-np.array([D(potential,d) for d in range(3)])
    omega=np.array([[D(g[j],i)-D(g[i],j) for j in range(3)] for i in range(3)])
    db=np.array([D(u[7:10],d) for d in range(3)])
    injection=gdot-np.array([D(tangent,d) for d in range(3)])+rate*e-ko(e,spacing,order,dissipation)
    curl=np.array([[D(gdot[j],i)-D(gdot[i],j) for j in range(3)] for i in range(3)])
    curl+=rate*omega-ko(omega,spacing,order,dissipation)
    for k in range(3):
        injection-=u[7+k]*D(e,k);curl-=u[7+k]*D(omega,k)
        for i in range(3):
            injection[i]-=db[i,k]*e[k]
            for j in range(3):curl[i,j]-=db[i,k]*omega[k,j]+db[j,k]*omega[i,k]
    for i in range(3):
        for j in range(3):curl[i,j]+=D(rate,i)*e[j]-D(rate,j)*e[i]
    return injection.reshape(30,*u.shape[1:]),np.array([curl[i,j] for i,j in [(0,1),(0,2),(1,2)]]).reshape(30,*u.shape[1:])


def derivative(f,d,spacing,order):
    if f.shape[-1-d]==1:return np.zeros_like(f)
    coefficients={2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[order]
    out=np.zeros_like(f)
    for j,c in enumerate(coefficients,1):
        out+=c*(np.roll(f,-j,axis=-1-d)-np.roll(f,j,axis=-1-d))/spacing[d]
    return out


def geometry(u):
    a,c,b,d,e=u[2:7];shape=u.shape[1:]
    t=np.zeros((3,3,*shape));t[0,0]=np.exp(a);t[1,1]=np.exp(c)
    t[2,2]=np.exp(-a-c);t[1,0]=b;t[2,0]=d;t[2,1]=e
    g=np.einsum('ik...,jk...->ij...',t,t)
    gu=np.moveaxis(np.linalg.inv(np.moveaxis(g,(0,1),(-2,-1))),(-2,-1),(0,1))
    ah=np.zeros_like(t);ah[0,0]=u[11];ah[0,1]=ah[1,0]=u[12]
    ah[0,2]=ah[2,0]=u[13];ah[1,1]=u[14];ah[1,2]=ah[2,1]=u[15]
    ah[2,2]=-u[11]-u[14]
    curvature=np.einsum('ir...,rs...,js...->ij...',t,ah,t)
    q=np.zeros((3,3,3,*shape))
    for k in range(3):
        s=u[26+5*k:31+5*k];dt=np.zeros_like(t)
        dt[0,0]=t[0,0]*s[0];dt[1,1]=t[1,1]*s[1];dt[2,2]=-t[2,2]*(s[0]+s[1])
        dt[1,0]=s[2];dt[2,0]=s[3];dt[2,1]=s[4]
        q[k]=np.einsum('ir...,jr...->ij...',dt,t)+np.einsum('ir...,jr...->ij...',t,dt)
    return g,gu,curvature,q


def diagnostics(u,spacing,order):
    D=lambda f,d:derivative(f,d,spacing,order)
    w=u[0];alpha=w*u[1];kval=u[10];shape=w.shape
    assert np.isfinite(u).all() and np.all(w>0)
    g,gu,a,q=geometry(u)
    dg=np.array([D(g,d) for d in range(3)])
    low=np.empty((3,3,3,*shape))
    for i in range(3):
        for j in range(3):
            for k in range(3):low[i,j,k]=(dg[j,i,k]+dg[k,i,j]-dg[i,j,k])/2
    gam=np.einsum('ir...,rjk...->ijk...',gu,low)
    dgam=np.array([D(gam,d) for d in range(3)])
    ricci=np.zeros_like(g)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ricci[i,j]+=dgam[k,k,i,j]-dgam[j,k,i,k]
                for l in range(3):ricci[i,j]+=gam[k,k,l]*gam[l,i,j]-gam[k,j,l]*gam[l,i,k]
    dw=np.array([D(w,d) for d in range(3)])
    hessian=np.array([[D(dw[j],i) for j in range(3)] for i in range(3)])
    hessian-=np.einsum('kij...,k...->ij...',gam,dw)
    mix=np.einsum('ik...,kj...->ij...',gu,a)
    physical_h=(2/3)*kval*kval-np.einsum('ij...,ji...->...',mix,mix)
    physical_h+=w*w*np.einsum('ij...,ij...->...',gu,ricci)
    physical_h+=4*w*np.einsum('ij...,ij...->...',gu,hessian)-6*np.einsum('ij...,i...,j...->...',gu,dw,dw)
    m=np.array([-(2/3)*D(kval,i) for i in range(3)])
    for i in range(3):
        for j in range(3):
            m[i]+=D(mix[j,i],j)
            for k in range(3):m[i]+=gam[j,j,k]*mix[k,i]-gam[k,j,i]*mix[j,k]
    contraction=np.einsum('ji...,j...->i...',mix,dw)
    physical_m=m-3*contraction/w
    weighted_m=alpha*m-3*u[1]*contraction
    potential=np.concatenate([u[0:1],alpha[None],u[2:7],u[7:10]])
    auxiliary=np.array([np.concatenate([u[20+d:21+d],u[23+d:24+d],u[26+5*d:31+5*d],u[41+3*d:44+3*d]]) for d in range(3)])
    reduction=auxiliary-np.array([D(potential,d) for d in range(3)])
    pairs=[(0,1),(0,2),(1,2)]
    curl=np.array([D(auxiliary[j],i)-D(auxiliary[i],j) for i,j in pairs])
    raw_q_curl=np.array([D(q[j],i)-D(q[i],j) for i,j in pairs])
    return dict(H_physical=physical_h[None],M_physical=physical_m,
                alpha_M_physical=weighted_m,C=u[19:20],Z=u[16:19],
                reduction=reduction.reshape(30,*shape),curl=curl.reshape(30,*shape),
                raw_Q_curl=raw_q_curl[:,[0,0,0,1,1,2],[0,1,2,1,2,2]].reshape(18,*shape))


def component_names():
    families=['w','alpha','a','c','b','d','e','betax','betay','betaz']
    return dict(H_physical=['H'],M_physical=['Mx','My','Mz'],
        alpha_M_physical=['alpha_Mx','alpha_My','alpha_Mz'],C=['C'],Z=['Zx','Zy','Zz'],
        reduction=[f'E_{d}_{f}' for d in ['x','y','z'] for f in families],
        curl=[f'Omega_{d}_{f}' for d in ['xy','xz','yz'] for f in families],
        raw_Q_curl=[f'OmegaQ_{d}_{ij}' for d in ['xy','xz','yz'] for ij in ['xx','xy','xz','yy','yz','zz']])


def norms(fields,volume):
    result={}
    for name,values in fields.items():
        v=values.reshape(values.shape[0],-1)
        winners=np.argmax(abs(v),axis=1)
        locations=np.array(np.unravel_index(winners,values.shape[1:])).T.tolist()
        result[name]=dict(components=component_names()[name],volume=float(volume),cells=v.shape[1],
            signed_at_abs_max=v[np.arange(len(v)),winners].tolist(),argmax_global_kji=locations,
            L1_integral=(volume*np.mean(abs(v),axis=1)).tolist(),
            L2_integral=np.sqrt(volume*np.mean(v*v,axis=1)).tolist(),
            RMS=np.sqrt(np.mean(v*v,axis=1)).tolist(),maximum=np.max(abs(v),axis=1).tolist())
    return result
