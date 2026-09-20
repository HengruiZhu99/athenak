"""20-field tracefree Cartesian operator with Fourier tangential directions.

No background gradients, AMR, block-edge one-sided tangent helper, or corners.
Tangential helper derivatives use their interior D2/D4 symbols.
"""
import numpy as np
from stencils import operators,active_derivative,char

names=['chi','hxx','hyy','hxy','hxz','hyz','k','theta','Axx','Ayy','Axy','Axz','Ayz','Gx','Gy','Gz','alpha','bx','by','bz']

def symbol(weights,offsets,angle,h,power=1):
    return sum(w*np.exp(1j*o*angle) for w,o in zip(weights,offsets))/h**power

def full_matrix(n=16,h=.25,angle_y=0.,angle_z=0.,degree=3,inner=4,outer=2,
                mode='zero_rate',tau=1.,diss=.5,alpha=1.,chi=1.,damping=False,
                beta=(0.,0.,0.),gauge_target='zero_rate',tt_target='zero_rate',
                paper_lapse_factor=1.,paper_actual_d0=False,shift=1.,lapse_damping=.1,sponge_rate=0.,sponge_cells=8.):
    if mode != 'zero_rate' or gauge_target != 'zero_rate' or tt_target != 'zero_rate':
        raise ValueError('This archived strip is validated only for original zero_rate.')
    D,Dxx,adv,KO,E=operators(n,h,degree,beta[0],diss,4)
    eye=np.eye(n);total=20*n
    state_eye=np.eye(total,dtype=complex);S=[state_eye[j*n:(j+1)*n] for j in range(20)]
    c=S[0];k=S[6];theta=S[7];lapse=S[16]
    H=[[S[1],S[3],S[4]],[S[3],S[2],S[5]],[S[4],S[5],-S[1]-S[2]]]
    A=[[S[8],S[10],S[11]],[S[10],S[9],S[12]],[S[11],S[12],-S[8]-S[9]]]
    G=S[13:16];B=S[17:20]
    angles=[angle_y,angle_z]
    d1=[D]+[eye*symbol([-1/60,3/20,-3/4,0,3/4,-3/20,1/60],range(-3,4),a,h) for a in angles]
    d2=[Dxx]+[eye*symbol([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90],range(-3,4),a,h,2) for a in angles]
    lap=sum(d2)
    def d(i,u):return d1[i]@u
    def dd(i,j,u):return (d2[i] if i==j else d1[i]@d1[j])@u
    divB=sum(d(i,B[i]) for i in range(3));divG=sum(d(i,G[i]) for i in range(3))
    V=np.zeros((total,total),complex)
    def put(name,u):j=names.index(name);V[j*n:(j+1)*n]=u
    put('chi',2*alpha*chi/3*(k+2*theta)-2*chi/3*divB)
    put('k',-chi*lap@lapse);put('theta',alpha*chi/2*divG+alpha*lap@c)
    put('alpha',-2*alpha*k-lapse_damping*lapse)
    hc=[[None]*3 for _ in range(3)];ac=[[None]*3 for _ in range(3)]
    for i in range(3):
      for j in range(3):
        hc[i][j]=-2*alpha*A[i][j]+d(i,B[j])+d(j,B[i])-(2/3*divB if i==j else 0)
        ac[i][j]=-chi*dd(i,j,lapse)+alpha*(-chi/2*lap@H[i][j]+chi/2*(d(i,G[j])+d(j,G[i]))+.5*dd(i,j,c))
    tr=sum(ac[i][i] for i in range(3))
    for (i,j),hn,an in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],names[1:6],names[8:13]):
        put(hn,hc[i][j]);put(an,ac[i][j]-(tr/3 if i==j else 0))
    for i in range(3):
        g=-4*alpha/3*d(i,k)-2*alpha/3*d(i,theta)+lap@B[i]+sum(dd(i,j,B[j]) for j in range(3))/3
        if damping:g-=.2*alpha*(G[i]-sum(d(j,H[i][j]) for j in range(3)))
        put(names[13+i],g);put(names[17+i],shift*G[i]-(2*B[i] if damping else 0))
    if damping:
        V[6*n:7*n]+=.1*alpha*theta;V[7*n:8*n]-=.2*alpha*theta
    # Principal targets use centered sixth-order derivatives, exactly as the
    # C++ TangentialPrincipal path; distinguish Dxx from D1@D1 here.
    Vp=V.copy()
    if damping:
        Vp[6*n:7*n]-=.1*alpha*theta;Vp[7*n:8*n]+=.2*alpha*theta
        for i in range(3):
            Vp[(13+i)*n:(14+i)*n]+=.2*alpha*(G[i]-sum(d(j,H[i][j]) for j in range(3)))
            Vp[(17+i)*n:(18+i)*n]+=2*B[i]
    center_adv=sum(beta[i]*d1[i] for i in range(3))
    for j in range(20):Vp[j*n:(j+1)*n]+=center_adv@S[j]
    Xp={}
    normal_adv=lambda u:sum(beta[j]*dd(0,j,u) for j in range(3))
    normal_divB=sum(dd(0,j,B[j]) for j in range(3))
    Xp[0]=normal_adv(c)+2*alpha*chi/3*d(0,k+2*theta)-2*chi/3*normal_divB
    Xp[16]=normal_adv(lapse)-2*alpha*d(0,k)
    for (i,j),field in zip([(0,0),(1,1),(0,1),(0,2),(1,2)],range(1,6)):
        Xp[field]=normal_adv(H[i][j])-2*alpha*d(0,A[i][j])+dd(0,i,B[j])+dd(0,j,B[i])-(2/3*normal_divB if i==j else 0)
    for j in range(3):Xp[17+j]=normal_adv(B[j])+shift*d(0,G[j])
    kot=KO+eye*sum(symbol([1,-8,28,-56,70,-56,28,-8,1],range(-4,5),a,h)*(-diss/256) for a in angles)
    advtot=adv.astype(complex)
    for velocity,angle in zip(beta[1:],angles):
        off=np.arange(-2,5);ww=np.array([1/30,-2/5,-7/12,4/3,-1/2,2/15,-1/60])
        if velocity<0:off=-off;ww=-ww
        advtot+=eye*velocity*symbol(ww,off,angle,h)
    for j in range(20):V[j*n:(j+1)*n]+=(kot+advtot)@S[j]
    depth=np.minimum(np.arange(n)+.5,n-np.arange(n)-.5)
    ss=np.clip(1-depth/sponge_cells,0,1)
    sponge=sponge_rate*ss**3*(10-15*ss+6*ss*ss)
    for j in range(20):V[j*n:(j+1)*n]-=sponge[:,None]*S[j]
    if inner=='volume':dm=d1
    else:
        coeff={2:([-.5,0,.5],range(-1,2)),4:([1/12,-2/3,0,2/3,-1/12],range(-2,3))}[inner]
        dm=[active_derivative(n,h,inner)]+[eye*symbol(*coeff,a,h) for a in angles]
    Q=[G[i]-sum(dm[j]@H[i][j] for j in range(3)) for i in range(3)]
    Do=D if outer=='volume' else active_derivative(n,h,outer)
    dout=d1 if outer=='volume' else [Do]+[eye*symbol([-.5,0,.5],range(-1,2),a,h) for a in angles]
    divQ=sum(dout[j]@Q[j] for j in range(3))
    Db=active_derivative(n,h,2)
    L=V.copy()
    if mode=='none':return L,dict(volume=V,Q=Q,E=E)
    for pos,s in [(0,-1),(n-1,1)]:
        left,_=char.scalar_left(alpha,chi,2*alpha,shift,1);lp=left[:,:4];ld=left[:,4:]
        qr=np.vstack([s*Db[pos]@V[j*n:(j+1)*n] for j in [0,1,16,17]]);qr[3]*=s
        pr=np.vstack([V[j*n+pos] for j in [6,7,8,13]]);pr[3]*=s
        rates=lp@pr+ld@qr;delta=-rates
        speed=alpha*np.sqrt(chi);transport=s*speed-beta[0];lam=s*beta[0]+speed
        if gauge_target=='tangential':
            pprincipal=np.vstack([Vp[j*n+pos] for j in [6,7,8,13]]);pprincipal[3]*=s
            qprincipal=np.vstack([s*Xp[j][pos] for j in [0,1,16,17]]);qprincipal[3]*=s
            dp=np.vstack([s*D[pos]@S[j] for j in [6,7,8,13]]);dp[3]*=s
            ddq=np.vstack([Dxx[pos]@S[j] for j in [0,1,16,17]]);ddq[3]*=s
            normalrates=lp@dp+ld@ddq
            lambdas=np.array([s*beta[0]+np.sqrt(2*alpha*chi),s*beta[0]+np.sqrt(4*shift/3),lam,lam])
            targets=lp@pprincipal+ld@qprincipal-lambdas[:,None]*normalrates
            delta[:2]+=targets[:2]
        tangent_transport=sum(-beta[j]*dout[j][pos,pos] for j in [1,2])
        fq=[q[pos]@V+transport*Do[pos]@q+tangent_transport*q[pos] for q in Q]
        ftheta=V[7*n+pos]+transport*Do[pos]@theta+tangent_transport*theta[pos]
        if mode=='radiation':
            delta[2]=-tau*lam*ftheta/alpha;delta[3]=tau*lam*s*fq[0]/speed
        correction=np.linalg.solve(lp,delta)
        if mode=='direct':
            sigma=.1*alpha if damping else 0.
            correction[0]=delta[0]/lp[0,0]
            correction[1]=-ftheta
            correction[3]=(delta[1]-lp[1,0]*correction[0]-lp[1,1]*correction[1])/lp[1,3]
            correction[2]=-np.sqrt(chi)*s*fq[0]+alpha*chi/3*divQ[pos]-np.sqrt(chi)*sigma*s*Q[0][pos]
        L[6*n+pos]+=correction[0];L[7*n+pos]+=correction[1]
        L[8*n+pos]+=correction[2];L[9*n+pos]-=.5*correction[2]
        L[13*n+pos]+=s*correction[3]
        for ai,gi,hi,bi,component in [(10,14,3,18,1),(11,15,4,19,2)]:
            gauge=np.sqrt(shift)*V[gi*n+pos]+s*Db[pos]@V[bi*n:(bi+1)*n]
            gt=0.
            if gauge_target=='tangential':
                fullrate=np.sqrt(shift)*Vp[gi*n+pos]+s*Xp[bi][pos]
                normalrate=(s*beta[0]+np.sqrt(shift))*(np.sqrt(shift)*s*D[pos]@S[gi]+Dxx[pos]@S[bi])
                gt=fullrate-normalrate
            dg=(gt-gauge)/np.sqrt(shift)
            rate=-2*s/np.sqrt(chi)*V[ai*n+pos]-V[gi*n+pos]+Db[pos]@V[hi*n:(hi+1)*n]
            dc=tau*lam*fq[component]/speed if mode=='radiation' else -rate
            da=-.5*s*np.sqrt(chi)*(dc+dg)
            if mode=='direct':da=-.5*s*np.sqrt(chi)*(fq[component]+sigma*Q[component][pos])
            L[ai*n+pos]+=da;L[gi*n+pos]+=dg
        if gauge_target=='paper':
            vl=np.sqrt(2*alpha*chi);vsl=np.sqrt(4*shift/3);vst=np.sqrt(shift)
            lapT=d2[1]+d2[2];divT=d(1,B[1])+d(2,B[2])
            d0k=-vl*s*d(0,k)-paper_lapse_factor*chi*lapT@lapse
            d0theta=-speed*s*d(0,theta)
            pk=center_adv@k+d0k
            if paper_actual_d0:
                d0k=d0k+V[6*n:7*n]-Vp[6*n:7*n]
                d0theta=L[7*n:8*n]-center_adv@theta
            pg=s*center_adv@G[0]-vsl*divG+s*lapT@B[0]-s*d(0,divT)
            pg-=4*alpha/(3*(vl*vl-vsl*vsl))*(vsl*d0k+vl*vl*s*d(0,k))
            pg-=2*alpha/(3*(speed*speed-vsl*vsl))*(vsl*d0theta+speed*speed*s*d(0,theta))
            if damping:pg+=8/(3*vsl)*divB
            # Replace principal K/Gamma RHS; retain pre-existing lower-order,
            # KO and upwind-minus-centered contributions explicitly.
            L[6*n+pos]=V[6*n+pos]-Vp[6*n+pos]+pk[pos]
            L[13*n+pos]=V[13*n+pos]-Vp[13*n+pos]+s*pg[pos]
            for j in [1,2]:
                ga=center_adv@G[j]-vst*s*(d(0,G[j])-d(j,G[0]))+lapT@B[j]
                ga+=4/3*dd(j,0,B[0])+d(j,divT)/3-2*alpha/3*d(j,2*k+theta)
                if damping:ga+=2/vst*s*(d(0,B[j])-d(j,B[0]))
                gi=13+j;L[gi*n+pos]=V[gi*n+pos]-Vp[gi*n+pos]+ga[pos]
        # Incoming tensor characteristic time rates are zero, as in the code.
        aplus=V[9*n:10*n]+.5*V[8*n:9*n]
        hplus=V[2*n:3*n]+.5*V[1*n:2*n]
        plusrate=-2*aplus[pos]/np.sqrt(chi)+s*Db[pos]@hplus
        plustarget=0.
        if tt_target=='tangential':
            pplus=Vp[9*n+pos]+.5*Vp[8*n+pos]
            hnormal=Xp[2][pos]+.5*Xp[1][pos]
            normalrate=lam*(-2*s*D[pos]@(S[9]+.5*S[8])/np.sqrt(chi)+Dxx[pos]@(S[2]+.5*S[1]))
            plustarget=-2*pplus/np.sqrt(chi)+s*hnormal-normalrate
        dplus=.5*np.sqrt(chi)*(plusrate-plustarget)
        if tt_target=='paper':
            ap=S[9]+.5*S[8]
            target=center_adv@ap-speed*s*d(0,ap)+.5*speed*s*(d(1,A[0][1])-d(2,A[0][2]))-.5*chi*(dd(1,1,lapse)-dd(2,2,lapse))
            dplus=target[pos]-(Vp[9*n+pos]+.5*Vp[8*n+pos])
        L[9*n+pos]+=dplus
        crossrate=-2*V[12*n+pos]/np.sqrt(chi)+s*Db[pos]@V[5*n:6*n]
        crosstarget=0.
        if tt_target=='tangential':
            normalrate=lam*(-2*s*D[pos]@S[12]/np.sqrt(chi)+Dxx[pos]@S[5])
            crosstarget=-2*Vp[12*n+pos]/np.sqrt(chi)+s*Xp[5][pos]-normalrate
        dcross=.5*np.sqrt(chi)*(crossrate-crosstarget)
        if tt_target=='paper':
            target=center_adv@S[12]-speed*s*d(0,S[12])+.5*speed*s*(d(1,A[0][2])+d(2,A[0][1]))-chi*dd(1,2,lapse)
            dcross=target[pos]-Vp[12*n+pos]
        L[12*n+pos]+=dcross
    return L,dict(volume=V,Q=Q,E=E)
