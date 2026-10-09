"""100-digit factored outer Schwarzschild gauge/null-rate oracle."""
import json
from pathlib import Path
import mpmath as mp
mp.mp.dps=100
S=mp.mpf(1);a=mp.mpf('.5');M=mp.mpf('.5');xi=mp.mpf('1.5')
nu=mp.mpf('1.5');regular_eta=mp.mpf(1)

def evaluate(O,mode='baseline',da1=mp.mpf(0),db1=mp.mpf(0),da2=mp.mpf(0),db2=mp.mpf(0),xi_rate=None,eta_rate=mp.mpf(10)):
    lapse_xi=xi if xi_rate is None else xi_rate
    r=mp.sqrt(S*S-2*a*S*O);op=-r/(a*S);opp=-1/(a*S)
    L=S/a-O;b=r/a;psi=1+M*O/(2*r);m=psi-1;N=(1-m)/psi;F=(1-m)/psi**3
    ahat=L;bhat=-b;alpha=N*L+da1*O+da2*O*O;beta=-b*F+db1*O+db2*O*O;chi=psi**-4
    def fields(x):
        rr=mp.sqrt(S*S-2*a*S*x);ll=S/a-x;mm=M*x/(2*rr);pp=1+mm;nn=(1-mm)/pp;ff=(1-mm)/pp**3
        return nn*ll+da1*x+da2*x*x,-rr/a*ff+db1*x+db2*x*x,pp**-4
    ap=mp.diff(lambda x:fields(x)[0],O)*op;bp=mp.diff(lambda x:fields(x)[1],O)*op;cp=mp.diff(lambda x:fields(x)[2],O)*op
    lp=-op;bhatp=-1/a
    dadiv=-M*L/(r*psi)+da1+da2*O
    dbdiv=M/(2*a)*(4+3*m+m*m)/psi**3+db1+db2*O
    pd=M/(2*a*r)*(8-m-6*m*m-3*m**3)/((1-m)*psi**3)
    chdiv=-M/(2*r)*(4+6*m+4*m*m+m**3)/psi**4
    what=-bhat*op/ahat
    dwn_div=-(dbdiv*op+what*dadiv)/alpha;wn=what+O*dwn_div
    kdiv=-3/(a*L)+pd-3*dwn_div
    shear=-M/(a*r)*(2-m)/(psi**2*(1-m*m));Ar=mp.mpf(2)/3*shear
    # Exact projection of full chi and metric equations; divergence terms
    # cancel algebraically, rather than dropping the geometric contribution.
    geometric=op*op*(beta*cp+2*alpha*chi*(Ar+kdiv/3)-2*chi*bp)
    adot=beta*ap-bhat*lp-alpha*nu*mp.log(alpha/ahat)
    adot-=alpha*alpha*pd+lapse_xi*(alpha+ahat)*dadiv+(alpha*dbdiv+bhat*dadiv)*op
    bdot=beta*bp-bhat*bhatp-regular_eta*O*dbdiv+alpha*alpha*chi*(cp/(2*chi)-ap/alpha+lp/ahat)
    deltaNull_div=chdiv*op*op-dwn_div*(2*what+O*dwn_div)
    if mode=='eta10':bdot-=eta_rate*dbdiv
    if mode in ['preferred','feedback']:
        f0refdiv=3/(a*L*L)
        f0pole_div=f0refdiv+(3*dwn_div-f0refdiv*O*dadiv)/alpha+((alpha*dbdiv+bhat*dadiv)*op+lapse_xi*(alpha+ahat)*dadiv)/alpha**3
        f0reg=(bhat*lp+alpha*nu*mp.log(alpha/ahat))/alpha**3
        source=(bhat*bhatp+regular_eta*O*dbdiv)/alpha**2-beta*f0reg-chi*lp/ahat
        nref=op*op/(L*L)
        whatBox=2*nref+O*(-3/(a*S*L*L)+r*r/(a*a*S*S*L**3))
        hess4=(chi-beta*beta/(alpha*alpha))*opp+2*chi*op/r
        deltaReg=hess4-O*whatBox-op*source
        bdot-=alpha*alpha/op*deltaReg+alpha*alpha*beta*f0pole_div
        if mode=='feedback':bdot+=5*alpha*alpha/op*deltaNull_div
    gauge=2*wn/alpha*(op*bdot+wn*adot)
    return {'Ndot':geometric+gauge,'geometry':geometric,'gauge':gauge,'alpha_dot':adot,'beta_dot':bdot,
            'N_raw':O*(O*op*op/(L*L)+deltaNull_div),'deltaNull_div':deltaNull_div,
            'alpha_deviation_div':dadiv,'beta_deviation_div':dbdiv}

if __name__=='__main__':
    rows=[]
    for mode in ['baseline','preferred','feedback','eta10']:
        for O in [mp.mpf('.0975'),mp.mpf('1e-4'),mp.mpf('1e-8'),mp.mpf('1e-16'),mp.mpf('1e-30'),mp.mpf(0)]:
            v=evaluate(O,mode);rows.append({'mode':mode,'Omega':str(O),**{k:mp.nstr(x,25) for k,x in v.items()}})
    expected={'baseline':24,'preferred':0,'feedback':0,'eta10':-56}
    for mode,value in expected.items():assert abs(evaluate(mp.mpf(0),mode)['Ndot']-value)<mp.mpf('1e-80')
    # Free gauge first jets a1=-4.5,b1=5.5 retain fixed ADM data and N_raw=O(O^2).
    # Relative to the original a1=-1,b1=2 they add -3.5 O and +3.5 O.
    da1=mp.mpf('-3.5');db1=mp.mpf('3.5')
    n0=evaluate(mp.mpf(0),'eta10',da1,db1)['Ndot'];assert abs(n0)<mp.mpf('1e-80')
    ncoef=mp.diff(lambda x:evaluate(x,'eta10',da1,db1)['Ndot'],mp.mpf(0))
    acoef=mp.diff(lambda x:evaluate(x,'eta10',da1,db1,mp.mpf(1))['Ndot'],mp.mpf(0))-ncoef
    bcoef=mp.diff(lambda x:evaluate(x,'eta10',da1,db1,mp.mpf(0),mp.mpf(1))['Ndot'],mp.mpf(0))-ncoef
    # Keeping N_raw=O(O^2) initial data does not impose a second coefficient; one
    # further compatibility condition cancels dt N_raw's O(O) coefficient.
    da2=-ncoef/acoef
    ncoef2=mp.diff(lambda x:evaluate(x,'eta10',da1,db1,da2)['Ndot'],mp.mpf(0))
    assert abs(ncoef2)<mp.mpf('1e-75')
    extras={'compatible_gauge_da1':str(da1),'compatible_gauge_db1':str(db1),
        'eta10_compatible_Ndot0':mp.nstr(n0,30),'eta10_compatible_Ndot_Omega_coefficient':mp.nstr(ncoef,30),
        'da2_linear_coefficient':mp.nstr(acoef,30),'db2_linear_coefficient':mp.nstr(bcoef,30),
        'example_da2_canceling_second_gate':mp.nstr(da2,30),'second_gate_residual':mp.nstr(ncoef2,30)}
    (Path(__file__).with_name('oracle.json')).write_text(json.dumps({'dps':mp.mp.dps,'rows':rows,'second_gate':extras},indent=2)+'\n')
    print('PASS 100-digit factored outer rates; original Ndot limits',expected)
    print(json.dumps(extras,indent=2))
