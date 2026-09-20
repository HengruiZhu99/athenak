"""Preparation only: independent Liu Kerr initial-data checks, no evolution.

Primary equations: https://arxiv.org/pdf/1001.4077, Eqs. 11,13--15.
Symbolically differentiate the spherical tensors, contract at 60 digits, and
test both ADM constraints and the complete stationary ADM geometric RHS.
"""
from pathlib import Path
import sys, json, time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] /
                       'vacuum-preservation-20260918/python-deps'))
import sympy as s
import mpmath as mp

mp.mp.dps = 60
r, th, ph = s.symbols('r theta phi', positive=True, real=True)
a = s.symbols('a', real=True)
xx = (r, th, ph)
rp, rm = 1+s.sqrt(1-a*a), 1-s.sqrt(1-a*a)
c = rp/4
R = (r+c)**2/r
Sigma = R*R+a*a*s.cos(th)**2
Delta = (R-rp)*(R-rm)
A = (R*R+a*a)**2-Delta*a*a*s.sin(th)**2
g = s.diag(Sigma*(r+c)**2/(r**3*(R-rm)), Sigma,
           A*s.sin(th)**2/Sigma)
P = 3*R**4+2*a*a*R*R-a**4-a*a*(R*R-a*a)*s.sin(th)**2
Krphi = a*s.sin(th)**2*P/(Sigma*s.sqrt(A*Sigma))*(1+c/r)/s.sqrt(r*(R-rm))
Ktphi = -2*a**3*R*s.cos(th)*s.sin(th)**3/(Sigma*s.sqrt(A*Sigma))*(r-c)*s.sqrt((R-rm)/r)
K = s.Matrix([[0,0,Krphi],[0,0,Ktphi],[Krphi,Ktphi,0]])
omega = -2*a*R/A
alpha_signed = (r-c)*s.sqrt((R-rm)*Sigma/(r*A))
det_cart = g.det()/(r**4*s.sin(th)**2)
alpha_precollapsed = det_cart**(-s.Rational(1,6))

def compile_jet(expr, second=True):
    es = [expr]+[s.diff(expr,x) for x in xx]
    if second: es += [s.diff(expr,x,y) for x in xx for y in xx]
    return s.lambdify((r,th,a), es, modules='mpmath', cse=True)

gj = [compile_jet(g[i,i]) for i in range(3)]
kj = {(i,j):compile_jet(K[i,j],False) for i,j in [(0,2),(1,2)]}
bj = compile_jet(omega,False)
aj = {'signed_stationary':compile_jet(alpha_signed),
      'precollapsed_zero_shift':compile_jet(alpha_precollapsed)}

def zeros(*dims):
    return [zeros(*dims[1:]) for _ in range(dims[0])] if dims else mp.mpf('0')

def evaluate(rad,theta,spin):
    args = (rad,theta,spin)
    gd = [f(*args) for f in gj]
    gd0 = [q[0] for q in gd]; gu = [1/q for q in gd0]
    dg,ddg = zeros(3,3,3),zeros(3,3,3,3)
    for i in range(3):
        for p in range(3):
            dg[p][i][i]=gd[i][1+p]
            for q in range(3): ddg[p][q][i][i]=gd[i][4+3*p+q]
    kval,dk = zeros(3,3),zeros(3,3,3)
    for (i,j),f in kj.items():
        v=f(*args);kval[i][j]=kval[j][i]=v[0]
        for p in range(3): dk[p][i][j]=dk[p][j][i]=v[1+p]
    Gamma,dGamma = zeros(3,3,3),zeros(3,3,3,3)
    for u in range(3):
        for i in range(3):
            for j in range(3):
                C=dg[i][u][j]+dg[j][u][i]-dg[u][i][j]
                Gamma[u][i][j]=gu[u]*C/2
                for p in range(3):
                    dC=ddg[p][i][u][j]+ddg[p][j][u][i]-ddg[p][u][i][j]
                    dGamma[p][u][i][j]=(-gu[u]**2*dg[p][u][u]*C+gu[u]*dC)/2
    Ricci=zeros(3,3)
    for i in range(3):
        for j in range(3):
            Ricci[i][j]=sum(dGamma[k][k][i][j]-dGamma[j][k][i][k] for k in range(3))
            Ricci[i][j]+=sum(Gamma[k][i][j]*Gamma[l][k][l]-Gamma[l][i][k]*Gamma[k][j][l] for k in range(3) for l in range(3))
    K2=sum(gu[i]*gu[j]*kval[i][j]**2 for i in range(3) for j in range(3))
    H=sum(gu[i]*Ricci[i][i] for i in range(3))-K2
    mom=[]
    for i in range(3):
        v=sum(-gu[j]**2*dg[j][j][j]*kval[j][i]+gu[j]*dk[j][j][i] for j in range(3))
        v+=sum(Gamma[j][j][l]*gu[l]*kval[l][i]-Gamma[l][j][i]*gu[j]*kval[j][l] for j in range(3) for l in range(3))
        mom.append(v)
    b=bj(*args)
    gauges={}
    for label in ['signed_stationary','absolute_stationary','precollapsed_zero_shift']:
        av=aj['precollapsed_zero_shift' if label=='precollapsed_zero_shift' else 'signed_stationary'](*args)
        if label=='absolute_stationary':
            if av[0]==0: continue  # |alpha| is not differentiable at the throat.
            sg=mp.sign(av[0]);av=[sg*x for x in av]
        bv=[mp.mpf(0)]*4 if label=='precollapsed_zero_shift' else b
        metric_rhs,curvature_rhs=zeros(3,3),zeros(3,3)
        for i in range(3):
            for j in range(3):
                lie_g=(gd0[2]*bv[1+i] if j==2 else 0)+(gd0[2]*bv[1+j] if i==2 else 0)
                metric_rhs[i][j]=lie_g-2*av[0]*kval[i][j]
                dda=av[4+3*i+j]-sum(Gamma[k][i][j]*av[1+k] for k in range(3))
                lie_K=kval[2][j]*bv[1+i]+kval[i][2]*bv[1+j]
                curvature_rhs[i][j]=-dda+av[0]*(Ricci[i][j]-2*sum(gu[k]*kval[i][k]*kval[k][j] for k in range(3)))+lie_K
        def norm(t): return mp.sqrt(sum(gu[i]*gu[j]*t[i][j]**2 for i in range(3) for j in range(3)))
        gauges[label]={'alpha':float(av[0]),'metric_rhs_norm':float(norm(metric_rhs)),
                       'curvature_rhs_norm':float(norm(curvature_rhs)),
                       'standard_oplog_lapse_driver_2alpha':float(2*av[0])}
    return {'r':float(rad),'theta':float(theta),'a_over_M':float(spin),
            'Hamiltonian':float(H),'momentum_norm':float(mp.sqrt(sum(gu[i]*mom[i]**2 for i in range(3)))),
            'K_trace':float(sum(gu[i]*kval[i][i] for i in range(3))),
            'g_diag_min':float(min(gd0)), 'gauges':gauges}

if __name__=='__main__':
    start=time.monotonic();rows=[]
    for spin in [mp.mpf('0'),mp.mpf('0.5')]:
        horizon=(1+mp.sqrt(1-spin*spin))/4
        for factor in ['0.1','0.5','0.99','1','1.01','2','10']:
            rows.append(evaluate(horizon*mp.mpf(factor),mp.pi/3,spin))
    threshold=1e-45
    assert max(abs(x['Hamiltonian']) for x in rows)<threshold
    assert max(x['momentum_norm'] for x in rows)<threshold
    assert max(x['gauges']['signed_stationary']['metric_rhs_norm'] for x in rows)<threshold
    assert max(x['gauges']['signed_stationary']['curvature_rhs_norm'] for x in rows)<threshold
    assert all(x['g_diag_min']>0 for x in rows)
    result={'method':'60-digit symbolic derivative/contraction; spherical coordinates; no evolution',
            'source':'https://arxiv.org/pdf/1001.4077','checks_passed':True,
            'wall_seconds':time.monotonic()-start,'points':rows}
    target=Path(__file__).with_name('geometry-results.json');target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='points'},indent=2))
