"""HELD analytic fixed-Lorentz-ray scalar derivatives, no inverse map.

No source/kernel/native import, numerical differentiation, CAS, spectral
operator or time stepping is used. values_context.py is an exact local copy
of the preserved values source; its main is never called. Every boost is
constant during differentiation. All derivatives are inertial Cartesian.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import mpmath as mp
import values_context as vc
from analytic_jets import Jet2, radial_lift, sumjet


HERE = Path(__file__).resolve().parent
PAIRS = [(a, b) for a in range(4) for b in range(a, 4)]
ZERO = lambda: mp.mpf(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def strings(values):
    return [vc.number(x) for x in values]


def delta(i, j):
    return mp.mpf(int(i == j))


def radial_table(layer, r):
    """Analytic cutoff through third derivative, ordinary r derivatives.

    The exact core/outer branches precede any logit or endpoint reciprocal.
    Both logistic tails and w(1-w) use the smaller exponential directly.
    """
    a = layer.a
    if r <= layer.r0:
        w, wp, wpp, wppp = [ZERO()]*4
    elif r >= layer.r1:
        w, wp, wpp, wppp = mp.mpf(1), ZERO(), ZERO(), ZERO()
    else:
        width = layer.r1-layer.r0
        s, t = (r-layer.r0)/width, (layer.r1-r)/width
        g = -1/s+1/t
        e = mp.exp(-abs(g))
        w, wc = (e/(1+e), 1/(1+e)) if g <= 0 else (1/(1+e), e/(1+e))
        p = e/(1+e)**2
        gp = (s**-2+t**-2)/width
        gpp = 2*(-s**-3+t**-3)/width**2
        gppp = 6*(s**-4+t**-4)/width**3
        wp = p*gp
        wpp = p*((wc-w)*gp**2+gpp)
        wppp = p*(((wc-w)**2-2*p)*gp**3+3*(wc-w)*gp*gpp+gppp)
    if r <= layer.r0:
        omega = Jet2(1, [ZERO()], [[ZERO()]])
        omega3 = ZERO()
        b = Jet2(0, [ZERO()], [[ZERO()]])
    else:
        outer, op, opp = (1-r)*(1+r)/(2*a), -r/a, -1/a
        if r >= layer.r1:
            ov = outer
        else:
            # wc was formed directly; no subtraction of a unit tail.
            ov = wc+w*outer
        omega = Jet2(ov, [wp*(outer-1)+w*op], [[wpp*(outer-1)+2*wp*op+w*opp]])
        omega3 = wppp*(outer-1)+3*wpp*op+3*wp*opp
        b = Jet2(r*w/a, [(w+r*wp)/a], [[(2*wp+r*wpp)/a]])
    h = (omega*omega+b*b)**mp.mpf(".5")
    L = Jet2(omega.v-r*omega.d[0], [-r*omega.h[0][0]], [[-omega.h[0][0]-r*omega3]])
    if not (omega.v > 0 and h.v > 0 and L.v > 0):
        raise ArithmeticError("nonpositive analytic reference coefficient")
    v = b/h
    rq = omega.v**2/L.v
    rqq = rq*(2*omega.v*omega.d[0]*L.v-omega.v**2*L.d[0])/L.v**2
    vq = v.d[0]*rq
    vqq = v.h[0][0]*rq*rq+v.d[0]*rqq
    return {"omega":omega, "b":b, "h":h, "L":L, "omega3":omega3, "rq":rq, "rqq":rqq, "v":v.v, "vq":vq, "vqq":vqq}


def lift_coefficient(jet, table, q, nu):
    first = jet.d[0]*table["rq"]
    second = jet.h[0][0]*table["rq"]**2+jet.d[0]*table["rqq"]
    return radial_lift(jet.v, first, second, q, nu)


def height_tensors(q, nu, v, vq, vqq):
    if q == 0:
        if v != 0 or vq != 0 or vqq != 0:
            raise ArithmeticError("use the smooth Cartesian graph branch at the center")
        return [ZERO()]*3, [[ZERO()]*3 for _ in range(3)], [[[ZERO()]*3 for _ in range(3)] for _ in range(3)]
    hi = [v*x for x in nu]
    hij = [[vq*nu[i]*nu[j]+v/q*(delta(i,j)-nu[i]*nu[j]) for j in range(3)] for i in range(3)]
    c3, c1 = vqq-3*vq/q+3*v/q**2, vq/q-v/q**2
    hijk = [[[c3*nu[i]*nu[j]*nu[k]+c1*(delta(i,j)*nu[k]+delta(i,k)*nu[j]+delta(j,k)*nu[i]) for k in range(3)] for j in range(3)] for i in range(3)]
    return hi, hij, hijk


def pulse_cartesian(x, omega, b, h, L, nu, amplitudes):
    """Return G=s*Omega directly, not Pi then a canceling Omega^3.

    In the outer branch f=(2a)^4 Omega^4 exp(-r^2/w^2) is used by
    NativeGraph.source. No live-lapse, data or logarithmic division occurs
    except the physical positive initial alpha in the original normal data.
    """
    rho = sumjet(v*v for v in x)
    f = ((1-rho)**4)*(-rho/mp.mpf(".35")**2).exp()
    shape_a = 1+mp.mpf(".2")*x[0]+mp.mpf(".3")*x[1]*x[2]
    shape_b = [1+mp.mpf(".3")*x[1]*x[2], mp.mpf(".2")*x[0], mp.mpf(".1")*x[0]*x[1]]
    aa, ab = [mp.mpf(v) for v in amplitudes]
    alpha = h+aa*f*shape_a
    if not alpha.v > 0:
        raise ArithmeticError("nonpositive initial native pulse lapse")
    if nu is None:
        # Exact Cauchy core, including y=0. Avoid radial normals and 1/q.
        return [-aa*f*shape_a/alpha]+[-ab*f*v/alpha for v in shape_b]
    bn = sumjet(shape_b[i]*nu[i] for i in range(3))
    tangent = [shape_b[i]-bn*nu[i] for i in range(3)]
    normal = b*aa*shape_a+L*ab*bn
    return [-f*(h*aa*shape_a+(b*L/h)*ab*bn)/alpha]+[-f*(normal*nu[i]+omega*ab*tangent[i])/alpha for i in range(3)]


class NativeGraph:
    def __init__(self, recipe, height_order, amplitudes):
        self.layer = vc.Layer(recipe, height_order)
        self.amplitudes = amplitudes
        self.a = self.layer.a
        self.root_context = {}

    def event(self, case):
        x = list(map(mp.mpf, case["xyz"]))
        re = vc.norm(x)
        omega, od, b, h, L = self.layer.reference(re)
        X = [v/omega for v in x]
        return self.layer.height(re)+mp.mpf(case["tau_reference"]), X, omega, h/omega, b/omega, x

    def root(self, T, X, k, initial=False):
        if initial:
            return ZERO()
        Re = vc.norm(X)
        if Re+T <= self.layer.r0:
            return T/k[0]
        eventkey=(T,tuple(X))
        if eventkey not in self.root_context:
            if len(self.root_context)>=8:
                raise RuntimeError("declared bounded root-context cache exceeded")
            re = self.layer.r_of_q(Re)
            tau = T-self.layer.height(re)
            if tau <= 0:
                raise ArithmeticError("future native graph event required")
            outer_only=False
            if Re:
                _, _, lower, upper = self.layer.source_bounds(re, tau)
                outer_only=lower>=self.layer.r1
            self.root_context[eventkey]=outer_only
        if self.root_context[eventkey]:
            tc = T-self.layer.outer_constant
            den = 2*(tc*k[0]-vc.dot(X,k[1:]))
            if not den > 0:
                raise ArithmeticError("nonpositive analytic outer root denominator")
            return (tc*tc-Re*Re-self.a*self.a)/den
        lo, hi = ZERO(), T/k[0]
        for _ in range(self.layer.root_iterations):
            ell = (lo+hi)/2
            y = [X[i]-ell*k[i+1] for i in range(3)]
            q = vc.norm(y)
            r = self.layer.r_of_q(q)
            residual = T-ell*k[0]-self.layer.height(r)
            if residual > 0:
                lo = ell
            else:
                hi = ell
            if hi-lo < self.layer.root_tolerance:
                return (lo+hi)/2
        raise ArithmeticError("fixed ray root iteration limit")

    def height(self, y):
        return self.layer.height(self.layer.r_of_q(vc.norm(y)))

    def source(self, y, k):
        q = vc.norm(y)
        Y = [Jet2.variable(y[i],i) for i in range(3)]
        if q <= self.layer.r0:
            one, zero = Jet2(1), Jet2(0)
            G = pulse_cartesian(Y,one,zero,one,one,None,self.amplitudes)
            H1,H2,H3 = height_tensors(ZERO(),None,ZERO(),ZERO(),ZERO())
            return {"G":G,"D":Jet2(k[0]),"h":one,"omega":one,"H1":H1,"H2":H2,"H3":H3,"r":q,"q":q,"s":G,"direct_D":Jet2(k[0])}
        r = self.layer.r_of_q(q)
        nuv = [v/q for v in y]
        table = radial_table(self.layer,r)
        rjet = radial_lift(r,table["rq"],table["rqq"],q,nuv)
        qjet = sumjet(v*v for v in Y)**mp.mpf(".5")
        nu = [v/qjet for v in Y]
        omega,b,h,L = [lift_coefficient(table[key],table,q,nuv) for key in ["omega","b","h","L"]]
        x = [rjet*v for v in nu]
        # Write the exact outer pulse by its Omega^4 factor before any
        # reference-normal or denominator algebra. NativeGraph is fixed a=.5.
        if r >= self.layer.r1:
            rho = sumjet(v*v for v in x)
            f = (2*self.a)**4*omega**4*(-rho/mp.mpf(".35")**2).exp()
            sa = 1+mp.mpf(".2")*x[0]+mp.mpf(".3")*x[1]*x[2]
            sb = [1+mp.mpf(".3")*x[1]*x[2],mp.mpf(".2")*x[0],mp.mpf(".1")*x[0]*x[1]]
            aa,ab = map(mp.mpf,self.amplitudes)
            alpha = h+aa*f*sa
            if not alpha.v > 0:
                raise ArithmeticError("nonpositive exact outer initial lapse")
            bn = sumjet(sb[i]*nu[i] for i in range(3))
            bt = [sb[i]-bn*nu[i] for i in range(3)]
            pn = b*aa*sa+L*ab*bn
            G = [-f*(h*aa*sa+(b*L/h)*ab*bn)/alpha]+[-f*(pn*nu[i]+omega*ab*bt[i])/alpha for i in range(3)]
        else:
            G = pulse_cartesian(x,omega,b,h,L,nu,self.amplitudes)
        diff = [k[i+1]-k[0]*nu[i] for i in range(3)]
        D = k[0]*omega*omega/(h+b)+b*sumjet(v*v for v in diff)/(2*k[0])
        direct_D = h*k[0]-b*sumjet(nu[i]*k[i+1] for i in range(3))
        H1,H2,H3 = height_tensors(q,nuv,table["v"],table["vq"],table["vqq"])
        return {"G":G,"D":D,"h":h,"omega":omega,"H1":H1,"H2":H2,"H3":H3,"r":r,"q":q,"s":[g/omega for g in G],"direct_D":direct_D}


def harmonic_polynomial(y, degree):
    if degree == 0:
        return y[0]*0+1
    if degree == 1:
        return y[0]+mp.mpf(".3")*y[1]-mp.mpf(".2")*y[2]
    if degree == 2:
        return y[0]*y[1]+mp.mpf(".3")*(y[1]**2-y[2]**2)
    raise ValueError("unadmitted harmonic degree")


class ControlGraph:
    def __init__(self, kind, degree=0):
        self.kind,self.degree,self.a = kind,degree,mp.mpf(".5")

    def height(self,y):
        return ZERO() if self.kind.startswith("flat") else mp.sqrt(vc.dot(y,y)+self.a*self.a)

    def root(self,T,X,k,initial=False):
        if initial:
            return ZERO()
        if self.kind.startswith("flat"):
            return T/k[0]
        return (T*T-vc.dot(X,X)-self.a*self.a)/(2*(T*k[0]-vc.dot(X,k[1:])))

    def source(self,y,k):
        Y=[Jet2.variable(y[i],i) for i in range(3)]
        zero,one=Jet2(0),Jet2(1)
        if self.kind.startswith("flat"):
            H1,H2,H3=height_tensors(ZERO(),None,ZERO(),ZERO(),ZERO())
            c=[mp.mpf(v) for v in [".7","-.3",".2","1.1"]]
            slopes=[[mp.mpf(v) for v in row] for row in [[".2","-.1",".3"],["-.4",".25",".1"],[".3",".2","-.2"],[".15","-.35",".4"]]]
            s=[Jet2(c[A])+(sumjet(slopes[A][i]*Y[i] for i in range(3)) if self.kind=="flat_affine" else zero) for A in range(4)]
            return {"G":s,"D":Jet2(k[0]),"h":one,"omega":one,"H1":H1,"H2":H2,"H3":H3,"r":vc.norm(y),"q":vc.norm(y),"s":s,"direct_D":Jet2(k[0])}
        q=vc.norm(y)
        H=(sumjet(v*v for v in Y)+self.a*self.a)**mp.mpf(".5")
        omega=1/(H+self.a)
        h=omega*H/self.a
        # b nu = omega y/a is smooth even at the exact center.
        bnu=[omega*v/self.a for v in Y]
        if q:
            nuv=[v/q for v in y]
            v=q/H.v
            vq=self.a*self.a/H.v**3
            vqq=-3*self.a*self.a*q/H.v**5
            H1,H2,H3=height_tensors(q,nuv,v,vq,vqq)
            b=omega*q/self.a
            nu=[v/(sumjet(t*t for t in Y)**mp.mpf(".5")) for v in Y]
            D=k[0]*omega*omega/(h+b)+b*sumjet((k[i+1]-k[0]*nu[i])**2 for i in range(3))/(2*k[0])
        else:
            H1=[ZERO()]*3
            H2=[[delta(i,j)/self.a for j in range(3)] for i in range(3)]
            H3=[[[ZERO()]*3 for _ in range(3)] for _ in range(3)]
            D=h*k[0]-sumjet(bnu[i]*k[i+1] for i in range(3))
        direct_D=h*k[0]-sumjet(bnu[i]*k[i+1] for i in range(3))
        p=harmonic_polynomial(Y,self.degree)
        scales=list(map(mp.mpf,["1","-.3",".2",".7"]))
        s=[2*(self.degree+1)*p*v/self.a for v in scales]
        return {"G":[v*omega for v in s],"D":D,"h":h,"omega":omega,"H1":H1,"H2":H2,"H3":H3,"r":q,"q":q,"s":s,"direct_D":direct_D}

    def exact(self,T,X):
        Y=[Jet2.variable(v,i,dimension=4) for i,v in enumerate([T]+X)]
        if self.kind.startswith("flat"):
            source=self.source(X,[mp.mpf(1),ZERO(),ZERO(),mp.mpf(1)])["s"]
            out=[]
            for s in source:
                # Independent closed form T*(c+d.X).
                affine=Jet2(s.v,[ZERO()]+s.d,[[ZERO()]*4]+[[ZERO()]+row for row in s.h])
                out.append(Y[0]*affine)
            return out
        z=Y[0]**2-sumjet((v*v for v in Y[1:]),dimension=4)
        p=harmonic_polynomial(Y[1:],self.degree)
        u=p*(1-self.a**(2*self.degree+2)/z**(self.degree+1))
        return [u*mp.mpf(v) for v in ["1","-.3",".2",".7"]]


def boost_matrix(axis, rapidity):
    axis=[v/vc.norm(axis) for v in axis]
    c,s=mp.cosh(rapidity),mp.sinh(rapidity)
    return [[c]+[s*v for v in axis]]+[[s*axis[i]]+[delta(i,j)+(c-1)*axis[i]*axis[j] for j in range(3)] for i in range(3)]


def matmul(A,B):
    return [[mp.fsum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def fixed_boost(X,gamma,boost,mode):
    axis=[v/vc.norm(X) for v in X] if vc.norm(X) else [ZERO(),ZERO(),mp.mpf(1)]
    base=[[gamma]+[boost*v for v in axis]]+[[boost*axis[i]]+[delta(i,j)+(gamma-1)*axis[i]*axis[j] for j in range(3)] for i in range(3)]
    if mode=="normal":
        return base
    if mode=="normal_then_fixed_z_0.2":
        return matmul(base,boost_matrix([ZERO(),ZERO(),mp.mpf(1)],mp.mpf(".2")))
    if mode=="laboratory":
        return [[delta(i,j) for j in range(4)] for i in range(4)]
    if mode=="fixed_oblique_0.35":
        return boost_matrix([mp.mpf(1),mp.mpf(2),mp.mpf(-1)],mp.mpf(".35"))
    raise ValueError("unknown fixed Lorentz frame")


def pullback(jet,ya,lab,k):
    d=[vc.dot(jet.d,ya[a]) for a in range(4)]
    h=[[mp.fsum(jet.h[i][j]*ya[a][i]*ya[b][j] for i in range(3) for j in range(3))-vc.dot(jet.d,k[1:])*lab[a][b] for b in range(4)] for a in range(4)]
    return Jet2(jet.v,d,h)


def ray_integrand(graph,T,X,k,initial=False):
    ell=graph.root(T,X,k,initial)
    y=[X[i]-ell*k[i+1] for i in range(3)]
    src=graph.source(y,k)
    h,D=src["h"],src["D"]
    H1,H2=src["H1"],src["H2"]
    if not (D.v>0 and k[0]>0):
        raise ArithmeticError("nonpositive factored ray denominator")
    la=[(h.v*int(a==0)-h.v*(H1[a-1] if a else ZERO()))/D.v for a in range(4)]
    ya=[[delta(a,i+1)-k[i+1]*la[a] for i in range(3)] for a in range(4)]
    lab=[[-h.v*mp.fsum(H2[i][j]*ya[a][i]*ya[b][j] for i in range(3) for j in range(3))/D.v for b in range(4)] for a in range(4)]
    elljet=Jet2(ell,la,lab)
    Dj=pullback(D,ya,lab,k)
    result=[elljet*pullback(G,ya,lab,k)/Dj for G in src["G"]]
    # Independently evaluate the unfactored pencil quotient with F=G/h,
    # K=D/h. No logarithmic division by source data is made.
    Kj=pullback(D/h,ya,lab,k)
    alt=[elljet*pullback(G/h,ya,lab,k)/Kj for G in src["G"]]
    first=[int(a==0)-k[0]*la[a]-vc.dot(H1,ya[a]) for a in range(4)]
    second=[-k[0]*lab[a][b]-mp.fsum(H2[i][j]*ya[a][i]*ya[b][j] for i in range(3) for j in range(3))+vc.dot(H1,k[1:])*lab[a][b] for a,b in PAIRS]
    # K_ab formula explicitly tests the supplied third graph derivatives.
    H3=src["H3"]
    Ki=[-mp.fsum(k[i+1]*H2[i][j] for i in range(3)) for j in range(3)]
    Kij=[[-mp.fsum(k[i+1]*H3[i][j][l] for i in range(3)) for l in range(3)] for j in range(3)]
    Kp=pullback(Jet2(D.v/h.v,Ki,Kij),ya,lab,k)
    metrics={"root":abs(T-ell*k[0]-graph.height(y)),"null":abs(-k[0]**2+vc.dot(k[1:],k[1:]))/max(1,k[0]**2),"first_graph":max(abs(v) for v in first),"second_graph":max(abs(v) for v in second),"factor_quotient":vc.scaled([v for row in result for v in row.flat()],[v for row in alt for v in row.flat()]),"K_jet":vc.scaled(Kj.flat(),Kp.flat()),"D_source_jet":vc.scaled(D.flat(),src["direct_D"].flat()),"minimum_D":D.v}
    for row in result:
        if not all(mp.isfinite(x) for x in row.flat()):
            raise ArithmeticError("nonfinite analytic ray jet")
    return result,metrics


def integrate_ray(graph,T,X,B,nmu,naz,initial=False):
    nodes,weights=vc.gauss(nmu,mp.mp.dps)
    accum=[[[] for _ in range(15)] for _ in range(4)]
    metrics={key:ZERO() for key in ["root","null","first_graph","second_graph","factor_quotient","K_jet","D_source_jet","Lorentz_matrix"]}
    metrics["Lorentz_matrix"]=max(abs(mp.fsum((-1 if l==0 else 1)*B[l][a]*B[l][b] for l in range(4))-(-1 if a==b==0 else int(a==b))) for a in range(4) for b in range(4))/max(1,*[abs(v)**2 for row in B for v in row])
    if not B[0][0]>0:
        raise ArithmeticError("boost not future oriented")
    minD=mp.inf
    for mu,weight in zip(nodes,weights):
        for j in range(naz):
            az=2*mp.pi*j/naz
            transverse=mp.sqrt((1-mu)*(1+mu))
            unit=[mp.mpf(1),transverse*mp.cos(az),transverse*mp.sin(az),mu]
            k=[vc.dot(row,unit) for row in B]
            rays,m=ray_integrand(graph,T,X,k,initial)
            for key in m:
                if key=="minimum_D":
                    continue
                metrics[key]=max(metrics[key],m[key])
            minD=min(minD,m["minimum_D"])
            for A,row in enumerate(rays):
                for l,v in enumerate(row.flat()):
                    accum[A][l].append(weight*v/(2*naz))
    result=[[mp.fsum(values) for values in row] for row in accum]
    metrics["minimum_D"]=minD
    return result,metrics


def initial_oracle(graph,X):
    """Independent graph-data + wave-equation initial Hessian, all ten.

    A=s/c; graph tangential u=0 yields u_T=A,u_i=-H_i A.
    The wave trace fixes u_TT; differentiating graph data fixes the rest.
    """
    src=graph.source(X,[mp.mpf(1),ZERO(),ZERO(),mp.mpf(1)])
    c=src["omega"]/src["h"]
    H1,H2=src["H1"],src["H2"]
    out=[]
    for s in src["s"]:
        A=s/c
        tt=(-2*vc.dot(H1,A.d)-A.v*mp.fsum(H2[i][i] for i in range(3)))/c.v**2
        ti=[A.d[i]-H1[i]*tt for i in range(3)]
        ij=[[-H1[i]*A.d[j]-H1[j]*A.d[i]-A.v*H2[i][j]+H1[i]*H1[j]*tt for j in range(3)] for i in range(3)]
        hess=[[tt]+ti]+[[ti[i]]+ij[i] for i in range(3)]
        out.append(Jet2(0,[A.v]+[-v*A.v for v in H1],hess).flat())
    return out


def jet_blocks(rows):
    return {"u":[r[0] for r in rows],"gradient":[v for r in rows for v in r[1:5]],"hessian":[v for r in rows for v in r[5:15]]}


def trace_error(rows):
    # Packed Hessian order TT,Tx,Ty,Tz,xx,xy,xz,yy,yz,zz.
    return max(abs(-r[5]+r[9]+r[12]+r[14])/max(1,*[abs(v) for v in r[5:15]]) for r in rows)


def run(recipe,out):
    rows,controls,initials,checks=[],[],[],[]
    def check(name,error,tolerance):
        checks.append({"name":name,"error":vc.number(error),"tolerance":tolerance,"passed":bool(error<=mp.mpf(tolerance))})
    def add_metrics(prefix,m):
        for key in ["root","null","first_graph","second_graph","factor_quotient","K_jet","D_source_jet","Lorentz_matrix"]:
            tol=recipe["tolerances"]["root"] if key=="root" else recipe["tolerances"]["local_identities"]
            check(prefix+"/"+key,m[key],tol)
        if not m["minimum_D"]>0:
            raise ArithmeticError("nonpositive quadrature ray denominator")
    for dps in recipe["precisions"]:
        mp.mp.dps=dps
        for level in recipe["levels"]:
            for case in recipe["events"]:
                graph=NativeGraph(recipe,level["height_order"],case["amplitudes"])
                T,X,omega,gamma,boost,x=graph.event(case)
                for mode in recipe["native_boosts"]:
                    start=time.monotonic()
                    B=fixed_boost(X,gamma,boost,mode)
                    result,metrics=integrate_ray(graph,T,X,B,level["polar_order"],level["azimuth_order"])
                    row={"name":case["name"],"dps":dps,"level":level["name"],"boost":mode,"T":vc.number(T),"X":strings(X),"Omega_event":vc.number(omega),"boost_matrix":[strings(r) for r in B],"jet":[strings(r) for r in result],"phi_values":strings([r[0]/omega for r in result]),"metrics":{k:vc.number(v) for k,v in metrics.items()},"seconds":time.monotonic()-start,"scope":"fixed reference evaluation event; inertial derivatives of u; no inverse/native target"}
                    rows.append(row)
                    add_metrics("native/%s/%s/%s/%s"%(dps,level["name"],case["name"],mode),metrics)
                    row["wave_trace_scaled_error"]=vc.number(trace_error(result))
                    if level["name"]==recipe["final_level"]:
                        check("wave/native/%s/%s/%s/%s"%(dps,level["name"],case["name"],mode),trace_error(result),recipe["tolerances"]["wave_trace"])
                    write(out/"partial-native-jets.json",rows)
                    print("native-complete",dps,level["name"],case["name"],mode,time.monotonic()-start,flush=True)
            # Closed-form controls retain all four gradients and ten Hessians.
            for ctrl in recipe["controls"]:
                graph=ControlGraph(ctrl["kind"],ctrl.get("degree",0))
                X=list(map(mp.mpf,ctrl["X"]))
                T=graph.height(X)+mp.mpf(ctrl["tau"])
                exact=[j.flat() for j in graph.exact(T,X)]
                for mode in recipe["control_boosts"]:
                    B=fixed_boost(X,mp.mpf(1),ZERO(),mode)
                    result,metrics=integrate_ray(graph,T,X,B,level["polar_order"],level["azimuth_order"])
                    controls.append({"name":ctrl["name"],"dps":dps,"level":level["name"],"boost":mode,"jet":[strings(r) for r in result],"exact":[strings(r) for r in exact],"metrics":{k:vc.number(v) for k,v in metrics.items()}})
                    add_metrics("control/%s/%s/%s/%s"%(dps,level["name"],ctrl["name"],mode),metrics)
                    if level["name"]==recipe["final_level"]:
                        for block,numer in jet_blocks(result).items():
                            check("exact/%s/%s/%s/%s"%(dps,ctrl["name"],mode,block),vc.scaled(numer,jet_blocks(exact)[block]),recipe["tolerances"]["convergence"])
                    controls[-1]["wave_trace_scaled_error"]=vc.number(trace_error(result))
                    if level["name"]==recipe["final_level"]:
                        check("wave/control/%s/%s/%s/%s"%(dps,level["name"],ctrl["name"],mode),trace_error(result),recipe["tolerances"]["wave_trace"])
                    write(out/"partial-controls.json",controls)
                    print("control-complete",dps,level["name"],ctrl["name"],mode,flush=True)
        final=next(level for level in recipe["levels"] if level["name"]==recipe["final_level"])
        # Exact lambda=0 boundary limit, without a finite-difference derivative.
        for point in recipe["initial_points"]:
            graph=NativeGraph(recipe,final["height_order"],recipe["native_amplitudes"])
            case={"xyz":point,"tau_reference":"0"}
            T,X,omega,gamma,boost,x=graph.event(case)
            expected=initial_oracle(graph,X)
            for mode in recipe["native_boosts"]:
                B=fixed_boost(X,gamma,boost,mode)
                result,metrics=integrate_ray(graph,T,X,B,final["polar_order"],final["azimuth_order"],initial=True)
                initials.append({"xyz":point,"dps":dps,"boost":mode,"initial_jet":[strings(r) for r in result],"independent_graph_wave_oracle":[strings(r) for r in expected],"metrics":{k:vc.number(v) for k,v in metrics.items()}})
                for block,numer in jet_blocks(result).items():
                    check("initial-limit/%s/%s/%s/%s"%(dps,point,mode,block),vc.scaled(numer,jet_blocks(expected)[block]),recipe["tolerances"]["convergence"])
                add_metrics("initial/%s/%s/%s"%(dps,point,mode),metrics)
                write(out/"partial-initial-limits.json",initials)
        # Values cross-binding invokes the copied independent coarea values
        # implementation only at identical declared events; no old receipt run.
        layer=vc.Layer(recipe,final["height_order"])
        for case in recipe["events"]:
            value,phi,meta=vc.coarea(layer,case,128,128)
            for mode in recipe["native_boosts"]:
                row=next(r for r in rows if r["dps"]==dps and r["level"]==recipe["final_level"] and r["name"]==case["name"] and r["boost"]==mode)
                got=[mp.mpf(r[0]) for r in row["jet"]]
                check("coarea-value/%s/%s/%s/u"%(dps,case["name"],mode),vc.scaled(got,value),recipe["tolerances"]["convergence"])
                check("coarea-value/%s/%s/%s/phi"%(dps,case["name"],mode),vc.scaled(list(map(mp.mpf,row["phi_values"])),phi),recipe["tolerances"]["convergence"])
    mp.mp.dps=max(recipe["precisions"])
    lookup={(r["dps"],r["level"],r["name"],r["boost"]):r for r in rows}
    for case in recipe["events"]:
        name=case["name"]
        for mode in recipe["native_boosts"]:
            for dps in recipe["precisions"]:
                final=lookup[dps,recipe["final_level"],name,mode]
                baseblocks=jet_blocks([[mp.mpf(v) for v in r] for r in final["jet"]])
                for other in recipe["comparison_levels"]:
                    row=lookup[dps,other,name,mode]
                    compare=jet_blocks([[mp.mpf(v) for v in r] for r in row["jet"]])
                    for block in baseblocks:
                        check("convergence/%s/%s/%s/%s/%s"%(dps,name,mode,other,block),vc.scaled(baseblocks[block],compare[block]),recipe["tolerances"]["convergence"])
                    check("convergence/%s/%s/%s/%s/phi"%(dps,name,mode,other),vc.scaled(list(map(mp.mpf,final["phi_values"])),list(map(mp.mpf,row["phi_values"]))),recipe["tolerances"]["convergence"])
            lo=lookup[recipe["precisions"][0],recipe["final_level"],name,mode]
            hi=lookup[recipe["precisions"][1],recipe["final_level"],name,mode]
            for block,a in jet_blocks([[mp.mpf(v) for v in r] for r in lo["jet"]]).items():
                b=jet_blocks([[mp.mpf(v) for v in r] for r in hi["jet"]])[block]
                check("precision/%s/%s/%s"%(name,mode,block),vc.scaled(a,b),recipe["tolerances"]["precision"])
            check("precision/%s/%s/phi"%(name,mode),vc.scaled(list(map(mp.mpf,lo["phi_values"])),list(map(mp.mpf,hi["phi_values"]))),recipe["tolerances"]["precision"])
            if case["amplitudes"]==["0","0"]:
                check("zero/%s/%s"%(name,mode),max(abs(mp.mpf(v)) for r in hi["jet"] for v in r),"0")
        # Independent constant Lorentz quadratures must produce the same
        # inertial component derivatives, not transformed output components.
        for dps in recipe["precisions"]:
            ra,rb=[lookup[dps,recipe["final_level"],name,m] for m in recipe["native_boosts"]]
            for block,a in jet_blocks([[mp.mpf(v) for v in r] for r in ra["jet"]]).items():
                b=jet_blocks([[mp.mpf(v) for v in r] for r in rb["jet"]])[block]
                check("boost/%s/%s/%s"%(dps,name,block),vc.scaled(a,b),recipe["tolerances"]["convergence"])
    for fn,data in [("native-jets.json",rows),("controls.json",controls),("initial-limits.json",initials),("checks.json",checks)]:
        write(out/fn,data)
    return {"passed_analytic_scalar_derivative_gate":all(c["passed"] for c in checks),"checks":len(checks),"failed_checks":[c for c in checks if not c["passed"]],"native_rows":len(rows),"control_rows":len(controls),"initial_rows":len(initials),"scope":"Finite reference-event scalar values, four inertial gradients and ten inertial Hessians. No inverse/native-target coverage, Jacobian/caustic verdict, PDE or native evolution, Einstein stability or black-hole gauge admission."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--recipe",required=True)
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    receipt={"kind":"Analytic fixed-Lorentz-ray derivative attempt","accepted_native":False,"inverse_map_attempted":False}
    before={}
    try:
        recipepath,authpath=Path(args.recipe).resolve(),Path(args.authorization).resolve()
        if recipepath!=(HERE/"derivative-recipe.json").resolve():
            raise PermissionError("consumed recipe must be the pinned local derivative-recipe.json")
        recipe_bytes=recipepath.read_bytes()
        recipe=json.loads(recipe_bytes)
        auth=json.loads(authpath.read_text())
        required={str(HERE/name):sha(HERE/name) for name in ["flat_ivp_derivatives.py","analytic_jets.py","values_context.py","PLAN.md","derivative-recipe.json"]}
        consumed_recipe_sha256=hashlib.sha256(recipe_bytes).hexdigest()
        if required[str(HERE/"derivative-recipe.json")]!=consumed_recipe_sha256:
            raise RuntimeError("consumed recipe changed during admission")
        if auth.get("analytic_scalar_derivative_execution_admitted") is not True or auth.get("source_pins")!=required:
            raise PermissionError("missing exact root analytic-derivative release")
        if str(out)!=auth.get("fresh_output_path"):
            raise PermissionError("unreleased output path")
        for key,value in recipe["dependency_pins"].items():
            if sha(key)!=value:
                raise RuntimeError("dependency pin mismatch: "+key)
        value_receipt=json.loads(Path(recipe["completed_values_receipt"]).read_text())
        if value_receipt.get("passed_scalar_values_only") is not True or value_receipt.get("sources_unchanged") is not True:
            raise PermissionError("completed values PASS prerequisite absent")
        for key,expected in recipe["required_environment"].items():
            if os.environ.get(key)!=expected:
                raise RuntimeError("unreviewed environment: "+key)
        runtime=Path(sys.executable).resolve()
        if str(runtime)!=recipe["python_runtime_path"] or sha(runtime)!=recipe["python_runtime_sha256"] or sys.version_info[:2]!=(3,9) or mp.__version__!="1.3.0":
            raise RuntimeError("unreviewed Python/mpmath runtime")
        package={str(f):sha(f) for f in sorted(Path(mp.__file__).resolve().parent.rglob("*.py"))}
        if package!=recipe["mpmath_python_pins"]:
            raise RuntimeError("mpmath source inventory mismatch")
        before={**required,**recipe["dependency_pins"],**package,str(runtime):sha(runtime),str(authpath):sha(authpath)}
        receipt.update({"source_before":before,"consumed_recipe":{"path":str(recipepath),"sha256":consumed_recipe_sha256},"authorization":str(authpath),"command":sys.argv,"environment":{k:os.environ.get(k) for k in recipe["required_environment"]},"python":platform.python_version(),"mpmath":mp.__version__})
        write(out/"before.json",receipt)
        result=run(recipe,out)
        receipt.update(result)
        if not result["passed_analytic_scalar_derivative_gate"]:
            raise ArithmeticError("fixed analytic-derivative gate failed; no retry or tolerance change")
    except BaseException as exc:
        receipt.update({"passed_analytic_scalar_derivative_gate":False,"error":str(exc),"exception_type":type(exc).__name__,"traceback":traceback.format_exc()})
        raise
    finally:
        def protected_after(key):
            try:
                return sha(key)
            except BaseException as exc:
                return "READ_ERROR:"+repr(exc)
        receipt["source_after"]={key:protected_after(key) for key in before}
        receipt["sources_unchanged"]=receipt["source_after"]==before
        if not receipt["sources_unchanged"]:
            receipt["passed_analytic_scalar_derivative_gate"]=False
        receipt["seconds"]=time.monotonic()-start
        receipt["output_pins"]={str(f):sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!="receipt.json"}
        write(out/"receipt.json",receipt)
        if not receipt["sources_unchanged"]:
            raise RuntimeError("protected source/runtime drift")


if __name__=="__main__":
    main()
