"""HELD complete analytic Gaussian inverse3 and compact ADM-consumed data."""
import mpmath as mp
from taylor3 import Jet, compose_radial, compose_jet, inverse
from reference3 import make_reference, height_physical_derivatives
from gaussian3 import wave, fderivative, outer_profiles, numerical_root, implicit_jet
from geometry import metric_from_embedding, adm_from_metric, extract_fields


def construct(event, layer, sigma, epsilon, settings):
    ref = make_reference(event,layer)
    r, omega, a = ref["rvalue"], ref["omega"], layer.a
    Q, B = ref["Q"], ref["B"]
    lower_j = 1-4*abs(epsilon)/mp.pi
    if lower_j <= 0:
        raise ArithmeticError("declared global inverse monotonicity bound fails")
    Rvalue = r/omega.v
    if r >= layer.r1:
        radius = ref["radius"]
        n = [v/radius for v in ref["x"]]
        p, z = n[0]*n[1], omega/radius
        s = ref["t"]+layer.outer_constant
        center = a*a/(mp.sqrt(1+a*a*z.v*z.v)+1)
        bound = abs(epsilon*p.v)*(2*sigma*sigma+6*abs(z.v)*sigma**3+6*z.v*z.v*sigma**4)
        def scalar(c):
            Phi, plus, minus = outer_profiles(s.v+z.v*c,z.v,sigma)
            J = 1+epsilon*p.v*z.v*(minus+z.v*plus)/2
            return c+epsilon*p.v*Phi-center,J
        root, root_record = numerical_root(scalar,center,bound,lower_j,settings)
        def equation(c):
            Phi, _, _ = outer_profiles(s+z*c,z,sigma)
            return c+epsilon*p*Phi-a*a/((1+a*a*z*z)**mp.mpf(".5")+1)
        c, inverse_residual = implicit_jet(root,equation,scalar(root)[1])
        u = s+z*c
        Phi, plus, minus = outer_profiles(u,z,sigma)
        square = (1+a*a*z*z)**mp.mpf(".5")
        h, eta = 1/square,a*a/(square*(square+1))
        J = 1+epsilon*p*z*(minus+z*plus)/2
        wr = h+epsilon*p*z*(minus-z*plus)/2
        km, jp = eta+epsilon*p*plus,1+h+epsilon*p*z*minus
        tangent = [n[1]-2*p*n[0],n[0]-2*p*n[1],-2*p*n[2]]
        Delta = km*jp-epsilon*epsilon*z*z*Phi*Phi*sum(v*v for v in tangent)
        D = z*z*Delta
        T = radius/omega+u
        if Delta.v <= 0:
            return dict(ref=ref, refused=True, J=J.v,D=D.v,Delta=Delta.v,
                        root=root_record, inverse_residual=inverse_residual,
                        branch="outer_c_ret")
        L = ref["L"]
        gamma = [[int(i==j)-n[i]*n[j]+L*L*km*jp/(radius*radius*J*J)*n[i]*n[j]
                  +epsilon*L*z*wr*Phi/(radius*J*J)*(n[i]*tangent[j]+tangent[i]*n[j])
                  -epsilon*epsilon*z**4*Phi*Phi/(J*J)*tangent[i]*tangent[j]
                  for j in range(3)] for i in range(3)]
        alpha = radius/Delta**mp.mpf(".5")
        beta = [-radius*radius*wr/(L*Delta)*n[i]+epsilon*omega*Phi/Delta*tangent[i] for i in range(3)]
        primary = "bounded_outer_conformal"
        auxiliary = dict(c=c,u=u,z=z,Phi=Phi,plus=plus,minus=minus,J=J,wr=wr,km=km,jp=jp,Delta=Delta)
    else:
        def scalar(Tvalue):
            T1 = Jet.variable(Tvalue,0,1)
            Q1 = [Jet(q.v,1) for q in Q]
            F = wave(T1,Q1,sigma)
            return Tvalue+epsilon*F.v-B.v,1+epsilon*F.derivative(0)
        bound = 16*abs(epsilon)*sigma/(3*mp.pi)
        if epsilon == 0 or r == 0 or ref["x"][0].v*ref["x"][1].v == 0:
            bound = mp.mpf(0)
        root, root_record = numerical_root(scalar,B.v,bound,lower_j,settings)
        T, inverse_residual = implicit_jet(root,lambda t:t+epsilon*wave(t,Q,sigma)-B,scalar(root)[1])
        embedding = [T]+Q
        physical = metric_from_embedding(embedding)
        bar = [[omega*omega*entry for entry in row] for row in physical]
        try:
            alpha,beta,gamma = adm_from_metric(bar)
        except ArithmeticError:
            # Negative control is recorded before any invalid ADM square root.
            Tq, Jq, Dq = graph_embedding(ref,T.v,sigma,epsilon)
            return dict(ref=ref,refused=True,J=Jq.v,D=Dq.v,root=root_record,
                        inverse_residual=inverse_residual,branch="Cartesian_T")
        primary, auxiliary = "Cartesian_embedding_conformal",{}
    embedding = [T]+Q
    compact = extract_fields(alpha,beta,gamma,omega)
    Tq,Jq,Dq = graph_embedding(ref,T.v,sigma,epsilon)
    if not (Jq.v > 0 and Dq.v > 0):
        raise ArithmeticError("a positive primary ADM branch disagrees with the graph signs")
    return dict(ref=ref,refused=False,root=root_record,inverse_residual=inverse_residual,
                embedding=embedding,compact=compact,Tq=Tq,J=Jq,D=Dq,
                primary=primary,auxiliary=auxiliary,sigma=sigma,epsilon=epsilon)


def graph_embedding(ref,Tvalue,sigma,epsilon):
    """Independent graph-chart implicit jets in (native t, physical Q)."""
    q = [Jet.variable(v.v,i+1) for i,v in enumerate(ref["Q"])]
    t = Jet.variable(ref["event"][0],0)
    if ref["rvalue"] <= mp.mpf(".05"):
        H = Jet(0)
    else:
        R = sum(v*v for v in q)**mp.mpf(".5")
        first,second,third = height_physical_derivatives(ref)
        H = compose_radial(ref["H"].v,first,second,third,R)
    T1 = Jet.variable(Tvalue,0,1)
    F1 = wave(T1,[Jet(v.v,1) for v in q],sigma)
    Jvalue = 1+epsilon*F1.derivative(0)
    Tq,_ = implicit_jet(Tvalue,lambda T:T+epsilon*wave(T,q,sigma)-t-H,Jvalue)
    # Evaluate partial inertial F derivatives at the event, independently from
    # the composed native derivatives. H has no graph-chart time dependence.
    inertial = [Jet.variable(Tvalue,0,2)]+[Jet.variable(v.v,i+1,2) for i,v in enumerate(q)]
    F = wave(inertial[0],inertial[1:],sigma)
    J = 1+epsilon*F.diff(0)
    w = [H.diff(i+1).truncate(1)-epsilon*F.diff(i+1) for i in range(3)]
    D = J*J-sum(v*v for v in w)
    return Tq,J,D


def graph_comparisons(data):
    """Independent metric/physical-K data; near-scri comparison is diagnostic."""
    ref, compact = data["ref"],data["compact"]
    increments = [ref['t']-ref['event'][0]]+[q-q.v for q in ref['Q']]
    Tback = compose_jet(data['Tq'],increments)
    physical = metric_from_embedding([Tback]+ref['Q'])
    graph_bar = [[ref["omega"]**2*v for v in row] for row in physical]
    ga,gb,gg = adm_from_metric(graph_bar)
    J,D,Tq = data["J"].v,data["D"].v,data["Tq"]
    Kq = [[-J*Tq.derivative(i+1,j+1)/mp.sqrt(D) for j in range(3)] for i in range(3)]
    Qjac = [[ref["Q"][I].derivative(i+1) for i in range(3)] for I in range(3)]
    Knative = [[mp.fsum(Qjac[I][i]*Qjac[Jj][j]*Kq[I][Jj] for I in range(3) for Jj in range(3))
                for j in range(3)] for i in range(3)]
    omega = ref["omega"].v
    Kcompact = [[compact["Kbar"][i][j].v/omega+compact["wn"].v*compact["gamma"][i][j].v/omega**2
                 for j in range(3)] for i in range(3)]
    return dict(alpha=(ga,compact["alpha"]),beta=list(zip(gb,compact["beta"])),
                gamma=[(gg[i][j],compact["gamma"][i][j]) for i in range(3) for j in range(3)],
                physical_K=[(Knative[i][j],Kcompact[i][j]) for i in range(3) for j in range(3)])
