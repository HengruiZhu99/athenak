"""HELD complete metric algebra, curvature terms and compact ADM extraction."""
import mpmath as mp
from taylor3 import Jet, determinant, inverse


def metric_from_embedding(embedding):
    return [[sum((-1 if A == 0 else 1)*embedding[A].diff(a)*embedding[A].diff(b)
                 for A in range(4)) for b in range(4)] for a in range(4)]


def metric_from_adm(alpha, beta, gamma):
    order = min(alpha.order, *(v.order for v in beta), *(v.order for row in gamma for v in row))
    result = [[Jet(0, order) for _ in range(4)] for _ in range(4)]
    result[0][0] = -alpha*alpha+sum(gamma[i][j]*beta[i]*beta[j] for i in range(3) for j in range(3))
    for i in range(3):
        result[0][i+1] = result[i+1][0] = sum(gamma[i][j]*beta[j] for j in range(3))
        for j in range(3):
            result[i+1][j+1] = gamma[i][j]
    return result


def adm_from_metric(metric):
    gi = inverse(metric)
    if gi[0][0].v >= 0:
        raise ArithmeticError("the native time level is not spacelike")
    alpha = (-gi[0][0])**mp.mpf("-.5")
    beta = [-gi[0][i+1]/gi[0][0] for i in range(3)]
    return alpha, beta, [[metric[i+1][j+1] for j in range(3)] for i in range(3)]


def connection(metric, axes):
    n = len(metric)
    if len(axes) != n:
        raise ValueError("metric dimension and derivative chart disagree")
    gi = inverse(metric)
    Gamma = [[[sum(gi[a][e]*(metric[e][c].diff(axes[b])+metric[e][b].diff(axes[c])
                    -metric[b][c].diff(axes[e]))/2 for e in range(n))
                for c in range(n)] for b in range(n)] for a in range(n)]
    return gi, Gamma


def curvature(metric, axes):
    """R^a_bcd with Ricci_bd=R^a_bad; retain fixed summands for scales."""
    n = len(metric)
    gi, Gamma = connection(metric, axes)
    terms, R = {}, [[[[None for _ in range(n)] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    pieces = [Gamma[a][d][b].diff(axes[c]).v,
                              -Gamma[a][c][b].diff(axes[d]).v]
                    pieces += [Gamma[a][c][e].v*Gamma[e][d][b].v for e in range(n)]
                    pieces += [-Gamma[a][d][e].v*Gamma[e][c][b].v for e in range(n)]
                    R[a][b][c][d] = mp.fsum(pieces)
                    terms[(a,b,c,d)] = pieces
    Ricci = [[mp.fsum(R[a][b][a][d] for a in range(n)) for d in range(n)] for b in range(n)]
    scalar = mp.fsum(gi[a][b].v*Ricci[a][b] for a in range(n) for b in range(n))
    return dict(inverse=gi, Gamma=Gamma, Riemann=R, terms=terms, Ricci=Ricci, scalar=scalar)


def embedding_connection(embedding):
    E = [[embedding[A].diff(a) for a in range(4)] for A in range(4)]
    Ei = inverse(E)
    return [[[sum(Ei[a][A]*embedding[A].diff(b).diff(c) for A in range(4))
              for c in range(4)] for b in range(4)] for a in range(4)]


def extract_fields(alpha, beta, gamma, omega):
    """All raw22 fields, with P/A/Lambda/Theta order one only."""
    alpha, beta = alpha.truncate(2), [v.truncate(2) for v in beta]
    gamma = [[v.truncate(2) for v in row] for row in gamma]
    gi = inverse(gamma)
    lie = [[sum(beta[k]*gamma[i][j].diff(k+1)
                +gamma[k][j]*beta[k].diff(i+1)+gamma[i][k]*beta[k].diff(j+1)
                for k in range(3)) for j in range(3)] for i in range(3)]
    Kbar = [[-(gamma[i][j].diff(0)-lie[i][j])/(2*alpha)
             for j in range(3)] for i in range(3)]
    K = sum(gi[i][j]*Kbar[i][j] for i in range(3) for j in range(3))
    wn = (-sum(beta[i]*omega.diff(i+1) for i in range(3))/alpha).truncate(1)
    P = omega*K+3*wn
    det = determinant(gamma)
    if det.v <= 0:
        raise ArithmeticError("nonpositive conformal spatial determinant")
    chi = det**(-mp.mpf(1)/3)
    gt = [[chi*gamma[i][j] for j in range(3)] for i in range(3)]
    A = [[chi*(Kbar[i][j]-gamma[i][j]*K/3) for j in range(3)] for i in range(3)]
    gti, Gamma = connection(gt, (1,2,3))
    Lambda = [sum(gti[j][k]*Gamma[i][j][k] for j in range(3) for k in range(3)) for i in range(3)]
    pairs = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    fields = [chi]+[gt[i][j] for i,j in pairs]+[P]+[A[i][j] for i,j in pairs]+Lambda+[Jet(0,1),alpha]+beta
    if len(fields) != 22 or any(v.order != (2 if k in list(range(7))+list(range(18,22)) else 1)
                                 for k,v in enumerate(fields)):
        raise RuntimeError("the raw22 explicit consumed-order schema is inconsistent")
    return dict(fields=fields, rates=[v.derivative(0) for v in fields],
                alpha=alpha, beta=beta, gamma=gamma, inverse=gi, Kbar=Kbar,
                K=K, wn=wn, chi=chi, gt=gt, A=A, Lambda=Lambda, det=det)


def adm_constraints(compact, omega):
    """Physical H/M from compact geometry; each row retains its terms.

    Kphys^i_j=Omega*Kbar^i_j+wn delta^i_j. Consequently the mixed trace term
    in Kphys^2-Kphys_ij Kphys^ij is 4 Omega wn Kbar (not 6).
    """
    gamma, gi = compact["gamma"], compact["inverse"]
    Kbar, K, wn = compact["Kbar"], compact["K"], compact["wn"]
    curv = curvature(gamma,(1,2,3))
    Gamma = curv["Gamma"]
    gradient2 = mp.fsum(gi[i][j].v*omega.derivative(i+1)*omega.derivative(j+1) for i in range(3) for j in range(3))
    laplace = mp.fsum(gi[i][j].v*(omega.derivative(i+1,j+1)
                   -mp.fsum(Gamma[k][i][j].v*omega.derivative(k+1) for k in range(3)))
                   for i in range(3) for j in range(3))
    squareK = mp.fsum(gi[i][k].v*gi[j][l].v*Kbar[i][j].v*Kbar[k][l].v
                     for i in range(3) for j in range(3) for k in range(3) for l in range(3))
    Hterms = [omega.v**2*curv["scalar"],omega.v**2*K.v*K.v,-omega.v**2*squareK,
              4*omega.v*laplace,4*omega.v*wn.v*K.v,-6*gradient2,6*wn.v**2]
    Kup = [[sum(gi[j][k]*Kbar[k][i] for k in range(3)) for i in range(3)] for j in range(3)]
    Mterms = []
    for i in range(3):
        divergence = [Kup[j][i].derivative(j+1) for j in range(3)]
        divergence += [Gamma[j][j][k].v*Kup[k][i].v-Gamma[k][j][i].v*Kup[j][k].v
                       for j in range(3) for k in range(3)]
        row = [omega.v*mp.fsum(divergence),-omega.v*K.derivative(i+1),
               -2*mp.fsum(omega.derivative(j+1)*Kup[j][i].v for j in range(3)),
               -2*wn.derivative(i+1)]
        Mterms.append(row)
    return [Hterms]+Mterms+[[mp.mpf(0)] for _ in range(4)]
