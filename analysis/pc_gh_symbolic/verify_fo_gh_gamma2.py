#!/usr/bin/env python3
"""Audit the full FO-GH gamma2 increment and reject an invalid production pullback.

The inverse-map oracle predates this audit. It receives changed Pi AND Phi, then
reconstructs the regular variables; it does not use the proposed increment formulas.
PASS means the identities/counterexamples were reproduced, not puncture qualification.
"""
import sympy as s

from verify_fo_gh_map import pc_to_fo, fo_to_pc


def zero(name, expr):
    entries = list(expr) if isinstance(expr, s.MatrixBase) else [expr]
    for value in entries:
        residual = s.factor(s.cancel(value))
        if residual != 0:
            raise AssertionError(f"{name}: {residual}")


def contracted(metric, q):
    inv = metric.inv()
    return s.Matrix([sum(inv[a, d]*inv[b, c]
        *(q[b][d, c] + q[c][d, b] - q[d][b, c])/2
        for b in range(3) for c in range(3) for d in range(3))
        for a in range(3)])


def pullback():
    w, rho, rate, eps = s.symbols("w rho lambda eps", positive=True)
    ap, aq, al, ab = s.symbols("amp_p amp_Q amp_L amp_B")
    alpha = rho*w
    # Use an exactly unimodular non-diagonal SPD metric.
    lower = s.Matrix([[1, 0, 0], [s.Rational(1, 2), 1, 0],
                      [0, s.Rational(1, 3), 1]])
    g = lower*lower.T
    gu = g.inv()
    beta = s.Matrix([s.Rational(1, 4), -s.Rational(1, 3), s.Rational(2, 5)])
    rw = ap*s.Matrix([1, -2, 3])
    ra = al*s.Matrix([2, 1, -1])
    rl = ra + 2*rho*rw
    rb = ab*s.Matrix([[1, 2, -1], [-3, 2, 1], [2, -1, -2]])
    rq = []
    for i in range(3):
        seed = s.Matrix([[i+1, 2-i, -1], [2-i, -i, 3], [-1, 3, 2]])
        rq.append(aq*(seed - g*s.trace(gu*seed)/3))
    pc = {"alpha": alpha, "chi": w*w, "gt": g, "beta": beta,
          "Q": [s.zeros(3) for _ in range(3)], "X": s.zeros(3, 1),
          "Y": s.zeros(3, 1), "B": s.zeros(3), "K": s.Rational(2, 7),
          "at": s.zeros(3), "pi": s.Rational(-3, 11), "Lambda": s.zeros(3, 1)}
    gauge = {"hperp": s.Rational(1, 7), "hvec": s.Matrix([1, -2, 3])/13}
    fo = pc_to_fo(pc, gauge)
    # Phi's map is linear in X,Q,Y,B. Build R_Phi independently via that map.
    err = dict(pc, X=2*w*rw, Q=rq, Y=alpha*rl, B=rb)
    mapped = pc_to_fo(err, gauge)
    rphi = [mapped["phi"][i] - fo["phi"][i] for i in range(3)]
    changed = dict(fo)
    changed["phi"] = [fo["phi"][i] - eps*rate*rphi[i] for i in range(3)]
    changed["Pi"] = fo["Pi"] + eps*rate/alpha*sum(
        (beta[i]*rphi[i] for i in range(3)), s.zeros(4))
    inv = fo_to_pc(changed, gauge, alpha)
    d = {name: s.diff(inv[name], eps).applyfunc(s.cancel)
         if isinstance(inv[name], s.MatrixBase) else s.cancel(s.diff(inv[name], eps))
         for name in ("X", "Y", "B", "K", "at", "pi", "Lambda")}
    dq = [s.diff(q, eps).applyfunc(s.cancel) for q in inv["Q"]]
    gamma2 = rate/alpha
    expected_k = -gamma2*(3*beta.dot(rw)/w + s.trace(rb))
    core = gamma2*(sum((beta[i]*rq[i] for i in range(3)), s.zeros(3))/2
                   - (rb*g + g*rb.T)/2)
    expected_at = core - g*s.trace(gu*core)/3
    expected_c = expected_k - gamma2*beta.dot(rl)/(2*alpha)
    expected_z = (-rate*contracted(g, rq) + rate/(alpha**2*w**2)*rb.T*beta
                  + gamma2*gu*ra/2)
    zero("p increment", d["X"]/(2*w) + rate*rw)
    zero("L increment", d["Y"]/alpha + rate*rl)
    zero("B increment", d["B"] + rate*rb)
    for i in range(3):
        zero(f"Q[{i}] increment", dq[i] + rate*rq[i])
    zero("coupled K increment", d["K"] - expected_k)
    zero("coupled Atilde increment", d["at"] - expected_at)
    zero("coupled Cperp increment", d["K"] + d["pi"] - expected_c)
    zero("coupled Z increment", contracted(g, dq) - d["Lambda"] - expected_z)
    for name in ("gt", "beta", "chi"):
        zero(f"unchanged configuration {name}", s.diff(inv[name], eps))

    inverse_length = s.symbols("inverse_length", positive=True)
    dimension_map = {a: inverse_length*a for a in (ap, aq, al, ab, rate)}
    for expr in (expected_k, expected_at, expected_c, expected_z):
        zero("geometric-unit dimension L^-2",
             expr.subs(dimension_map, simultaneous=True) - inverse_length**2*expr)

    # A sufficient vanishing rate makes ALL gamma2 increments denominator free.
    # This does not repair the separately rejected zero-gamma2 baseline.
    f = s.symbols("f", nonnegative=True)
    for name, expr in {"K": expected_k, "At": expected_at,
                       "C": expected_c, "Z": expected_z}.items():
        for entry in list(expr) if isinstance(expr, s.MatrixBase) else [expr]:
            den = s.denom(s.cancel(entry.subs(rate, rho**2*w**4*f)))
            if den.has(w, rho):
                raise AssertionError(f"regularized {name} denominator: {den}")
    print("PASS: independent inverse FO-GH map verifies all coupled gamma2 increments")
    print("PASS: independent p/Q/L/B amplitudes and non-diagonal conformal metric")
    print("PASS: all primary damping increments have dimension inverse length squared")
    print("PASS: lambda=rho^2*w^4*f regularizes the increment, not the baseline")


def characteristics():
    alpha, beta, g2, c = s.symbols("alpha beta gamma2 c", positive=True)
    # One physical-unit normal. A is defined by u_t + A u_x = lower order.
    matrix = s.Matrix([[0, 0, 0, 0, 0],
                       [g2*beta, -beta, alpha, 0, 0],
                       [-alpha*g2, alpha, -beta, 0, 0],
                       [0, 0, 0, -beta, 0], [0, 0, 0, 0, -beta]])
    left = s.Matrix([[1, 0, 0, 0, 0], [-g2, 1, 1, 0, 0],
                     [-g2, 1, -1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    zero("characteristic fields", left*matrix
         - s.diag(0, -beta+alpha, -beta-alpha, -beta, -beta)*left)
    assert left.det() == -2
    sym = s.eye(5)
    sym[0, 0], sym[0, 1], sym[1, 0] = c+g2*g2, -g2, -g2
    zero("symmetrizer", sym*matrix - matrix.T*sym)
    assert [s.factor(sym[:i, :i].det()) for i in range(1, 6)] == [c+g2*g2, c,c,c,c]
    # The Pi partner is necessary for this specific positive energy.
    guessed = matrix.copy()
    guessed[1, 0] = 0
    assert s.simplify((sym*guessed - guessed.T*sym)[0, 1]) != 0
    print("PASS: exact FO-GH fields, speeds and positive symmetrizer on alpha>0")
    print("PASS: deleting the Pi partner breaks the FO-GH symmetrizer identity")


def subsidiary():
    x, y, t = s.symbols("x y t")
    coords = (x, y)
    psi = s.Function("psi")(x, y, t)
    pi = s.Function("Pi")(x, y, t)
    phi = [s.Function(f"Phi{i}")(x, y, t) for i in range(2)]
    alpha = s.Function("alpha")(x, y, t)
    beta = [s.Function(f"beta{i}")(x, y, t) for i in range(2)]
    rate = s.Function("lambda")(x, y, t)
    ar = [s.Function(f"aR{i}")(x, y, t) for i in range(2)]
    br = [[s.Function(f"bR{i}{j}")(x, y, t) for j in range(2)] for i in range(2)]
    r = [phi[i] - s.diff(psi, coords[i]) for i in range(2)]
    curl = [[s.diff(phi[j], coords[i])-s.diff(phi[i], coords[j])
             for j in range(2)] for i in range(2)]
    psit = -alpha*pi + sum(beta[k]*phi[k] for k in range(2))
    phit = [sum(beta[k]*s.diff(phi[i], coords[k]) for k in range(2))
            - alpha*s.diff(pi, coords[i]) - (s.diff(alpha, coords[i])+ar[i])*pi
            + sum((s.diff(beta[k], coords[i])+br[i][k])*phi[k] for k in range(2))
            - rate*r[i] for i in range(2)]
    source = [-ar[i]*pi + sum(br[i][k]*phi[k] for k in range(2)) for i in range(2)]
    rt = [sum(beta[k]*curl[k][i] for k in range(2))+source[i]-rate*r[i]
          for i in range(2)]
    for i in range(2):
        zero("full reduction subsidiary", phit[i]-s.diff(psit, coords[i])-rt[i])
    ct = (sum(beta[k]*s.diff(curl[0][1], coords[k])
              + s.diff(beta[k], x)*curl[k][1] - s.diff(beta[k], y)*curl[k][0]
              for k in range(2))
          + s.diff(source[1], x)-s.diff(source[0], y)-rate*curl[0][1]
          -s.diff(rate, x)*r[1]+s.diff(rate, y)*r[0])
    zero("full curl subsidiary", s.diff(phit[1], x)-s.diff(phit[0], y)-ct)
    # Nonlinear configuration maps carry Hessian terms in their reductions.
    # Check the general M=f'(psi) pullback on a non-affine scalar map.
    M = 3*psi**2
    transformed_curl = s.diff(M*r[1], x)-s.diff(M*r[0], y)
    zero("nonlinear curl pullback", transformed_curl-M*curl[0][1]
         -s.diff(M, x)*r[1]+s.diff(M, y)*r[0])
    zero("nonlinear reduction time pullback",
         (M*rt[0]+6*psi*psit*r[0])
         -(M*(phit[0]-s.diff(psit, x))+6*psi*psit*r[0]))

    # Ralpha is the code's diagnostic, whereas RL=L-2*d(alpha) is the
    # true lapse-gradient reduction. Damping must include their p coupling.
    rho = s.Function("rho")(x, y, t)
    rw = [s.Function(f"Rw{i}")(x, y, t) for i in range(2)]
    ra = [s.Function(f"Ra{i}")(x, y, t) for i in range(2)]
    rl = [ra[i]+2*rho*rw[i] for i in range(2)]
    for i in range(2):
        zero("diagnostic lapse reduction damping", -rate*rl[i]
             -2*rho*(-rate*rw[i])+rate*ra[i])
    zero("lapse curl product rule",
         s.diff(-rate*rl[1], x)-s.diff(-rate*rl[0], y)
         +rate*(s.diff(rl[1], x)-s.diff(rl[0], y))
         +s.diff(rate, x)*rl[1]-s.diff(rate, y)*rl[0])
    bn, bt, lam, wave = s.symbols("beta_n beta_t lambda wave", real=True)
    fourier = s.Matrix([[-lam, -s.I*wave*bt], [0, -lam+s.I*wave*bn]])
    # Compare algebraically: SymPy may factor I*(beta_n*wave + I*lambda).
    spectral = s.symbols("spectral")
    zero("flat reduction damping spectrum", fourier.charpoly(spectral).as_expr()
         -(spectral+lam)*(spectral+lam-s.I*wave*bn))
    print("PASS: exact reduction/curl subsidiary including spatial lambda gradients")
    print("PASS: nonlinear constraint pullback, lapse/p coupling, and flat damping spectra")
    print("INFO: gamma1=-1 longitudinal reduction speed is zero; curl speed is -beta.n")


def baseline_and_puncture():
    w, rho, beta, p, drho, L, K, B, driver = s.symbols(
        "w rho beta p drho L K B driver", positive=True)
    wt = beta*p + w*(rho*w*K-B)/3
    production = beta*drho + rho*(driver-(rho*w*K-B)/3)
    standard = (beta*L/2 + rho*w*driver - rho*wt)/w
    ra = L - 2*(w*drho + rho*p)
    zero("exact baseline discrepancy", standard-production-beta*ra/(2*w))
    r, amplitude, rate = s.symbols("r amplitude lambda", positive=True)
    q = s.Rational("0.091297265")
    # B trace error on a vanishing-lapse, zero-shift wormhole point.
    ksource = -rate*amplitude/r**2
    assert s.limit(ksource, r, 0, dir="+") == -s.oo
    # Stationary-trumpet beta~r, w~r, rho~r^q, independent bounded B error.
    zsource = rate*amplitude*r/(r**(2*q)*r**4)
    exponent = s.simplify(r*s.diff(zsource, r)/zsource)
    assert exponent == -3-2*q
    print("PASS: required zero-gamma2 rho correction = beta.Ralpha/(2*w)")
    print("EXPECTED REJECTION: constant lambda gives wormhole K source ~r^-2")
    print(f"EXPECTED REJECTION: constant lambda gives trumpet Z source ~r^{exponent}")
    print("NOT QUALIFIED: the production baseline is not an exact off-constraint FO-GH pullback")


def moving_gauge_map():
    # The metric-only-source invertibility hypothesis cannot be reused after
    # substituting the moving-puncture source: its lapse feedback cancels pi.
    C, K, alpha, chi = s.symbols("C K alpha chi", positive=True)
    pc = {"alpha": alpha, "chi": chi, "gt": s.eye(3),
          "beta": s.Matrix([1, -2, 3])/4,
          "Q": [s.zeros(3) for _ in range(3)], "X": s.Matrix([1, 2, -1])/7,
          "Y": s.Matrix([2, -1, 3])/11, "B": s.eye(3)/13,
          "K": K, "at": s.diag(1, -2, 1)/17, "pi": C-K,
          "Lambda": s.Matrix([1, -2, 1])/19}
    gauge = {"hperp": alpha*pc["pi"]+2*K,
             "hvec": (1-alpha**2*chi)*pc["Lambda"]
                      -(alpha**2*pc["X"]-chi*pc["Y"])/2 - 2*pc["beta"]}
    fo = pc_to_fo(pc, gauge)
    for name in ("g", "Pi"):
        zero("moving gauge null Cperp column", s.diff(fo[name], C))
    for phi in fo["phi"]:
        zero("moving gauge null Cperp column in Phi", s.diff(phi, C))
    print("PASS: the moving-gauge forward map has an identically zero Cperp column")
    print("EXPECTED REJECTION: no invertible 50-field FO-GH point map for this gauge substitution")


if __name__ == "__main__":
    pullback()
    characteristics()
    subsidiary()
    baseline_and_puncture()
    moving_gauge_map()
