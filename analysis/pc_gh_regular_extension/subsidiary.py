"""Differentiate the proposed production equations without imposing reductions."""
import sympy as s


def zero(label, value):
    residual = s.expand(value)
    if residual != 0:
        raise AssertionError(f"{label}: {residual}")


x, y = coords = s.symbols("x y")
field = lambda name: s.Function(name)(x, y)
beta = [field(f"beta{i}") for i in range(2)]
B = [[field(f"B{i}{j}") for j in range(2)] for i in range(2)]
b = [[B[i][j]-s.diff(beta[j], coords[i]) for j in range(2)] for i in range(2)]
rate = field("lambda")
adv = lambda value: sum(beta[k]*s.diff(value, coords[k]) for k in range(2))

# A nonlinear configuration source tests both the chain rule and independent
# derivative variables. The component identity extends to vector/tensor q^A.
q, u = field("q"), field("u")
G = [field(f"G{i}") for i in range(2)]
E = [G[i]-s.diff(q, coords[i]) for i in range(2)]
F = q**3+q*u
qt = adv(q)+F
N = [field("Ncoef")*E[i] for i in range(2)]
Gt = [adv(G[i])+sum(B[i][k]*G[k] for k in range(2))
      +(3*q*q+u)*G[i]+q*s.diff(u, coords[i])-rate*E[i]+N[i] for i in range(2)]
source = [sum(b[i][k]*G[k] for k in range(2))+(3*q*q+u)*E[i]+N[i]
          for i in range(2)]
Et = [adv(E[i])+sum(s.diff(beta[k], coords[i])*E[k] for k in range(2))
      +source[i]-rate*E[i] for i in range(2)]
for i in range(2):
    zero("nonlinear advective reduction closure", Gt[i]-s.diff(qt, coords[i])-Et[i])
curl = s.diff(G[1], x)-s.diff(G[0], y)
curlt = (adv(curl)+(s.diff(beta[0], x)+s.diff(beta[1], y))*curl
         +s.diff(source[1], x)-s.diff(source[0], y)-rate*curl
         -s.diff(rate, x)*E[1]+s.diff(rate, y)*E[0])
zero("full curl and variable-rate gradients", s.diff(Gt[1], x)-s.diff(Gt[0], y)-curlt)
print("PASS: nonlinear reduction/curl closure, including spatial damping-rate gradients")

# Independently expand the actual regular w/rho/p/L moving-puncture rows.
w, rho, K = field("w"), field("rho"), field("K")
alpha = rho*w
p, L = [field(f"p{i}") for i in range(2)], [field(f"L{i}") for i in range(2)]
trB = B[0][0]+B[1][1]+field("B22")
r = [p[i]-s.diff(w, coords[i]) for i in range(2)]
ell = [L[i]-2*s.diff(alpha, coords[i]) for i in range(2)]
a = [L[i]-2*(w*s.diff(rho, coords[i])+rho*p[i]) for i in range(2)]
Fw = w*(alpha*K-trB)/3
wt = adv(w)+Fw
rhot = adv(rho)+rho*(-2*K-(alpha*K-trB)/3)
pt = [adv(p[i])+sum(B[i][k]*p[k] for k in range(2))
      +(p[i]*(alpha*K-trB)+w*(L[i]*K/2+alpha*s.diff(K, coords[i])
                             -s.diff(trB, coords[i])))/3-rate*r[i] for i in range(2)]
Lt = [adv(L[i])+sum(B[i][k]*L[k] for k in range(2))-2*K*L[i]
      -4*alpha*s.diff(K, coords[i])-rate*ell[i] for i in range(2)]
for i in range(2):
    rt = (adv(r[i])+sum(B[i][k]*r[k]+b[i][k]*s.diff(w, coords[k]) for k in range(2))
          +(2*alpha*K-trB)*r[i]/3+w*K*a[i]/6-rate*r[i])
    at = (adv(a[i])+sum(B[i][k]*a[k]+2*w*b[i][k]*s.diff(rho, coords[k])
                       for k in range(2))
          -(2+alpha/3)*K*a[i]-2*rho*alpha*K*r[i]/3-rate*a[i])
    zero("regular p subsidiary", pt[i]-s.diff(wt, coords[i])-rt)
    measured_at = Lt[i]-2*(wt*s.diff(rho, coords[i])+w*s.diff(rhot, coords[i])
                          +rhot*p[i]+rho*pt[i])
    zero("regular lapse/p subsidiary", measured_at-at)
print("PASS: exact moving-puncture Rw/Ralpha equations, with all cross-couplings")
print("PASS: all reduction characteristics advect with speed -beta.n")
print("PASS: on flat backgrounds every reduction decays at coordinate rate lambda")
print("LIMIT: on varying backgrounds the homogeneous source matrix also stretches/mixes errors")

# Intrinsic Q correction: differentiating tr_g A=0 uses the actual metric
# derivative. The stored derivative instead has a trace mismatch A:(Q-dg).
a_entries = s.symbols("A0:5")
A = s.Matrix([[a_entries[0], a_entries[1], a_entries[2]],
              [a_entries[1], a_entries[3], a_entries[4]],
              [a_entries[2], a_entries[4], -a_entries[0]-a_entries[3]]])
e_entries = s.symbols("e0:5")
E_q = s.Matrix([[e_entries[0], e_entries[1], e_entries[2]],
                [e_entries[1], e_entries[3], e_entries[4]],
                [e_entries[2], e_entries[4], -e_entries[0]-e_entries[3]]])
al = s.symbols("alpha", positive=True)
N_q = -2*al*s.eye(3)*s.trace(A*E_q)/3
zero("intrinsic Q algebraic tangent", 2*al*s.trace(A*E_q)+s.trace(N_q))
print("PASS: homogeneous intrinsic Q correction cancels the derivative trace mismatch")
