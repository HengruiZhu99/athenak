"""HELD 283 fixed analytic controls; no finite differences or native imports."""
import math
import mpmath as mp
from taylor3 import Jet,indices,factorial,inverse
from reference3 import radial_coefficients
from gaussian3 import implicit_jet,wave,fderivative
from oracle import construct
from diagnostics import zero_row,comparison


def unit_checks(layer,settings):
    checks = []
    x = [Jet.variable(0,k) for k in range(4)]
    linear = sum((k+1)*x[k] for k in range(4))
    polynomial = (2+linear)**3
    reciprocal = 1/(2+x[0]-2*x[1]+3*x[2]-4*x[3])
    exponential = (2+linear).exp()
    for m in indices(3):
        d = sum(m)
        factor = math.prod((k+1)**m[k] for k in range(4))
        expected = mp.mpf(math.factorial(3))/math.factorial(3-d)*2**(3-d)*factor
        checks.append(comparison("unit_polynomial/"+str(m),polynomial.c[m]*factorial(m),expected,True,"exact_coefficients"))
        expected = mp.mpf((-1)**d*math.factorial(d))*math.prod([1,-2,3,-4][k]**m[k] for k in range(4))/2**(d+1)
        checks.append(comparison("unit_reciprocal/"+str(m),reciprocal.c[m]*factorial(m),expected,True,"closed_rational_derivative"))
        checks.append(comparison("unit_exp/"+str(m),exponential.c[m]*factorial(m),mp.exp(2)*factor,True,"closed_exp_derivative"))
    target = 2+x[0]+x[1]*x[2]+x[3]**3
    result,_ = implicit_jet(mp.mpf(2),lambda c:c+c*c-target-target*target,mp.mpf(5))
    explicit = {(0,0,0,0):mp.mpf(2),(1,0,0,0):mp.mpf(1),
                (0,1,1,0):mp.mpf(1),(0,0,0,3):mp.mpf(6)}
    for m in indices(3):
        checks.append(comparison("unit_implicit/"+str(m),result.c[m]*factorial(m),explicit.get(m,mp.mpf(0)),True,"nonlinear_implicit_polynomial"))
    E = [[Jet(int(i==j),1) for j in range(4)] for i in range(4)]
    for i in range(4):
        E[i][(i+1)%4] += Jet.variable(0,i,1)
    Ei = inverse(E)
    for i in range(4):
        for j in range(4):
            residual = sum(E[i][k]*Ei[k][j] for k in range(4))-int(i==j)
            for m in indices(1):
                checks.append(zero_row("unit_inverse/%d%d/%s"%(i,j,m),[residual.c[m]*factorial(m)],True,"matrix_chain"))
    origin = [Jet.variable(mp.mpf(".1"),0)]+[Jet.variable(0,k) for k in (1,2,3)]
    F = wave(origin[0],origin[1:],mp.mpf(".35"))
    checks.append(comparison("unit_origin_xy",F.derivative(1,2),-mp.mpf(2)/15*fderivative(origin[0].v,mp.mpf(".35"),5),True,"Cartesian_origin"))
    checks.append(comparison("unit_origin_txy",F.derivative(0,1,2),-mp.mpf(2)/15*fderivative(origin[0].v,mp.mpf(".35"),6),True,"Cartesian_origin"))
    checks.append(zero_row("unit_origin_wave",[-F.derivative(0,0)]+[F.derivative(k,k) for k in (1,2,3)],True,"Cartesian_origin"))
    for n,event in enumerate(((".2",".3",".1","-.2"),("1","2","-1",".5"),("0",".025",".01",".02"))):
        inertial = [Jet.variable(mp.mpf(v),k,2) for k,v in enumerate(event)]
        F = wave(inertial[0],inertial[1:],mp.mpf(".35"))
        checks.append(zero_row("unit_scalar_wave/%d"%n,[-F.derivative(0,0)]+[F.derivative(k,k) for k in (1,2,3)],True,"nonzero_inertial"))
    for label,r in (("inner",layer.r0),("outer",layer.r1)):
        coeff = radial_coefficients(r,layer.a,layer.r0,layer.r1)
        for field in ("w","wc"):
            for d in (1,2,3):
                checks.append(zero_row("unit_cutoff/%s/%s/%d"%(label,field,d),[coeff[field].derivative(*([0]*d))],True,"exact_endpoint"))
    core = construct([mp.mpf(".1"),mp.mpf(".025"),mp.mpf(0),mp.mpf(0)],layer,mp.mpf(".35"),mp.mpf(0),settings)
    expected = [1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
    for k,(field,rate) in enumerate(zip(core["compact"]["fields"],core["compact"]["rates"])):
        checks.append(comparison("unit_core_field/%d"%k,field.v,mp.mpf(expected[k]),True,"exact_core_harmonic"))
        checks.append(zero_row("unit_core_rate/%d"%k,[rate],True,"exact_core_harmonic"))
    radius = mp.mpf(".75")
    q = radius/mp.sqrt(2)
    negative = construct([mp.mpf(".1"),q,q,mp.mpf(0)],layer,mp.mpf(".5"),mp.mpf(".75"),settings)
    checks.append(comparison("unit_negative_refusal",mp.mpf(int(negative["refused"] and negative["J"]>0 and negative["D"]<0)),mp.mpf(1),True,"no_ADM_square_root"))
    if len(checks) != 283:
        raise RuntimeError("fixed analytic unit count drift")
    return checks
