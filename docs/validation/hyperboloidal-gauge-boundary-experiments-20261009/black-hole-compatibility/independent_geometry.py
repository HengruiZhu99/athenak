"""Independent Schwarzschild geometry for a detached boost and fixed Omega."""
import sympy as sp
r,M=sp.symbols('r M',positive=True)
o=sp.Function('Omega')(r);b=sp.Function('b')(r)
ell=o-r*sp.diff(o,r);a2=o*o+b*b
R=r/o;m=M/(2*R);psi=1+m;N=(1-m)/(1+m)
rho=psi**2*R
E=psi**4*ell**2/(o*o*a2)
kr=psi**-2*((-o*sp.diff(b,r)+b*sp.diff(o,r))/ell-2*b*m/(r*(1-m*m)))
kt=-psi**-2*b*N/r
mass=rho/2*(1-sp.diff(rho,r)**2/E+rho*rho*kt*kt)
assert sp.simplify(mass-M)==0
momentum=-2*sp.diff(kt,r)+2*sp.diff(rho,r)/rho*(kr-kt)
assert sp.simplify(momentum)==0
scalar=2*(1-sp.diff(rho,r)**2/E)/rho**2-4*(sp.diff(rho,r,2)/E-sp.diff(rho,r)*sp.diff(E,r)/(2*E*E))/rho
assert sp.simplify(scalar+4*kr*kt+2*kt*kt)==0
diff=(-sp.diff(b,r)+b/r)/ell
shear=psi**-2*(diff-b*M*(2-m)/(r*r*(1-m*m)))
assert sp.simplify((kr-kt)-o*shear)==0
assert sp.simplify(1-N*N-4*m/(1+m)**2)==0
# Derive gamma, lapse and shift from the four-metric t_static=t+h(R),
# h_R=psi^2*b/(N*sqrt(Omega^2+b^2)). Check the curvature again from the
# stationary ADM definition K_ij=(L_beta gamma)_ij/(2 alpha), rather than
# assuming the formulas above. The symbolic calculation cancels b/N factors;
# the exact Cauchy branch below handles their excluded zero points directly.
h2=psi**4*b*b/(N*N*a2)
assert sp.simplify((psi**4-N*N*h2)*sp.diff(R,r)**2-E)==0
ba=-o*b/(psi**2*ell)  # beta^r / alpha_physical
dlogbeta=sp.diff(N,r)/N-2*sp.diff(psi,r)/psi+sp.diff(b,r)/b+sp.diff(a2,r)/(2*a2)-sp.diff(ell,r)/ell
kr_from_ADM=ba*(sp.diff(E,r)/(2*E)+dlogbeta)
kt_from_ADM=ba*sp.diff(rho,r)/rho
assert sp.simplify(kr_from_ADM-kr)==0
assert sp.simplify(kt_from_ADM-kt)==0
beta2=N*N*psi**-4*b*b*a2/ell**2
assert sp.simplify(1/E-beta2/(N*N*a2/o**2)-o**4/(psi**4*ell**2))==0
# At the exact Cauchy/throat branch b=0 before evaluating any 1/N factor.
E0=psi**4*ell**2/o**4
scalar0=2*(1-sp.diff(rho,r)**2/E0)/rho**2-4*(sp.diff(rho,r,2)/E0-sp.diff(rho,r)*sp.diff(E0,r)/(2*E0*E0))/rho
assert sp.simplify(scalar0)==0
print('PASS detached boost: four-metric/ADM curvature/physical inverse, H/M/Mass/shear, and independent exact-Cauchy throat scalar curvature')
