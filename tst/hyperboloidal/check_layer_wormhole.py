#!/usr/bin/env python3
"""Independent exact vacuum constraints for mass-corrected layer wormhole data.

This proves geometric initial data, not gauge stationarity or evolution stability.
Conventions are K_ij=Lie_beta(gamma_ij)/(2*alpha) for a stationary height metric.
The Minkowski reference and fixed Omega are untouched.
"""

import sympy as sp


R, mass, a = sp.symbols('R mass a', positive=True)
v = sp.Function('v')(R)
psi = 1 + mass / (2 * R)
N = (1 - mass / (2 * R)) / psi
B = psi**4
d = 1 - v**2
E = B * d
rho = psi**2 * R
alpha_geometric = N / sp.sqrt(d)
normal_shift = -v / (psi**2 * sp.sqrt(d))
beta = alpha_geometric * normal_shift
kr = -(sp.diff(v, R) / d**sp.Rational(3, 2)
       + v * sp.diff(N, R) / (N * sp.sqrt(d))) / psi**2
kt = normal_shift * sp.diff(rho, R) / rho
kr_stationary = (sp.diff(beta, R) + beta * sp.diff(E, R) / (2 * E)) / alpha_geometric
assert sp.simplify(kr_stationary - kr) == 0
# Scalar curvature of E*dR^2+rho^2*dOmega_sphere^2, independently of Z4 jets.
R3 = (2 * (1 - sp.diff(rho, R)**2 / E) / rho**2
      - 4 * (sp.diff(rho, R, 2) / E
             - sp.diff(rho, R) * sp.diff(E, R) / (2 * E**2)) / rho)
hamiltonian = R3 + (kr + 2 * kt)**2 - kr**2 - 2 * kt**2
momentum_R = -2 * sp.diff(kt, R) + 2 * sp.diff(rho, R) * (kr - kt) / rho
assert sp.factor(hamiltonian) == 0
assert sp.factor(momentum_R) == 0
mass_invariant = rho * (1 - sp.diff(rho, R)**2 / E + rho**2 * kt**2) / 2
assert sp.factor(mass_invariant - mass) == 0
print('PASS exact Hamiltonian and momentum constraints for arbitrary smooth boost')

# The mass-dependent height includes the outgoing Schwarzschild optical factor.
v_outer = R / sp.sqrt(R**2 + a**2)
height_prime = psi**2 * v_outer / N
assert sp.simplify(B - N**2 * height_prime**2 - B * a**2 / (R**2 + a**2)) == 0
z = sp.symbols('z', positive=True)
outer_series = sp.series(height_prime.subs(R, 1 / z), z, 0, 2).removeO()
naive_defect = sp.series((B - N**2 * v_outer**2).subs(R, 1 / z), z, 0, 2).removeO()
assert sp.expand(outer_series - 1 - 2 * mass * z) == 0
assert sp.expand(naive_defect - 4 * mass * z) == 0
print('PASS regular outer height h_prime=1+2M/R+O(R^-2); naive height is singular')

# The implemented compact-coordinate shear avoids dividing by Omega.
r, omega, ell, w, wr = sp.symbols('r omega ell w wr', positive=True)
m = mass * omega / (2 * r)
psi_compact = 1 + m
N_compact = (1 - m) / psi_compact
eta = r * omega * wr / ell
kr_compact = -(w + eta + 2 * w * m / (1 - m**2)) / (a * psi_compact**2)
kt_compact = -w * N_compact / (a * psi_compact**2)
shear = -(r * wr / ell + w * mass * (2 - m) / (r * (1 - m**2))) / (a * psi_compact**2)
assert sp.factor(kr_compact - kt_compact - omega * shear) == 0
assert sp.factor(shear.subs({omega: 0, w: 1, wr: 0}) + 2 * mass / (a * r)) == 0
print('PASS factored shear identity and finite nonzero mass shear at scri')
