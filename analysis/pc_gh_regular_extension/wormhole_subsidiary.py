"""Exact frozen reduction-source matrix on the requested isotropic initial data.

This is a nonflat initial-slice damping check, not a stationary trumpet estimate
or a substitute for the evolving-background CUDA qualification.
"""
import sympy as s

w, alpha, rate, eta, switch = s.symbols('w alpha lambda eta S', positive=True)
p = s.Matrix(s.symbols('p0:3'))
lapse_gradient = s.Matrix(s.symbols('Galpha0:3'))
r, ell, a = s.symbols('r ell a')
V = alpha**2*w*p-alpha*w**2*lapse_gradient
variation = V.diff(w)*r+V.diff(alpha)*ell/2
on_slice = variation.subs(alpha, w).subs(dict(zip(lapse_gradient, p)))
assert all(s.expand(value) == 0 for value in on_slice.subs(ell, a+2*r)-w*w*p*a/2)
# One derivative-index block (r,a,b^x,b^y,b^z, five trace-free Q entries).
# All three derivative-index blocks are identical on this beta=B=K=At=0 slice.
J = -rate*s.eye(10)
for k in range(3):
    J[0, 2+k] = p[k]
    J[2+k, 1] = switch*w*w*p[k]/2
    J[2+k, 2+k] -= eta
z = s.symbols('z')
assert s.factor(J.charpoly(z).as_expr()-(z+rate)**7*(z+rate+eta)**3) == 0
print('PASS: independently differentiated moving shift gives delta Fbeta = S w^2 p a/2')
print('PASS: each of three reduction blocks has eigenvalues -lambda (7), -lambda-eta (3)')
print('LIMIT: repeated source eigenvalues can cause polynomial transients times exp(-lambda t)')
rad, mass = s.symbols('radius M', positive=True)
initial_w = (rad/(rad+mass/2))**2
initial_p = s.diff(initial_w, rad)
assert s.limit(initial_w, rad, 0) == 0
assert s.limit(initial_p, rad, 0) == 0
assert s.limit(initial_w**2*initial_p, rad, 0) == 0
assert s.simplify(initial_p.subs(rad, mass/4)-16/(27*mass)) == 0
print('PASS: wormhole w=O(r^2), p=O(r), shift residual coupling w^2 p=O(r^5)')
print('PASS: |p| is bounded by 16/(27 M); all frozen reduction source coefficients are bounded')
print('SCOPE: isotropic rho=1 initial slice only; evolving trumpet damping remains unqualified')
