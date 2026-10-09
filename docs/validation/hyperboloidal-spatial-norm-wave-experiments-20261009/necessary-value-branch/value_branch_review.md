# Independent read-only review of root's general-rho value branch

Reviewed `build-layer-research/spatial-norm-zerojet/proof.py`, SHA256
`fa80b680bee0f4a68127e80936ba8e98a8b3e6f1343d28213317de85b0344bff`.
No correction to its algebra, domain, monotonicity or stated limits is needed.
The rho1.5 special case is independently proved by polynomial elimination in
this directory's executed `norm_leading.py`.

For xi*a=1 and W=1, normalize x=alpha/alpha_ref>0,
y=beta_rad/alpha_ref and z=sqrt(G/Ghat)>0, and write d=1-1/rho. At scri,
Omega_r=-1/a and alpha_ref=S/a. The actual live shift numerator vanishes
exactly when y=-[d*z^2+1-d]. Since 1<=rho<=5/2 gives 0<=d<=3/5,
the bracket is strictly positive; this selects negative radial shift and
omega_n. Tangential shift components vanish separately by their pole.

Finite Q and Theta_phys=0 give P0=3*omega_n0. Substituting that equality
into the actual physical-P lapse numerator and dividing by -alpha_ref^2/a
gives 2*x*y+4*x^2-2. The null condition gives y=-x*z. Hence
x^2*(2-z)=1, which requires 0<z<2; positive lapse gives
x=1/sqrt(2-z). The remaining equation is

```text
f(z) = z / [sqrt(2-z)*(d*z^2+1-d)] = 1.
d(log f)/dz = [d*z^2*(3z-4)+(1-d)*(4-z)]
                 / [2z*(2-z)*(d*z^2+1-d)].
```

Every denominator factor is positive on 0<z<2. For 0<z<4/3,
the numerator decreases with d because its d derivative is
3z^2*(z-4/3)+z-4<0. The worst permitted case d=3/5 has five times
the numerator equal to 9z^3-12z^2-2z+8. On [0,1] its Bernstein
coefficients are 8,22/3,8/3,3, all positive. On [1,4/3], substituting
z=1+t/3 gives 3+t/3+5t^2/3+t^3/3>0. For 4/3<=z<2 both
original numerator terms are nonnegative and (1-d)*(4-z)>0.
Thus f is strictly increasing throughout the admitted domain.

Its limits are f(0+)=0 and f(2-)=infinity, since 1-d>=2/5 and
d*z^2+1-d stays finite and positive. Also f(1)=1. Therefore z=1 is
the sole root, followed by x=1,y=-1. This fixes G, alpha and beta's
boundary values on the specified regular vacuum/null branch. It fixes
neither chi nor gtilde separately, no conformal shear/Lambda limits,
and no spatial or time derivatives. Evolution preservation, off-constraint
finite-Q data and angular closure require separate analysis.
