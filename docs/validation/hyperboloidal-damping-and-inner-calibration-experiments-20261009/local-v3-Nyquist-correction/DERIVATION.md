# Prescribed C0 kappa2 profile: local analytical gate

This scratch candidate changes only the existing second damping argument of the actual `ConformalRHS`. The first argument remains `kappa_input/alpha`; physical-P lapse stabilization, the private spatial-norm shift control, storage, derivative operators, ghosts, KO and algebraic projection are unchanged. There are no C1/covector-repair additions, double poles, Omega floors or Theta falloff conditions.

Let `kcrit=2S/a^2`, `d=kappa_input-kcrit`, and

```
kappa2 = ((kcrit-kappa_input)/kappa_input)*(1-Omega)
kappa_eff = kappa_input*(1+kappa2) = kcrit+d*Omega.
```

The shared device-compatible `damping_profile.hpp` is the authoritative candidate formula. Its factored form is exactly zero at Omega=1. For S=1,a=.5,kappa_input=10 it gives kappa2=-.2(1-Omega), kappa_eff=8+2Omega. For the tested a=.5,.75,1,2 and 0<=Omega<=1, -1<kappa2<=0. More generally this flat interval requires 0<kappa_eff<=kappa_input; kappa_input>=2S/a^2 suffices with this core-matched d. Other parameters have not been admitted by this gate.

## Actual nonlinear kernel identity

Write m=kappa_input*kappa2. Directly in the actual production conformal RHS parts,

```
Delta pole.P = Delta pole.Theta = -m*Theta
all other regular/pole entries are unchanged.
```

Thus at Omega>0, Delta P_t=Delta Theta_t=-m*Theta/Omega. Physical K=P+2Theta changes by -3m*Theta/Omega, and the unchanged tracefree/metric equations give Delta Kij_t=-(m*Theta/Omega)*gammaij. The additions vanish exactly for Theta=0, a condition already met in the Einstein sector; no assumption about its asymptotic order is used. The reference has Theta=0, so its RHS identity is preserved without a new reference subtraction. This is a lower-order change; the actual 360-case extracted principal system remains unchanged and complete.

## Physical eight-constraint correction, including coefficient derivatives

The independent ordering is H,Mi,Zi,Theta, where H/M are the physical ADM constraints and Zi is the physical spatial covector. Relative to the existing C0 subsidiary with constant kappa_input and kappa2=0:

```
Delta H_t = -4 K m Theta/Omega
Delta Theta_t = -m Theta/Omega
Delta Mi_t = 2 partial_i(m Theta/Omega)
           = 2m partial_i Theta/Omega
             +2[d Omega_i/Omega - m Omega_i/Omega^2] Theta
Delta Zi_t = 0.
```

The isolated derivative of kappa2 is the first value term, `+2d Omega_i Theta/Omega`. Because m=-d(1-Omega), the *sum* of both value terms is `+2d Omega_i Theta/Omega^2`. Dropping the isolated derivative is incorrect even though the complete expression has a simple alternative form. The coefficient-aware actual dual20 chain differentiates the analytic profile at every displaced coefficient point and agrees with these equations. Its negative control deliberately omits only d kappa2.

In the exact outer CMC reference,

```
Omega=(S^2-r^2)/(2aS), alpha=S/a-Omega,
alpha*w=-S/a^2+2Omega/a.
sigma_C0=(-2alpha*w-kappa_eff)/Omega = -4/a-d.
sigma_C1=(2alpha/a-kappa_eff)/Omega = -2/a-d.
```

Consequently `-2 partial_i(sigma Theta)` has no coefficient-gradient Theta/Omega^2 term there. This exact cancellation is a reference-background statement. On general live off-constraint data, alpha*w and the physical extrinsic curvature vary; this profile does not prove a nonlinear regularity closure or a constraint energy estimate. Other C0 geometric/Z couplings, transition coefficient gradients, numerical product-rule defects and global boundary effects remain.

## Flat damping theorem scope and a printed-prose caveat

The mapping rho=kappa2 follows from the physical damping coefficients in Eqs.6–8 of [Gundlach et al., gr-qc/0504114](https://arxiv.org/pdf/gr-qc/0504114). The *printed Eq.19 matrix* gives longitudinal determinant

```
(s^2+kappa(2+rho)s+w^2)(s^2+kappa s+w^2)-kappa^2 rho w^2.
```

Its Hurwitz determinant Delta3 is `2 kappa^4(3+rho)^2 w^2(1+rho)`, while its constant coefficient is `w^2(w^2-kappa^2 rho)`. Hence the all-nonzero-frequency flat Hurwitz interval is **-1<rho<=0**, stricter than the paper's subsequent prose rho>-1: positive rho has a low-frequency positive root. Our tested profile satisfies the stricter interval. This result assumes constant parameters and a frozen flat inertial background; it supplies no stability theorem for the variable hyperboloidal C0 system. The independent literature audit is pinned by index SHA 3a2b38c659820243b4d850afeff0d819e4c946f01e25cf22ae6d8fb5ababd5d6.
