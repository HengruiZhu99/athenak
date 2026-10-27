# Geometric coordinate-sphere mass diagnostic

`surface_mass.hpp` supplies a local, Kokkos-compatible physical surface-expansion
kernel and a separate integral-to-Hawking-mass function. It does not modify
evolution, elliptic data or any constraint. Surface quadrature and interpolation
errors remain the caller's responsibility. This component test is not yet a
coupled mass comparison or a late-time Schwarzschild validation.

For a coordinate sphere centered at a chosen position, let rho=|x−center| and
n_i=(x−center)_i/rho. All field components and derivatives remain global
Cartesian; centering the diagnostic does not transform or reset the fields.
The physical ADM metric is g_phys=g_tilde/(chi Omega²). Write

```
v^i = g_tilde^{ij} n_j,
q = n_i v^i,
P^{ij} = g_tilde^{ij} − v^i v^j/q,
C = P^{ij} (partial_i n_j − Gamma_tilde^k_ij n_k).
```

The physical mean curvature of this surface is

```
k = sqrt(chi/q) [Omega(C − v^i partial_i chi/chi)
                − 2 v^i partial_i Omega].
```

This includes the full metric connection and the conformal-gradient terms.
The expanded expression avoids a standalone1/Omega connection term. With the
exact curvature normalization used by `ToPhysicalADM`, and without projecting
the trace of A or identifying the independent Z4 connection,

```
p = K_phys_ij s^i s^j − K_phys
  = Omega [A_ij v^i v^j/q − g_tilde^{ij} A_ij]
    − 2(u.trace + 2Theta)/3,
theta_plus = p+k,   theta_minus = p−k,
dA/d(solid angle) = rho² sqrt(det(g_tilde) q)/(chi Omega²).
```

The null normals are n_time±s, with inner product−2. For a sphere of areaA,

```
R_A = sqrt(A/(4pi)),
M_H = R_A/2 [1 + integral(theta_plus theta_minus dA)/(16pi)].
```

This follows the Hawking surface functional, with its null-normal normalization
converted explicitly from the norm−1 convention in
[Hayward's primary paper, section2](https://arxiv.org/html/gr-qc/9303030v1).
It is not Hayward's additional shear/twist-corrected energy. In spherical
symmetry M_H agrees with Misner–Sharp mass. Generic nonspherical fields retain
the surface functional interpretation; a coordinate sphere can be a nonround
physical surface even in flat space, so its Hawking mass need not vanish.
No generic3D field is labeled as having a Misner–Sharp mass.

Omega must be strictly positive for this finite-area diagnostic. The puncture
position and actual Omega=0 boundary are rejected rather than floored. A Bondi
limit requires a separate regularized extraction analysis. Near large physical
radius, cancellation in the surface integral can lose relative mass accuracy;
negative roundoff results are not clipped or hidden.

## Independent kernel checks

`test_surface_mass.cpp` checks36 exact spherical cases at six radii and three
directions: hyperboloidal Minkowski mass0 and time-symmetric isotropic
Schwarzschild mass1/3. The latter is only a normalization unit test, not a
replacement for the selected hyperboloidal wormhole benchmark. Forty further
nonflat/off-diagonal metric cases retain nontracefree A and independent Theta;
the compact formulas agree with an independently constructed full physical ADM
metric, connection and curvature. Maximum mass error is1.47251e-14, and the
independent ADM error is1.77636e-15. Endpoint/invalid-area checks reject.
The process takes.52s on Serial CPU. No GPU run is claimed.

Evidence: `evidence/surface-mass-kernel-20261027`. Reproduce:

```sh
cmake -S . -B build-shift-3d
cmake --build build-shift-3d --target elliptic_shift_3d_surface_mass_tests -j2
ctest --test-dir build-shift-3d -R surface_mass -V
```
