# Actual 3D radiation-v2 complete-step response

This bounded 24-vector Arnoldi search finds rapidly amplifying directions concentrated on physical outer faces. It does **not** converge an eigenpair, so its Ritz growth rates are not measured eigenvalues. This is a negative stability result for the experimental closure; no production or queue changes were made.

The immutable v2 hook executable, input and seed provenance are in `manifest.json`. The map is the entire RK3 evolution, including projection, recasting and boundary updates, on one 16³ block in [-2,2]³ with dx=.25M, dt=.0375M, G=2, linear ghost extrapolation, and the opt-in physical-constraint radiation boundary. Other sources use original/default settings (not the covariant-source prototype). The direction seed combines the finite old G2 t=500.025M state with reproducible relative .001 noise. It is rescaled about exact background for each response; this is not a late-time checkpoint evolution.

## Map validity

- Zero residual stays bitwise zero through 20 complete steps.
- A perturbed 1+1-step composition equals the uninterrupted 2-step map bitwise.
- Central responses at peak amplitudes 1e-4 and 3e-5 agree to 1.08e-9 in active relative L2 for the leading candidate (8.35e-9 including ghosts).

## Leading candidate: actual amplification, not a converged eigenvalue

The 24-vector projection gives μ(.3M)=1.448546 and γ=1.2352/M, but direct eigen residuals are .1134 globally and .06453 on active cells. Independent one-step response has active Rayleigh γ=1.2661/M and shape residual .01232. These rates remain candidate diagnostics only. The direct 0.3M **active norm gain is 1.46338**. Other saved candidates are less converged. No further Arnoldi solve was run.

For the leading candidate:

- 97.78% of active unweighted state norm squared lies on outer face cells; .088% on edges and .0018% on corners. 99.849% of Theta norm squared lies on faces. These field norms are diagnostic Euclidean norms, not a continuum symmetrizer energy.
- Active Theta peaks at (-.125,-.125,1.875)M, r≈1.8833M, gid0/rank0/level0. This is the mode-direction maximum, not a claim about the earliest injection point.
- Theta carries46.03% of active field norm squared; diagonal A carries39.14%; off-diagonal A8.11%.
- Using the actual combined-face normal, boundary A Frobenius norm is70.88% scalar,29.08% vector and.0446% transverse traceless tensor. Boundary Gamma is98.85% normal; beta99.70% normal. The fast direction is predominantly scalar/longitudinal constraint content with a substantial A vector component.
- Independently evaluated **physical** Hamiltonian and momentum perturbations, after subtracting background by central ± state probes, grow by1.46471 and1.45952 over0.3M. The volume-D6 Q perturbation grows1.45710. Their face fractions are98.998%,98.248%,90.596%. This is not merely a Theta-only perturbation. Q here uses the actual volume D6 metric connection, distinct from the helper's D4 functional.

`leading-candidate-localization.png` shows the midplane, outer face, and normal profiles. `validation.json` and `physical-constraints.json` retain all numerical details. The two other candidates have larger direct eigen residuals and must not be interpreted as independent measured eigenmodes.

The result rules out a purely internal-MPI-interface or geometric corner explanation: it occurs in a single block and is dominated by face interiors. It does not identify the exact mismatched stencil term or prove a continuum instability. A boundary differential compatibility repair remains necessary; zero preservation and pointwise manufactured residual accuracy alone are insufficient.
