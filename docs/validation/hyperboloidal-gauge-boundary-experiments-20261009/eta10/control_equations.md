# Scratch outer-shift restoring control

Use production physical-P lapse with preferred source off and append
`S_beta^i = -eta_p W (beta^i-beta_ref^i)` to the shift pole, assembling the
new pole explicitly. Eta_p=10 is the weaker tested rate that removes all
sampled outer frozen Fourier growing roots at kappa1=10. The existing regular
eta damping, Gamma driver, live speed bound and core W=0 remain unchanged.

This is a gauge choice. It is zero exactly on the reference and depends only
on field values and prescribed W, so the complete principal symbol is retained.
The actual full20 kernel audit confirms its change is precisely the beta
block diagonal `-eta_p W/Omega`, independent of wave number. Full20 leading
poles at all four radii have 15 negative and five semisimple zero roots.
The kappa5 family still has growing derivative-coupled roots even at eta_p=20.
At kappa10, eta_p=10 and 20 remove all sampled outer growth through k=256;
the wide-reference r=.75 k=0 frozen geometric mode still grows at .485/.479.
No global PDE spectrum or finite native stability follows from these gates.

The GH-source modification is exact off constraint:
`Delta F^0=0`,
`Delta F^i=eta_p W (beta^i-beta_ref^i)/(alpha^2 Omega)`,
`Delta Box(Omega)=-eta_p W (beta-beta_ref).dOmega/(alpha^2 Omega)`.
The temporal and spatial Z contributions in Gamma+2Z cancel from this
comparison because the live state/geometric RHS are held fixed. This control
does not impose the preferred Box(Omega) source. Arbitrary beta deviations
can make the added source singular; retaining a preferred O(Omega) Box source
would require stronger asymptotic matching, which is not assigned here.
The finite-Q counterexample still survives when beta=beta_ref.

Nothing here changes tracked production files. The private native wrapper
uses eta_p=10 and all same wide-reference pulse parameters as the baseline.

The eta10-alone actual native N24/RK3 run reached t=2 with all 81 saved active-array snapshots finite and positive alpha/chi/SPD physical metric. Final H/M/Z=1.266556/1.636471/.375338; relative to the matching wide kappa10 source-off production control these are 1.05595/.82995/.84629. The growing norms and worse H reject this as demonstrated finite-pulse stabilization. It also has a separate required later-BH null-tangency gate: a beta-only restoring pole generally changes Nraw and the Box source, even on null initial data. A tentative detached M=.5, a=.5 geometric-gauge first-jet calculation gives reference/base null rate +24 and eta10 rate -56; independent BH compatible initial gauge jets are still being checked. No BH run or nonlinear compatibility claim follows.
