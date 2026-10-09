# Held radial projection/SAT constraint readback

This is a source-only plan. No new kernel query, compiler invocation, eigenvalue
or propagation is admitted until root reviews the driver and supplies its exact
source authorization. Original API, sources, executables, matrices and frozen
constraint-rate evidence remain unchanged.

Use only the existing J0/N8/rb=.98 operator
`boundary/total-j-finite-rb-control-20261009/J0-N8-rb.98-segmentedQ64-a12x24-refinement001/operator.npz`,
SHA `2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742`.
Its separate real 64x64 arrays are Jbulk=E^-1 Kweak and Jsat=E^-1 SATload.
DOFs are channel-major modal degree 0..7, with physical channel order
alpha,metric_trace,P,Theta_physical,beta,Lambda,metric_STF,independent_A.
No configuration part of the Riesz lift is omitted.

For channel orbital L, B=rb², the unchanged envelope modes are

```
v_k(rho) = sqrt(2(2k+L+3/2)/B^(L+3/2))
           P_k^(0,L+1/2)(2rho/B-1).
```

Use the analytic rho derivatives in the frozen assembly source. The solid basis
already carries r^L; no additional r^L is applied. Common nodes are the eight
roots_jacobi(0,.5) mapped to [0,B]. The stored nodal_from_modal map T is checked
against the analytic modes before source queries. Each initial witness is
interpolated once, X=solve(T,W_at_nodes), then held fixed while differentiating.
Continuum source and semidiscrete source use this SAME polynomial interpolant.
Scalar envelope interpolation errors versus the analytic original are recorded
separately; the original analytic source is not substituted into the projection
defect comparator.

The 14 minimal witnesses are five polynomial cases (alpha W=1, P W=rho,
Theta W=rho², beta W=rho³, and the fixed all-channel alternating W=1+rho/3−rho²/5+rho³/7
mixture), two gauge-only exp(-8rho) interpolants (alpha or beta), six individual
nongauge shell interpolants and their fixed alternating mixture. Shell envelopes
are exp(-((rho-.49)/.16)²); channels are metric trace,P,Theta,Lambda,metric STF,A.
No source/field coefficient is Euclidean-normalized or tuned.

Use the same 21 transition/collar centers, directions and five h values as the
completed rb=.98 continuum-rate gate. Every complete Cartesian stencil remains
inside rb. The existing --source-batch rejects the origin because it also forms
a radial frame; no origin query is attempted. The separately frozen exact core
oracle remains separate.

To avoid repeated queries, build one shared point cache for every stencil and
center. At each actual Cartesian point and channel, query the three independent
WJets (1,0,0),(0,1,0),(0,0,1). The unchanged source-batch schema has 150 outputs:
actual raw22 RHS [0:22], configuration source derivative [22:33], complete raw22
input values [33:55], configuration input derivative [55:66], normalized fields
[66:116], normalized sources[116:146], and four algebraic normals [146:150].
Contract these complete pointwise maps with analytic polynomial W/Wrho/Wrhorho;
the maps are freshly evaluated at every Cartesian point, not spatially frozen.
Reference/angular coefficient variation therefore remains in the subsequent FD.
Gate this linear mapping against direct arbitrary-WJet queries at every center,
all channels, using three fixed jet choices, at the frozen 5e-11 scaled threshold.
No field/source formula is replaced by a guessed principal symbol.
The first two algebraic normals use max(1,raw-input22 norm), while the last two
use max(1,actual-RHS22 norm); separate absolute/scaled maxima are preserved. The
saved NPZ is loaded with allow_pickle=False, and every relevant real64x64 array
is explicitly checked finite before arithmetic.

For each X, form Ybulk=Jbulk X, Ysat=Jsat X and Ytotal=Ybulk+Ysat. From the complete
raw value/source functions form five independent Cartesian FD jets, then apply
the unchanged --constraint-rate-batch physical8 functional:

```
continuum = DC_ref[L_actual Phi(X)]
bulk      = DC_ref[Phi(Ybulk)]
SAT       = DC_ref[Phi(Ysat)]
total     = DC_ref[Phi(Ytotal)]
initial   = DC_ref[Phi(X)]
bulk projection defect = bulk - continuum
total defect           = total - continuum.
```

All source/field jets are recovered from complete analytic-jet function values
using the frozen fourth-order Cartesian stencils. No third/fourth analytic
reference jets are invented. Record every five-h sequence/increment/order and
extrapolation, including projection defects. Resolve each sequence with the
same 2e-7 final increment/convergence-or-unclassified rule; stop and preserve an
unresolved case. Check total=bulk+SAT numerically. Initial gauge constraints and
continuum gauge rates retain the exact-zero gates. Matrix/source binding and
interpolation are checked before interpreting defects.

Bulk defect and SAT contribution are measured separately and are NOT required
to vanish. This is not CPBC. Component sample RMS/peaks and absolute/scaled values
remain in physical Cartesian H/M/Z/Theta order, without Omega rescaling. Boundary
owns integrated Penrose/physical covector contractions and radial refinement.
No N12/N16, rb=.995, J>=1, spectrum, propagation or native action is included.
