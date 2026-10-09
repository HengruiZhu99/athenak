Source-only J0 N12/N16 analytic projected-point adapter
=====================================================

Status: HELD pending separate exact root authorization for each N. No kernel
query, compilation, eigenvalue, exponential, propagation or new continuum-rate
batch. Original N8 frozen evidence970a117d and both failed ordinary-FD attempts
remain byte-identical. The generic nongauge C_ref[L_actual Phi(X)] comparator
remains unresolved; this stage evaluates C_ref[Phi(Y)] only.

Inputs and normalization
------------------------

Shared `point_constraints.py` pins the21-center analytic physical8 maps44a6e015,
analytic receipt98f18f39 and original index970a117d. These maps are independent
of radial polynomial degree: they act on complete channel W,Wrho,Wrhorho jets.
They inherit actual C0 physicalP/spatial-norm reference S1/a.5, geometry .05-.95,
kinput10,k2=0,xi2. H,Mxyz,Zxyz,Theta_physical are Cartesian physical components,
without Omega rescaling or a moving frame. Angular solid fields already carry
r^L; no additional radial power is inserted.

N12 operator1420df92/report1ebc6c87 and N16 operator466c6386/reportf8247ff6 are
the exact primary degree controls under boundary/total-j-finite-rb-degree-control-
20261009/J0-N{N}-rb.98-segmentedQ64-a12x24-primary001. The report must declare
passed_single_quadrature_algebra=true,J0,N,rb=.98 and the matching operatorhash.
Do not follow later matrix files automatically. Pin the original witness source
abd42a99 as lineage, copying its14 exact witnesses in the adapter.

At each N use normalized Jacobi modes
sqrt(2(2k+L+1.5)/B^(L+1.5))*P_k^(0,L+.5)(2rho/B-1),B=.98^2,
with analytic rho derivatives. Common nodes are roots_jacobi(N,0,.5), mapped
to[0,B]. Channel-major layout alpha,metric_trace,P,Theta_phys,beta,Lambda,
metric_STF,independent_A has L=0,0,0,0,1,1,2,2.

Fixed comparisons
-----------------

Reuse the14 witnesses: constantalpha, rho P, rho^2 Theta, rho^3 beta;
alternating weighted all-channel cubic mixture; exp(-8rho) alpha/beta;
six separate nongauge Gaussian shells exp(-((rho-.49)/.16)^2); weighted nongauge
shell mixture. Weights are(-1)^c/(c+1). The initial X is obtained from each
N-specific nodal interpolation; Ybulk=Jbulk X,Ysat=Jsat X,Ytotal=Ybulk+Ysat.
Keep X fixed when applying the analytic maps. This compares the actual polynomial
interpolant, not an analytic-original nonpolynomial forcing family.

At the original21centers record C_ref Phi(X), C_ref Phi(Ybulk),
C_ref Phi(Ysat), C_ref Phi(Ytotal) for294 point/witness rows per N. For pure
alpha/beta seeds initial constraints must be exactly zero. The continuum gauge
rate zero is the already derived positive-Omega Einstein-sector ADM tangency,
not a numerical pass of either failed FD sequence. Bulk/SAT point constraints may
be nonzero and are measured without a smallness threshold or CPBC assertion.
For nongauge data do not assign a continuum defect without the missing comparator.

Gates and receipts
------------------

allow_pickle=False; finite relevant matrices, shapes and metadata. Independently
recheck EJbulk=Kweak,EJsat=SATload and saved Cholesky LL^T=E (2e-9), nodal T from
analytic Jacobi modes (5e-11), and seed nodal solve (2e-9). Independent scalar
math.fsum contractions of the saved maps must agree at5e-11. Check total point
linearity at5e-11; all84 initial gauge rows must be exactly zero. The scalar
envelope interpolation errors at2N+3 held rho samples are recorded separately,
without an accuracy/admission threshold. No new spectral calculation is used.

Outputs:294 cases per N, complete physical8 vectors, polynomial coefficients,
sample-coordinate Euclidean component summaries, algebra/source checks and
before/after pin receipts. Every NPZ/NPY is tagged large_payload regardless of
size. The reported sample norms are not integrated energy or geometric covector
norms; N changes both polynomial interpolation and matrix projection, so these
rows alone prove neither an order nor convergence of the continuum PDE.

Authorization schema: degree_projected_point_readback_admitted=true,N=12 or16,
and exact driver_sha256,helper_sha256,plan_sha256,operator_sha256. A fresh output
directory is mandatory and failures are retained. No production/source adoption;
finite-pulse stability and later single-BH wormhole-to-trumpet formation with the
Minkowski hyperboloidal reference remain unresolved.
