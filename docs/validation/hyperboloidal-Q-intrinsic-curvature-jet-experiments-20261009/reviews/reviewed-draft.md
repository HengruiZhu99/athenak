# Conditional joint null and intrinsic-curvature boundary jets

The actual linearized sigma3 Q/null gauge has a conditional joint boundary
tangency that the earlier unrestricted spatial-pullback test did not expose.
On the stated finite Einstein/null/shear jets,

```
(Nraw_t)_1=-deltaR[q]/a^2.
```

Intrinsic roundness deltaR[q]=0 cancels that first-time null defect, and the
actual first-time curvature map vanishes on the same initial conditions.
This is a finite-order linear result at four reference scales. The complete
compatibility hierarchy, nonlinear closure and finite-pulse stability remain
unproved. No production gauge or boundary condition is changed.

## Coordinate frame and intrinsic geometry

Let q_AB be the induced Penrose two-metric at scri,nu=|D Omega|,s the outward
unit spatial normal and Y=beta+alpha*s. On Einstein compatible data,

```
q_t=Lie_Y q+[alpha*Box(Omega)_0/(2nu)]*q.
```

Preferred conformal compatibility makes Box(Omega)_0=0. Then coordinate
q components need not be fixed, while intrinsic roundness R[q]=2 transports
under the angular diffeomorphism. Keeping coordinate components fixed would
instead require Killing Y; preserving only the conformal class allows CKV Y.
These restrictions are distinct from intrinsic roundness.

The actual pure-gauge test beta=Omega*T,reference lapse/geometry,has zero
initial Einstein constraints and leading residues. Its first metric RHS
changes the normal, giving Y_t=2T/a^2 and q_tt=Lie_(2T/a^2)q. A non-Killing
T therefore changes q components while preserving intrinsic roundness.
The16-control/192-point frame gate agrees with the moving-normal formula
within2.70e-10 and with q_tt within1.42e-11. Initial/next leading-residue
errors remain below6.22e-15. The Killing control has q_tt below7.12e-12.

For the previously tested stationary spatial pullback xi=Omega*X with
Y=n.X on the unit sphere,the sigma3 result is

```
N1_t=(2/a^3)*(Delta_S+2)Y,
deltaR[q]=-(2/a)*(Delta_S+2)Y.
```

Roundness excludes the scale and higher-multipole curvature changes in that
family while retaining its l1 kernel. This resolves the distinction from
the earlier [unrestricted spatial-pullback obstruction](hyperboloidal-Q-angular-geometric-jet-audit.md).
It does not establish that a restricted nonlinear formulation exists.

## Complete cubic chart and available conditions

The fresh actual-kernel gate uses S1,a=.5/.75/1/2,kappa_input10,kappa2=0,
sigma3,eta_outer1 and the exact harmonic outer branch. Physical-P storage
and evolution remain unchanged. Twenty independent primitive tangent fields
times all20 local Cartesian monomials of degree at most3 supply400 columns.
Both north and rational oblique orientations are evaluated. The nonsingular
chart composition r=sqrt(1-2a Omega),n=(sqrt(1-y^2-z^2),y,z) retains normal
and frame derivatives. Coefficients are Taylor coefficients,not repeated
derivatives. The determinant/trace-free tangent chart is imposed throughout.

The246 initial condition rows E contain physical H through degree3,M/Z
covectors through degree2,physical Theta through degree3,Nraw coefficients
Omega0/Omega1 with available angular derivatives,Qnum0=P-3omega_n,and all20
leading actual pole numerators through angular degree2. These are exactly
the orders available without fourth state jets: H's second metric derivatives
carry Omega^2,whereas M/Z have unsuppressed first derivatives. Einstein
Theta=0 is used here; this supplies no arbitrary off-constraint Theta falloff.
No physical-norm rescaling of M/Z conceals their covector coefficients.

E defines necessary finite compatible jets. Its kernel is not proven to
consist entirely of jets extendible to exact vacuum Einstein solutions.
No stronger metric/A/Lambda Omega falloffs are imposed.

## Actual time maps and rowspace identities

The actual regular/pole kernel gives R+S/Omega. Kinematics assembles
Nraw_t=N_R+N_S/Omega,where

```
Nraw=chi*gtilde^ij*Omega_i*Omega_j-omega_n^2,
omega_n=-beta.Omega_i/alpha,
omega_n,t=-(beta_t.Omega_i+omega_n*alpha_t)/alpha.
L=(Nraw_t)_1=(N_R)_1+(N_S)_2.
```

L consumes only second state jets. All pure cubic columns vanish in its
regular and pole contributions separately. Under the initial compatibility
conditions,the finite first RHS has F0=R0+S1 and F1=R1+S2. Feeding its complete
available Cartesian value/first jets into the actual pole map tests all20
next leading residues. Induced metric second angular jets also supply the
actual deltaR[q]_t map. Unavailable higher rates of E are not invented.

At each tested a,the rationally reconstructed E has rank127/nullity273.
L+deltaR[q]/a^2 lies in rowspace(E),as do (Nraw_t)_0,(Qnum_t)_0,deltaR[q]_t
and all20 next leading pole numerators. L and deltaR[q] separately do not.
Adding deltaR[q]=0 raises rank to128,leaving272 finite-jet freedoms.
The null/curvature identity already follows from a119-row lower-order subset
of rank59; a sparse exact combination uses only H/M/Z/Theta/N rows.
The archive gives all four parameter-specific identities and an explicit
compact expression verified at those four values. M/Z coefficients use the
fixed local Cartesian frame at the base point,including angular derivatives.
No symbolic-in-a or uniform-a theorem is asserted.

Exact reconstructed controls retain finite metric,A and Lambda boundary
components. They demonstrate that roundness is not an all-fields-Omega
restriction. They are not identified as radiative Weyl data or Einstein germs.

## Validation and limits

The actual3200 columns agree between orientations within2.27e-13; rational
reconstruction changes entries by at most8.53e-14. An independent induced
curvature formula agrees exactly. It includes graph/frame derivatives and
the curvature term -2trace(h_AB),rather than using fixed Cartesian components
as a cut metric.

At a=.5,three positive-Omega point offsets test all400 columns against direct
actual point kernels. Halving offsets reduces regular/pole and constraint
remainders by about16 and assembled Ndot remainders by about8. Finest absolute
remainders are6.27e-12/1.23e-12/3.30e-8. A separate nonlinear completed-state
central-difference check covers80 columns at the oblique orientation; finest
regular/pole and assembled Ndot errors are7.19e-10/1.11e-6. This implementation
oracle has narrower parameter scope than the four-a rowspace result.

Release and ASan/UBSan cubic outputs are byte identical. Five commands pass
with empty stderr in283.36s;373 source/build inputs are pinned. Earlier
preparations/pilots are retained. Root and independent reviews accompany
the frozen evidence. The frame gate separately has six passing commands
and identical Release/ASan outputs.

The result establishes conditional first-time joint tangency,not invariance
of E or of the full hierarchy. It supplies no exact nonlinear Omega0 assembly,
radiative-data admission,finite-Q amplitude bound,characteristic boundary
energy estimate or stable evolution. Production remains implementation
27c19d20696ea6dd4704032c51dfd026218f64f2. Stable finite Minkowski pulses and
the later inner wormhole-to-trumpet transition with a Minkowski hyperboloidal
reference throughout remain the acceptance targets.

Archive: PENDING.
