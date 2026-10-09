# Conditional joint null/curvature tangency of finite Einstein jets

This private linear audit uses the actual Cartesian C0 ConformalRHS and the byte-pinned factored Q/preferred/null-feedback helper. Parameters are S=1,kappa_input=10,kappa2=0,sigma=3,eta_outer=1 and curvature radii a=.5,.75,1,2. The analytic reference has Omega=(1-r^2)/(2a), alpha_ref=(1+r^2)/(2a), beta_ref=-x/a, gtilde=I,chi=1,A=Theta=Lambda=0,P=-3/a. All evaluations concern its exact harmonic outer branch W=V=1. The physical-P storage/evolution is unchanged; the already rejected sigma5/global sources and every previous frozen receipt remain unchanged. No source, boundary prescription, native/global evolution or sigma3 admission is introduced.

## Complete local cubic chart and exact finite-order requirements

At a scri point choose a rational orthonormal frame (n,e_y,e_z). Local Cartesian offsets (X,Y,Z) give x=n(1+X)+e_y Y+e_z Z. Twenty monomials of total degree at most three times twenty primitive tangent variables give 400 independent columns. The fields are alpha,chi,P,Theta,beta(3),five tracefree gtilde components,five tracefree A components,Lambda(3), in that order. At this reference the differentiated determinant/trace conditions are exactly the ordinary tracefree linear chart. The nonlinear point finite-difference oracle completes det(gtilde)=1 and tr_g(A)=0, including their consumed spatial derivatives.

The Cartesian chart is composed with

    r=sqrt(1-2a Omega), n_local=(sqrt(1-y^2-z^2),y,z),
    (X,Y,Z)=(r*n_local.x-1,r*y,r*z).

Its nonsingular linear Jacobian shows that this spans the full cubic jet in (Omega,n_y,n_z), including nonfree normal/frame derivatives. Taylor coefficients, rather than ordinary repeated derivatives, are reported in the matrices. Pure quadratic monomials therefore have coefficient one and second derivative two. Both the north orientation and n=(.36,-.48,.8),e_y=(.8,.6,0),e_z=(-.48,.64,.6) are independently passed through the actual tensor kernel.

The 246-row initial condition matrix E contains exactly:

* H through total boundary-chart degree3:20 rows.
* Each native momentum covector M_i and spatial Z covector Z_i through degree2:30+30 rows.
* Stored physical Theta through degree3:20 rows.
* Nraw coefficient Omega^0 through angular degree3:10 rows; coefficient Omega^1 through angular degree2:6 rows.
* Qnum=P-3omega_n coefficient Omega^0 through angular degree3:10 rows.
* All20 actual singular numerators R0 at Omega=0 through angular degree2:120 rows. Some rows are identically zero or dependent; no independent lapse/shift Dirichlet values are prescribed.

These are the coefficients determined without fourth state jets. H's two metric derivatives have an Omega^2 prefactor, and its first derivatives an Omega prefactor. M/Z contain unsuppressed first derivatives, so their degree3 coefficients are unavailable from a general cubic state. Einstein Theta is zero as a field; using its known finite coefficients here is not an imposed falloff for arbitrary off-constraint Theta. Physical norm boundedness alone would not imply these covector coefficients vanish.

E defines necessary finite Einstein/null/shear-compatible jets. It is not a theorem that every vector in its kernel extends to an exact vacuum Einstein germ. The complete E and the explicit finite-value controls are archived, so neither higher constraint conditions nor all A/Lambda/metric deviations O(Omega) are hidden in the selection.

## Actual first-time maps and derivative order

For each column the actual kernel returns analytic regular and pole numerators R,S. The kinematic metric/chi/lapse/shift contribution to Nraw_t is assembled algebraically as

    Nraw=chi*gtilde^ij Omega_i Omega_j-omega_n^2,
    omega_n=-beta.Omega/alpha,
    omega_n,t=-(beta_t.Omega+omega_n alpha_t)/alpha.

Thus Nraw_t=N_R+N_S/Omega. Its coefficient L=(Nraw_t)_1 equals (N_R)_1+(N_S)_2. In these actual kinematic rows R has at most first state derivatives and S has only values. L therefore consumes second state jets. Every pure cubic column is zero in both contributions separately; it is not a cancellation obtained by setting unknown third/fourth jets to zero.

When the initial S0 and its required tangential jets vanish, the finite first RHS has coefficients F0=R0+S1 and F1=R1+S2. Its complete value/first Cartesian jets are supplied back to the actual R0 kernel. This computes all20 available first-time pole residues without imposing them as initial conditions. The first RHS's metric/chi boundary second angular jets similarly give the intrinsic curvature time rate. These use at most the supplied cubic state jets. Testing evolution of every row of E, including its higher angular/shear derivatives, would require additional state jets and is not performed.

Intrinsic delta R[q] is computed from the induced Penrose two-metric q, retaining its first/second angular coefficient jets and frame derivatives. An independent Cartesian-to-induced metric variation formula checks the complete curvature row. At the north graph coordinates qref_AB=delta_AB+y_A*y_B/(1-y^2-z^2),

    deltaR[q]=sum_AB partial_A partial_B h_AB
              -Delta(sum_A h_AA)-2sum_A h_AA,

where h_AB is the induced two-metric perturbation, not unprojected fixed Cartesian components.

## Rowspace result and its limited implication

At each of the four audited curvature radii the rationally reconstructed E has rank127 and nullity273. The exact reconstructed rows satisfy

    L+deltaR[q]/a^2 is in rowspace(E),
    (Nraw_t)_0, (Qnum_t)_0, deltaR[q]_t and all20 (R0)_t are in rowspace(E).

L and deltaR[q] separately are not in that rowspace. Adjoining deltaR[q]=0 raises rank to128, leaving272 free finite jets. Consequently imposing intrinsic roundness supplies conditional joint first-time tangency of (N1,deltaR[q]) on these stated initial Einstein/null/shear jets. It does not prove that E itself is invariant, close the full hierarchy, or establish nonlinear/global stability. At the stationary linear reference, preferred Einstein compatibility gives q_t=Lie_Y q and Rref=2, so deltaR_t=0; the actual complete first-RHS map confirms this conditional statement rather than assuming full hierarchy invariance.

The null/curvature identity already follows from a119-row lower-order subset (rank59): H through degree2,M/Z through degree1,Theta through degree2,N/Q through degree2,and R0 through first angular order. The saved sparse combinations use only H/M/Z/Theta/N rows. For a=.5, with coefficient notation [Omega power,y power,z power], one such exact identity is

    L+4 deltaR = H000/6-4H100+4H200+(H020+H002)/3
      -4M_n000+8M_n100-2M_y010-2M_z001
      -48Z_n000+32Z_n100+8Theta000-48Theta100+32Theta200
      +N000+4N100.

This combination is for the rationally reconstructed analytic-reference matrices. All four parameter-specific combinations, reconstruction errors and orientation comparisons are recorded by check_curvature.py. No exact nonlinear Omega0 assembly or uniform-a PDE theorem is claimed.

## Controls, validation and unresolved work

The full cubic chart includes every previous compatible gauge-only and spatial-pullback local jet. Its curvature identity recovers the m1 spatial-pullback obstruction N1_t=-deltaR[q]/a^2 and the gauge-only sigma3 cancellation. Exact reconstructed kernel vectors with finite boundary A, Lambda and metric components demonstrate that roundness here does not force every field deviation to vanish as Omega. They remain finite necessary-jet examples, not a claim of exact radiative vacuum data.

Three direct positive-Omega point offsets at a=.5 check the shared formal Taylor/dual implementation against independently evaluated actual point kernels for every400 columns. R/S and physical-constraint remainders shrink16-fold with half offset, and the assembled Ndot remainder shrinks8-fold because of the explicit Omega denominator. A separate nonlinear completed-state finite-difference oracle checks80 columns at the oblique orientation. Release and ASan/UBSan outputs, source hashes, compiler flags, all command exits/stderr and exact matrix statements are retained by the final receipt. The four-a rowspace check and the a=.5 implementation oracle have distinct scopes.

This local result makes intrinsic roundness more defensible than fixing coordinate q_AB/Y values, but does not adopt it. The earlier gauge-only q_tt=Lie_(2T/a^2)q counterexample preserves roundness and remains consistent. No freely radiative tensor family, higher-order Einstein existence, evolution-preserved conformal frame, boundary characteristic energy estimate, finite-Q amplitude bound or puncture/trumpet hyperbolicity is established. The later single-BH goal still requires the inner wormhole-to-trumpet transition with the Minkowski hyperboloidal reference retained throughout; none of these local outer jets establishes that goal.
