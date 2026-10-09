# Conformal Minkowski reference wave-map gauge: pencil feasibility assessment

This is a fresh mathematical assessment, not a gauge implementation, source query, principal extraction, spectrum, evolution or adoption. The physical-P storage and geometric evolution remain unchanged. The same Minkowski hyperboloidal reference and prescribed time-independent Omega are used throughout. No full-Q subtraction, floor, imposed Theta falloff or BH RHS subtraction is proposed. All accepted and failed prior evidence remains byte-preserved.

There is a conditional negative stationary result. Replacing the physical reference connection by its conformal reference connection removes that gauge source's explicit off-reference simple pole when the live conformal inverse metric is bounded. It does **not** by itself repair the missing Schwarzschild mass logarithm for the normalized stationary, spherical, Killing-aligned smooth end considered below. This is not a finite-time blowup result, a native-failure explanation, or an exclusion of time-dependent BH coordinates. In particular it does not answer the later requirement of surviving a wormhole-to-trumpet inner transition with the Minkowski reference retained.

## 1. Full four-dimensional transformation and conventions

Let g=Omega^-2 barg and ghat=Omega^-2 barghat, signature -+++, with the same positive Omega. Physical and conformal Z vectors obey Zphysical^a=Omega^2 Zbar^a. Let

```text
Hphysical^a = g^{bc}(Gamma[g]^a_bc-Gamma[ghat]^a_bc)+2Zphysical^a,
Hconformal^a = barg^{bc}(Gamma[barg]^a_bc-Gamma[barghat]^a_bc)+2Zbar^a,
s = barg^{bc} barghat_bc.
```

The exact connection transformation is

```text
Gamma[g]^a_bc = Gamma[barg]^a_bc
 - (delta^a_b Omega_c+delta^a_c Omega_b-barg_bc barg^{ad}Omega_d)/Omega.
```

On taking the connection difference the two Kronecker-delta terms cancel, but the metric-contracted term does not. Contracting with the LIVE inverse gives

```text
Hphysical^a/Omega^2 = Hconformal^a+U^a/Omega,
U^a = (4barg^{ad}-s barghat^{ad})Omega_d.
```

The numeral four is the spacetime dimension. The tempting replacement of U by twice an inverse-metric difference is incorrect for a fixed physical reference. Both U and the connection-difference gauges vanish at the same reference; their off-reference values differ.

The proposed conformal gauge would impose Hconformal=0. In the universal conformal GH convention GammaBar^a+2Zbar^a=Fbar^a its source is

```text
Fc^a=barg^{bc} Gamma[barghat]^a_bc.
```

The physical reference gauge instead has Fp=Fc-U/Omega. Therefore the finite-positive-Omega changes of the assembled gauge rows are exactly

```text
Delta(D0 alpha)=-alpha^3 U^0/Omega,
Delta(D0 beta^i)=-alpha^2(U^i+beta^i U^0)/Omega,
D0=partial_t-beta^i partial_i.
```

All four-dimensional reference Christoffel components enter Fc. Unlike the physical reference connection in the reference-inertial embedding, Gamma[barghat]^a_0b generally is nonzero. It must not be silently discarded. A stationary ADM four-metric built from the complete reference lapse, shift and Penrose spatial metric supplies this connection from their first derivatives, without unspecified higher jets.

## 2. Physical-P gauge rows and robust factoring

Use the actual conventions

```text
P=Kphysical-2ThetaPhysical,
omega_n=-B/alpha,  B=beta^i Omega_i,
Q=(P-3omega_n)/Omega,
C^{ij}=chi*gtildeInv^{ij},
Lambda^i=GammaTilde^i+2gtildeInv^{ij}Z_j.
```

The universal identities, including the full Z coupling, give the conformal-reference rows

```text
alpha_t = beta^i alpha_i-alpha^3 Fc^0
          -(alpha^2 P+3alpha B)/Omega,
beta_t^i = beta^j beta^i_j+alpha^2 chi Lambda^i
           +(alpha^2/2)gtildeInv^{ij}chi_j-alpha C^{ij}alpha_j
           -alpha^2(Fc^i+beta^i Fc^0).
```

Thus the un-factored split has a lapse simple pole and a regular shift. This is a different gauge from the physical-P restoring driver; calling it physical-P refers to the retained evolved trace/storage, not retention of that driver's lower-order stabilization. It is also different from the earlier conformal-Q/preferred-projection/null-feedback gauges.

The following factoring is algebraic and does not require evaluating Q or dividing by the LIVE lapse. Write a=alpha, h=alpha_hat, da=a-h, dP=P-Phat, db=beta-betahat, dB=db.dOmega. Define

```text
V=a^2 C, L=V-beta beta,
J^{00}=-1, J^{0i}=J^{i0}=beta^i, J^{ij}=L^{ij},
T^a=J^{bc}Gamma[barghat]^a_bc.
```

Then a^3 Fc^0=a T^0 and a^2(Fc^i+beta^i Fc^0)=T^i+beta^i T^0. These contractions are polynomial in the live lapse/shift and inverse spatial metric and contain no inverse live lapse. With dT=T-That obtained by contracting dJ, the source differences factor as

```text
a T^0-h That^0 = da That^0+a dT^0,
(T^i+beta^i T^0)-(That^i+betahat^i That^0)
 =dT^i+beta^i dT^0+db^i That^0.
```

The lapse pole numerator S=-a(aP+3B) has the exact difference

```text
S-Shat=-a[a dP+da Phat+3dB]-da(h Phat+3Bhat).
```

Since h Phat+3Bhat=Omega*h*Kbarhat (Theta_hat=0), the final term can be put analytically into the regular part. A reference-deviation representation is

```text
pole_alpha=-a[a dP+da Phat+3dB],
regular_alpha=beta.dalpha-betahat.dh
              -(da That^0+a dT^0)-da*h*Kbarhat.
```

The shift can likewise use reference-deviation expansions of its displayed regular terms and the factored source difference. Both returned parts vanish algebraically at the reference. The stationary identity

```text
betahat.dh-h^3 Fchat^0+Shat/Omega=0
```

justifies this representation; it is not a numerical subtraction of a complete reference RHS. In the outer S,a CMC reference, Phat=-3/a, h Phat+3Bhat=-3Omega/a, Kbarhat=-3/(a h), and h^3 Fchat^0=3h/a-Bhat. At the exact geometric Cauchy plateau Gamma[barghat]=0 and Omega=1, so these are the usual complete harmonic lapse/shift rows. No moving-puncture blend is selected.

## 3. Finite-Omega derivative principal relation and closure limits

Fc and U depend algebraically on the live metric values and on fixed reference/Omega jets. At any fixed Omega>0, positive lapse/chi and SPD conformal metric, changing Fp to Fc therefore does not change the derivative-order harmonic principal matrix. The full-Z Lambda/P convention is essential to this statement. The previously audited physical-reference harmonic principal family motivates an equality check, but this assessment is not an actual new-helper symbol gate, a raw22 normal-dynamics result, or a uniform estimate as Omega tends to zero.

For a pure physical coordinate perturbation delta g=Lie_xi ghat at fixed prescribed Omega,

```text
delta barg=Lie_xi barghat-2f barghat, f=(xi.dOmega)/Omega.
```

Linearization of Hconformal on the Einstein sector is consequently

```text
(delta Hconformal)^a=Box_hatbar xi^a+Ric_hatbar^a_b xi^b+2 hatbar_nabla^a f.
```

The last term comes from contracting the conformal metric perturbation -2f barghat in four dimensions. It must be retained. These are not four independently conformally covariant scalar equations (Box-R/6)phi=0, nor simply the physical-reference inertial wave equations. At finite Omega the second derivative principal part is the wave operator; near scri f and its derivatives require their own coupled boundary-jet compatibility. Bounded physical inertial perturbations alone must not be replaced by arbitrary compact-coordinate monomials.

The source identity for the conformal gauge is

```text
Box_barg Omega=barg^{bc} hatbar_nabla_b hatbar_nabla_c Omega
                +2Zbar^a Omega_a.
```

The Minkowski reference in the preferred conformal outer frame has hatbar_nabla dOmega=0 at scri. Conditional on a bounded live conformal inverse and Z=0 this yields Box Omega=0 there. It does not enforce a chosen finite-Omega Box source or its next Taylor coefficient. Off constraints,

```text
Zbar^i=C^{ij}Z_j-beta^i ThetaPhysical/(Omega alpha),
```

so a finite physical Theta may leave a singular contraction. No additional Theta falloff is imposed here. The pole compatibility aP+3B=O(Omega), the null residue/order, geometric shear, full Z/Theta hierarchy and their time tangencies remain to be derived jointly. Leading preferred Box alone is not a closure theorem.

The existing coupled conformal-wave isolate is a two-field scalar test on a fixed LayerReference. It evolves no geometry, no differential Z4 constraint and no gauge pole. Its boundary/spectral/pulse evidence cannot be used as this conformal-reference GAUGE gate. The prior Q/preferred-source artifacts also use a different source and lapse split.

## 4. Conditional stationary Schwarzschild end: the mass logarithm remains

The following uses the same restrictive but useful assumptions as the prior physical-reference stationary audit: M>0, spherical stationarity, partial_t equal to the normalized Schwarzschild Killing field, a standard smooth nondegenerate future conformal end, and the fixed ordinary Minkowski reference compactification. It neither prescribes a black-hole initial reference nor proves nonexistence of time-dependent or polyhomogeneous alternatives.

Let F=1-2M/R and write physical Schwarzschild in areal radius R. Introduce the REFERENCE inertial target coordinates

```text
Y^0=T+psi(R)=t+hhat(f(R)),
Y^I=f(R)n^I,
psi=hhat(f)-hBH,
hhat(f)=sqrt(f^2+a^2)+constant.
```

In these target coordinates barghat_AB=Omega(f)^2 eta_AB, while barg_AB=Omega(f)^2 g_AB. The conformal gauge becomes the wave-map equation

```text
Box_g Y^A+(4g^{AB}-s eta^{AB})partial_B logOmega(f)=0,
s=g^{AB}eta_AB.
```

One can derive it either from Section1 or directly from the domain/target conformal scalar/connection transformations; both yield the same factor four and sign. Set k(f)=dlogOmega/df. The inverse physical metric in Y components is

```text
g^{00}=-F^-1+F psi'^2,
g^{0I}=F psi' f' n^I,
g^{IJ}=F f'^2 n^I n^J+(f^2/R^2)(delta^{IJ}-n^I n^J),
s=F^-1-F psi'^2+F f'^2+2f^2/R^2.
```

The temporal and radial spatial equations are therefore

```text
(R^2 F psi')'/R^2+4F psi'f' k=0,
[(R^2F f')'-2f]/R^2+(4F f'^2-s)k=0.
```

The first equation integrates EXACTLY to

```text
R^2F psi' Omega(f)^4=D.
```

For the fixed CMC outer reference Omega(f)=S/[sqrt(f^2+a^2)+a], hence Omega~S/f and k=-1/f+O(f^-2). With f/R tending to1 a nonzero D forces psi'~D R^2/S^4. Such growth is incompatible with the assumed normalized smooth future end; its height difference has at most the inverse-radius behavior needed for the mass logarithm. Thus D=0 and psi'=0 in this stationary class. In particular hBH'=hhat'(f) f'.

The remaining spatial equation reduces to

```text
(R^2F f')'-2f+R^2 k[3F f'^2-F^-1-2f^2/R^2]=0.
```

To see whether it can supply the required mass logarithm, allow

```text
f=R+c log(R/ell)+d+o(1),
f'=1+c/R+o(R^-1),
f''=-c/R^2+o(R^-2),
```

with differentiated remainders sufficient for the displayed equation. Its first term is c-2M-2c log(R/ell)-2d+o(1). Its bracket is [6c-8M-4c log(R/ell)-4d]/R+o(R^-1), so multiplication by R^2 k gives -6c+8M+4c log(R/ell)+4d+o(1). The total residual is

```text
2c log(R/ell)-5c+6M+2d+o(1).
```

It follows that c=0 and d=-3M. The physical-reference gauge instead selected f=R-M+..., so the coordinate mass offset changes. Neither source supplies the height logarithm. Indeed smooth future hyperboloidal spatial data require

```text
gammaPhysical_RR=F^-1-F hBH'^2=aB^2/R^2+O(R^-3),
hBH'=1+2M/R+O(R^-2),
hBH=R+2M log(R/ell)+O(1).
```

Here hhat'(f)=1+O(R^-2) and f'=1+O(R^-2) on the derived regular expansion, so their product misses 2M/R. It yields gammaPhysical_RR=4M/R+O(R^-2). With f=r/Omega and L=Omega-rOmega', the Penrose radial metric then has the same leading pole 4M L^2/(r Omega). Allowing f=R+2M log R would repair the temporal height relation but conflicts with the conformal spatial gauge equation just derived, and also introduces Omega logOmega in ordinary angular coefficients.

Thus boundedness of Fc for bounded conformal fields is insufficient to establish a smooth stationary massive end. This calculation leaves open a different derived outer gauge, a differently justified asymptotic domain/target, or time-dependent coordinates. It gives no reason to alter the user's Minkowski reference, and no choice for the wormhole-to-trumpet inner gauge.

## 5. What is and is not ready

This assessment supplies exact pencil identities and a conditional asymptotic obstruction. It does not supply an admitted new source helper or a finite-amplitude conformal wave-map oracle. A later concrete candidate would still require independent four-dimensional/reference factoring checks, the full actual constrained20 principal gate, actual off-constraint finite/pole/RK checks and a justified domain of coupled boundary jets. The already-failed physical-reference N16 pulse and C0 N16 pulse have separate saved diagnostics; none of their observed behavior is attributed to this stationary argument.

The source inventory pins the earlier reviewed proposal/helper, conditional inner/mass audit, scalar-control distinction and relevant production conventions. File hashing and source reads are provenance operations only. No CAS, numerical asymptotic experiment, kernel query, compilation, operator, propagation or native evolution was used for this assessment.

Independent pencil scrutiny by the literature agent agrees with the complete temporal/spatial ODEs and the expansion. Before setting D=0 the spatial bracket is 3F f'^2-F^-1+F psi'^2-2f^2/R^2; the psi term has only been removed after the temporal argument. For the assumed smooth inverse-radius end, the necessary leading result is f=R-3M+O(R^-1); it is not a constructed solution. The inverse-radius remainder makes the displayed O(R^-2) height statement appropriate. A time-dependent end, different asymptotic target/frame or a separately justified polyhomogeneous domain remains outside this calculation.
