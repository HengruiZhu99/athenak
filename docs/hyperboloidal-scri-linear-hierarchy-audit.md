# Actual linear scri first-jet compatibility

Cancelling all leading C0 poles, preserving finite Q and imposing the scalar first-corner rates does not close the smooth scri compatibility hierarchy. The actual full tensor kernel supplies a next-level A/Λ counterexample. This is a smooth-corner obstruction, not proof of finite-Q amplitude blowup or a closed boundary prescription.

The audit uses the physical-P lapse and spatial-norm shift on the outer Minkowski CMC reference, S1, a=.5,.75,1,2 and κinput10, with baseline κ2=0 and the live coefficient's outer reference value. Write δu=u0(n)+Ωu1(n)+Ω²u2(n)+…, where u1 is the Ω Taylor coefficient. Eight compiled20×80 residue maps act on (u0,u1,∂y u0,∂z u0) at n=(1,0,0). All have exact reconstructed rank15. All640 basis tests agree with the explicit formula to3.553e−15;160 admissible finite-A/Λ first-jet cases pass to4.441e−16 in Release and ASan/UBSan.

Let h=δgtilde, c=δχ, p=δP, θ=δΘphys, da=δαbar, b=δβ, x=c−h_nn, d=δΛ−δΓtilde and H_ij=h_ij+(δΓbar^n_ij)^TF. All fields here denote boundary values unless differentiated. Set η=1.5/a², C=1/(3a), κ=κinput, and k2*=0 or 2/(κa²)−1. The actual leading residues include

```
R0_alpha=(-p−3da+b_n)/a²,
R0_chi=2(p+2θ−3da−3b_n)/(3a),
R0_P=−2(p+2θ)/a²−3x/a³+κ(1−k2*)θ,
R0_Theta=−2p/a²−[1/a²+κ(2+k2*)]θ−3x/a³,
R0_beta=−η[b+C n x],  R0_g=0,
R0_A=2[H−(d n+n d)^TF/2−A]/a²,
R0_Lambda_i=−(κ−2/a²)d_i−2(2∂i p+∂i θ)/(3a)+4A_in/a².
```

For the audited κa²>1, R0=0 requires da0=p0=θ0=b0=0 and c0=h_nn0. Angular derivatives of these equalities must also hold, including ∂A c0=n^i n^j∂A h_ij+2h_nA. Remaining residues allow finite boundary A/Λ:

```
d_A=4H_nA/(κa²),
d_n=(12H_nn+4P1+2Θ1)/(3κa²+2),
A=H−(d n+n d)^TF/2,  Λ0=Γtilde0+d.
```

No universal A/Λ=O(Ω) condition is imposed. Five zero modes of the value-only pole matrix do not determine this first-jet map.

For Θ=Ωτ, another necessary smooth-corner condition is

```
Θ_t0=(2/a)δLap_bar Ω−(2/a²)δQ−κ(2+k2*)Θ1−(3/a)Nraw1=0,
δQ=P1−3(α1+β_n1),
Nraw1=(χ1−h_nn1)/a²+2(α1+β_n1)/a.
```

Here Nraw=χgtildeinv(dΩ,dΩ)−w². Quadratic-null regularity requires Nraw0=Nraw1=0 and is separate from R0 cancellation and finite Q. The initially omitted −3Nraw1/a term fails an actual random-state test by7.16784; the exact failed check is retained. The corrected expression agrees to3.842e−11.

A first witness δP=Ωq with suitable finite radial A/Λ cancels every leading pole but produces nonzero Θ_t0, ΩQ_t and Nraw_t0. It violates asymptotic differential constraints and is not a vacuum Einstein counterexample. A stronger witness with only δP=Ω²q initially satisfies R0, scalar corner rates and boundary H/M/Z/Θ values. Feeding its exact first Cartesian RHS, including spatial derivatives, back into the actual kernel gives

```
(d_t R0)_A,nn=−8q/(3a⁴),
(d_t R0)_Lambda,n=(12−8κa²)q/(3a⁴).
```

Thus d_t R0=B(F0,F1,tangential F0)=0 adds second-jet conditions. No finite closed nonlinear hierarchy or constraint-compatible ghost continuation has yet been derived.

The actual value-only matrix also explains the limit of this counterexample. Its five-dimensional semisimple kernel projector Π and group inverse D satisfy exact reconstructed identities; Θ and δ(P−3w) rows annihilate Π. For constant frozen finite forcing and zero initial perturbation, u_t=P u/Ω+f has u=tΠf+ΩD(exp(tP/Ω)−I)(I−Π)f. Those two amplitudes can remain O(Ω) despite O(1) initial derivatives through a time layer t~Ω. This is a finite matrix statement; the full spatial pole operator, variable coefficients and forcing are not controlled by it.

See the byte-preserved [complete derivation](validation/hyperboloidal-live-damping-and-scri-hierarchy-experiments-20261009/scri-linear-hierarchy/DERIVATION.md), compiled maps, exact projector check and all failed/successful receipts in the [archive](validation/hyperboloidal-live-damping-and-scri-hierarchy-experiments-20261009/README.md). Original index SHA256 is `bdaaa422b7906a7237130f146a687c1be4ed55ced61605c48f57eaa7efe4cae9`. These results reject a value-only ghost projection as a justified closure; they do not impose new production falloffs.

The [Einstein Taylor and gauge-corner follow-up](hyperboloidal-mode-and-gauge-corner-audit.md)
derives Θ1=0 from M0=Z0=0 on compatible pole jets and relates Θ_t0 to H1.
It also identifies the nonzero higher Einstein coefficients of the Ω²P witness.
A distinct initially constraint-free lapse/shift witness still violates null
time compatibility, so those constraint conditions alone do not close the gauge
hierarchy.

The subsequent [Q-gauge first-jet audit](hyperboloidal-Q-early-and-jet-audit.md)
derives its complete leading residue map and separates the missing Theta/shear
conditions from null-jet time tangency. A genuinely Einstein initial quadratic
shift violates the latter under sigma-five despite cancellation of the next
leading RHS pole. No exact-scri closure or amplitude blowup theorem follows.
