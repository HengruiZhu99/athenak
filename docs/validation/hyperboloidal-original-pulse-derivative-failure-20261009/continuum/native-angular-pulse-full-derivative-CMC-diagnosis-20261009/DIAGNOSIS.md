The original full derivative gate remains FAILED: 493,568 completed ray/root calls, 2,360 checks, and 156 failed checks. Its recorded child time is 9,664.044208541 s (root reported 9,664.333227334 s for the enclosing process). Completion of all calls is not acceptance.

The independent stdlib saved-check readback found 2,204 passing and 156 failing records. All failures belong to CMC_l0/l1/l2: 60 `K_jet`, 60 `D_source_jet`, 12 exact-gradient, 12 exact-Hessian, and 12 scalar-wave-trace checks. The 120 local failures cover both precisions, all five angular grids, and both fixed boosts. No saved native, initial-limit, coarea, native convergence/precision/boost, zero, or flat-control check failed. This is a classification of saved records, not a newly evaluated derivative oracle.

The first failure is `control/80/full16/CMC_l0/laboratory/K_jet`, error 1.98821953249403357966 at tolerance 1e-50. The next is its `D_source_jet`, error 1.99947579866821534384. The largest saved CMC exact-Hessian error is 0.70247214999510920964. Both 80- and 110-digit evaluations exhibit this systematic error.

The specific source defect is in `ControlGraph.source`, in the noncenter CMC branch. `q = vc.norm(y)` is a scalar evaluated at the base point; `Y` contains three independent ordinary jets. The code forms

```
H = sqrt(sum(Y_i^2) + a^2),  omega = 1/(H+a),
h = omega*H/a,              bnu_i = omega*Y_i/a,
b = omega*q/a.             # q is frozen: defective jet construction
```

The factored denominator identity requires the jet radius `Q = sqrt(sum(Y_i^2))`:

```
b = omega*Q/a,
nu_i = Y_i/Q,
D = k0*omega^2/(h+b) + b*sum((k_i-k0*nu_i)^2)/(2*k0)
  = h*k0 - sum(bnu_i*k_i).
```

The equality follows from `h^2-b^2=omega^2` and the null identity `sum(k_i^2)=k0^2`. It is an equality of functions in the source chart, hence of their first and second jets. With scalar q, only the equality of base values holds. In particular, the implemented b jet lacks `omega*nu_i/a` in its first derivative and the derivatives of that term in its Hessian. The stable quotient then differentiates a different denominator; the independent smooth `direct_D` retains the missing terms through `omega*Y_i/a`. The CMC value checks passing while its derivative and wave checks fail is consistent with this defect. It is not evidence for a sign change of the Kirchhoff integral.

The minimal proposed correction is restricted to the existing q>0 CMC branch:

```
- b = omega*q/self.a
+ b = omega*(sumjet(t*t for t in Y)**mp.mpf(".5"))/self.a
```

The exact-center branch already uses the smooth Cartesian bnu expression and remains unchanged. NativeGraph uses analytically lifted coefficient jets and does not contain this scalar-radius construction; its body is unchanged by the proposal. This identifies a defect in the control implementation, not a proof of correctness of every other source formula.

The literature agent independently confirmed the source/pencil argument without imports or evaluations: retaining the radius as a jet is necessary for the factorization to hold at derivative order, and the q>0-only replacement is the minimal algebraic correction. The original source, full failed attempt and all 156 failed records are preserved. No large jet payload was decoded. Streaming hashes record those payloads as metadata only.

The next proposal is a fresh control-only gate with all five original controls, both precisions, five angular grids, both boosts, and unchanged local/exact/wave thresholds: 189,440 control rays and 880 checks. A separate saved-check qualification can retain the other 1,480 passing records, conditioned on the exact original 2,360/156 classification and unchanged native/initial/source functions. Neither step relabels the original full attempt, supplies an inverse map, diagnoses a native abort, or proves global coordinate regularity. Execution of the new control source is held for root and independent source review.
