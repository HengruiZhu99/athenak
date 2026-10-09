# Additive finite-jet tensor-freedom check

After the original frame/cubic draft review, a separate exact algebraic probe
checked the retained tensor freedoms. Its original frozen index SHA256 is
`c16dbe6eb947a9c5f8710c297c042868c3dd74581742df9cf81786875316380d`.
That bundle and independent/root additive reviews are included here without
modifying either original frame/cubic index or the reviewed draft.

At all four tested a values,ker(E;deltaR[q]) maps onto both tracefree first-normal
cut-metric coefficients q1_TF. Here TF uses the fixed reference cut metric.
Differentiating the tracefree part with respect to the perturbed live leading
cut metric instead gives q1_TF+2a*h0_TF. The transparent h0_TF=0 representatives
coincide under both definitions,so two directions remain under either choice.
The actual imposed shear relation is

```
A0_TF-q1_TF/(2a)-2h0_TF-S=-(a^2/2)*R0_A_TF,
S_plus=(d_y h_ny-d_z h_nz)/2,
S_cross=(d_y h_nz+d_z h_ny)/2.
```

The selected transparent representatives additionally have h0_TF=S=0 and
A0_TF=q1_TF/(2a). These are representative choices,not extra necessary
conditions,boundary values or falloffs. The base rank/nullity remains128/272.
This retains two finite Taylor-jet tensor freedoms; genuine radiative Weyl
data,Einstein-germ existence and full hierarchy invariance are unproved.

The readable audit also applies the independent notation clarification
F0=regular_0+pole_1,F1=regular_1+pole_2. The original reviewed draft is kept
byte for byte under reviews/ so its historical hash remains verifiable.
