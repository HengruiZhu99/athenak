# Incoming-trace retention audit

## Finding and scope

**Subsequent evidence:** the compact controls in `../compact-theta/trace-comparison.md` start with zero incoming data but acquire a quadratic trace during evolution. Retained initial Gaussian tails are therefore not a complete explanation of every plateau.

The historical `characteristic_cpbc` / `zero_rate` closure enforces zero **weighted RHS rate**, not zero incoming state. At constant linearization coefficients it therefore preserves an initially nonzero incoming characteristic trace. The direct-Theta Gaussian has a nonzero tail at the boundary. This is one contribution to a tiny late boundary plateau. The completed C/D traces also contain an acquired increment, and do not invalidate earlier independently verified unstable roots.

The rank7 local preflight checkpoints below demonstrate retention at a fixed first-active +x face cell. Aurora access has since been restored and the final C/D traces have been collected: **all three cases reached 50000M**, with 783 complete eight-rank Theta snapshots per case. The updated quantitative evidence is in [GPU_FINAL.md](GPU_FINAL.md). The earlier 373-snapshot collection through 23808M remains in `archive-23808/`; this document retains the initial mathematical/local audit.

## Code audit

In `src/z4c/z4c_Sbc.cpp`, `scalar_p` comprises residual `(Khat,Theta,A_nn^TF,Gamma_n)` and `scalar_d` comprises active-only normal derivatives of `(chi,h_nn^TF,alpha,beta_n)`. The light-speed scalar incoming amplitudes are

```
C1 = sqrt(chi) Theta + chi Gamma_n/2 + D_n chi_res
C2 = (4 Khat/3 + 2 Theta/3 - 2 A_nn^TF)/sqrt(chi)
     - Gamma_n + D_n h_nn^TF .
```

The normal and trace removal use the full conformal metric, while the fields being differentiated are residuals. `zero_rate` constructs `Lp p_rhs + Ld D_n q_rhs` from the volume/source RHS and chooses target zero. The sparse correction solve changes momentum RHS fields to enforce that condition. Configuration RHS fields and their active derivative stencils are immutable in this solve. At flat Minkowski background, the first two formulas reduce to the stated flat characteristic combinations. With the selected G=1 adapted gauge (f=2), the longitudinal gauge row initially equals4Theta/9 for a pure Theta perturbation. The mode signs here follow the source convention; both state values and time rates must use that same convention.

`src/z4c/z4c_tasks.cpp` orders

```
CalcRHS -> ApplyUserRHS (including outer sponge) -> Z4cBoundaryRHS
        -> ExpRKUpdate -> exchanges/ghosts -> EnforceAlgConstr .
```

Consequently the final boundary correction can cancel the sponge's incoming characteristic-rate contribution at the boundary. A sponge in the volume does not imply that every incoming boundary-state combination decays.

For constant matrices and a fixed linear derivative operator, the condition is `d_t C=0`, giving `C(t)=C(0)`. That statement has three limits in the actual code: the coefficients and normal depend on the evolving state; their time derivatives are not included in the weighted RHS rate; and RK updates are followed by nonlinear algebraic projection/recasting. The state diagnostic is therefore a measurement, not a promised exact invariant. The check below retains both the actual nonlinear basis and a fixed-flat-reference trace to distinguish these effects.

## Quantitative local checkpoint check

The cell is `(2016,32,32)M`, rank7, block7, relative level0, first active +x face and the lower tangential edges of that block. The Gaussian is `Theta=A exp(-r^2/(2*384^2))`. Initially `Gamma=chi_res=h_res=A_ij=Khat=0`, so C1=Theta0, C2=2Theta0/3 and the longitudinal gauge trace=4Theta0/9.

For A=1e-6, Theta0=C1(0)=**1.027692618952470e-12**. At t=9.6M after3 full RK steps:

| C1 term | Measured value |
|---|---:|
| sqrt(chi) Theta |1.104384038141563e-12|
| chi Gamma_n/2 |1.647577363466781e-14|
| D_n chi_res |−9.316852844776918e-14|
| Sum C1 |1.027691283328462e-12|

Theta itself has risen7.462486%, while C1/C1(0)=**0.999998700366**. C2/C2(0)=0.999998084399 and the longitudinal gauge ratio=0.999998874525. The nonlinear-basis minus fixed-flat C1 is only7.7312e-24. The C1 deviation from its initial value is−1.3356e-18; this larger discrepancy must not be blamed on basis evolution alone. Projection/recasting and floating-point error are possible contributors; this endpoint comparison does not isolate the exact operation.

The matched A=1e-7 gate has Theta0=1.027692618952470e-13 and C1/C1(0)=1.000066932479 at9.6M. Its C1 sum comprises1.104387734140186e-13 +1.647627978235921e-15 −9.310260895567592e-15. Actual-minus-flat basis is7.7296e-26. The absolute drift is6.8786e-18, not a clean amplitude-scaled nonlinear drift. The raw double checkpoints and script hashes are recorded in `face-trace-local-gate.json` and `face-trace-local-amplitude1e7.json`. These are short CPU MPI8 preflights, not final GPU evidence or a stability validation.

## Follow-up tests and their status

1. **Completed:** read C/D rank7 checkpoints at 0 and approximately 5k intervals through the 50k target, with final suffix `.00050`. All C1/C2/gauge terms are recorded at the same cell; the all-rank checkpoint validator independently passed. The total C1 has both the retained initial trace and an acquired increment, whose tenfold-amplitude ratio is 0.00935. See `GPU_FINAL.md` for uncertainty and physical-constraint limits.
2. **Completed locally with an opt-in compact seed:** the pulse starts with exactly homogeneous incoming data but acquires a nonzero trace, with amplitude ratio 0.0099092 in the matched comparison. See `../compact-theta/trace-comparison.md`. A further possible pulse design would keep initial incoming data compatible with homogeneous boundary data. One controlled design keeps the inner Gaussian unchanged and smoothly tapers its initial tail between1024 and1536M, leaving all stencils at the physical boundary initially zero. This is a different initial-data test, not zeroing any evolved perturbation. A narrower Gaussian also removes the boundary tail approximately but changes the excited wavelengths more strongly.
3. Compare the historical boundary with the candidate physical constraint-radiation closure, keeping outgoing/tangential/lower-order terms and testing the actual constraint subsystem. The existing initial gauge trace must also be recorded; removing the two light-speed traces alone need not remove a separately retained gauge trace.
4. Record characteristic rates before/after CPBC, then the state traces after RK and projection, at the fixed face and an interior control location. This distinguishes frozen initial data, true exponential amplification, and projection-induced trace drift.

## Why a simple −nu C target is not a complete stability fix

Changing the old condition from `d_t C=0` to `d_t C=−nu C` removes the retained initial trace in the constant-coefficient linear model. It is a useful compatibility/plateau experiment. It does **not** by itself eliminate an existing unstable homogeneous surface eigenmode.

For a Laplace-mode state `exp(lambda*t) U`, the old boundary row is `lambda B(lambda,k)U=0`. The relaxed version is `(lambda+nu) B(lambda,k)U=0`. A root with `Re(lambda)>0` already satisfying `B U=0` remains a root; multiplication by lambda+nu cannot remove it. This argument applies when the modification only multiplies the same full boundary operator. A truly different constraint-preserving operator changes the coupling/tangential/lower-order closure or exterior map; it must be screened and evolved independently. No claim is made that state relaxation is equivalent to a complete physical constraint-radiation condition.

## Reproduction

`checkpoint_face_trace.py RUN OUTPUT --prefix PREFIX --indices 0 5 10 15 20 25 30 35 40 45 50` reads one fixed rank per cohort and emits all terms and file hashes. It asserts the uniform cube geometry (global 32³ or 64³, 8 blocks), M=0, adapted gauge and G=1; seed parameters and the fixed face coordinates come from the input/header. Missing requested files are listed explicitly.

`aggregate.py` and `refresh.py` now accept repeatable `--case NAME=PREFIX` for the amplitude ablation without changing the parser's full eight-rank validation. `refresh.py` verifies an ALCF hostname and the expected Flare project directory before any remote writes, to reject an accidentally reused control socket for another cluster. The synthetic binary-reader regression still passes constant-field coverage and rejects truncated payloads, time/cycle mismatch, duplicate coverage and NaN payloads.
