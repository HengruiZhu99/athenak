# Stability diagnosis

## Classification

```text
bulk instability:          not established
axis instability:          disfavored
outer-boundary instability: excluded as first cause
AMR-interface instability: strongly supported
under-resolution:          not ruled out as an amplifier
```

The campaign verdict is `VC_AMR_INTERFACE_LIMITED`, not
`VC_INTRINSIC_INSTABILITY`.

## Why the axis is not leading

The exact active-axis regularity telemetry stays at roundoff and never reports
a nonfinite correction. Axis field/constraint convergence remains positive
well after trusted-core/interface convergence is negative. The N512 terminal
failure is at `rho=3.421875`, not at the axis.

## Why the boundary is not leading

The N512 failure radius is `3.5419`, while the conservative N256 speed
integral still leaves a protected radius `9.0027` at a later time. Outer-layer
orders are usually positive when the inner AMR-enabled region is already
negative. The user's permission to lower extrapolation for another outer-face
failure therefore does not apply; no extrapolation change was made.

## Why AMR handling leads

1. The initial slice and pre-turnover interval converge.
2. The first catastrophic change is locked to authority event 3, a replayed
   derefinement.
3. Its constraint jump increases by roughly three orders of magnitude per
   resolution step for C/H and similarly for M/Z.
4. Event 4, one N256 cycle later, starts a rapid refinement cascade.
5. The previous matched fixed-grid control has positive protected-core order
   over the same early interval.
6. At `t=1`, the axis and outer layers remain convergent while seams and
   coarse/fine neighborhoods are strongly resolution-worsening.
7. N512 eventually fails one fine cell from a MeshBlock edge.

These observations implicate the transaction/interface path. They do not yet
identify whether restriction, redistributed active state, cache/ghost refresh,
q6 interface closure, or the first post-event RHS is the first writer.

The retained history and quarter-time field cadence also cannot establish the
first variable/location of a high-k branch at event 3. No such spectral claim
is made. The bounded replay below must add the native-resolution spectrum or a
local O6/O4 disagreement census at each writer phase.

## Parent under-resolution

It is not mathematically ruled out. Native shadow sensors do not select the
same schedule at all resolutions, and a coarse representation may already
contain short-scale structure. But simple global parent under-resolution is
not sufficient to explain why an exact derefinement causes a much larger jump
at higher cells-per-MeshBlock resolution. Treat under-resolution as a possible
amplifier until the event-3 writer census and pre/post spectra are available.

## Smallest decisive next diagnostic

Replay only the event-3 window from an authenticated pre-event checkpoint and
stop after the first post-event RHS. Without changing numerics, census all 25
state fields and H/M/Z/C after:

1. pre-transaction active state and derivatives;
2. restriction/derefinement candidates;
3. redistributed accepted leaf state;
4. shared-node synchronization;
5. coarse caches and physical/axis ghost fill;
6. algebraic projection/ADM reconstruction;
7. first post-event RHS and RK update.

Record the first writer, location, level, block-edge/coarse-fine distance,
high-order value, lower-order shadow value, and O6/O4 derivative disagreement.
This bounded replay distinguishes state-transfer injection from derivative
closure amplification without another long collapse run.
