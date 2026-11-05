# Puncture reference variables and perturbation controls

The r² reference-variable representation removes the observed N512 puncture
failure without floors and permits a four-resolution 10M comparison. It does
not resolve differential D compatibility or the late growing gauge response.
No robust spherical candidate or fixed-error performance claim is admitted.

## A spatial representation of the same equations

The v14 direct-chi representation failed at the first N512 puncture node near
3.142678M, independently of halving the timestep. At the last accepted state,
the negative one-sided chi derivative drove its value through zero; KO opposed
the crossing. This motivated differentiating v=chi/f with f=r^p, for p=2 or 4.

`reference_variables.hpp` supplies Cartesian scalar-jet product and quotient
rules, with fixed f and all first/second derivatives. They are valid for
nonspherical jets as well as the spherical adapter:

```
chi_i  = f_i v + f v_i
chi_ij = f_ij v + f_i v_j + f_j v_i + f v_ij
```

The laboratory's `--chi-reference-power 2|4` differentiates the normalized
residual v-v_initial, adds the consistently transformed initial-reference
derivatives, and recovers chi jets through those shared rules. Biased advection
uses the same transformation. The stored RK values and the KO term remain
physical chi, so this isolates a spatial-representation change; changing the
stored RK coordinate by this fixed linear factor would be algebraically
equivalent. There is no sign projection, constraint reset, modified continuum
source, artificial stationary core, or changed scri policy. The singular
inverse factor at r=0 is not a puncture regularity proof; r=0 is excluded by
the existing puncture chart.

Verification precedes the evolution controls. A new CTest evaluates 198
nonspherical Cartesian polynomials independently by their monomial
coefficients, including mixed derivatives and quotient round trips. The
existing spherical/Cartesian equation CTests also pass. On 257 off-constraint
radial jets, normalized and independently expanded physical RHS agree within
2.85e-15 scaled. Quotient round-trip error reaches 3.52e-10 near r=.001; this
conditioning is measured rather than assumed harmless. The actual initial
jets change by at most 5.69e-14, and the core RHS remains nonzero. The default
compiled initializer, sources and RHS exactly match the v14 library.

## Resolution and failed control

All following r² runs use the shared compact b=.97 profile, the same central
20M scale, uniform grid, separated speeds, exact scri reference jets,
h_reference=.000125, analytic pole normal, KO=.5 and penalty=.6510416666660156.
The explicit timestep corresponds to initial C_char=.4. No floors are enabled.
The exterior indicator is the largest dimensionless physical H/M/Theta/Z
log-R RMS on the common initial-reference domain R/M=25–100.

| N | 10M exterior indicator | D differential defect | Clock residual magnitude | Horizon mass |
|---:|---:|---:|---:|---:|
|64|2.83955e-3|147.886|.213343|.312636|
|128|5.40649e-4|5.37362|.063864|.333268|
|256|5.37199e-5|5.93514|.006081|.334045|
|512|1.38384e-6|6.60158|.003328|.333204|

The finest three-grid field ratios correspond to orders 3.54–5.08; lapse,
shift, Khat, Lambda and D are approximately fourth order (3.86, 3.91, 3.97,
4.00, 4.04). Interpolation sensitivity and all field differences are saved in
`reference-chi-spatial-v16`. These are useful resolved exterior differences,
not a proof that every boundary quantity converges. In particular, the
differential D defect does not decrease over N128/256/512 even though exterior
constraints improve sharply. The N512 pole maximum is .00506.

The r⁴ control completes the short 1M pilot but fails near r=.035294 at
6.70149M on N128. Halving its timestep fails at 6.70344M at the same node.
Both failed stages are retained. This representation is rejected for the
tested long-time transient; normalization alone is not a stability mechanism.

## Perturbations and characteristic probes

The runner now supports compact-support C-infinity pulses strictly inside the
domain. Gauge pulses add the same bump to lapse and radial shift; constraint
pulses add it to independent Theta and Lambda. The bump is
`exp(1-1/(1-z²))` for |z|<1, zero otherwise, with z=(r-.9)/.025 and amplitude
1e-5. Reference fields, derivatives and sources remain unperturbed. Algebraic
constraints are retained, but geometric constraints are not solved or reset.
The pulse is mixed; it is not claimed to be a pure outgoing eigenmode.

Optional probes save total fields and radial derivatives at actual fixed grid
nodes every accepted step. Paired baseline/pulse runs at identical timesteps
allow subtraction of the evolving background. `analyze_pulses.py` projects
the resulting first-order principal variables onto local incoming/outgoing
eigenspaces of the full shared 26-field symbol. It uses transport speeds
`-eig(P)` for the convention U_t=P U_r, and groups repeated eigenspaces with
SVD before inversion. The initial implementation omitted repeated values
from groups; its failed analysis and source are preserved in the v16 response
directories. The corrected v17 analysis passes projector reconstruction and
has maximum eigenbasis residual below 8e-14 in these probes.

At r=.960784, coordinate speeds range approximately -.403 to 2.325. These
are measured mapped speeds, not unit-speed assumptions. Basis conditioning
reaches about 575. The projector norms are disclosed Euclidean diagnostics,
not a continuum energy or a unique reflection coefficient: the initial
inward component and bulk scattering must be distinguished from returning
boundary signals. Gauge-response amplitude doubling has relative L2
nonlinearity 1.74e-5 (absolute residual 2.41e-8). Plots are saved and inspected
for both gauge and constraint pulses.

Both pulse types complete 10M at C_char=.4 and .5, and 20M at .5. This does
**not** admit .5 as robust: the gauge pulse's exterior maximum field response,
dominated by the auxiliary/connection sector, grows from 1.36e-6 at 4M to
3.30e-5 near 10M and .02855 at 20M. The lapse/shift response at 20M is much
smaller, about 3.06e-5. The constraint pulse gives a maximum exterior response
2.06e-4 at 20M. The unperturbed 20M exterior indicator is .001305, with D
defect 6.35 and clock magnitude .1023. Halving C_char from .5 to .25 in both
baseline and gauge-pulse runs gives final response .0285496 instead of .0285479;
the response changes by only 5.68e-5 relative L2. The delayed growth is therefore
not explained by RK4 error. This does not distinguish a continuum source
instability from the spatial/boundary discretization.

At N128, the separate C_char=.4/.2/.1 temporal check gives maximum exterior
field differences 2.13e-9 and 1.88e-10. Chi has ratio corresponding to order
4.13, but several fields have mixed orders near this small error level. The
check bounds temporal contamination far below the spatial error without
claiming a clean all-field asymptotic temporal fit.
The N64 control at the same .4/.2/.1 targets resolves larger temporal
differences, 1.248e-7 and 7.533e-9, and gives orders 4.007–4.189 across all
12 physical fields. This establishes fourth-order temporal behavior for the
new representation at that resolution; it does not remove the boundary or
late-response failures.

## Penalties and frozen operators

At N128 and 10M, increasing the penalty coefficient from .65104 to 1 and 2
leaves the exterior indicator near 5.41e-4. The clock residual magnitude drops
.06386 → .04503 → .02529, but D defects are 5.37, 4.35, 4.79. Larger penalty
strength alone does not cure differential compatibility. These stronger-penalty
controls use explicit C_char=.1 to separate boundary accuracy from a tighter
RK4 limit. Paired gauge-response probes at penalty 1 use C_char=.2.
At r=.960784 their late incoming projected norm is 1.18675e-5, essentially
the same as 1.18685e-5 at penalty .65104; the stronger penalty has not removed
the returning signal. This remains a mixed-pulse, evolving-background proxy,
not a pure reflection coefficient.

The r² N64 initial and 10M semidiscrete Jacobians agree under perturbation-size
refinement to 2.33e-11 and 3.09e-11 relative. Largest real eigenvalues are
.64517 and .39875 in code-time units. The evolved leading mode is an
oscillatory pair. These positive modes and measured nonnormality prevent
calling the frozen RK4 decaying-spectrum bound a stability theorem. Full
radial/transverse/oblique symbols are also sampled on N512 states through 10M;
there are no numerically defective sampled clusters, maximum imaginary part
7.88e-16, and eigenspace conditioning reaches 3.65e5. These remain local
interior tests, not endpoint uniformity or nonspherical evolution evidence.

## Reproduction and remaining gates

Use the CMake Release build `build-spherical-v16`; its library SHA256 is
`666c8d7b0faf414aacc619abaa7108a2c3ec3613a79b9c04a1573606b97b9a15`.
The source/configuration/archive for every run is immutable in its evidence
directory. To reproduce a baseline, add `--chi-reference-power 2` to the
compact-profile command and select the recorded explicit timestep. For a
pulse add `--perturbation gauge|constraint --probes`; all pulse parameters
are recorded in the manifest. `test_reference_chi.py` verifies the adapter;
`analyze_chart.py`, `analyze_convergence.py` and `analyze_pulses.py` reconstruct
the saved comparisons and plots. Use fresh output directories.

The next substantive gate is the auxiliary/clock boundary system and late
gauge-response growth, with consistent damping/reference-source controls and
normal/characteristic boundary analysis. Preserve the r² spatial improvement
and the failed direct/r⁴ cases. Controlled serial cost-at-error measurements
and a robust operating point remain outstanding. No Aurora job or parent
campaign edit has been made, and no spherical result validates boosted/3D
stability.
