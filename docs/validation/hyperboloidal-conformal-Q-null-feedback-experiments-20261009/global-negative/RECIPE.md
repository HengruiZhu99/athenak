# Held conformal-Q/null-feedback full-tensor recipe

This fresh directory is reserved for the distinct candidate from
`continuum/conformal-q-null-feedback`. Source preparation copies the original
frozen C0 templates and records their hashes. It does not bind an unfrozen
helper, compile an executable, export a matrix, or propagate a state.

Admission requires the final immutable local-gate index, exact helper and
dependency pins, all-command success and unchanged input checks, plus an
independent review from the literature agent or root. A future binding script
must enforce these requirements before compilation. The candidate is separate
from C1, prescribed/live kappa2 profiles, and earlier lapse/trace controls.

## Fixed comparison

- N16, cube span 2.2, S=1, a=.5, reference transition .05 to .95.
- Gauge cutoff .45 to .85; candidate null feedback cutoff .85 to .95,
  sigma=5, physical-inner lapse blending enabled.
- C0, kappa_input=10, kappa2=0, symmetric quadratic strict-interior,
  nonrecursive spherical continuation, native KO=.1.
- Original point-major active k/j/i free20 ordering, reference lift/restrict,
  original gauge-pulse and shell seeds, and all original grid/mask/stencils.
- Compare against the original C0 spatialnorm operator and saved histories;
  no original output is regenerated or overwritten.

The planned API is `qnf::Gauge(p,u,g,qnf::Parameters{.85,.95,5,true})` and
`qnf::Assemble(parts,Omega,rhs)`, with `g.physical_trace_lapse=false` and
`g.preferred_source=true`. This remains provisional until the final frozen
helper is verified. Only alpha is blended between physical-P and conformal-Q
lapse formulas. The candidate also replaces the spatialnorm shift source with
the preferred-Q regular shift source and an explicit null-feedback beta pole.
The geometric P storage/evolution and C0 damping remain original.

## Binding and source identity gates

1. Pin the immutable gate index/receipt, shared helper and all included private
   math headers. Verify every indexed small file and every gate input against
   its saved hash. Preserve the independent review and root admission.
2. Replace both copied gauge wrappers consistently. Do not compose the old
   spatialnorm wrapper with the new preferred-Q gauge. Both local cached Point
   methods and the actual copied CartesianPatch RHS must call the exact same
   candidate helper. Retain the existing reference subtraction lifecycle and
   use the same candidate gauge for live and reference evaluations.
3. Explicitly assemble the beta pole once. The original production assembler
   does not add it. Preserve geometric regular/simple/double-pole assembly;
   no floors, SPD repair, algebraic ghost projection, or Lambda reconstruction.
4. Save exact copied headers, unified patches, compiler version/flags, all
   compiler dependency hashes, link archive hashes, executable hashes, launch
   HEAD and runtime implementation 27c19d20. Confirm untouched production.

## Actual full22 and reference validation

Run the original Prepared reference/strict-donor/ghost-weight audit and actual
RHS/Jv amplitude controls on all22 fields, including algebraic normal
directions. Validate native projection derivative, free20 lift/restrict and
actual ghost interpolation. Compare raw22 cache action against actual native
RHS centered differences using multiple amplitudes and smooth/random/shell
seeds. Reference stationarity must remain at roundoff, with native initial
H/M/Z and donor/ghost support recorded.

Derive the exact local gauge-row Jacobian difference from the final helper at
the analytic reference and independently build its sparse global action.
Only alpha/beta rows may change; all geometric rows must match C0. Verify which
value columns change, exact inner/outer behavior, all derivative-slot entries
and the final gate's principal claim. Do not infer this from a finite matrix
subtraction alone. Include the reference geometry dependence in the null
residue, determinant-one tangent lift and the preferred shift extension.

For the stage lifecycle, retain raw22 `A`, the exact reference lift `L` and
restriction `P`. Compare the actual nonlinear one-step derivative against
`P R3(dt A) L`, with projection only after the native final SSPRK3 stage.
Also compare with `R3(dt P A L)` at nominal, half and quarter dt. These maps
are distinct; propagation of `P A L` is projected continuous evolution, not
the exact finite native step. Record state-unit residuals and per-time units
separately.

## Propagation sequence and honest limits

After all source/Jv/stage/attribution gates pass, run a short independent
canonical Taylor action at t=.025/.05 against the inexpensive Arnoldi pilot.
Use the exact constant configuration 1/h similarity if needed, then undo it
before every comparison and diagnostic. Retain local pair errors, actual
Arnoldi residuals, finite checks and guards.

Then run the inexpensive projected-continuous t2 screen for the original
gauge and shell seeds. Compare every sampled native H/M/Z, Theta, reference
contracted field-group value/derivative norm, component content and radial
localization against the saved C0 histories. The field-group norm is not a
physical energy or a proved symmetrizer. If t2 is not catastrophic, extend in
a separate fresh output path to t6 with an exact saved-prefix comparison;
t2 transient differences alone do not classify late growth. Stop and retain
receipt/data on overflow or a 1e12 norm guard.

No native evolution, global eigenvalue certification, physical energy bound,
continuum instability claim, scri regularity acceptance or production adoption
is authorized by this recipe. Root owns independent native preflights.
