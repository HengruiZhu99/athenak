# Completed principal N1025 duration trial

The fresh attenuation candidate finishes coordinate t500 after10,240,000 steps,
with no restart, parameter change, or relaxed guard. The final checkpoint agrees
exactly with the last saved profile; both the worker and dispatcher have exited.
`assessment.json` passes the unchanged predeclared budget over all10,001 saved
states, including the initial transient. Final relativeAH error is1.14804e-6;
its maximum is2.64272e-6. Exterior RMS is normalized over the radial weights
outside the instantaneous apparent horizon. Interior errors remain recorded
but are excluded from that budget. No positive-radius evolution nodes are excised.

The final exterior(H,M,Theta,Z) RMS is(1.79430e-7,2.43679e-6,4.99922e-8,5.21145e-7).
Raw inner lapse profiles retain five negative neighboring differences; the maximum
fractional change between25 and500 is3.07971e-4. Initial flooring and later RK
clipping are separately quantified in floor-accounting.json and its NPZ arrays.
At500 the passive inward rays coalesce at subnormal binary64 radius3.7233e-320;
ray-resolution.json documents the loss of diagnostic separation without asserting
continuum characteristic crossing or noncrossing. This does not feed back into
field evolution.

This is a single-grid duration-and-accuracy result, not a convergence or continuum
stability proof. The N513 control also reaches500 but has0.109% finalAH drift and
is not the principal accuracy certificate. Historical unfloored, floor-only,
and failed repair controls retain their separate inputs, bindings, and outcomes.

summary.json.gz contains the exact, unrounded, unthinned diagnostic JSON bytes;
the redundant raw summary.json is retained locally but excluded from Git and the
portable package. The collector validates original full profiles and saves their
hashes; profiles.npz contains selected raw curves, not every full evolution state.
The assessor and plot script accept either summary representation. Source hashes,
protocol, stage extrema, process/checkpoint verification, and terminal scalar
measurements are retained alongside the histories.
