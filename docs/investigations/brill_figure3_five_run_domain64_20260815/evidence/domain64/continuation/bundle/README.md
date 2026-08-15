# Enlarged-domain zero-shift restart continuation

This is a one-case continuation of the enlarged-domain zero-shift Brill-wave
control from Perlmutter job `57004715`.  That job ran its fixed-shift case to
the expected strict-chi endpoint, then ran the zero-shift case alone until its
inner `srun --time=01:55:00` limit expired at coordinate time
`7.42498168945451`.  The timeout was operational: the zero-shift case had not
reached its original-domain endpoint or a numerical fatal condition.

This successor therefore resumes only the zero-shift case from its latest
complete authenticated restart (`.00163.rst`).  It does not rerun the fixed
case.  It requests exactly one `shared_interactive` node allocation with one
A100-SXM4-80GB, one MPI rank, and 32 CPU cores, and launches exactly one
numbered science step.  The step gets `03:45:00` inside a `04:00:00`
allocation.  No other campaign run may overlap it.

The resumed case preserves the exact v7 executable/source, IrisK
`A=-0.047` initial data, enlarged domain `rho=[0,64]`, `z=[-64,64]`, base
spacing `0.25`, N128/O6/RK4 setup, pre-collapsed lapse, scale-invariant
telegrapher lapse `(tau,kappa)=(1,1)`, zero shift, zero Z4c constraint
damping, KO dissipation `0.5`, `dchi_max=0.02`, levels 0--20, strict positive
chi policy, and target `t=20`.  Only the output basename and continuation
directory are new.

The login preflight authenticates the terminal v7 root and detached
manifests, settled accounting, comparison JSON, zero-shift history/log, exact
restart bytes, source tree, executable, and coefficient file before promotion.
The compute wrapper repeats those checks before launching the sole step.  A
failure is retained as evidence; no capacity, floor, threshold, gauge, or
physics adjustment and no adaptive retry is authorized.

For final plotting, the v7 zero-shift history before the restart and this
continuation history are one scientific curve.  The merger must discard the
overlap at the restart time, retain exact provenance for both segments, and
must not describe the continuation as a sixth physical configuration.
