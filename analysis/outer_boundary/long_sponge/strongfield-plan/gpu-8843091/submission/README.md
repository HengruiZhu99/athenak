# Launch copy: strong-field SMR vacuum control

Derived from the committed prepared package at85d99c3f. Physics, mesh and immutable executable are unchanged. Restart cadence is50M, full active Theta binary cadence5M, and constraint binary cadence10M, all per rank. This supports spatial localization and validated recovery from a finite earlier checkpoint if needed; no automatic restart is enabled.

Aurora hostname, Flare path, queue limits, executable SHA, MPI wrapper and Python/NumPy availability were rechecked after access renewal. Account MHDTidal, debug,2nodes24ranks,1hour. Fresh3cycle zero gate must pass all24rank checks before fresh pulse target1000M with -t00:55:00. Final stop must be classified explicitly. This is the reduced-damping configuration using original zero_rate, not the new linear exterior-memory boundary.

The original prepared documentation and its hash manifest are preserved in the repository and prepared-package-manifest.json. Local CPU/validator checks are not a24rank GPU evolution pass. No job has been submitted by merely creating this launch copy.
