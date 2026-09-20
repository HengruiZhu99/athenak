# Evidence archive

This directory contains small evidence and reproduction sources from the isolated boundary investigation. Full application logs, binary stage snapshots, immutable executables and mode seeds remain in the local review directory recorded in archive-manifest.json. No executables or multi-gigabyte stage arrays are committed.

The frozen Cartesian matrix scripts are portable within this repository and require NumPy/SciPy. Run verify_operator.py from its directory to keep generated verification.json there. The archived copy was independently checked after relocation; archive-verification.json records that pass. first_order.py imports the previously committed radial continuum operators from evidence/modes/constraint-lower-order.

Complete-timestep replay and original pilot-launch scripts intentionally retain the immutable executable/seed paths and hashes used for the measurements. They are forensic records, not commands to submit production jobs. Adapt those paths explicitly when reproducing on another machine. source-v2/full-map-hook.patch is the isolated diagnostic hook; it is not applied to production src/main.cpp.

Exact-zero/stage-repeatability passes are separate from perturbation stability. The candidate fails the vacuum stability gate and must not be selected for a star or spinning evolution on the basis of these stage tests.

Original run inputs, histories and the diagnostic patch retain their exact whitespace for provenance. Whitespace-only diff warnings in those records are expected; they do not alter parsing or patch contents.
