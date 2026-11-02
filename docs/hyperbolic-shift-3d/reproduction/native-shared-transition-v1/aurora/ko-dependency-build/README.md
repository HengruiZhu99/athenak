# Independent KO10 donor-repair build

The three namespace-adapted files and package provenance here exactly match the uploaded replacement launcher. Other launcher files are unchanged from the source commit recorded in package-provenance.json and individually hash-bound there. The native source is commit e7d55b008c86774580386dbc3e13028fa72b7e24, with pinned Kokkos; the only scientific change from the preceding controls is the MPI donor dependency reach.

The remote build is detached on aurora-uan-0012, PID2584700, under the existing guarded build wrapper. Read remote ko-dependency-build-start.json, native-transition-state.json next_build and build-logs/native-transition-ko-dependencies-v1/result.json before further action. Existing builds remain immutable. Actual GPU hashes will only be known when the build completes and freeze succeeds. Current scope is compilation plus fresh repaired C/D/E GPU parity and matched N16 controls, not a new resolution campaign.
