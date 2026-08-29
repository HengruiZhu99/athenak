# Aurora job 8790895 pre-build failure

Job `8790895` exited one after 32 seconds before configuration, build, GPU
mapping, or numerical work.  The original silent bootstrap emitted no failing
command.  It is recorded as an unresolved transient/bootstrap failure, not a
PVC or numerical failure.  Commit `68bc1ee3` added fail-line logging; the
otherwise unchanged rerun `8790897` passed the bootstrap and full requested
Phase-2/3 workload.
