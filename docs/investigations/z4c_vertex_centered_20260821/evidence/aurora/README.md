# Aurora SYCL evidence

Two bounded PVC campaigns are retained.

- `v2-dualview-failure/` is the first run at source `684402ae...`.  Tests
  1--16 pass and the first dynamic-AMR test faults after the accepted refine
  transaction.
- `v3-final-source-failure/` is the one authorized retry at exact production
  source `5d37b5e5...`.  It has the same 16-test prefix and the same production
  phase failure despite the narrower device-view capture.

Each subdirectory contains the campaign's original absolute-path
`SHA256SUMS` and detached checksum.  Verification after relocation replaces
only the original campaign-root prefix with the local subdirectory; all
payload hashes remain exact.  The `qstat-final.txt.local-refresh` file in v3
was collected after PBS moved the job to historical state and is therefore
bound by the repository-level manifest rather than the original in-job raw
manifest.

The final-source failure is a qualification result.  It is not a basis for a
second blind source repair or for relabeling the host/CUDA numerical evidence
as SYCL evidence.
