# Initial puncture evolution results (campaign ongoing)

Interface diagnosis remains stopped at the user's request. No further interface repair is included. The frozen issues remain failures, not satisfied prerequisites. User-directed puncture and subsequent binary screens are proceeding with ordinary transfer.

CPU intrinsic single-hole runs reached 2M with all stage health checks satisfied:

| Layout | Finest h/M | Cycles | CPU seconds | Full-domain H RMS | H RMS, .5<=r/M<1 | H RMS, 1<=r/M<2 |
|---|---:|---:|---:|---:|---:|---:|
| Uniform 32 cubed | .5 | 200 | 185.313 | .019401540 | .052945498 | .026473203 |
| Uniform 64 cubed | .25 | 238 | 991.802 | .006538425 | .054139724 | .006850018 |
| SMR root32, 120 leaves | .25 | 222 | 558.362 | .006685643 | .054139880 | .006850148 |

All domains are [-8,8]^3; the SMR interface is fixed at +/-4M. Regional and full-domain physical diagnostics use primary fields and independent leaf stencils; the complete 89-component results are supplied. The .5--1M annulus has not improved in the first uniform refinement. Lower full-domain RMS alone is not convergence evidence. The third resolution, signed-difference alignment and middle-resolution temporal controls remain in progress.

The coarse uniform timestep-halving control completed 2M in 401 steps (a tiny terminal step results from floating-point time accumulation), versus 200 steps. The combined primary-field RMS difference is 1.5077669813e-9; all-50 RMS difference is 2.5223378087e-9. This is not a substitute for the middle-resolution temporal test. Outflow puncture split/restart evolution is bitwise identical at .02M. Initial single-hole native values match an independent psi-based calculation to <=4.45e-16, and all 89 initial physical/reduction diagnostic RMS values match independent stencils to <=2.85e-16. Outflow Minkowski stays exactly constant.

GPU binary SHA256: 151d28e40f2014526e99aab6a0570265d3ca4b49c3f160215604e489c5bff5f4. CUDA smoke checks passed; uniform CPU/CUDA maximum normalized state difference is 3.3750821462e-14. Direct A10040 SMR continuation from the CPU 2M checkpoint targets 20M and is ongoing. CPU checkpoint SHA256: fdc10087319a12e8476bdb17822d169b16578d435081dd2d429357f3c5e4da19.

Scheduler routing: specifying partition gputest was rejected. A six-hour gpu-test request was silently classified gpu-short/partition gpu (job 13585391); it was cancelled while pending, with zero runtime and empty NodeList. Replacement job 13585475 requests four A10080 GPUs for one hour and is confirmed QoS gpu-test/partition gputest. A 55-minute aggregate application wall limit preserves checkpoints for continuation. Unscoped sacct returned a historical reused job ID from 2017; the live scontrol record is authoritative for this submission.

Binary preparation only: exact equal-bare-mass, nonspinning, zero-momentum Brill-Lindquist data, total bare M=1 and separation5, alpha=psi^-2. Complex-step gradients match all native initial fields to 4.45e-16. The source input is byte-identical to b81b44d6's archived long input. Reusing its geometric boxes yields 1464 leaves with counts 448/496/112/112/112/120/64 at physical levels1--7. The prepared native binary freezes this initial hierarchy; AMR, legacy tracker, and waveform hooks are not yet enabled for the 50-field layout. It is not a matched binary qualification, and no binary evolution has yet been run.

Implementation checkpoint 07434a28 is the single-hole/outflow source. The separately prepared head-on initializer changes initial data only. CPU production smoke/evolution binary SHA256 7bff8fb5ce2cddc9612b76742dde5d2c736704e0f2ba536d3f849b4e4149f8de is preserved as external athena-cpu; head-on initial-only binary has SHA2561292143e3b5c849d9477e4d40651e303556bf39d3dd62c07d6bd0f22ea4280e2.

Raw data, inputs, manifests and controllers: /Users/hz0693/research/pcgh-clean-reduction-tests-20260907/intrinsic-puncture-001 and /scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-intrinsic-mesh-001/puncture-001. The interrupted build and source timestamp repair are preserved; all physics runs were left untouched. The unfinished earlier prolongation control is not in these binaries.
