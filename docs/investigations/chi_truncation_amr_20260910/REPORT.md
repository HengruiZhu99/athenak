# χ truncation-error AMR series — in progress

Source branch codex/vc-chi-truncation-amr-20260910; implementation SHA4041c73a73eb8a8ea1f2e787623b167b7c2a6ff7, committed/pushed. Immutable executable SHA256 bd6a029f3d6fb91087ac0b5b6c72d24f410bccac8168def9eee399067b1af5e9.

The paper-inspired fourth-order derivative-error proxy estimates first- and second-derivative errors from D5χ/D6χ. Active-only seven-point stencils avoid lower-order ghosts. A fixed normalization length1 combines the two dimensional quantities. This is not full-evolution Richardson error. It does not provide a rigorous global error bound.

Analytic Taylor and C++ stencil tests passed, including shifted stencils and Nyquist response. CUDA build passed. dχ trajectories were exactly unchanged through t6; bootstrap trajectory is identical through t5. A final exact-executable GPU check58148426 COMPLETED0:0 reproduced the selected calibration history exactly. Initial ghost-inclusive and over-strict tolerance probes failed or were cancelled for excessive refinement; their evidence is preserved.

Chosen reference tolerance1e-4 at root nx1=64; N128/N256/N512 effective tolerances1e-4,6.25e-6,3.90625e-7. Fixed dχ bootstrap until t5. At t6 the selected N128 calibration had317blocks/maxlevel7 versus254/maxlevel6 for dχ. This selection preceded all late-time TE classifications.

All production cases use A=-0.04896875, identical spectral initial coefficients, and unchanged CFL.15,diss.50,telegraph lapse/zero shift/zero constraint damping, corrected CPBC+linear outer ghosts,device synchronization,liveAMR,physicalradiusfloors and lapse stopping criteria. Root grids64x128/128x256/256x512; meshblocks16²/32²/64². Targett200 unless a native lapse stop occurs. No evolution crash is reclassified or restarted automatically.

N128 TE job58148434 RUNNING shared_interactive. N256/N512 shared submissions were rejected by the two-job-per-user QoS limit; preserved logs record that. Both were packed into interactive allocation58148544, currentlyqueued, with independent one-GPU steps. Original dχ N512 job58146650 remainsrunning. N128 dχ completed as lapse recovery at74.2801; N256 dχ failedat58.23184. The bisection and its monitor remainpaused.

Full convergence comparison is pending completion of the runs. [Partial histories](comparison.png) and [mesh comparison](mesh_comparison.png) are snapshots only. Do not treat successful submission or short qualification as established convergence. Remote runroot:/pscratch/sd/h/hzhu/chi-truncation-amr-20260910/production.
