# LG1: move the gauge transition outward, keep damping fixed

The active goal is to diagnose the DO1 constraint growth and admit N24 only
following reviewed N16 accuracy success. The damping-only DG1 run is still
running. At the matching4.65M diagnostic its full Hamiltonian norm is.331950,
compared with DO1 .335618 and IT1 .065295; both DG1 and DO1 lose horizon acceptance.
This provisional result indicates that stronger damping alone does not remove
the growth. Final10M common-domain analysis remains required.

LG1 changes only the inner endpoint of the gauge blend, from.2 to.4, retaining
outer endpoint.9 and C4 smoothstep9. The physical telegrapher gauge is exact
throughr=.4. The previous restored IT1 constraint-damping interpolation remains
fixed on.2--.56, including its donor enhancement; it does not move with the gauge.
All outer gauge terms, restoring collar and scri treatment are unchanged. Compared
with DG1, the only scientific change is this gauge-weight placement. Characteristic
sampling on preserved IT1/DO1 states is recorded in direct-outer-investigation-v1;
it does not prove global hyperbolicity or future evolution stability.

The new executable hyperbolic_shift_3d_mpi_late_gauge uses unique checkpoint magic
HYP3D_MPI_LG1,168-byte header,scheme3. It must not load DG1 or any old evolved state.
Fresh original N16 wormhole text,dt=.0005,targetcoordinate10/3=10M,direct chi/alpha
RK/KO,physical lapsefloor1e-10,positivechi floor1e-8 inside r<.03 and finite
nonpositivechi rescue everywhere remain unchanged. Keep original FD4,KO,grid,
angular epsilon2,scri scale1,horizon lmax12/ntheta24/tolerance.001 and diagnostic
interval.05. Retain negative/zero candidate counts and horizon quality flags.

The N16-only helper and controller reject N24,wrong schemas,modified dt/settings,
and failed/unadmitted checkpoints. Local point tests cover physical gauge rates,
independent .4--.9 placement,unchanged outer limit and fixed damping; device parity
also passes. Build/evolution validation and remote execution records will be added
when completed. This is a prepared experiment, not a validated formulation.
No LG1 remote job is submitted yet. Do not duplicate or interrupt the live DG1 run.

Local fresh LG1 reached coordinate.001 in2steps; its LG1 restart reached.0015
in3total steps. Both completed with null residual below1.3e-15. These smoke
checks establish executable/schema/restart operation, not duration accuracy.

All125 controller/campaign tests passed. Frozen scientific source commit:
e7c0b8ce2bdd639cfb9532b196fa5badb2b1c3b0. Remote build uses the existing
SYCL/PVC+MPI configuration,oneapi/release/2025.3.1 and make -j64.
