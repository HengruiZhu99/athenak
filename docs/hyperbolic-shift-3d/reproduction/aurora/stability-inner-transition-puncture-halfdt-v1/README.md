# V7 puncture timestep-halving control: first segment submitted

Fresh job **8984574** runs n16 V7 with **dt=.00025**, half the previous timestep,
from the same isotropic initial data to coordinate t=.5=1.5M. Spatial operators,
bulk KO.5, surface epsilon2, horizon settings, diagnostic cadence and all
physical acceptance limits are unchanged. It uses one CompactBinaryMerger/debug
node,12 MPI ranks and12 GPU tiles. The native expected step count is2001.

This is the first segment of the timestep control. The target is chosen to
permit a subsequent separately scheduled continuation past the known2.9M
failure while respecting the debug allocation bound. Only an independently
verified, scientifically accepted checkpoint may be used for continuation.
A failed or rejected segment will not be restarted. **This first segment alone
cannot settle the cause of the later lapse failure.** No result is available yet.

The previous V7 allocation8984472 is terminal, independently verified and
released before this submission. The retained plan/binding/submission and
preflight admission pin the source-v14 binary and initial data. Campaign
footprint at binding is3,838,943,547 bytes, with the declared output and atomic
checkpoint reserves fitting below5GB. No standard Z4c campaign is run.
