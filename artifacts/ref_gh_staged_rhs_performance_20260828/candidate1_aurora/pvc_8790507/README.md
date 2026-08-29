# Candidate 1 Aurora PVC discriminator 8790507

Status: **FAIL**.  No performance benchmark was launched.

At commit `7294f2210770bdb3f04a8f61c4870d552f57c300`, the compact atomic q
accumulation passed at time zero and in the first evolved stage.  The complete
first-stage reference update, Ref-GH RHS, RK update, communication, and
prolongation also passed.  PVC then raised a `NotPresent` device write in the
analytic projected-metric physical-boundary kernel, before its fence.  The
eight-rank cycle was not attempted.

This distinguishes the corrected q portability issue from a second private
memory problem in the physical-boundary projection.  The boundary kernel
returned a 50-Real projected state while also retaining coordinate metric jets;
the next correction stages the same point algebra directly into the state view.

Remote campaign and retained large build:

`/lus/flare/projects/CompactBinaryMerger/hzhu/refgh_staged_rhs_20260828_candidate1_qdevice_7335058c`

