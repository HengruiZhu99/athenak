# V7 puncture timestep-halving control: first segment passed

Aurora job8984574 completed n16 V7 with dt=.00025 through coordinate.5=1.5M,
2001 accepted steps. All46 predefined checks passed and were independently
replayed; all worker artifact hashes and all four finite checkpoint arrays
were verified. PBS finished with exit0 and the allocation was released.

The horizon was found with radius.06423476739, mass.33330139565 and expansion
RMS.00063235288. Minimum lapse was.04229094. Maximum surface/ghost null errors
were1.11e-15/1.23e-15. This is a bounded early-time result, not evidence of
stability through the known2.9M failure or through100M.

The same source-v14 V7 binary, initial data, spatial operators, bulk KO.5,
surface epsilon2, horizon settings and physical limits as the full-dt control
were retained. One CompactBinaryMerger/debug node used12 MPI ranks and12 GPU
tiles. The independently accepted checkpoint and horizon seed are explicitly
hash-bound inputs of continuation job8984609, target coordinate1=3M.
No rejected state is resumed. Raw checkpoints remain outside Git.
