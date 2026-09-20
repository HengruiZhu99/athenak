# Direct Theta constraint-pulse GPU pair

A new independent diagnostic pair; no previous checkpoint is restarted. Each case has its own3-cycle exact-zero GPU gate. Primary: shift eta0.02/lapse residual damping0.01. Ablation: eta0.02/lapse damping0.1. Both kappa1=0, radial sponge start512/ramp1280/rate0.001, full Minkowski cube±2048/global64³/eight32³blocks, dt3.2, target50000. Primary wall cap28minutes, ablation25minutes, each zero gate1minute; one-node/eight-rank/MHDTidal/debug allocation1hour.

The seed is the regular centered monopole Gaussian Theta=1e−6 exp[−r²/(2·384²)]. Its Gaussian tails overlap the smooth sponge ramp;30.93% of initial proper-volumeTheta² lies outside the core. This differs from A/B's compact lapse bump.

A history-only radius512 spherical mask splits core Theta-int2 from exterior Theta-norm every10time units. It does not excise or modify any evolved field. Theta-norm/Volume gives EXTERIOR RMS. Theta-int2 is the CORE squared integral; normalization by initial core volume must be labeled explicitly rather than called instantaneous core RMS. Full checkpoints and active regional profiles provide whole-domain quantities separately.

Final input hashes match the independently validated named coremask inputs. CPU masked/unmasked3-cycle full payloads were bitwise identical;8-rank zero gates exact; dt3.2 versus1.6 through320 had full residual relativeL2 difference8.405e−8 and Theta relativeL2 difference1.006e−7. Long GPU results are pending. Every1000 and final checkpoints are per-rank and checked across all8ranks, including all fields and ghost metric SPD.

The actual frozen outputs are per-rank checkpoints every1000, full Theta bins every64 and constraint bins every128 code-time units. Reported throughput includes this I/O.
