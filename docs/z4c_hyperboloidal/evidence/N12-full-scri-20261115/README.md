# Full-overlap N12 rerun with scri recording

This directory records the requested fresh rerun of the latest validated
full-overlap binary. The target is approximately `60 M0`; `campaign.json`
contains its actual target cycle, step, source revision and executable hash.
The initial data, overlap strength/taper, dissipation, time step and optimized
GPU arithmetic are unchanged. `run.py` extends the copied input's time/cycle
limits and adds native scri output every `.01` code units (`.03 M0`).

## GPU admission

Job `9092194` compared the selected prior GPU executable against the cleaned
executable with recording enabled, for eight steps from both fresh initial
data and the original `26 M0` checkpoint. Both complete accepted restart
payloads were bitwise identical: **1,361,093,760 bytes per comparison**.
See `gpu-admission.json`. The isolated build reused only objects whose source
dependency hashes were unchanged; `build-receipt.json` records the
dependency audit, compile/link commands and executable digest.

`fresh-scri-admission.json` and `late-scri-admission.json` additionally compare
every recorded bulk, edge and endpoint value with its same-cycle checkpoint:
**423,026 scalar values per case**, all bitwise identical. Independent native
row and normal-lift reconstruction gave maximum errors of zero and
`1.55e-12`, respectively, below the `2e-11` admission tolerance. These checks
include all 24 MPI ranks.

The first science launch failed before evolution with a scheduler/process
launcher error, `No reply from shepherd after 108s` (exit 127). Job `9092227`
reused the successful admission and began the fresh evolution. Sequential
continuations use verified final checkpoints, with one science job active at a
time. Final segment and waveform receipts will accompany the completed run.

## Offline extraction

The reusable tools and conventions are documented in
[`scri-news.md`](../../scri-news.md). The analytic logs here were produced with
NumPy 2.1.3 and Python 3.11 on Aurora. The full pipeline recovered a known
quadrupolar news signal from angle-dependent native-time cuts with maximum
absolute error `1.7e-12`. This validates the manufactured extractor cases;
it does not establish convergence or vacuum regularity of the N12 evolution.

Bulk checkpoint and recorder archives remain at the remote root recorded in
`campaign.json`; they are not committed to Git. Scripts here are campaign
receipts with fixed Aurora paths, not general cluster submission commands.
The copied base input retains its historical limits; `run.py` supplies the
actual target and output block. Its inherited `configuration.json` is used
only to verify the initial-data manifest; `campaign.json` is the authoritative
description of this rerun.
