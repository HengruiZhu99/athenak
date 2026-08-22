# Native VC unsupported-feature inventory

The following boundaries are deliberate and fail closed before unsupported
data can be interpreted through cell-centered geometry.

## Physics

- Any nonvacuum VC run is rejected unless a separately qualified CC-to-VC
  matter adapter exists.  Hydro, MHD, radiation, particles, and other matter
  sources are not qualified with native VC Z4c.
- Cartoon SO(2) remains vacuum-only even in CC mode under this contract.

## Consumers

- Optional consumers that directly sample Z4c through CC geometry are rejected
  in VC mode unless they use an explicit centering-aware adapter.
- Cartoon m=0 FastFlow is the documented exception: it uses the explicit
  VC-to-CC ADM adapter and a centering-aware lapse sampler.

## Restart

- Native VC restart is supported only when immutable centering/layout/carrier
  metadata match.
- CC-to-VC and VC-to-CC state restarts are rejected.  AMR-history
  `cell_to_vertex` compatibility is topology-only and does not convert state.
- Legacy restart files without VC carrier metadata are interpreted as CC.

## Output

- Qualified Cartoon output types: `tab`, `hst`, `log`, `vtk`, `pdf`, `bin`,
  and `rst`.
- Cartoon `cart`, `sph`, `cbin`, `pvtk`, and `trk` outputs are rejected before
  output construction.
- Native VC VTK is point data.  A consumer that assumes cell data must convert
  explicitly.

## Dimensional and mesh contracts

- Cartoon SO(2) requires `nx3=meshblock/nx3=1`, exact `x1min=0`, and `axis` only
  at inner x1.
- Multilevel Cartoon requires at least `2*nghost` x1 intervals per MeshBlock so
  coarse axis parity ghosts have local authoritative data.
- O2/O4/O6 require the corresponding validated ghost widths.

## Scientific qualification

- The fixed-grid O4 bounded interval is qualified; common-tree Brill collapse
  is not.
- No Figure-3 reproduction, critical collapse, horizon, matter coupling,
  long-time stability, or production-readiness claim follows from this branch.
