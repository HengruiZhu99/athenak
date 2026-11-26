# Radial profile of the constraints in the history output

The history output of the hyperboloidal module can report the constraints in an arbitrary set of
bands in compact radius, to resolve their radial dependence without field output.

```
<output1>
file_type = hst
hyperboloidal_radial_edges = 0.004,0.01,0.02,0.05,0.1,0.3,0.6,0.9,0.97,1
```

The value is a comma-separated, strictly increasing list of edges within [0, 1]. Each pair of
consecutive edges defines a band `[edge_i, edge_{i+1})` and adds one history file
`<basename>.radial-<iii>.hst` with the columns of the `scri-shell` histories: the coordinate volume
of the band, the volume-weighted RMS of H, M, Theta and Z, and the RMS and mean differences of
chi, lapse and normal shift from the initial reference. The header records the band edges.

- All evolved points in the band are included, the interiors of the horizons too (the exterior mask
  of the main history is not applied). L2 norms follow as `rms * sqrt(Volume)`.
- A band that contains no grid point is written with zero volume and zero values.
- The option is off unless the parameter is present; existing decks and outputs are unchanged.
- On a restart the parameter is supplied with `-i` like any other new parameter. A restart with
  `nlim` equal to the restored cycle writes one row for the restored state without stepping.
