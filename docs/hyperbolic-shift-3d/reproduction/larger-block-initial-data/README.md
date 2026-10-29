# Larger-block puncture initial data

Prepared locally using the unchanged `initial_data.py` donor and exactly the n16 native library. No evolution, upload, or checkpoint admission was performed.

| Grid | Active cells | Initial text bytes | Preparation seconds | Min chi | Min lapse | 2049/4097 shift difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n24 | 262656 | 47640768 | 6.27 | 1.4095923e-4 | .0078604084 | 6.1772531e-6 |
| n32 | 622592 | 112446689 | 11.38 | 5.0656436e-5 | .0047124483 | 5.6847572e-6 |

Both use M=1/3, 96-node isotropic-wormhole quadrature, a 4097-node initial shift, and the same donor configuration as n16. Values are finite; chi is positive. The fixed 19-block order comes from `checkpoint_export.coordinates`: core half-width .125, shell radii .45/.75/1. At n16 it reproduces native x/y/z/r bit-for-bit; its largest coordinate-volume difference is 1.0842e-19 and volume is unused by the initial-data generator.

The raw local folders under `output/hyperbolic-core-shell/initial-data-larger-blocks-v1/` contain `coordinates.csv`, `initial.txt`, the generator's `initial.json`/`initial.log`, and `provenance.json`. This small committed package retains the JSON metadata and provenance, binding generator, imported donor dependencies, library, coordinates, initial data, configuration and n16 provenance. `summary.json` aggregates their hashes. Initial files have a leading scalar-count line followed by one 31-field row per active cell, in block/k/j/i order with i fastest.

The recorded shift difference compares the same 2049/4097 quadratures at the grid's sampled radii. Its larger sampled maximum relative to n16 is not a changed donor configuration. No initial-data convergence claim is made. Large text files remain local.
