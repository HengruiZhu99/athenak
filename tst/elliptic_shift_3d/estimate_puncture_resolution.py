#!/usr/bin/env python3
"""Geometry-only lower-bound estimate; allocates no mesh and forecasts no speedup."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
n=8;half_width=.25;radius=4.0394e-7;target=radius/8
levels=math.ceil(math.log2((half_width/n)/target))
blocks=8+56*levels+6*7;cells=blocks*n**3
out=dict(scope='Illustrative fixed-origin nested-octree geometry estimate, not a validated collapse grid or native performance forecast',
 radius_of_recorded_t10_flux_peak=radius,assumed_cells_across_peak_radius=8,
 unresolved_feature_width='Unknown; radius is not width. A narrower front or continuum endpoint can require more refinement.',
 cells_per_block_axis=n,base_cartesian_blocks=8,shell_layers=7,
 base_cartesian_cell_spacing=half_width/n,additional_center_refinement_levels=levels,
 finest_spacing=half_width/n/2**levels,leaf_blocks=blocks,active_cells=cells,
 proposed_pilot_budget_cells=50000,cell_count_ratio_to_budget=cells/50000,
 minimum_bytes_for_one_22field_state=cells*22*8,
 minimum_bytes_for_49_threecomponent_krylov_vectors=cells*49*3*8,
 omitted_memory='ghosts, interpolation rows, metric jets, multigrid levels, RK registers, communication, and other working arrays',
 current_dense_coarse_guard_cells=1024,current_n2_per_leaf_coarse_cells=blocks*8,
 missing_algorithm='Global AMR topology coarsening and transfers are needed; retaining all refined leaves exceeds the current bounded dense coarse solve.',
 conditional_cpu_cost=dict(measured_pilot_cells=13312,measured_two_step_restart_seconds=60.438802584,
   same_diagnostic_linear_per_cell_seconds=60.438802584*cells/13312,
   qualification='Only arithmetic extrapolation for the same short diagnostic, with equal iteration counts assumed. No timing measurement at this size, no scaling claim, and no extrapolation to a validated t10 timestep.'),
 action='No allocation, large grid, GPU job or remote run requested or started. Continue local source/derivative and hierarchy fixes first.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
