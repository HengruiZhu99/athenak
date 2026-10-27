#!/usr/bin/env python3
"""Same all-node Hermite equations with a full-action boundary-aware coarse solve."""
import runpy
from pathlib import Path
import enriched_outer_gauge
from enriched_coarse_preconditioner import EnrichedCoarsePreconditioner

original=enriched_outer_gauge.EnrichedPreconditioner
def coarse(enriched,base):
    triangular=original(enriched,base)
    result=EnrichedCoarsePreconditioner(enriched,base,triangular,degree=1)
    result.diagnostic={'full_coarse':result.diagnostic,'triangular':triangular.diagnostic}
    return result
enriched_outer_gauge.EnrichedPreconditioner=coarse
runpy.run_path(str(Path(__file__).with_name('solve_local_hermite.py')),run_name='__main__')
