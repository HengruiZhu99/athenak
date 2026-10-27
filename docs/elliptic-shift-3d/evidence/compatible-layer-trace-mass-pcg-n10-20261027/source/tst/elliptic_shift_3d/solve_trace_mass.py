#!/usr/bin/env python3
"""Same layer-affine gauge experiment with a trace-norm block preconditioner."""
import runpy
from pathlib import Path
import mapped_sbp_preconditioner
from trace_mass_preconditioner import TraceMassPreconditioner

mapped_sbp_preconditioner.MappedPreconditioner=TraceMassPreconditioner
runpy.run_path(str(Path(__file__).with_name('solve_layer_affine.py')),run_name='__main__')
