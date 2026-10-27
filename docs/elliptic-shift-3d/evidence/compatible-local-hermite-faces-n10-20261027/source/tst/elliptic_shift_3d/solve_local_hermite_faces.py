#!/usr/bin/env python3
"""Same Hermite gauge with exact per-face bubble-block preconditioning."""
import runpy
from pathlib import Path
import enriched_outer_gauge
from hermite_face_preconditioner import HermiteFacePreconditioner

enriched_outer_gauge.EnrichedPreconditioner=HermiteFacePreconditioner
runpy.run_path(str(Path(__file__).with_name('solve_local_hermite.py')),run_name='__main__')
