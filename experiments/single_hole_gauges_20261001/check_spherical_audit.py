#!/usr/bin/env python3
"""The spherical audit must reject uncovered horizons and changing masks."""
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch
import advance

case=json.loads((advance.WORKFLOW/'cases.json').read_text())['cases'][0]
with tempfile.TemporaryDirectory() as tmp:
    run=Path(tmp)
    (run/'input.athinput').write_bytes((advance.WORKFLOW/'inputs'/(case['name']+'.athinput')).read_bytes())
    (run/'run.log').write_text('Terminating on time limit\n')
    (run/'test.hst').touch()
    (run/'horizon').mkdir()
    (run/'horizon/BHaHAHA_diagnostics.ah1.gp').touch()
    volume=4*math.pi/3*(16**3-2**3)
    row=[0.0]*20;row[14]=volume;row[17]=volume*1.0001
    rows=[row[:],row[:]];rows[1][0]=4
    horizons=[[0,0,0.7,0,0,0.1,0.6]]
    summary=dict(exit_status='0',evolution_time=4,successes=1,failures=0,
                 pending_searches=0,reported_successes_above_rms_tolerance=0)
    def read(path): return rows if path.suffix=='.hst' else horizons
    def audit():
        with patch.object(advance,'analyze',return_value=summary.copy()), \
             patch.object(advance,'numeric_rows',side_effect=read):
            return advance.audit(run,case)
    assert audit()['passed']
    rows[1][16]=1000  # Violations confined inside excision do not invalidate the exterior.
    assert audit()['passed']
    rows[1][19]=.01
    assert not audit()['fixed_safe_shell']
    rows[1][19]=0
    rows[1][17]+=1
    assert not audit()['fixed_safe_shell']
    rows[1][17]=rows[0][17]
    horizons[0][6]=1.4
    assert not audit()['horizon_excised']
    horizons[0][6]=.6
    rows[1][18]=.01
    assert not audit()['fixed_safe_shell']
print('6 spherical audit checks passed.')
