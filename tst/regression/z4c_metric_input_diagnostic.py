#!/usr/bin/env python3
"""Check that enabling pre-C2P metric inspection leaves finite evolution unchanged."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve()
    repo = Path(__file__).resolve().parents[2]
    template = (repo / 'analysis/tde_stability/inputs/vacuum_pulse.athinput').read_text()
    template = template.replace('nx1 = 32', 'nx1 = 16')
    template = template.replace('nx2 = 32', 'nx2 = 16')
    template = template.replace('nx3 = 32', 'nx3 = 16')
    template = template.replace('min = -4', 'min = -2').replace('max = 4', 'max = 2')
    template = re.sub(r'(?m)^tlim\s*=.*$', 'tlim = 5', template)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, enabled in [('off', 'false'), ('on', 'true')]:
        path = args.output / name
        path.mkdir()  # Refuse to overwrite existing run data.
        (path / 'input.athinput').write_text(template.replace(
            '<mhd>', '<mhd>\ndebug_metric_before_c2p = '+enabled))
        with (path / 'run.log').open('w') as stream:
            subprocess.run([str(exe), '-i', 'input.athinput'], cwd=path,
                           env=dict(os.environ, OMP_NUM_THREADS='2', OMP_PROC_BIND='false'),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
    checks = {}
    for name in ('ks_background.user.hst', 'ks_background.z4c.user.hst',
                 'ks_background.mhd.hst'):
        checks[name] = ((args.output/'off'/name).read_bytes() ==
                        (args.output/'on'/name).read_bytes())
    restarts = list((args.output/'off/rst/rank_00000000').glob('*.rst'))
    assert restarts, 'Expected per-rank double-precision checkpoints'
    for path in restarts:
        paired = args.output/'on/rst/rank_00000000'/path.name
        left, right = path.read_bytes(), paired.read_bytes()
        marker = b'<par_end>\n'
        assert left.count(marker) == right.count(marker) == 1
        # The parameter header intentionally differs; all binary metadata and
        # evolved fields, including ghosts, must match exactly.
        checks[path.name] = left.split(marker, 1)[1] == right.split(marker, 1)[1]
    log = (args.output/'on/run.log').read_text()
    checks['no_invalid_metric_event'] = 'C2P_INVALID_ADM_INPUT' not in log
    checks['no_invalid_active_state'] = 'Z4C_INVALID_STATE' not in log
    (args.output/'results.json').write_text(json.dumps(checks, indent=2)+'\n')
    assert all(checks.values()), checks
    print('PASS: histories and full checkpoint payloads are unchanged')


if __name__ == '__main__':
    main()
