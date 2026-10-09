"""Record the two small advisory algebra checks; constructs no PDE operator."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = [P/'check_identities.py', P/'check_boundary_principal.py',
           P/'ASSESSMENT.md', P/'FINITE-RADIUS.md', P/'VARIATIONAL-SAT.md']
before = {str(p.relative_to(ROOT)): sha(p) for p in sources}
commands = []
start_all = time.perf_counter()
for name in ['check_identities', 'check_boundary_principal']:
    cmd = [sys.executable, str((P/(name+'.py')).relative_to(ROOT))]
    start = time.perf_counter()
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    (P/(name+'.stdout.log')).write_text(out.stdout)
    (P/(name+'.stderr.log')).write_text(out.stderr)
    commands.append({'command': cmd, 'returncode': out.returncode,
                     'seconds': time.perf_counter()-start,
                     'stdout_sha256': sha(P/(name+'.stdout.log')),
                     'stderr_sha256': sha(P/(name+'.stderr.log'))})
    assert out.returncode == 0 and not out.stderr, commands[-1]
assert before == {str(p.relative_to(ROOT)): sha(p) for p in sources}
reports = ['identity-report.json', 'boundary-principal-report.json']
for name in reports:
    json.loads((P/name).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError('nonfinite JSON '+x)))
receipt = {
    'passed_advisory_checks': True,
    'scope': ('Mathematical representation/source and local frozen harmonic '
              'principal/SAT algebra only; no radial PDE operator, new kernel '
              'run, global eigensolve, propagation or boundary admission.'),
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        text=True).strip(),
    'python_executable': sys.executable,
    'python_executable_sha256': sha(Path(sys.executable).resolve()),
    'python_version': sys.version,
    'runner_sha256': sha(Path(__file__).resolve()),
    'seconds': time.perf_counter()-start_all, 'commands': commands,
    'source_hashes_before_equals_after': before,
    'report_hashes': {name: sha(P/name) for name in reports},
    'failure_history': {
        'failed-structural-kin-assertion':
            'Initial exact source/tool-observed structural assertion retained; '
            'mathematically equivalent square factors now compared by simplify.'}}
(P/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps(receipt, indent=2, allow_nan=False))
