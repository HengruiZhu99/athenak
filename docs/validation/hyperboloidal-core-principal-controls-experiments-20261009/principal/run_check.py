"""Save one exact-algebra check; neither compiles nor runs a tensor kernel."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = HERE/'check_sectors.py'
pins = [source, Path(__file__),
        ROOT/'tst/hyperboloidal/kernel_symbol.cpp',
        ROOT/'tst/hyperboloidal/check_kernel_symbol.py',
        ROOT/'src/z4c/hyperboloidal/conformal_rhs.hpp',
        ROOT/'src/z4c/hyperboloidal/conformal_constraints.hpp',
        ROOT/'src/z4c/hyperboloidal/layer_gauge.hpp']
before = {str(p.relative_to(ROOT)): sha(p) for p in pins}
command = [sys.executable, str(source.relative_to(ROOT))]
start = time.perf_counter()
done = subprocess.run(command, cwd=ROOT, capture_output=True)
seconds = time.perf_counter()-start
(HERE/'check.stdout.log').write_bytes(done.stdout)
(HERE/'check.stderr.log').write_bytes(done.stderr)
after = {str(p.relative_to(ROOT)): sha(p) for p in pins}
assert before == after
receipt = {
    'passed': done.returncode == 0 and not done.stderr,
    'scope': 'Exact algebra and historical actual-symbol readback only.',
    'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=ROOT, text=True).strip(),
    'python_executable': sys.executable,
    'python_executable_sha256': sha(Path(sys.executable)),
    'python_version': sys.version,
    'command': command, 'returncode': done.returncode, 'seconds': seconds,
    'stdout_sha256': sha(HERE/'check.stdout.log'),
    'stderr_sha256': sha(HERE/'check.stderr.log'),
    'source_hashes_before_equals_after': before,
    'report_sha256': sha(HERE/'report.json') if done.returncode == 0 else None}
(HERE/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
assert receipt['passed'], done.stderr.decode()
print(json.dumps(receipt, indent=2, allow_nan=False))
