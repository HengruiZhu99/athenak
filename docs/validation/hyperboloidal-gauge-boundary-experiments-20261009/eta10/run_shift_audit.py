"""Reproduce the private frozen shift-control family, without production edits.

The receipt retains exact compiler/include/library flags. The separate native
overlay is rebuilt using its saved CMake and target-build commands; native
reference, short and long runs are driven by native_phase.py.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

p = Path(__file__).resolve().parent
receipt = json.loads((p/'receipt.json').read_text())
compile_command = receipt['commands'][0]
assert hashlib.sha256((p/'shift_fourier.cpp').read_bytes()).hexdigest() == receipt['source_sha256']
with (p/'build.log').open('w') as log:
    subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
commands = [compile_command]
for flag, destination in (('--shift-poles', 'shift-poles.json'), ('--shift', 'shift-fourier.json')):
    command = [str(p/'shift_fourier'), flag]
    with (p/destination).open('w') as out:
        subprocess.run(command, stdout=out, check=True)
    commands.append(command)
for checker in ('check_shift.py', 'check_group.py'):
    command = [sys.executable, str(p/checker)]
    with (p/('check.log' if checker == 'check_shift.py' else 'group.log')).open('w') as out:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
    commands.append(command)
receipt = json.loads((p/'receipt.json').read_text())
receipt['commands'] = commands
receipt['executable_sha256'] = hashlib.sha256((p/'shift_fourier').read_bytes()).hexdigest()
receipt['reproducer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt['artifacts_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in p.glob('*.json') if f.name != 'receipt.json'}
(p/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print('PASS rebuilt actual full20 frozen shift-control audit; no production edits')
