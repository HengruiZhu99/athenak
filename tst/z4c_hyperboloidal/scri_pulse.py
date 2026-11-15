#!/usr/bin/env python3
"""Build the test-only pulse checkpoint writer and verify its native payloads."""
import argparse, hashlib, json, re, shlex, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from restart_roundtrip import parse, payload, compare, run

p = argparse.ArgumentParser()
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
b, out = args.build.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
flags = (b/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
obj, exe = out/'pulse.o', out/'scri_pulse'
compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off',
                   *includes, '-c', str(root/'tst/z4c_hyperboloidal/scri_pulse.cpp'), '-o', str(obj)]
subprocess.run(compile_command, check=True)
link = shlex.split((b/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if v == 'CMakeFiles/athena.dir/main.cpp.o' else v for v in link]
link[link.index('-o')+1] = str(exe)
subprocess.run(link, cwd=b/'src', check=True)
text = (root/'build-scri-experiment/inputs/A-N12.athinput').read_text()
text += '\n<time>\nnlim = 0\n<fastflow>\nnum_horizons = 0\n'
text += '<output1>\ndt = .0033333333333333333\n'
inp = out/'input.athinput'; inp.write_text(text)
states = {}
for name, amp in [('zero', 0), ('pulse', 1e-6), ('half', 5e-7)]:
    print('Writing', name, flush=True)
    d = out/name; d.mkdir()
    with (d/'run.log').open('w') as log:
        subprocess.run(['mpiexec', '-n', '2', str(exe), str(inp), str(amp)],
                       cwd=d, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    states[name] = parse(next((d/'rst').glob('*.rst')))
standard = run(b/'src/athena', out, 'standard', 2, ['-i', str(inp)])
result = {'zero_equals_production': compare(states['zero'], standard)}
zero, full, half = [payload(states[k]) for k in ('zero', 'pulse', 'half')]
untouched = [f for f in range(39) if f not in (13, 18)]
assert np.array_equal(zero[:, :, untouched], full[:, :, untouched])
assert np.array_equal(zero[:, :, untouched], half[:, :, untouched])
delta, delta_half = full-zero, half-zero
error = np.max(np.abs(delta-2*delta_half))
assert np.isfinite(full).all() and error < 5e-15
assert np.max(np.abs(delta[:, :, 18])) > .9e-6
result.update(unchanged_other_fields=True, amplitude_linearity_error=float(error),
              pulse_peak=float(np.max(np.abs(delta[:, :, 18]))),
              geometry='radial C-infinity bump, center .92, half-width .04',
              polarization='outgoing leading lapse/trace gauge; not full coupled eigenmode',
              compile=compile_command, link=link,
              test_source_sha256=hashlib.sha256((root/'tst/z4c_hyperboloidal/scri_pulse.cpp').read_bytes()).hexdigest(),
              production_executable_sha256=hashlib.sha256((b/'src/athena').read_bytes()).hexdigest())
(out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
