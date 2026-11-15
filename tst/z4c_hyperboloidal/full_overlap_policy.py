#!/usr/bin/env python3
"""Check full-overlap defaults, rejected retired settings, and old-full restarts.

The optional --reference executable must predate removal of alternative overlap
policies. Its explicit full-overlap evolution must match the current executable
bitwise, including a restart repartitioned from two to eight MPI ranks.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
from restart_roundtrip import parse, compare

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reference', type=Path)
a = parser.parse_args()
out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
exe = a.build.resolve()/'src/athena'
fixture = (root/'tst/z4c_hyperboloidal/n12_initialize.athinput').read_text()
base = re.sub(r'(?m)^(overlap_mode|agreement_kappa|agreement_taper_width)\s*=.*\n', '', fixture)
base += '\n<time>\nevolution = dynamic\nintegrator = rk4_classical\nnlim = 2\ntlim = 1\n<output1>\nfile_type = rst\ndt = 100\n'
full = '\n<z4c_hyperboloidal>\noverlap_mode = agreement_full\nagreement_kappa = 1\nagreement_taper_width = .015\n'
report = {}
def run(name, text, binary=exe, ranks=2, options=None, error=None):
    work = out/name; work.mkdir(); inp = work/'input'; inp.write_text(text)
    cmd = ['mpiexec', '-n', str(ranks), str(binary), *(options or ['-i', str(inp)])]
    with (work/'run.log').open('w') as log:
        result = subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    log = (work/'run.log').read_text()
    if error:
        assert result.returncode != 0 and error in log, (name, result.returncode, log[-2000:])
    else:
        assert result.returncode == 0, (name, log[-2000:])
    report[name] = {'command': cmd, 'exit_code': result.returncode}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return work

def checkpoint(work, final=True):
    paths = sorted(work.glob('rst/*.rst'))
    assert paths, work
    return paths[-1] if final else paths[0]

def match(name, left, right):
    report[name] = compare(parse(left), parse(right))
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')

explicit = run('explicit', base+full)
default = run('default', base)
match('defaults_bitwise', checkpoint(explicit), checkpoint(default))
for mode in ['legacy', 'masked', 'agreement', 'unknown']:
    run('reject-'+mode, base+'\n<z4c_hyperboloidal>\noverlap_mode = '+mode+'\n',
        error='Only overlap_mode=agreement_full is supported')
for key in ['cartesian_cutoff', 'agreement_inner', 'agreement_outer']:
    run('reject-'+key, base+f'\n<z4c_hyperboloidal>\n{key} = .1\n',
        error='Obsolete masked-overlap parameter: '+key)
if a.reference:
    reference = run('reference', base+full, binary=a.reference.resolve())
    match('reference_bitwise', checkpoint(reference), checkpoint(explicit))
    restarted = run('old-full-restart-8', '', ranks=8,
                    options=['-r', str(checkpoint(reference, final=False))])
    match('old_full_restart_bitwise', checkpoint(explicit), checkpoint(restarted))
print(json.dumps(report, indent=2))
