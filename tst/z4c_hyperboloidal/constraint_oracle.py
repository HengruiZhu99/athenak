#!/usr/bin/env python3
"""Compare every accepted constraint point and saved N12 diagnostics to legacy."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--legacy', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--checkpoint', type=pathlib.Path)
parser.add_argument('--ranks', type=int, choices=[2, 8], default=2)
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--reuse-executable', type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
checkpoint = (args.checkpoint or build/'n12-reference-step3551').resolve()
output.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot(paths):
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}

def unchanged(manifest):
    return all(pathlib.Path(p).exists() and sha(pathlib.Path(p)) == h for p, h in manifest.items())

for name, expected in [('fields.bin','7d37c940eea84a04dc471bb90178c3e2facf153f14382cd2a47cf1fba6a320be'),
                       ('boundary.bin','343d4b04c6dc6456eb8bdf84bf1ddaa97fc1aee9531bdb1299c4d8573394e6ef')]:
    assert sha(checkpoint/name) == expected, 'saved N12 checkpoint provenance differs'
if args.reuse_executable:
    exe = args.reuse_executable.resolve()
    provenance = json.loads((exe.parent/'build-provenance.json').read_text())
    assert str(exe) == provenance['executable'] and sha(exe) == provenance['executable_sha256']
    for key in ['source_sha256','dependency_sha256','object_sha256']:
        assert unchanged(provenance[key]), f'changed {key} from frozen build'
else:
    tracked = [p for p in (root/'src').rglob('*') if p.suffix in ['.cpp','.hpp']]
    tracked += [root/'tst/z4c_hyperboloidal'/p for p in [
        'constraint_oracle.cpp','constraint_oracle.py','constraint_oracle_contract.md','n12_initialize.athinput']]
    source_manifest = snapshot(tracked)
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
    options = ['BROAD_SMOOTH_A','CONFIGURED_FORMULATION','DIRECT_OUTER','INNER_TELEGRAPHER',
        'NORMALIZED_INNER_SHIFT','RESTORE_INNER_DAMPING','SHORT_INNER_GAUGE','SMOOTH_GAUGE_BLEND','SPHERICAL_POLICY']
    obj, exe, dep = output/'constraint_oracle.o', output/'constraint_oracle', output/'oracle.d'
    compile_command = ['/usr/bin/c++','-std=c++17','-O2','-ffp-contract=off',*includes,
        *['-DHYPERBOLIC_'+x+'=1' for x in options],'-I'+str(legacy/'src'),'-I'+str(legacy),
        '-MMD','-MF',str(dep),'-c',str(root/'tst/z4c_hyperboloidal/constraint_oracle.cpp'),'-o',str(obj)]
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
    assert str(obj) in link
    link[link.index('-o')+1] = str(exe)
    objects = [(build/'src'/x).resolve() for x in link if x.endswith('.o') and x != str(obj)]
    object_manifest = snapshot(objects)
    makefile = output/'oracle.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make','-j64','-f',str(makefile)], cwd=build/'src', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    dependencies = shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
    dependency_manifest = snapshot([(build/'src'/p).resolve() for p in dependencies])
    assert unchanged(source_manifest) and unchanged(object_manifest), 'source changed during build'
    provenance = {'executable':str(exe),'executable_sha256':sha(exe),
        'source_sha256':source_manifest,'dependency_sha256':dependency_manifest,
        'object_sha256':object_manifest,'compile_command':compile_command,
        'native_compile_flags':flags,'checkpoint_sha256':snapshot([checkpoint/'fields.bin',checkpoint/'boundary.bin'])}
    (output/'build-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
if args.build_only:
    print('Built',exe)
    raise SystemExit(0)
command = ['mpiexec','-n',str(args.ranks),str(exe),
    str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput'),str(checkpoint)]
with (output/'run.log').open('w') as log:
    result = subprocess.run(command,cwd=output,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
text = (output/'run.log').read_text()
metrics = {}
for m in re.finditer(r'^(constraints\d) comparisons=(\d+) max_scaled=(\S+) max_absolute=(\S+) '
                     r'max_tolerance_ratio=(\S+) failures=(\d+) weighted_rms=(\S+) weight_sum=(\S+)$',text,re.M):
    metrics[m[1]] = {'count':int(m[2]),'max_scaled':float(m[3]),'max_absolute':float(m[4]),
        'max_tolerance_ratio':float(m[5]),'failures':int(m[6]),'weighted_rms':float(m[7]),
        'weight_sum':float(m[8]),'components':[]}
for m in re.finditer(r'^(constraints\d) field=(\d+) max_absolute=(\S+) max_scaled=(\S+) weighted_rms=(\S+) failures=(\d+)$',text,re.M):
    metrics[m[1]]['components'].append({'field':int(m[2]),'max_absolute':float(m[3]),
        'max_scaled':float(m[4]),'weighted_rms':float(m[5]),'failures':int(m[6])})
status = {'passed':result.returncode==0 and 'PASS native constraints' in text,
    'ranks':args.ranks,'executable':str(exe),'executable_sha256':sha(exe),'command':command,
    'metrics':metrics,'source_unchanged':all(unchanged(provenance[k]) for k in
        ['source_sha256','dependency_sha256','object_sha256']),
    'surface_sha256':snapshot(list(output.glob('surface-*.txt'))),
    'scope':'initial, nonspherical, saved3551 constraints; fixed N12 includes central SMR; not AMR lifecycle or GPU'}
(output/'results.json').write_text(json.dumps(status,indent=2)+'\n')
print(text)
result.check_returncode()
assert status['passed'] and status['source_unchanged']
assert len(metrics)==3 and all(v['count']==19*229392 and len(v['components'])==19 for v in metrics.values())
