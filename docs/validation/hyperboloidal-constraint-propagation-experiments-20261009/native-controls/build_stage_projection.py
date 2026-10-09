"""Build a private stage-projection control without changing production or base objects."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT/'build-layer-spatial-norm-native'
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PRIVATE = HERE/'stage-projection-build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


PRIVATE.mkdir(exist_ok=False)
receipt_path = FAMILY/'native-build-receipt.json'
receipt = json.loads(receipt_path.read_text())
assert all(sha(ROOT/k) == v for k, v in receipt['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in receipt['overlay_sha256'].items())
base_exe = BUILD/'src/athena'
assert sha(base_exe) == 'dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d'
source = ROOT/'src/z4c/z4c_tasks.cpp'
old = ('TaskStatus Z4c::EnforceAlgConstr(Driver *pdrive, int stage) {\n'
       '  if (pmy_pack->pdyngr != nullptr || stage == pdrive->nexp_stages) {')
new = ('TaskStatus Z4c::EnforceAlgConstr(Driver *pdrive, int stage) {\n'
       '  if (hyperboloidal_patch || pmy_pack->pdyngr != nullptr\n'
       '      || stage == pdrive->nexp_stages) {')
text = source.read_text()
assert text.count(old) == 1
private_source = PRIVATE/'z4c_tasks.cpp'
private_source.write_text(text.replace(old, new))
assert private_source.read_text().replace(new, old) == text
commands = json.loads((BUILD/'compile_commands.json').read_text())
entry, = [x for x in commands if Path(x['file']) == source]
compile_cmd = shlex.split(entry['command'])
source_index = compile_cmd.index('-c')+1
object_index = compile_cmd.index('-o')+1
base_object = Path(entry['directory'])/compile_cmd[object_index]
private_object = PRIVATE/'z4c_tasks.cpp.o'
compile_cmd[source_index] = str(private_source)
compile_cmd[object_index] = str(private_object)
link_cmd = shlex.split((BUILD/'src/CMakeFiles/athena.dir/link.txt').read_text())
link_cwd = BUILD/'src'
link_inputs = {}
for arg in link_cmd:
    if arg.endswith(('.o', '.a')):
        path = (link_cwd/arg).resolve()
        link_inputs[str(path.relative_to(ROOT))] = {'sha256': sha(path),
                                                   'bytes': path.stat().st_size}
        if path == base_object.resolve():
            link_cmd[link_cmd.index(arg)] = str(private_object)
assert str(private_object) in link_cmd
assert len([x for x in link_inputs if x.endswith('.o')]) == 182
executable = PRIVATE/'athena-stage-projection'
link_cmd[link_cmd.index('-o')+1] = str(executable)
started = time.monotonic()
with (PRIVATE/'build.log').open('w') as log:
    compile_result = subprocess.run(compile_cmd, cwd=entry['directory'],
                                    stdout=log, stderr=subprocess.STDOUT)
    assert compile_result.returncode == 0
    link_result = subprocess.run(link_cmd, cwd=link_cwd, stdout=log,
                                 stderr=subprocess.STDOUT)
    assert link_result.returncode == 0
assert all(sha(ROOT/k) == v['sha256'] for k, v in link_inputs.items())
assert all(sha(ROOT/k) == v for k, v in receipt['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in receipt['overlay_sha256'].items())
out = {
    'scope': ('Private native control: call the existing active-cell algebraic projector '
              'at every explicit RK stage for hyperboloidal patches. Same spatial-norm '
              'gauge, all other production source and reused compiled objects unchanged. '
              'This is an experiment, not an adopted stability fix.'),
    'launch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=ROOT, text=True).strip(),
    'compiled_implementation': receipt['implementation_commit'],
    'base_build_receipt': str(receipt_path.relative_to(ROOT)),
    'base_build_receipt_sha256': sha(receipt_path),
    'base_executable_sha256': sha(base_exe),
    'source_before_sha256': sha(source),
    'private_source_sha256': sha(private_source),
    'only_source_replacement': {'before': old, 'after': new},
    'script_sha256': sha(Path(__file__)),
    'compile_command': compile_cmd, 'compile_cwd': entry['directory'],
    'link_command': link_cmd, 'link_cwd': str(link_cwd),
    'compile_exit_status': compile_result.returncode,
    'link_exit_status': link_result.returncode,
    'elapsed_seconds': time.monotonic()-started,
    'reused_base_link_inputs_sha256': link_inputs,
    'base_source_overlay_and_link_inputs_unchanged': True,
    'private_object_sha256': sha(private_object),
    'executable': str(executable.relative_to(ROOT)),
    'executable_sha256': sha(executable),
    'build_log_sha256': sha(PRIVATE/'build.log'),
    'compiler_version': subprocess.check_output([compile_cmd[0], '--version'], text=True)}
(PRIVATE/'build-receipt.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print('PASS private build', out['executable_sha256'], out['elapsed_seconds'])
