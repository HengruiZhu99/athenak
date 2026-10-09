"""Metadata and AST preparation only; never imports generated gate modules."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
NEW = P / 'inner-joint-principal-gate-v2-held-20261009'
OLD = P / 'inner-joint-principal-gate-held-20261009'
PRIOR = P / 'reference-wave-map-principal-core-20261009'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


base = json.loads((PRIOR / 'release-recipe.json').read_text())
accepted = PRIOR / 'attempts/1791564714189871000/receipt.json'
old_receipt = json.loads(accepted.read_text())
headers = {}
for key, values in old_receipt.items():
    if key.endswith('_dependencies'):
        for path in values:
            p = Path(path)
            if p.suffix not in ('.cpp', '.cc', '.c'):
                headers[str(p.resolve())] = pin(p)
production = []
for relative in base['inputs']:
    if relative.startswith('src/') or relative == 'CMakeLists.txt':
        production.append(pin(ROOT / relative))
compiler, python = pin(base['compiler_path']), pin(base['python_path'])
local = ['probe.cpp', 'gauge_proposal.hpp', 'reference_wave_map.hpp',
         'check_exact.py', 'analyze.py', 'run_gate.py', 'PLAN.md',
         'PREPARATION.md', 'COUNT-CORRECTION.md', 'count-only.diff']
for name in ('check_exact.py', 'analyze.py', 'run_gate.py'):
    ast.parse((NEW / name).read_text(), filename=str(NEW / name))
recipe = dict(scope='coupled-inner-principal-118-and-exact18',
              execution_admitted=False, attempt_relative='attempts/gate001',
              production_implementation=base['production_implementation'],
              compiler=compiler, python=python,
              release_flags=base['release_flags'], debug_flags=base['debug_flags'],
              local_sources=local,
              protected_inputs=sorted(headers.values(), key=lambda x: x['path']) + production + [compiler, python],
              actual20_cases=118, exact_scalar_cases=18,
              limits=['constant reference principal only', 'no native/evolution/puncture/nonflat-source admission'])
write(NEW / 'recipe.json', recipe)
shutil.copyfile(__file__, NEW / 'prepare_recipe_source.py')
shutil.copyfile(P / 'prepare_inner_gate_v2.py', NEW / 'prepare_count_source.py')
external = [pin(OLD / 'source-index.json')] + [pin(x['path']) for x in
            json.loads((OLD / 'source-index.json').read_text())['files']]
external += [pin(PRIOR / 'release-recipe.json'), pin(accepted)]
note = P / 'inner-joint-principal-candidate-pencil-20261009'
external += [pin(note / name) for name in ('ASSESSMENT-v2.md', 'ERRATUM.md', 'index-v2.json')]
for relative in ('tst/hyperboloidal/kernel_symbol.cpp', 'tst/hyperboloidal/check_kernel_symbol.py'):
    external.append(pin(ROOT / relative))
files = [pin(NEW / name) for name in local + ['recipe.json', 'prepare_recipe_source.py', 'prepare_count_source.py']]
index = dict(source_only=True, execution_admitted=False, files=files,
             external_inputs=external, actual20_cases=118, exact_scalar_cases=18,
             original_count_failure_preserved=True,
             no_generated_module_import_compile_query_or_exact_execution=True,
             readiness='HELD exact root authorization and independent source review')
write(NEW / 'source-index.json', index)
write(NEW / 'readiness.json', dict(source_only=True, execution_admitted=False,
      source_index_sha256=sha(NEW / 'source-index.json'), recipe_sha256=sha(NEW / 'recipe.json'),
      file_count=len(files), protected_header_count=len(headers), production_pin_count=len(production),
      compiler_sha256=compiler['sha256'], python_sha256=python['sha256'],
      source_modules_AST_only=['check_exact.py', 'analyze.py', 'run_gate.py'],
      launch_command_held=[python['path'], str(NEW / 'run_gate.py'), '--authorization',
                           str(NEW / 'root-authorization.json')],
      no_scientific_execution=True))
print(json.dumps({'source_index_sha256': sha(NEW / 'source-index.json'),
                  'recipe_sha256': sha(NEW / 'recipe.json'), 'readiness_sha256': sha(NEW / 'readiness.json'),
                  'files': len(files), 'header_pins': len(headers), 'production_pins': len(production),
                  'probe_sha256': sha(NEW / 'probe.cpp'), 'analyzer_sha256': sha(NEW / 'analyze.py'),
                  'runner_sha256': sha(NEW / 'run_gate.py'), 'no_scientific_execution': True}, indent=2))
