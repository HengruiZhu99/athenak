"""Source-only guard repair preparation. No gate module is imported/executed."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
OLD = P / 'inner-joint-principal-gate-v2-held-20261009'
NEW = P / 'inner-joint-principal-gate-v3-held-20261009'
NEW.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


old_index = json.loads((OLD / 'source-index.json').read_text())
old_recipe = json.loads((OLD / 'recipe.json').read_text())
for row in old_index['files']:
    assert sha(row['path']) == row['sha256']
for name in old_recipe['local_sources']:
    shutil.copyfile(OLD / name, NEW / name)
exact = (NEW / 'check_exact.py').read_text()
exact = exact.replace('import json\n', 'import json\nimport sys\n', 1)
exact = exact.replace("if __name__=='__main__':main()", "if __name__=='__main__':\n    if sys.flags.optimize != 0:\n        raise RuntimeError('optimized Python cannot execute assertion gates')\n    main()")
assert exact != (OLD / 'check_exact.py').read_text()
(NEW / 'check_exact.py').write_text(exact)
analyze = (NEW / 'analyze.py').read_text()
analyze = analyze.replace("if __name__ == '__main__':\n    main(sys.argv[1])", "if __name__ == '__main__':\n    if sys.flags.optimize != 0:\n        raise RuntimeError('optimized Python cannot execute assertion gates')\n    main(sys.argv[1])")
assert analyze != (OLD / 'analyze.py').read_text()
(NEW / 'analyze.py').write_text(analyze)
runner = (NEW / 'run_gate.py').read_text()
runner = runner.replace('        checkpoint()\n        parser = argparse.ArgumentParser', "        checkpoint()\n        if sys.flags.optimize != 0:\n            raise RuntimeError('optimized Python cannot execute assertion gates')\n        receipt['python_flags'] = dict(optimize=sys.flags.optimize, isolated=sys.flags.isolated)\n        parser = argparse.ArgumentParser", 1)
runner = runner.replace("[recipe['python']['path'], '--version']", "[recipe['python']['path'], '-I', '--version']")
runner = runner.replace("[recipe['python']['path'], str(attempt / 'check_exact.py')]", "[recipe['python']['path'], '-I', str(attempt / 'check_exact.py')]")
runner = runner.replace("[recipe['python']['path'], str(attempt / 'analyze.py'), str(raw)]", "[recipe['python']['path'], '-I', str(attempt / 'analyze.py'), str(raw)]")
assert runner != (OLD / 'run_gate.py').read_text()
(NEW / 'run_gate.py').write_text(runner)
diff = []
for name in ('check_exact.py', 'analyze.py', 'run_gate.py'):
    diff.extend(difflib.unified_diff((OLD / name).read_text().splitlines(True),
                (NEW / name).read_text().splitlines(True),
                fromfile=str(OLD / name), tofile=str(NEW / name)))
(NEW / 'optimization-guard-only.diff').write_text(''.join(diff))
(NEW / 'OPTIMIZATION-GUARD.md').write_text('''# Optimization guard correction

The original v2 source/index/readiness and independent admission failure remain
unchanged. Its exact and saved-JSON checks used assert while the wrapper inherited
PYTHONOPTIMIZE. Thus an uncontrolled optimized interpreter could skip checks and
print passed=true. No v2 scientific gate was executed.

This fresh source-only revision rejects sys.flags.optimize!=0 in the outer
runner and both checker direct entrypoints. Python child commands use -I, so
inherited PYTHONOPTIMIZE/PYTHONPATH cannot change assertion semantics or module
resolution. The runner records its optimize/isolated flags. The recommended
root command also uses -I. The exact diff changes guards/launch flags only;
the probe grid,118-case registry,18 Fraction cases, source formulas, all matrix
thresholds and required Release/ASanUB byte equality remain unchanged.

No generated source was imported, compiled or scientifically executed during
preparation. Exact root review/authorization and independent guard review are
still required. This revision does not authorize a kernel query or evolution.
''')
for name in ('check_exact.py', 'analyze.py', 'run_gate.py'):
    ast.parse((NEW / name).read_text(), filename=str(NEW / name))
recipe = dict(old_recipe)
recipe['local_sources'] = old_recipe['local_sources'] + ['OPTIMIZATION-GUARD.md', 'optimization-guard-only.diff']
recipe['guard_revision'] = 'v3: reject optimized runner/checkers; isolated checker children'
write(NEW / 'recipe.json', recipe)
shutil.copyfile(__file__, NEW / 'prepare_guard_source.py')
external = old_index['external_inputs'] + [pin(OLD / 'source-index.json'), pin(OLD / 'readiness.json')]
external += [pin(row['path']) for row in old_index['files']]
files = [pin(NEW / name) for name in recipe['local_sources'] + ['recipe.json', 'prepare_guard_source.py']]
write(NEW / 'source-index.json', dict(source_only=True, execution_admitted=False,
      files=files, external_inputs=external, actual20_cases=118, exact_scalar_cases=18,
      original_count_and_optimization_guard_failures_preserved=True,
      no_generated_module_import_compile_query_or_exact_execution=True,
      readiness='HELD root source review/authorization and independent guard review'))
write(NEW / 'readiness.json', dict(source_only=True, execution_admitted=False,
      source_index_sha256=sha(NEW / 'source-index.json'), recipe_sha256=sha(NEW / 'recipe.json'),
      files=len(files), unchanged_science=['probe.cpp', 'gauge_proposal.hpp', 'reference_wave_map.hpp'],
      python_children_isolated=True, optimized_runner_and_direct_checker_rejected=True,
      launch_command_held=[recipe['python']['path'], '-I', str(NEW / 'run_gate.py'),
                           '--authorization', str(NEW / 'root-authorization.json')],
      no_scientific_execution=True))
print(json.dumps(dict(index_sha256=sha(NEW / 'source-index.json'), recipe_sha256=sha(NEW / 'recipe.json'),
      readiness_sha256=sha(NEW / 'readiness.json'), runner_sha256=sha(NEW / 'run_gate.py'),
      diff_sha256=sha(NEW / 'optimization-guard-only.diff'), files=len(files), no_scientific_execution=True), indent=2))
