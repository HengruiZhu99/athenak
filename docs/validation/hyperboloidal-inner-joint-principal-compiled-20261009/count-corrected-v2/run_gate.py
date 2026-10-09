"""HELD one-shot root-authorized principal/exact gate, standard library only."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

P = Path(__file__).resolve().parent
ROOT = P.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def bad(value):
        raise ValueError('nonfinite JSON token ' + value)
    return json.loads(Path(path).read_text(), parse_constant=bad)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_pin(pin):
    path = Path(pin['path'])
    if sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
        raise ValueError('input drift: ' + str(path))


def main():
    attempt = P / 'attempts' / 'gate001'
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = dict(completed=False, passed=False, returncode=1, commands=[],
                   inputs_unchanged=None, scientific_scope='constant-reference principal118 plus exact scalar18',
                   no_native_evolution_or_puncture_admission=True)
    protected = []

    def checkpoint():
        save(attempt / 'receipt.json', receipt)

    def run(command, name):
        start = time.monotonic()
        outpath, errpath = attempt / (name + '.stdout'), attempt / (name + '.stderr')
        receipt['active_command'] = dict(name=name, command=command)
        checkpoint()
        with outpath.open('w') as out, errpath.open('w') as err:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=False)
        row = dict(name=name, command=command, returncode=result.returncode,
                   seconds=time.monotonic()-start, stdout_sha256=sha(outpath),
                   stderr_sha256=sha(errpath), stderr_bytes=errpath.stat().st_size)
        receipt['commands'].append(row)
        receipt.pop('active_command', None)
        checkpoint()
        if result.returncode != 0 or errpath.stat().st_size != 0:
            raise RuntimeError(name + ' nonzero exit or stderr')
        return outpath

    try:
        checkpoint()
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--authorization', required=True)
        parser.add_argument('--recipe', default=str(P / 'recipe.json'))
        args = parser.parse_args()
        # Local recipe identity is required before consumed-recipe parsing.
        local_recipe = P / 'recipe.json'
        if Path(args.recipe).resolve() != local_recipe.resolve():
            raise ValueError('only the pinned local recipe can be consumed')
        index = load(P / 'source-index.json')
        recipe = load(local_recipe)
        authorization = load(args.authorization)
        if not (authorization.get('execution_released') is True and
                authorization.get('scope') == recipe['scope'] and
                authorization.get('source_index_sha256') == sha(P / 'source-index.json') and
                authorization.get('recipe_sha256') == sha(local_recipe)):
            raise ValueError('exact source/recipe authorization is absent')
        protected = index['files'] + index['external_inputs'] + recipe['protected_inputs']
        for path in (P / 'source-index.json', Path(args.authorization).resolve()):
            protected.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        for pin in protected:
            check_pin(pin)
        if recipe['attempt_relative'] != 'attempts/gate001':
            raise ValueError('attempt differs from fixed fresh destination')
        receipt.update(authorization_sha256=sha(args.authorization),
                       recipe_sha256=sha(local_recipe), consumed_recipe_path=str(local_recipe),
                       source_index_sha256=sha(P / 'source-index.json'),
                       production_implementation=recipe['production_implementation'],
                       protected_input_count=len({p['path'] for p in protected}))
        for name in recipe['local_sources'] + ['recipe.json', 'source-index.json']:
            shutil.copyfile(P / name, attempt / name)
        shutil.copyfile(args.authorization, attempt / 'authorization.json')
        receipt['launch_HEAD'] = run(['git', 'rev-parse', 'HEAD'], 'launch-HEAD').read_text().strip()
        run([recipe['compiler']['path'], '--version'], 'compiler-version')
        run([recipe['python']['path'], '--version'], 'python-version')
        exact_path = run([recipe['python']['path'], str(attempt / 'check_exact.py')], 'exact-scalar')
        exact = load(exact_path)
        if exact.get('passed') is not True or exact.get('exact_scalar_cases') != 18:
            raise ValueError('exact scalar gate did not pass all18 cases')
        receipt['exact_scalar_summary'] = exact
        analyses = {}
        declared = {x['path']: x for x in protected}
        for mode in ('release', 'debug'):
            exe, dep = attempt / ('probe-' + mode), attempt / ('probe-' + mode + '.d')
            command = [recipe['compiler']['path']] + recipe[mode + '_flags'] + [
                '-MD', '-MF', str(dep), str(attempt / 'probe.cpp'), '-o', str(exe)]
            run(command, 'compile-' + mode)
            dependencies = []
            for word in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]):
                q = Path(word)
                q = q if q.is_absolute() else ROOT / q
                q = q.resolve()
                pin = dict(path=str(q), sha256=sha(q), bytes=q.stat().st_size)
                if q.parent == attempt:
                    original = P / q.name
                    if q.name not in recipe['local_sources'] or sha(q) != sha(original):
                        raise ValueError('unexpected copied source dependency: ' + str(q))
                elif str(q) not in declared:
                    raise ValueError('undeclared compile dependency: ' + str(q))
                else:
                    check_pin(declared[str(q)])
                dependencies.append(pin)
            save(attempt / ('dependencies-' + mode + '.json'), dependencies)
            receipt[mode + '_executable_sha256'] = sha(exe)
            receipt[mode + '_dependency_count'] = len(dependencies)
            checkpoint()
            raw = run([str(exe)], 'probe-' + mode)
            summary = run([recipe['python']['path'], str(attempt / 'analyze.py'), str(raw)],
                          'analyze-' + mode)
            analyses[mode] = load(summary)
            if analyses[mode].get('passed') is not True or analyses[mode]['actual20_cases'] != 118:
                raise ValueError('actual20 gate did not pass all118 cases')
        receipt['release_debug_stdout_byte_equal'] = (
            (attempt / 'probe-release.stdout').read_bytes() ==
            (attempt / 'probe-debug.stdout').read_bytes())
        if not receipt['release_debug_stdout_byte_equal']:
            raise ValueError('Release/ASanUB probe outputs are not byte-identical')
        receipt['analysis_summaries'] = analyses
        receipt['passed'] = True
        receipt['completed'] = True
        receipt['returncode'] = 0
    except BaseException as error:
        receipt['exception'] = repr(error)
        (attempt / 'exception.txt').write_text(traceback.format_exc())
    finally:
        try:
            drift = []
            for pin in protected:
                try:
                    check_pin(pin)
                except BaseException as error:
                    drift.append(dict(path=pin['path'], error=repr(error)))
            receipt['inputs_unchanged'] = not drift
            receipt['input_drift'] = drift
            if drift:
                receipt.update(passed=False, completed=False, returncode=1)
        except BaseException as error:
            receipt.update(passed=False, completed=False, returncode=1,
                           finalization_exception=repr(error))
        checkpoint()
    print(json.dumps(dict(attempt=str(attempt), passed=receipt['passed'],
                          receipt_sha256=sha(attempt / 'receipt.json')), indent=2))
    return receipt['returncode']


if __name__ == '__main__':
    sys.exit(main())
