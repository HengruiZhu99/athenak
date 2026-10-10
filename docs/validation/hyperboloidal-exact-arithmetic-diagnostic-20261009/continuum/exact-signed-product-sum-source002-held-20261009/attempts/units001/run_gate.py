"""HELD one-shot standalone CPU integer-product units; exact release required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

P = Path(__file__).resolve().parent
ROOT = P.parents[2]


def require(test, message):
    if not test:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError('nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check(pin):
    p = Path(pin['path'])
    require(p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'input drift: ' + str(p))


def pin(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def main():
    attempt = P / 'attempts' / 'units001'
    # Existing destinations are never reused, even for failed admission.
    attempt.mkdir(parents=True, exist_ok=False)
    receipt = dict(completed=False, passed=False, returncode=1, inputs_unchanged=None,
                   stage='exact-short-signed-product-sum-CPU-units70', commands=[],
                   no_RWM_adoption_or_saved_failure_upgrade=True,
                   compiler_or_arithmetic_executed_before_authorization=False)
    protected = []

    def checkpoint():
        save(attempt / 'receipt.json', receipt)

    def run(command, name, env, input_path=None):
        outpath, errpath = attempt / (name + '.stdout'), attempt / (name + '.stderr')
        receipt['active_command'] = dict(name=name, command=command)
        checkpoint()
        start = time.monotonic()
        with outpath.open('wb') as out, errpath.open('wb') as err:
            if input_path is None:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
            else:
                with Path(input_path).open('rb') as inp:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdin=inp,
                                            stdout=out, stderr=err, check=False)
        receipt['commands'].append(dict(name=name, command=command, returncode=result.returncode,
                                        seconds=time.monotonic() - start,
                                        stdout=pin(outpath), stderr=pin(errpath),
                                        input=None if input_path is None else pin(input_path)))
        receipt.pop('active_command', None)
        checkpoint()
        require(result.returncode == 0 and errpath.stat().st_size == 0, name + ': exit/stderr failure')
        return outpath

    try:
        checkpoint()
        require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
                'isolated unoptimized -I -B launch required')
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument('--authorization', required=True)
        ap.add_argument('--recipe', default=str(P / 'recipe.json'))
        args = ap.parse_args()
        require(Path(args.recipe).resolve() == P / 'recipe.json', 'only exact local recipe is admitted')
        recipe = load(P / 'recipe.json')
        index = load(P / 'source-index.json')
        auth = load(args.authorization)
        require(auth.get('execution_released') is True and
                auth.get('scope') == recipe['scope'] and
                auth.get('source_index_sha256') == sha(P / 'source-index.json') and
                auth.get('recipe_sha256') == sha(P / 'recipe.json'), 'exact root release absent')
        require(auth.get('source_review_passed') is True and
                isinstance(auth.get('review_receipt'), dict), 'pinned passed source review required')
        protected = index['files'] + load(P / 'external-pins.json') + [auth['review_receipt'],
                         pin(P / 'source-index.json'), pin(args.authorization)]
        for item in protected:
            check(item)
        review = load(auth['review_receipt']['path'])
        require(review.get('passed') is True or review.get('passed_source_review') is True,
                'source review receipt does not explicitly pass')
        require(review.get('reviewed_source_index_sha256') == sha(P / 'source-index.json'),
                'passed review does not bind this exact source index')
        require(str(Path(sys.executable).resolve()) == recipe['python']['path'] and
                sha(sys.executable) == recipe['python']['sha256'], 'Python interpreter identity')
        # Keep the literal clang++ driver basename in argv; bind its target too.
        require(recipe['compiler']['path'] == '/Library/Developer/CommandLineTools/usr/bin/clang++',
                'literal C++ driver invocation path required')
        require(str(Path(recipe['compiler']['path']).resolve()) == recipe['compiler']['resolved_path'],
                'C++ driver resolved target changed')
        require(sha(recipe['compiler']['path']) == recipe['compiler']['sha256'] and
                sha(recipe['compiler']['resolved_path']) == recipe['compiler']['resolved_sha256'],
                'C++ driver invocation/target byte identity')
        for key, value in recipe['environment'].items():
            require(os.environ.get(key) == value, 'launch environment mismatch: ' + key)
        for key in recipe['sanitized_environment']:
            require(key not in os.environ, 'forbidden inherited environment: ' + key)
        require(recipe['attempt_relative'] == 'attempts/units001' and recipe['fixed_cases'] == 70,
                'fixed attempt/count mismatch')
        receipt.update(source_index=pin(P / 'source-index.json'), recipe=pin(P / 'recipe.json'),
                       authorization=pin(args.authorization), source_review=auth['review_receipt'],
                       consumed_recipe_path=str(P / 'recipe.json'),
                       python=pin(sys.executable), python_flags=dict(optimize=sys.flags.optimize,
                           isolated=sys.flags.isolated, dont_write_bytecode=sys.dont_write_bytecode),
                       environment=recipe['environment'], protected_count=len(protected))
        env = os.environ.copy()
        for name in recipe['local_sources'] + ['recipe.json', 'source-index.json', 'external-pins.json']:
            shutil.copyfile(P / name, attempt / name)
        shutil.copyfile(args.authorization, attempt / 'authorization.json')
        run(['git', 'rev-parse', 'HEAD'], 'launch-HEAD', env)
        run([recipe['compiler']['path'], '--version'], 'compiler-version', env)
        run([recipe['python']['path'], '-I', '-B', '--version'], 'python-version', env)
        declared = {x['path']: x for x in protected}
        reports = {}
        for mode in ('release', 'debug'):
            exe = attempt / ('probe-' + mode)
            dep = attempt / ('probe-' + mode + '.d')
            cmd = [recipe['compiler']['path']] + recipe[mode + '_flags'] + [
                '-MD', '-MF', str(dep), str(attempt / 'probe.cpp'), '-o', str(exe)]
            run(cmd, 'compile-' + mode, env)
            dependencies = []
            for word in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]):
                q = Path(word)
                q = (q if q.is_absolute() else ROOT / q).resolve()
                item = pin(q)
                if q.parent == attempt:
                    require(q.name in recipe['local_sources'] and sha(q) == sha(P / q.name),
                            'unexpected copied-source dependency: ' + str(q))
                else:
                    require(str(q) in declared, 'undeclared compiler dependency: ' + str(q))
                    check(declared[str(q)])
                dependencies.append(item)
            save(attempt / ('dependencies-' + mode + '.json'), dependencies)
            receipt[mode + '_executable'] = pin(exe)
            receipt[mode + '_dependencies'] = pin(attempt / ('dependencies-' + mode + '.json'))
            checkpoint()
            raw = run([str(exe)], 'probe-' + mode, env, attempt / 'cases.txt')
            reportpath = attempt / ('oracle-' + mode + '.json')
            run([recipe['python']['path'], '-I', '-B', str(attempt / 'fraction_oracle.py'),
                 '--registry', str(attempt / 'registry.json'), '--output', str(raw),
                 '--report', str(reportpath)], 'oracle-' + mode, env)
            reports[mode] = load(reportpath)
            require(reports[mode].get('passed') is True and reports[mode]['fixed_cases'] == 70,
                    'all fixed70 exact controls must pass')
            receipt[mode + '_oracle_report'] = pin(reportpath)
        receipt['release_debug_probe_byte_equal'] = ((attempt / 'probe-release.stdout').read_bytes() ==
                                                      (attempt / 'probe-debug.stdout').read_bytes())
        require(receipt['release_debug_probe_byte_equal'], 'Release/ASanUB result bytes differ')
        receipt.update(completed=True, passed=True, returncode=0, fixed_cases=70,
                       scalar_cases=44, dual_cases=26)
    except BaseException as error:
        receipt['exception'] = repr(error)
        (attempt / 'exception.txt').write_text(traceback.format_exc())
    finally:
        drift = []
        for item in protected:
            try:
                check(item)
            except BaseException as error:
                drift.append(dict(path=item['path'], error=repr(error)))
        receipt['inputs_unchanged'] = not drift
        receipt['input_drift'] = drift
        if drift:
            receipt.update(completed=False, passed=False, returncode=1)
        checkpoint()
    print(json.dumps(dict(attempt=str(attempt), passed=receipt['passed'], receipt=pin(attempt / 'receipt.json'))))
    return receipt['returncode']


if __name__ == '__main__':
    sys.exit(main())
