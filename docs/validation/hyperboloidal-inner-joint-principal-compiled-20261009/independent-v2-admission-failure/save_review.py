"""Record source-only inspection and file hashes; never import candidate modules."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / 'inner-joint-principal-gate-v2-held-20261009'
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


assert digest(SUITE / 'source-index.json') == 'd8df2620a1c8d4c8542caf4aeb52699d10b016cc700b8c2d88657a372e262479'
assert digest(SUITE / 'recipe.json') == 'e14ba9a0bf913babddba1c6fdba252cdedd6897d6bf84184f13f4b0dd548c676'
source_index = json.loads((SUITE / 'source-index.json').read_text())
recipe = json.loads((SUITE / 'recipe.json').read_text())
declarations = source_index['files'] + source_index['external_inputs'] + recipe['protected_inputs']
unique = {}
for declaration in declarations:
    key = declaration['path']
    if key in unique:
        assert unique[key] == declaration, key
    unique[key] = declaration
before = [pin(path) for path in sorted(unique)]
assert before == [unique[x['path']] for x in before]
assert len(before) == 1438
source_copy_dir = HERE / 'source-copies'
source_copy_dir.mkdir(exist_ok=False)
copied = []
for declared in source_index['files']:
    original = Path(declared['path'])
    destination = source_copy_dir / original.name
    shutil.copyfile(original, destination)
    assert digest(destination) == declared['sha256']
    copied.append(dict(original=declared, copy=pin(destination)))
for name in ('source-index.json', 'readiness.json'):
    original = SUITE / name
    destination = source_copy_dir / name
    shutil.copyfile(original, destination)
    copied.append(dict(original=pin(original), copy=pin(destination)))
save(HERE / 'verified-inputs.json', before)
after = [pin(path) for path in sorted(unique)]
assert after == before
receipt = dict(
    source_review_completed=True,
    mathematical_source_review_passed=True,
    probe_analyzer_source_review_passed=True,
    unconditional_execution_admission_passed=False,
    admission_issue='Inherited PYTHONOPTIMIZE can disable scientific assertions; recipe/runner lack an optimization guard.',
    source_index=pin(SUITE / 'source-index.json'),
    recipe=pin(SUITE / 'recipe.json'),
    source_inputs_unchanged=True,
    verified_unique_declared_inputs=len(before),
    local_indexed_files=len(source_index['files']),
    external_context_entries=len(source_index['external_inputs']),
    recipe_dependency_entries=len(recipe['protected_inputs']),
    source_copies=copied,
    reviewed_actual_case_count=118,
    reviewed_exact_case_count=18,
    scientific_counts_executed=0,
    no_candidate_import_compile_CAS_numeric_kernel_array_or_evolution=True,
    no_frozen_source_mutation=True,
    review_operations=['source text inspection', 'standard-library JSON/file hash reads', 'private review artifact writes', 'git HEAD read'],
    launch_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    scope='Constant-reference finite-positive constrained20 principal source review only; no scientific execution or puncture admission.',
    later_goal='Wormhole-to-trumpet with Minkowski hyperboloidal reference retained.'
)
save(HERE / 'receipt.json', receipt)
files = [pin(x) for x in sorted(HERE.rglob('*')) if x.is_file() and x.name != 'index.json']
save(HERE / 'index.json', dict(source_only_review=True, scientific_execution=False,
                              files=files, reviewed_inputs=pin(HERE / 'verified-inputs.json')))
print(json.dumps(dict(index=pin(HERE / 'index.json'), receipt=pin(HERE / 'receipt.json'),
                      report=pin(HERE / 'REVIEW.md'), indexed_file_count=len(files)), indent=2))
