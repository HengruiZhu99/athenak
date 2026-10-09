"""Single-use source/failed-receipt metadata preparation; no array reads."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT / 'build-layer-research/reference-wave-map-native-held-20261009'
LONG = ROOT / 'build-layer-research/wave-map-native-t2-root-20261009'
COMPLETED = ROOT / 'build-layer-research/reference-wave-map-t2-readback-held-20261009/recipe.json'
READER = ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py'
REVIEW = ROOT / 'build-layer-research/boundary/reference-wave-map-native-seam-snapshot-independent-review-20261009/guard-revision002/receipt.json'
FAILED = {
    'wave-map-N16-large-t2': '9b57613601546288c06b22a0c8ac34bc76fd1379c4224e3cf1706c5dc2d9b7e7',
    'c0-N16-large-t2': '4d69b3d78f2fc7db17bdf191f01486a76069cc63bf8dd2e23aa041442ddc2c5f',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
def dump(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def main():
    assert not (HERE / 'recipe.json').exists()
    assert sha(COMPLETED) == '47777eb5188406dad342e8f7045c29a216adaeb780936fba66824b4abadd5cca'
    template = load(COMPLETED)
    prefixes = [str(NATIVE), str(LONG)]
    fixed = {p: digest for p, digest in template['fixed_pins'].items()
             if any(p.startswith(prefix + '/') for prefix in prefixes)
             or p in [str(READER), str(READER.with_name('abi.json')), str(REVIEW), '/Library/Developer/CommandLineTools/usr/bin/python3']}
    context = HERE / 'source-context'
    context.mkdir(exist_ok=False)
    copies = {}
    initial_failures = {}
    for name, digest in FAILED.items():
        p = LONG / 'batch001' / name / 'launch-receipt.json'
        assert sha(p) == digest
        failed = load(p)
        assert failed['returncode'] == -6 and failed['passed_native_process_and_provenance'] is False
        assert failed['sources_before_after_equal'] is True
        destination = context / (name + '-original-failed-launch.json')
        destination.write_bytes(p.read_bytes())
        copies[str(destination)] = {'source': str(p), 'sha256': digest}
        fixed[str(p)] = digest
        initial_failures[name] = {'original_failed_launch_receipt': str(p), 'sha256': digest,
                                 'returncode': failed['returncode'], 'output_inventory_files': len(failed['outputs'])}
    execution = load(LONG / 'release.json')
    assert len(execution['cases']) == 9
    cases = {spec['name']: spec for spec in execution['cases']}
    for spec in cases.values():
        for p, digest in spec['required_hashes'].items(): assert sha(p) == digest, p
        source = Path(spec['input_path'])
        destination = context / source.name
        destination.write_bytes(source.read_bytes())
        copies[str(destination)] = {'source': str(source), 'sha256': sha(source)}
    for label, source in {
        'accepted-analyzer.py': NATIVE / 'analyze_snapshots.py',
        'native-t2-release.json': LONG / 'release.json',
        'native-t2-launcher.py': LONG / 'run_preflights.py',
        'accepted-native-probe-recipe.json': NATIVE / 'probe-build-held/recipe.json',
        'accepted-binary64-reader.py': READER,
        'accepted-binary64-abi.json': READER.with_name('abi.json'),
        'accepted-independent-guard-review.json': REVIEW,
    }.items():
        destination = context / label
        destination.write_bytes(source.read_bytes())
        copies[str(destination)] = {'source': str(source), 'sha256': sha(source)}
    local = [HERE / name for name in ['observe_partial.py', 'PLAN.md', 'release-schema.json', 'prepare_recipe.py']]
    for p in local:
        if p.suffix == '.py': compile(p.read_text(), str(p), 'exec')
    fixed.update({str(p): sha(p) for p in local if p.name != 'observe_partial.py'})
    for p, digest in fixed.items(): assert sha(p) == digest, p
    recipe = {'held': True, 'source_only_preparation': True,
        'prepared_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'observer_sha256': sha(HERE / 'observe_partial.py'), 'fixed_pins': fixed,
        'cases': cases, 'initial_review_failures': initial_failures,
        'native_execution_release_sha256': sha(LONG / 'release.json'),
        'probe_executable': template['probe_executable'], 'probe_executable_sha256': template['probe_executable_sha256'],
        'probe_recipe': template['probe_recipe'], 'probe_recipe_sha256': template['probe_recipe_sha256'],
        'seam_receipt': template['seam_receipt'], 'seam_receipt_sha256': template['seam_receipt_sha256'],
        'reader': str(READER), 'reader_sha256': sha(READER), 'abi_sha256': sha(READER.with_name('abi.json')),
        'unchanged_completed_analyzer': template['analyzer'], 'unchanged_completed_analyzer_sha256': template['analyzer_sha256'],
        'source_context_copies': copies,
        'scope': 'Held stopped failed-process observations; always accepted_native_run=false. No completed analyzer execution or gate relaxation.'}
    dump(HERE / 'recipe.json', recipe)
    readiness = {'prepared_source_only': True, 'observer_sha256': sha(HERE / 'observe_partial.py'),
        'recipe_sha256': sha(HERE / 'recipe.json'), 'local_source_files': {str(p): sha(p) for p in local},
        'source_context_copies': copies, 'initial_review_failures': initial_failures,
        'new_native_calls': 0, 'new_probe_calls': 0, 'new_analyzer_calls': 0,
        'native_arrays_or_histories_read': False, 'root_execution_release_required': True}
    dump(HERE / 'source-only-readiness.json', readiness)
    print(json.dumps({'observer_sha256': readiness['observer_sha256'], 'recipe_sha256': readiness['recipe_sha256'],
        'readiness_sha256': sha(HERE / 'source-only-readiness.json'), 'initial_failures': list(FAILED), 'execution': 'HELD'}))
if __name__ == '__main__': main()
