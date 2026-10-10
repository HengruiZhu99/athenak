"""HELD root metadata preparation. Do not run before root source review."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SRC = BASE/'boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009'
REVIEW = BASE/'continuum/reference-wave-map-v9-mass-measure-independent-source-review-20261009'
EXPECTED = {
    SRC/'source-index.json':'c8d1b760ec7dea2711719078e434139af28496f5050c3ea7a336d6e39ff2103f',
    SRC/'recipe.json':'66b29ff7f20668bce2bcdee035dc97900ab95f565e8bd2dee2fd8c174ddd7e78',
    SRC/'diagnose_mass.py':'2d6d0b00b794b0f2144224e49e3f80b992cbbe70735b5c8e4f34bc7effd9d702',
    SRC/'run_once.py':'31cb3df6b839776d9cc561965211a57236ae686fee40c780358cd9bc80538d87',
    REVIEW/'index.json':'307340b0796ae26463e8cfc612f3fe81915ba38a9740135336d1ea4f7f57202a',
    REVIEW/'receipt.json':'3e0f8c6a539714775306191dcdbdae1005104b6e51aea055c8e976ea862503d0',
}
ENVIRONMENT = {
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
    'PYTHONDONTWRITEBYTECODE':'1','PYTHONOPTIMIZE':'0',
    'PYTHONPATH':'/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/python-deps:'
        '/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/lib/python3.9/site-packages',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    path = Path(path)
    if path.suffix == '.jsonl' or path.stat().st_size > 1 << 20:
        raise RuntimeError('compact metadata JSON only')
    return json.loads(path.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def merge(pins,path,digest):
    path = str(Path(path).absolute())
    if path in pins and pins[path] != digest:
        raise RuntimeError('conflicting pin: '+path)
    pins[path] = digest


def verify(pins):
    for path,digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed protected input: '+path)


def scipy_inventory(recipe):
    search = [Path(x) for x in recipe['environment']['PYTHONPATH'].split(':')]
    first = search[0]
    if first != BASE/'boundary/python-deps' or not (first/'scipy/__init__.py').is_file():
        raise RuntimeError('exact first SciPy package path absent')
    if (SRC/'scipy').exists() or (SRC/'scipy.py').exists() or (first/'scipy.py').exists():
        raise RuntimeError('unexpected SciPy shadow path')
    roots = [first/'scipy']
    roots += sorted(x for x in first.glob('scipy-*.dist-info') if x.is_dir())
    roots += sorted(x for x in first.iterdir() if x.is_dir() and
        ((x.name.startswith('scipy') and x.name.endswith('.libs')) or x.name=='.dylibs'))
    if len([x for x in roots if x.name.endswith('.dist-info')]) != 1:
        raise RuntimeError('expected one installed SciPy distribution')
    files = {}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc','.pyo'}:
                merge(files,path,sha(path))
    if not any('/linalg/_fblas.' in p for p in files):
        raise RuntimeError('SciPy BLAS extension absent')
    return dict(roots=[str(x) for x in roots],files=files,bytecode_excluded=True,
                binaries_and_scientific_payloads_metadata_only=True,imports_executed=False)


def main():
    if not (sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize == 0):
        raise RuntimeError('root metadata preparation requires unoptimized -I -B')
    for path,digest in EXPECTED.items():
        if sha(path) != digest:
            raise RuntimeError('exact reviewed pin differs: '+str(path))
    pins = load(HERE/'source-pins001.json')
    independent = load(REVIEW/'pins-before.json')
    if len(pins) != 3283 or pins != independent or independent != load(REVIEW/'pins-after.json'):
        raise RuntimeError('all3283 root/independent source pins required')
    for path,digest in EXPECTED.items():
        merge(pins,path,digest)
    for index_path in [REVIEW/'index.json',HERE/'launcher-source-index.json']:
        for row in load(index_path)['files']:
            path = Path(row['path'])
            if path.stat().st_size != row['bytes']:
                raise RuntimeError('source/review byte size differs')
            merge(pins,path,row['sha256'])
        merge(pins,index_path,sha(index_path))
    recipe = load(SRC/'recipe.json')
    if recipe['environment'] != ENVIRONMENT:
        raise RuntimeError('exact pinned environment differs')
    if recipe['python'] != '/Library/Developer/CommandLineTools/usr/bin/python3':
        raise RuntimeError('literal interpreter differs')
    if str(Path(recipe['python']).resolve()) != recipe['resolved_python']:
        raise RuntimeError('resolved interpreter differs')
    if (recipe['first_radius'],recipe['last_radius'],recipe['expected_components']) != (609,640,131072):
        raise RuntimeError('fixed diagnostic registry differs')
    for key in ['attempt','outer_attempt']:
        if Path(recipe[key]).exists():
            raise RuntimeError('fixed owner output already exists: '+recipe[key])
    if (HERE/'outer-invocation001').exists():
        raise RuntimeError('root destination already exists')
    review = load(REVIEW/'receipt.json')
    if not (review['passed'] is True and review['reviewed_source_index_sha256']==EXPECTED[SRC/'source-index.json']):
        raise RuntimeError('passed independent source review required')
    if load(HERE/'source-review001.json').get('root_source_math_and_admission_review_passed') is not True:
        raise RuntimeError('root source review absent')
    runtime = scipy_inventory(recipe)
    for path,digest in runtime['files'].items():
        merge(pins,path,digest)
    verify(pins)
    for name in ['prepare_release001.py','launch.py']:
        ast.parse((HERE/name).read_text())
    write(HERE/'scipy-runtime-pins001.json',runtime)
    merge(pins,HERE/'scipy-runtime-pins001.json',sha(HERE/'scipy-runtime-pins001.json'))
    write(HERE/'pins-prepared001.json',pins)
    authorization = dict(bounded_saved_mass_diagnostic_authorized=True,
        diagnostic_source_sha256=EXPECTED[SRC/'diagnose_mass.py'],
        wrapper_source_sha256=EXPECTED[SRC/'run_once.py'],
        source_index_sha256=EXPECTED[SRC/'source-index.json'],recipe_sha256=EXPECTED[SRC/'recipe.json'],
        independent_review_receipt=dict(path=str(REVIEW/'receipt.json'),sha256=EXPECTED[REVIEW/'receipt.json']),
        independent_review_index_sha256=EXPECTED[REVIEW/'index.json'],
        only_selected_operands_no_E_accumulation_no_SVD_no_queries=True,
        root_process_group_cap_seconds=180,launcher_source_index_sha256=sha(HERE/'launcher-source-index.json'),
        scipy_runtime_pins_sha256=sha(HERE/'scipy-runtime-pins001.json'),
        pins_prepared_sha256=sha(HERE/'pins-prepared001.json'))
    write(HERE/'release.json',authorization)
    verify(pins)
    receipt = dict(prepared=True,science_executed=False,protected_pins=len(pins),
        original_root_source_pins=3283,scipy_metadata_files=len(runtime['files']),
        root_process_group_cap_seconds=180,inputs_unchanged=True,
        authorization_sha256=sha(HERE/'release.json'))
    write(HERE/'preparation-receipt001.json',receipt)
    print(json.dumps(receipt,sort_keys=True))


if __name__ == '__main__':
    main()
