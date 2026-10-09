from pathlib import Path
import json, hashlib

BASE = Path('/Users/hz0693/research/hyperboloidal/build-layer-research')
HERE = Path(__file__).resolve().parent
OLD = BASE/'manufactured-angular-Gaussian-screen-v2-held-20261009'
OWNER = BASE/'manufactured-angular-Gaussian-a2-physical-screen-held-20261009'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


fixed = {
    'source-index.json': '287f96ced076617c877d4bfbd7d5d63e484dd8b2d2eb61a53c96e1e2f60fbf7b',
    'recipe.json': '5b22ca77da59d52357d77bc9e00b20d5aace8ecfaa1c9c78dbbc4c84a4df0561',
    'screen.py': '6c9cae79f014d30a81717d05320c865ff406415afd2c0bf249b3571737636c8d',
}
for name, digest in fixed.items():
    if sha(OLD/name) != digest:
        raise RuntimeError('old source differs '+name)
receipt = OLD/'attempts/screen001/receipt.json'
result = OLD/'attempts/screen001/result.json'
if (sha(receipt) != '3b46677afd4dce07e38c18ddf3d41e0497e9e44a88759b2929b1ee30cee12979'
        or sha(result) != 'e005ba473f652ea762aa15b91ee1d90d1075eaeb3755627139d0370242001f7e'):
    raise RuntimeError('actual previous screen differs')
prior = load(receipt)
if not (prior['completed'] and prior['returncode'] == 0 and prior['inputs_unchanged']
        and load(result)['checks_passed']):
    raise RuntimeError('actual prior consistency success required')
recipe = load(OLD/'recipe.json')
recipe['a'] = '2'
recipe['pins'].update({str(OLD/name): digest for name, digest in fixed.items()})
for path in [receipt, result, HERE/'prepare_release001.py', HERE/'preparation-history.json']:
    recipe['pins'][str(path)] = sha(path)
for path, digest in recipe['pins'].items():
    if sha(path) != digest:
        raise RuntimeError('protected input differs '+path)
OWNER.mkdir(exist_ok=False)
(OWNER/'screen.py').write_bytes((OLD/'screen.py').read_bytes())
write(OWNER/'recipe.json', recipe)
(OWNER/'PLAN.md').write_text('''# Fixed a=2 physical-event height control

One new parameter control of the accepted Gaussian physical-event screen.
Only a changes from1/2 to2. The complete source, 28032 events per precision,
80/110 digits, origin series40/60 and identity/precision gates are unchanged.
Original negative cases and receipts remain preserved. S1/a2 satisfies a>=S/2.
The generic formulas h=R/sqrt(R^2+a^2) and referenceD=a^2/(R^2+a^2) retain
their meanings. This tests a gentler CMC slope at the same physical events.
No native inverse, curvature/connection jets, kernel query or evolution is run.
Nonpositive D records remain saved. Finite sampling cannot establish global
positivity, an efficient native candidate or PDE stability.
''')
rows = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(OWNER.iterdir()) if path.is_file()]
write(OWNER/'source-index.json', dict(files=rows, source_only=True,
      parameter_control=dict(a_before='1/2', a_after='2'),
      scientific_source_byte_identical=True))
pins = dict(recipe['pins'])
for row in rows:
    pins[row['path']] = row['sha256']
pins[str(OWNER/'source-index.json')] = sha(OWNER/'source-index.json')
write(HERE/'source-review001.json', dict(passed=True,
      verified_unique_pins=len(pins), source_byte_identical_to_previous_accepted=True,
      only_scientific_parameter_change='a:1/2->2',
      actual_prior_consistency_pass_required=True, root_reviewed_generic_a_formula=True,
      no_scientific_import_or_execution_by_preparation=True,
      scope='One finite physical-event parameter control; no global/native/PDE admission'))
auth = dict(finite_Gaussian_screen_authorized=True,
            recipe_sha256=sha(OWNER/'recipe.json'),
            source_index_sha256=sha(OWNER/'source-index.json'),
            screen_source_sha256=sha(OWNER/'screen.py'),
            root_review_sha256=sha(HERE/'source-review001.json'),
            scope='One unchanged-code fixed a2 finite physical-event control')
write(HERE/'authorization.json', auth)
write(HERE/'release.json', dict(owner=str(OWNER), pins=pins,
      authorization_sha256=sha(HERE/'authorization.json')))
print(json.dumps(dict(passed=True, pins=len(pins),
      authorization_sha256=sha(HERE/'authorization.json'),
      source_index_sha256=sha(OWNER/'source-index.json'),
      recipe_sha256=sha(OWNER/'recipe.json'))))
