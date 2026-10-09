"""One-shot compact collection after complete source/review/hash preflight."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = ROOT/'docs/validation/hyperboloidal-Q-sigma3-local-screen-experiments-20261009'
FILES = {}
OMITTED = {}
PENDING = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def queue(source, target):
    data = source.read_bytes()
    assert target not in FILES and target not in OMITTED
    assert not Path(target).is_absolute() and '..' not in Path(target).parts
    spec = {'source': str(source.relative_to(ROOT)),
            'sha256': sha(data), 'bytes': len(data)}
    if source.suffix == '.json':
        finite(json.loads(data))
    # All executable objects and arrays are metadata-only, regardless of size.
    executable_magic = data[:4] in {
        b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
        b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe',
    }
    if (executable_magic or data[:8] == b'!<arch>\n'
            or source.suffix in {'.npz','.npy','.o','.a','.bin','.rst','.raw'}
            or len(data) > 1048576):
        OMITTED[target] = spec
        return
    FILES[target] = spec
    PENDING[target] = data


def frozen(folder, expected, target):
    base = ROOT/'build-layer-research'/folder
    path = base/'index.json'
    assert sha(path.read_bytes()) == expected
    record = json.loads(path.read_text())
    finite(record)
    for name, spec in record['files'].items():
        source = base/name
        data = source.read_bytes()
        assert sha(data) == spec['sha256']
        assert len(data) == spec['bytes']
        queue(source, target+'/'+name)
    queue(path, target+'/index.json')


assert not DEST.exists(), 'Refuse to mutate an existing archive'
head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
assert head == '1d53345821c035f62ea15010076bd9e5235c7923'
assert (HERE/'archive-README-final.md').is_file()
assert (HERE/'independent-draft-review.json').is_file()
assert json.loads((HERE/'root-review.json').read_text())['required_corrections'] == []
frozen('continuum/q-sigma3-frozen-fourier/immutable-Q-sigma3-raw22-intrinsic20-frozen-Fourier-20261009',
       '97eca136515520bb7a551b32013ce4faa7bafde0d3a079cee2620f8367e95aa3','original-v1')
frozen('continuum/q-sigma3-frozen-fourier-deterministic/immutable-Q-sigma3-deterministic-reanalysis-20261009',
       '943db2465bb361802fdd237f7658f11f66f142ecff6d147920d0c8e9393a4077','accepted-additive')
for name in ('root-review.json','independent-draft-review.json','review_frozen.py',
             'independent_review.py','independent-review.stdout','independent-review.stderr'):
    queue(HERE/name,'reviews/'+name)
queue(HERE/'audit-draft.md','reviews/reviewed-draft.md')
queue(HERE/'archive-README.md','reviews/reviewed-README.md')
queue(HERE/'archive-README-final.md','README.md')
queue(HERE/'summary.json','summary.json')
queue(HERE/'ROOT-NOTATION-ADDENDUM.md','ROOT-NOTATION-ADDENDUM.md')
queue(Path(__file__),'collect_sigma3_stage.py')
queue(ROOT/'docs/validation/hyperboloidal-constraint-propagation-experiments-20261009/verify_constraint_archive.py',
      'verify_archive.py')
catalog = {
    'scope': 'Negative actual C0/Q sigma3 local primitive Fourier comparison. No source adoption, native/global run, subsidiary classification, stability or BH admission.',
    'compiled_production_implementation': '27c19d20696ea6dd4704032c51dfd026218f64f2',
    'collection_head': head, 'files': FILES, 'omitted_large_payloads': OMITTED,
    'large_artifacts': 'All executables, objects, binary arrays and files larger than1MiB remain metadata-only. Original indexes unchanged.',
}
finite(catalog)
catalog_bytes = (json.dumps(catalog,indent=2,allow_nan=False)+'\n').encode()
# No destination writes occur before every preflight/read/hash/JSON succeeds.
DEST.mkdir(parents=True)
for target, data in PENDING.items():
    p = DEST/target
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(data)
    assert p.read_bytes() == data
(DEST/'catalog.json').write_bytes(catalog_bytes)
print(json.dumps({'files':len(FILES),'bytes':sum(v['bytes'] for v in FILES.values()),
                  'catalog_sha256':sha(catalog_bytes),'omitted_payloads':len(OMITTED)}))
