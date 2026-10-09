"""Read-only source capture; no scientific helper/checker import or execution."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT / 'build-layer-research/reference-wave-map-native-held-20261009'
HERE = Path(__file__).resolve().parent
INDEX = OWNER / 'source-review-index-v2.json'
EXPECTED = '23f2323dcc42cd70fa93b221b92a5eff4da312ac459b8f451b3fa698a46bf71e'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

assert sha(INDEX) == EXPECTED
data = json.loads(INDEX.read_text())
copies = HERE / 'source-copies'
copies.mkdir(exist_ok=False)
shutil.copyfile(INDEX, copies / INDEX.name)
for relative, item in data['files'].items():
    src = OWNER / relative
    assert sha(src) == item['sha256'] and src.stat().st_size == item['bytes'], relative
    dst = copies / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    assert sha(dst) == sha(src) == item['sha256'], relative
for item in data['compiled_native_executables'].values():
    assert sha(Path(item['path'])) == item['sha256']
assert sha(INDEX) == EXPECTED
(HERE / 'capture.json').write_text(json.dumps({
    'status': 'SOURCE_ONLY_CAPTURED', 'owner_index_sha256': EXPECTED,
    'copied_source_file_count': len(data['files']),
    'native_executables_rehashed_not_executed': data['compiled_native_executables'],
    'source_queries_compiles_analyzer_or_probe_calls': 0,
    'capture_source_sha256': sha(Path(__file__)),
}, indent=2, allow_nan=False) + '\n')
print('PASS source-only capture:', len(data['files']), 'files; no scientific execution')
