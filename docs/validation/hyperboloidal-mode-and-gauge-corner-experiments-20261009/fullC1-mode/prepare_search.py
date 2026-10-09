"""Prepare a separate C1 saved-state search with source-level method changes recorded."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
OLD = ROOT / 'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD / 'index.json') == '396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69'
source = (OLD / 'reduced_modes.py').read_text()
edits = []


def replace(before, after):
    global source
    assert before in source
    edits.append({'before': before, 'after': after})
    source = source.replace(before, after)


replace('C0 N16', 'C1 N16')
replace("OLD = B/'full-tensor-propagator/full22-v2'\nLONG = B/'full-tensor-C0-long-window-20261009'\nFROZEN = LONG/'immutable-C0-long-window-20261009'\nMATRIX = OLD/'spatialnorm-projected-J20.npz'\nSTATES = LONG/'spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz'",
        "config = json.loads((P/'input-pins.json').read_text())\nMATRIX = ROOT/config['matrix']\nSTATES = ROOT/config['states']")
begin = source.index("assert sha(FROZEN/'index.json')")
end = source.index('started = time.monotonic()', begin)
old_pins = source[begin:end]
new_pins = """pins = {str((P/'input-pins.json').relative_to(ROOT)): sha(P/'input-pins.json'),
        str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
for path, wanted in config['catalogs'].items():
    index = ROOT/path
    assert sha(index) == wanted
    catalog = json.loads(index.read_text())
    for name, entry in catalog['files'].items():
        file = index.parent/name
        assert sha(file) == entry['sha256'] and file.stat().st_size == entry['bytes']
    pins[path] = wanted
for path, entry in config['artifacts'].items():
    file = ROOT/path
    assert sha(file) == entry['sha256'] and file.stat().st_size == entry['bytes']
    pins[path] = entry['sha256']
"""
replace(old_pins, new_pins)
replace('RR = U.T@JU', "RR = np.einsum('ki,kj->ij', U, JU, optimize=False)\nassert np.isfinite(RR).all()")
replace('dmd_matrix = Ur.T@Y@vh[:rank, :].T/singular[:rank][None, :]',
        "uy = np.einsum('ki,kj->ij', Ur, Y, optimize=False)\n    dmd_matrix = np.einsum('ki,ji->kj', uy, vh[:rank], optimize=False)/singular[:rank][None, :]\n    assert np.isfinite(dmd_matrix).all()")
replace('residual_vectors = JUr@vectors-(Ur@vectors)*eig[None, :]',
        "lifted = np.einsum('ij,jk->ik', Ur, vectors, optimize=False)\n        action = np.einsum('ij,jk->ik', JUr, vectors, optimize=False)\n        residual_vectors = action-lifted*eig[None, :]\n        assert np.isfinite(residual_vectors).all()")
replace('v = Ur@vectors[:, k]', "v = np.einsum('ij,j->i', Ur, vectors[:, k], optimize=False)")
(P / 'search.py').write_text(source)
record = {'scope': 'Same SVD/Ritz/DMD/rank/window method as frozen C0; new C1 pinned inputs and explicit dense contractions',
          'old_index_sha256': sha(OLD / 'index.json'),
          'old_source_sha256': sha(OLD / 'reduced_modes.py'),
          'new_source_sha256': sha(P / 'search.py'), 'literal_edits': edits}
(P / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
print('PREPARED', record['new_source_sha256'])
