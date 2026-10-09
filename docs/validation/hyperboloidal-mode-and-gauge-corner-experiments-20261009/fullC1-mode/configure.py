"""Pin completed C1 saved-state inputs; refuse to replace an existing config."""
from pathlib import Path
import argparse
import hashlib
import json

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
B = ROOT / 'build-layer-research/boundary'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--long-index', required=True)
parser.add_argument('--long-sha', required=True)
args = parser.parse_args()
assert not (P / 'input-pins.json').exists()
long_index = ROOT / args.long_index
assert sha(long_index) == args.long_sha
catalogs = {
    args.long_index: args.long_sha,
    'build-layer-research/boundary/full-tensor-covariant-c1/immutable-C1-global-screen-20261009/manifest.json': '5a44232529dd90e71f493895a6742817d73ee016066ab0cdead159dd19cdee4b',
    'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009/index.json': '396899199b3e94afdf28133c3db094690c18c094aa86e38ad42edfc8829c1d69',
}
for path, wanted in catalogs.items():
    index = ROOT / path
    assert sha(index) == wanted
    data = json.loads(index.read_text())
    for name, q in data['files'].items():
        assert sha(index.parent / name) == q['sha256']
        assert (index.parent / name).stat().st_size == q['bytes']
C1 = B / 'full-tensor-covariant-c1'
FULL = C1 / 'full22-candidate'
matrix = FULL / 'spatialnorm-projected-J20.npz'
states = B / 'full-tensor-C1-long-window-20261009/spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz'
assert sha(matrix) == '1093d7c07a71019cd69e578d52ca0c47635dc190b32ccb371683fd162132241e'
assert sha(states) == 'e9a8fbf271cb2ec67e23d7071d46f8afe2b544af2630ffda30d6f7bcca3ed62b'
paths = [matrix, states, FULL / 'spatialnorm-cache0.0001-metadata.json',
         FULL / 'spatialnorm-cache0.0001-lift.bin', C1 / 'server-spatialnorm',
         C1 / 'tangent_server.cpp', C1 / 'build-provenance.json',
         B / 'full-tensor-C0-long-window-20261009/spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz']
record = {'scope': 'C1 N16 projected continuous saved-state mode diagnostic versus frozen C0; no propagation/native evolution/LU',
          'matrix': str(matrix.relative_to(ROOT)), 'states': str(states.relative_to(ROOT)),
          'metadata': str((FULL / 'spatialnorm-cache0.0001-metadata.json').relative_to(ROOT)),
          'lift': str((FULL / 'spatialnorm-cache0.0001-lift.bin').relative_to(ROOT)),
          'native20_server': str((C1 / 'server-spatialnorm').relative_to(ROOT)),
          'catalogs': catalogs,
          'artifacts': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths},
          'source_sha256': sha(Path(__file__))}
(P / 'input-pins.json').write_text(json.dumps(record, indent=2) + '\n')
print('CONFIGURED', sha(P / 'input-pins.json'))
