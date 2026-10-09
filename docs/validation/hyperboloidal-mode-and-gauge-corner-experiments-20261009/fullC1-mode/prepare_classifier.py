"""Reuse the frozen C0 descriptive diagnostic with explicitly recorded C1 pins."""
from pathlib import Path
import hashlib
import json

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
OLD = ROOT / 'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = (OLD / 'classify_candidate.py').read_text()
config = json.loads((P / 'input-pins.json').read_text())
meta = json.loads((P / 'candidate-metadata.json').read_text())
edits = []


def replace(before, after):
    global source
    assert before in source
    source = source.replace(before, after)
    edits.append({'before': before, 'after': after})


replace("B = ROOT/'build-layer-research/boundary/full-tensor-propagator'", "B = ROOT/'build-layer-research/boundary/full-tensor-covariant-c1'")
replace("OLD = B/'full22-v2'", "OLD = B/'full22-candidate'")
replace('fc88b20d4953f5088aed97d04dce41ad0af039fc794a401a21cceb53280e1eee', meta['candidate_vectors_sha256'])
replace('e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1', sha(P / 'candidate-metadata.json'))
replace('495647e847aa77cca2c51615ed1fd0e007d71cdbb8b5ed310b37e332c7812bf0', config['artifacts'][config['native20_server']]['sha256'])
replace('767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6', config['artifacts'][config['matrix']]['sha256'])
replace('full-tensor-C0-long-window-20261009', 'full-tensor-C1-long-window-20261009')
replace('coefficient = orth.T@y', "coefficient = np.einsum('ij,i->j', orth, y, optimize=False)")
replace("(P/'candidate-vectors.npz', P/'candidate-metadata.json', B/'server-spatialnorm',", "(P/'candidate-vectors.npz', P/'candidate-metadata.json', P/'input-pins.json', B/'server-spatialnorm',")
(P / 'classify_candidate.py').write_text(source)
(P / 'classifier-preparation.json').write_text(json.dumps({
    'scope': 'Same descriptive native diagnostic as frozen C0, new C1 pins and explicit pair projection',
    'original_source_sha256': sha(OLD / 'classify_candidate.py'),
    'new_source_sha256': sha(P / 'classify_candidate.py'), 'literal_edits': edits}, indent=2) + '\n')
print('PREPARED', sha(P / 'classify_candidate.py'))
