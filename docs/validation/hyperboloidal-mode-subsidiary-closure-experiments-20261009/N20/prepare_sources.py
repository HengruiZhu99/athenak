"""Grid-only replay of frozen N16 comparator; no old files modified."""
from pathlib import Path
import hashlib,json,difflib
W=Path(__file__).resolve().parent;R=W.parents[2]
P=W.parent/'mode-subsidiary-defect/immutable-mode-subsidiary-defect-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'index.json')=='0ccc0eb70207cfdc7ba14d7da156063902c0850119690215f65bc0c8b3c3321b'
for name,row in json.loads((P/'index.json').read_text())['files'].items():assert sha(P/name)==row['sha256'],name
meta=json.loads((W/'candidate-metadata.json').read_text());assert sha(W/'candidate-vectors.npz')==meta['candidate_vectors_sha256']
changes={}
source=(P/'comparator.cpp').read_text();grid_old='g.n[a]=22;g.h[a]=2.2/16;g.first[a]=-.5*21*g.h[a];';grid_new='g.n[a]=26;g.h[a]=2.2/20;g.first[a]=-.5*25*g.h[a];'
assert source.count(grid_old)==1
cpp=source.replace(grid_old,grid_new);(W/'comparator.cpp').write_text(cpp)
(W/'inputs').mkdir(exist_ok=True)
(W/'N16-to-N20-callback.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),cpp.splitlines(True),fromfile='frozen-N16/comparator.cpp',tofile='fresh-N20/comparator.cpp')))
build=(P/'prepare_build.py').read_text().replace("M = R/'build-layer-research/continuum/discrete-mode-identification'","M = W")
build=build.replace('fc88b20d4953f5088aed97d04dce41ad0af039fc794a401a21cceb53280e1eee',sha(W/'candidate-vectors.npz')).replace('e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1',sha(W/'candidate-metadata.json'))
(W/'prepare_build.py').write_text(build)
driver=(P/'run_comparator.py').read_text()
driver=driver.replace("V2=B/'full22-v2'","V2=R/'build-layer-research/boundary/full-tensor-C0-N20-20261009/full22'")
driver=driver.replace("MODE=R/'build-layer-research/continuum/discrete-mode-identification'","MODE=W")
for old,new in [('fc88b20d4953f5088aed97d04dce41ad0af039fc794a401a21cceb53280e1eee',sha(W/'candidate-vectors.npz')),
 ('e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1',sha(W/'candidate-metadata.json')),
 ('767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6','bfc114a9495b49d01b9275f4efd8a1fe49d167273f8d849631d25cddf9c2a103'),
 ('98b7b7f459cce564398769cff48545cb6e89eb3708d01b2018a81b3a393d4e26','cd72ca5486a65e90744ed04f00a4da58f7246e2f63e5743147fba19209babb86'),
 ("'16','2.2','0.0001'","'20','2.2','0.0001'")]:
 assert old in driver;driver=driver.replace(old,new)
(W/'run_comparator.py').write_text(driver)
(W/'source-preparation.json').write_text(json.dumps({'frozen_N16_index_sha256':sha(P/'index.json'),
 'frozen_N16_callback_sha256':sha(P/'comparator.cpp'),'fresh_N20_callback_sha256':sha(W/'comparator.cpp'),
 'callback_literal_only_change':{grid_old:grid_new},'candidate_vectors_sha256':sha(W/'candidate-vectors.npz'),
 'candidate_metadata_sha256':sha(W/'candidate-metadata.json'),
 'scope':'Grid-only callback geometry change; same frozen subsidiary/native lifts/primitive and constraint stencils. Driver changes only pinned N20 inputs, metadata, lift and native callback grid argument.'},indent=2)+'\n')
print('PREPARED',sha(W/'comparator.cpp'),sha(W/'candidate-vectors.npz'))
