"""Root metadata/source review and narrow one-shot release, no science calls."""
import ast, hashlib, json, subprocess
from pathlib import Path
P=Path(__file__).resolve().parent; R=P.parents[1]
S=R/'build-layer-research/continuum/inner-joint-principal-gate-v3-held-20261009'
V=S.with_name('inner-joint-principal-gate-v2-held-20261009')
def load(p): return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def pin(p):
 p=Path(p).resolve(); return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def check(x):
 assert pin(x['path'])=={k:x[k] for k in ('path','sha256','bytes')},x['path']
def save(name,x):
 q=P/name
 with q.open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert not (P/'authorization.json').exists() and not (S/'attempts/gate001').exists()
assert pin(S/'source-index.json')['sha256']=='ea06abf5a285ede3fb4224904a7b8932f2664dbc07c5b31718c19353570c7b4a'
assert pin(S/'recipe.json')['sha256']=='9123c2182d8df050f4fe41a8545667c3d1626d3ed40149cda3aa22ceb0a4a4a5'
i=load(S/'source-index.json');r=load(S/'recipe.json');v=load(V/'recipe.json')
for name in ('probe.cpp','gauge_proposal.hpp','reference_wave_map.hpp','PLAN.md'):
 assert (S/name).read_bytes()==(V/name).read_bytes()
for k in v:
 if k!='local_sources': assert r[k]==v[k],k
assert r['actual20_cases']==118 and r['exact_scalar_cases']==18
for name in ('check_exact.py','analyze.py','run_gate.py'):
 t=ast.parse((S/name).read_text()); assert t
 assert 'sys.flags.optimize != 0' in (S/name).read_text()
runner=(S/'run_gate.py').read_text()
assert "[recipe['python']['path'], '-I', str(attempt / 'check_exact.py')]" in runner
assert "[recipe['python']['path'], '-I', str(attempt / 'analyze.py'), str(raw)]" in runner
assert "release_debug_stdout_byte_equal" in runner
review=R/'build-layer-research/continuum/inner-joint-principal-gate-v3-independent-guard-review-20261009/receipt.json'
assert pin(review)['sha256']=='fe95614231988f7a1e8e3eb0a8b806c46323a3cab8c7887c0fa65875a8f0d022'
assert load(review)['passed'] is True and load(review)['source_inputs_unchanged'] is True
prior=R/'build-layer-research/continuum/inner-joint-principal-gate-v2-independent-source-review-20261009/receipt.json'
assert pin(prior)['sha256']=='ac2b4b5b4f7bec0b7fd5b0b467de27dc81a58cf37d92e88ad7bb80e317b70707'
rows=i['files']+i['external_inputs']+r['protected_inputs']+[pin(S/'source-index.json'),pin(review),pin(prior),pin(__file__)]
unique={}
for x in rows:
 check(x)
 if x['path'] in unique: assert x==unique[x['path']]
 unique[x['path']]=x
prod=subprocess.run(['git','diff','--exit-code',r['production_implementation'],'--','src','CMakeLists.txt'],cwd=R,capture_output=True)
(P/'production-diff.stdout').write_bytes(prod.stdout);(P/'production-diff.stderr').write_bytes(prod.stderr)
assert prod.returncode==0 and prod.stderr==b''
report=dict(root_review_passed=True,science_unchanged_from_reviewed_v2=True,
 source_index=pin(S/'source-index.json'),recipe=pin(S/'recipe.json'),
 independent_guard_review=pin(review),prior_conditional_math_review=pin(prior),
 verified_inputs=list(unique.values()),production_unchanged=True,
 reviewed_scope='constant-reference full20 principal, positive lapse/chi; finite118 plus exact18',
 limitations=['no nonflat reference identity','no nonlinear regularity','no puncture-uniform proof','no native or BH evolution'])
save('review.json',report)
save('authorization.json',dict(execution_released=True,scope=r['scope'],
 source_index_sha256=pin(S/'source-index.json')['sha256'],recipe_sha256=pin(S/'recipe.json')['sha256'],
 root_review=pin(P/'review.json'),isolated_runner=True,optimization=0))
print(json.dumps(dict(root_review_passed=True,verified_inputs=len(unique),authorization=pin(P/'authorization.json')),indent=2))
