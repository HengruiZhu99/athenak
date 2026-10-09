#!/usr/bin/env python3
"""Additive local evidence freeze; no Git operation or scientific rerun."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile,time
O=Path(__file__).resolve().parent;R=O.parents[1]
P=R/'boundary/total-j-finite-rb-control-20261009'
F=O/'immutable-finite-rb-C0-constraint-rates-20261009'
assert not F.exists();F.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
origins={};start=time.monotonic();head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def copy(src,dst):
    src=Path(src).resolve();dst=F/dst;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst);assert sha(src)==sha(dst);origins[str(dst.relative_to(F))]=str(src)
for src in sorted(O.iterdir()):
    if src.is_file() and src.suffix in ('.py','.md','.json') or src.is_file() and src.name=='proof.stderr':copy(src,src.name)
for name in ('debug-api-binding','debug-api-binding-v2'):
    for src in sorted((O/name).rglob('*')):
        if src.is_file():copy(src,src.relative_to(O))
for name in ('constraint_rate_api.hpp','run_constraint_rates.py','actual_bridge.cpp','all_m_data.hpp',
             'baseline_dual_spatial.hpp','spatial_dual.hpp','configuration_rows.hpp','generic_gauge.hpp',
             'radial_bridge.cpp','radial_normalization.hpp','point_energy.hpp','build.py',
             'build-release-latest.json','build-debug-latest.json','constraint-rate-api-authorization.json'):
    copy(P/name,Path('context/boundary')/name)
for name in ('inputs/basis/total_j_basis.hpp','inputs/basis/reference_conversion.hpp'):
    copy(P/name,Path('context/boundary')/name)
for name in ('report.json','receipt.json','stdout.json','stderr'):
    copy(P/'source-gate-release-003'/name,Path('context/source-gate-release-003')/name)
for name in ('FINAL-ADDENDUM.md','final-addendum-receipt.json','RECIPE.md'):
    copy(P.parent/'total-j-finite-rb-control-held-20261009'/name,Path('context/held')/name)
for src in (R/'continuum/finite-rb-subsidiary-context').rglob('*'):
    if src.is_file() and '__pycache__' not in src.parts:copy(src,Path('context/subsidiary-capsule')/src.relative_to(R/'continuum/finite-rb-subsidiary-context'))
copy(P.parent/'total-j-flat-core-envelope-20261009/inputs/flat_formula.hpp','context/frozen-flat-formula.hpp')
for mode in ('release','debug'):copy(P/('radial-bridge-'+mode),Path('executables')/('radial-bridge-'+mode))
names={'core':'attempt-1791555596293519000','gauge':'attempt-1791556101260565000','shell':'attempt-1791556133586266000'}
for stage,name in names.items():
    a=P/'constraint-rate-attempts'/name
    for src in sorted(a.iterdir()):
        if not src.is_file():continue
        if src.name in ('cases.jsonl','calls.json','calls.jsonl'):
            copy(src,Path('payloads')/(stage+'-'+src.name))
        else:copy(src,Path('results')/stage/src.name)
    archive=F/'payloads'/(stage+'-raw-calls.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        tar.add(a/'calls',arcname='calls',recursive=True)
    origins[str(archive.relative_to(F))]=str(a/'calls')+' (lossless tar.gz)'
files=[]
for p in sorted(F.rglob('*')):
    if not p.is_file():continue
    rel=str(p.relative_to(F));role='large_payload' if rel.startswith(('payloads/','executables/')) else 'source_or_receipt'
    files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p),'role':role,'origin':origins.get(rel)})
index={'scope':'immutable local finite-rb C0 stationary-reference continuum constraint-rate evidence; no native/global/eigenvalue/production changes',
       'freeze_HEAD':head,'public_runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
       'launch_HEAD':'2e0aa3b0d5ade2fef4d86807a4569fef61f6b202','files':files,
       'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),
       'large_payload_policy':'Keep all payload bytes locally. A compact Git archive may omit only entries explicitly tagged large_payload while retaining this original index verbatim.',
       'source_context':'Boundary build receipts pin full compiler/dependency context; subsidiary capsule preserves original helper/preamble. Executables copied with permissions.',
       'seconds':time.monotonic()-start}
(F/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
print(json.dumps({'index':str(F/'index.json'),'sha256':sha(F/'index.json'),'file_count':len(files),'bytes':index['total_bytes']},indent=2))
