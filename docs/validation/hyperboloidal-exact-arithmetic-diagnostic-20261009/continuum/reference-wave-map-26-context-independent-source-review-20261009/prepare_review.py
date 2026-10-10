"""Stdlib source review inventory only; no candidate imports or targets."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OWNER=ROOT/'boundary/reference-wave-map-far-dual-metric-flux-diagnostic-held-20261009'
ORIGINAL=ROOT/'boundary/reference-wave-map-far-dual-source002-held-20261009'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    if (HERE/'index.json').exists():raise RuntimeError('one-shot fresh review only')
    index=load(OWNER/'source-index.json')
    if sha(OWNER/'source-index.json')!='3da906bcddd42dafcee590a775ee7c142cf78767784c08fae7deb924134ff6a5':raise RuntimeError('owner index mismatch')
    if sha(OWNER/'recipe.json')!='022a924a8958d0a85d401476272316fe69d1d714dd9180f24f0445a7b3e7241a':raise RuntimeError('owner recipe mismatch')
    pins=load(OWNER/'input-pins.json')
    for row in index['files']:pins[row['path']]=row['sha256']
    pins[str(OWNER/'source-index.json')]=sha(OWNER/'source-index.json')
    before={}
    for path,wanted in pins.items():
        actual=sha(path)
        if actual!=wanted:raise RuntimeError('review input drift '+path)
        before[path]=actual
    paths=[Path(row['path']) for row in index['files']]
    paths += [OWNER/'source-index.json',ORIGINAL/'probe.cpp']
    paths += [ORIGINAL/'inputs'/name for name in ('reference_wave_map.hpp','reference_wave_map_legacy.hpp','arithmetic_traits.hpp','original_state.hpp','original_cast.hpp','dual_helpers.hpp','nonlinear_values.hpp')]
    repo=ROOT.parent
    paths += [repo/'src/z4c/hyperboloidal/conformal_constraints.hpp',repo/'src/z4c/hyperboloidal/layer_reference.hpp']
    copies=[]
    for i,path in enumerate(paths):
        if path.stat().st_size>1048576:continue
        if path.suffix in ('.jsonl','.npz','.npy'):continue
        destination=HERE/'source-copies'/('%03d-'%i+path.name)
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,destination)
        if sha(destination)!=sha(path):raise RuntimeError('source snapshot mismatch')
        copies.append(dict(original=str(path),copy=str(destination),sha256=sha(path),bytes=path.stat().st_size))
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
    r=load(OWNER/'recipe.json')
    old=load(r['original_oracle_report'])
    failures=old['failures']
    if len(failures)!=63:raise RuntimeError('saved failure count mismatch')
    zeros=sum(item['target'] in ('0','0.0','0.00') for item in failures)
    # Literal saved strings only: no rational conversion or target arithmetic.
    if zeros!=8:raise RuntimeError('saved zero-label count mismatch')
    if any(item['label']['component']<8 for item in failures if item['check']=='native-rhs-dual'):raise RuntimeError('saved RHS offset assumption false')
    final={path:sha(path) for path in pins}
    if final!=before:raise RuntimeError('source drift during independent review')
    save(HERE/'protected-inputs.json',before)
    save(HERE/'source-inventory.json',copies)
    save(HERE/'mechanical-read-history.json',dict(missing_file_reads=[str(ORIGINAL/name) for name in ('reference_wave_map.hpp','original_state.hpp','original_cast.hpp')],corrected_inputs_subdirectory=True,scientific_source_changed=False))
    save(HERE/'receipt.json',dict(status='PASS_SOURCE_ONLY_HELD',source_math_admission_passed=True,execution_released=False,
        owner_index_sha256=sha(OWNER/'source-index.json'),owner_recipe_sha256=sha(OWNER/'recipe.json'),
        owner_file_count=len(index['files']),protected_inputs=len(pins),protected_inputs_before_after_equal=True,
        copies=len(copies),original_failure_labels=63,original_saved_zero_targets=8,
        source_imports_compiles_queries_fraction_targets_array_decodes=False,
        original_payloads='streaming hashes only; no JSONL decoding',
        launch_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        scientific_scope='Observational26-context scalar arithmetic attribution only; no repair or gauge qualification',
        read_review_sha256=sha(HERE/'REVIEW.md')))
    files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(HERE.rglob('*')) if p.is_file()]
    save(HERE/'index.json',dict(status='immutable source-only independent review',files=files,file_count=len(files)))
    print(json.dumps(dict(index_sha256=sha(HERE/'index.json'),receipt_sha256=sha(HERE/'receipt.json'),protected_inputs=len(pins))))
if __name__=='__main__':main()
