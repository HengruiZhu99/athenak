"""Read-only independent provenance and declared local gate review."""
from pathlib import Path
import hashlib,json,subprocess,time,math
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=ROOT/'build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009'
def main():
    start=time.monotonic();assert sha(P/'index.json')=='25f09ff04e1f7067147b9bb75ff916750f989beb4366477b1db8748fde74dd31'
    idx=json.loads((P/'index.json').read_text());checked=0
    for e in idx['files']:
        p=P/e['path'];assert sha(p)==e['sha256'] and p.stat().st_size==e['bytes'];checked+=1
    source=idx['external_source_inputs']
    for path,h in source.items():assert sha(ROOT/path)==h,path
    prod={k:h for k,h in source.items()if k=='CMakeLists.txt'or k.startswith('src/')}
    assert len(prod)==365
    for path,h in prod.items():assert hashlib.sha256(subprocess.check_output(['git','show','27c19d20696ea6dd4704032c51dfd026218f64f2:'+path],cwd=ROOT)).hexdigest()==h,path
    deps=0
    for mode,mapping in idx['external_compiler_dependencies'].items():
        for path,h in mapping.items():assert sha(path)==h,path;deps+=1
    A=P/idx['accepted_attempt'];r=json.loads((A/'receipt.json').read_text());recipe=json.loads((A/'release-recipe.json').read_text())
    assert sha(A/'receipt.json')=='45f789e122a78a3c3b651dcc1317d83d7f09cb4931032ca841952ece881ed8d2'
    assert r['source_before']==r['source_after']==recipe['inputs']==source
    assert r['passed_local_gate']and r['sources_unchanged']and r['release_debug_equal']and not r['operators_or_evolution_run']
    out=json.loads((A/'release.json').read_text());assert out==json.loads((A/'debug.json').read_text())
    for key,tol in recipe['thresholds'].items():
        v=out[key];v=v[-1]if isinstance(v,list)else v;assert math.isfinite(v)and v<=tol,key
    assert out['factored_reference_max']==0
    assert [out[k]for k in ['reference_rows','offconstraint_rows','core_rows','tiny_gauge_rows','dual_directions']]==[756,756,132,480,1440]
    assert all(q['returncode']==0 and q['stderr_sha256']==hashlib.sha256(b'').hexdigest()for q in r['commands'])
    # The independent reviewer has read the frozen helper, the complete C0
    # metric-time/source construction, and the embedding double-double oracle.
    receipt={'passed':True,'scope':'source algebra inspection, saved thresholds and complete hash/dependency recheck; no source query or rerun',
      'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'capsule_index_sha256':sha(P/'index.json'),'source_sha256':sha(__file__),
      'checked_frozen_files':checked,'checked_source_inputs':len(source),'checked_compiler_dependencies':deps,'production_files_identical_to_27c19':len(prod),
      'accepted_receipt_sha256':sha(A/'receipt.json'),
      'math_review':['Stationary reference scaled connection equals the embedding pullback connection; temporal connection slots vanish.',
       'Physical P lapse and shift formulas assemble both poles exactly once and retain Lambda/Z terms.',
       'Deviation expansion uses inverse-live times metric-deviation times inverse-reference and exact stationary identities. It is not a numerical reference RHS subtraction.',
       'C0 tensor/chi time rows plus candidate lapse/shift time rows reconstruct the full four-metric source identity off constraints.',
       'Explicit FMA double-double improves only independent embedding oracle arithmetic; it is not certified interval arithmetic or a production arithmetic change.'],
      'limitations':['Global harmonic core is diagnostic; no puncture/blended BH gauge selected.','Finite local gates do not establish principal completeness, continuum growth control, exact-scri closure, or native evolution stability.'],
      'seconds':time.monotonic()-start}
    dest=HERE/'receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
