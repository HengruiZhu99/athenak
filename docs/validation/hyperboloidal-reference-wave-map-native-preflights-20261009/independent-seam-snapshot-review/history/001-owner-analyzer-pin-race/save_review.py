"""Save source-review findings only; do not import/run owner scientific code."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OWNER=ROOT/'build-layer-research/reference-wave-map-native-held-20261009'
COPIES=HERE/'source-copies'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
index=json.loads((COPIES/'source-review-index-v2.json').read_text())
for relative,item in index['files'].items():
    assert sha(COPIES/relative)==sha(OWNER/relative)==item['sha256'],relative
additional=['src/outputs/restart.cpp','src/outputs/history.cpp','src/outputs/io_wrapper.hpp',
 'src/mesh/mesh.hpp','src/mesh/mesh.cpp','src/driver/driver.cpp','src/athena.hpp',
 'src/z4c/z4c.hpp','src/z4c/z4c.cpp','src/z4c/z4c_hyperboloidal.cpp',
 'src/z4c/hyperboloidal/cartesian_patch.hpp','src/z4c/hyperboloidal/athenak_bridge.hpp',
 'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py',
 'build-layer-research/time-projection-controls/rst-reader-gate/abi.json']
pins=[]
for name in additional:
    p=ROOT/name;dst=HERE/'additional-source-copies'/name
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    assert sha(p)==sha(dst)
    pins.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size})
receipt={
 'passed_source_math_layout_transport_review':True,
 'passed_scientific_seam_or_snapshot_execution':False,
 'scope':'Independent exact-byte source review only. No source/helper/probe query, compilation, snapshot readback, scientific recomputation or evolution.',
 'reviewed_owner_index_sha256':sha(COPIES/'source-review-index-v2.json'),
 'source_file_count':len(index['files']),
 'seam_source_sha256':sha(COPIES/'native_seam_and_snapshot.cpp'),
 'analyzer_source_sha256':sha(COPIES/'analyze_snapshots.py'),
 'probe_recipe_sha256':sha(COPIES/'probe-build-held/recipe.json'),
 'report_sha256':sha(HERE/'REVIEW.md'),
 'independent_source_sha256':{'capture_review_inputs.py':sha(HERE/'capture_review_inputs.py'),'save_review.py':sha(Path(__file__))},
 'additional_production_and_RST_context':pins,
 'fixed_seam_scope':{'stored_cells':6,'t0_states':3,'rows':18,'exact_flat_core_available':False,'reference_RHS_max_scope':'six fixed reference cells'},
 'findings':[
  {'id':'successful-seam-prerequisite','severity':'snapshot-admission-provenance','status':'reported-before-execution','detail':'Analyzer/auth schema do not require a pinned successful --seam receipt; root gate must provide it or analyzer must enforce it.'},
  {'id':'end-of-case-source-pins','severity':'snapshot-provenance','status':'reported-before-execution','detail':'Source/executable/auth pins checked before case but not after; declared mid-case immutability needs post-readback or external receipt.'},
  {'id':'early-failure-capture','severity':'driver-scope-note','status':'reported-before-execution','detail':'Guards/history parsing before try can fail without internal failure.json; preserve with outer command receipt.'}],
 'source_query_compile_probe_snapshot_evolution_calls':0,
 'native_executable_hashes_reverified_not_executed':index['compiled_native_executables'],
 'gate_owner':'root',
 'owner_source_bytes_unchanged_at_review_completion':True}
(HERE/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({'receipt_sha256':sha(HERE/'receipt.json'),'report_sha256':receipt['report_sha256'],'passed_source_review':True,'scientific_execution':False},indent=2))
