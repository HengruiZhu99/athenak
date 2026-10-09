"""Read back stable as-built inputs and reconstruct hash-identical pointers."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
P=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
deps={};counts=[];pointer=[]
for mode,attempt,gate in [('release','release-004','source-gate-release-002'),('debug','debug-002','source-gate-debug-001')]:
    bp=P/'build-attempts'/attempt/'receipt.json';b=json.loads(bp.read_text());g=json.loads((P/gate/'receipt.json').read_text())
    assert b['exit_code']==g['exit_code']==0
    assert b['sources_before']==b['sources_after']
    assert b['executable_sha256']==g['executable_sha256']
    assert b['sources_before']['radial_bridge.cpp']==g['source_sha256']
    for name,digest in (b['compiler_dependency_hashes']|b['link_archive_hashes']).items():
        original=Path(name);source=original
        if source.parent==P and (P/'build-attempts'/attempt/source.name).is_file():source=P/'build-attempts'/attempt/source.name
        assert sha(source)==digest,name
        if name in deps:assert deps[name]==digest
        deps[name]=digest
        if original.is_relative_to(ROOT/'src'):
            relative=original.relative_to(ROOT)
            data=subprocess.check_output(['git','show','27c19d20696ea6dd4704032c51dfd026218f64f2:'+str(relative)],cwd=ROOT)
            assert hashlib.sha256(data).hexdigest()==digest
    assert (P/gate/'stderr').stat().st_size==0
    assert sha(P/gate/'stdout.json')==g['stdout_sha256']
    data=(json.dumps({'attempt':str(P/'build-attempts'/attempt),'receipt_sha256':sha(bp),'executable_sha256':b['executable_sha256']},indent=2)+'\n').encode()
    assert hashlib.sha256(data).hexdigest()==g['build_latest_sha256']
    out=HERE/('reconstructed-build-'+mode+'-latest.json');assert not out.exists();out.write_bytes(data)
    pointer.append({'mode':mode,'sha256':sha(out),'matches_original_gate_record':True,'method':'Additive reconstruction from retained attempt path, exact receipt hash and historical executable hash; original superseded pointer not rewritten.'})
    counts.append({'mode':mode,'compiler_dependencies':len(b['compiler_dependency_hashes']),'link_archives':len(b['link_archive_hashes']),'historical_executable_sha256':b['executable_sha256'],'old_executable_retained':False,'fresh_executable_readback':False})
assert (P/'source-gate-release-002/stdout.json').read_bytes()==(P/'source-gate-debug-001/stdout.json').read_bytes()
review={'status':'PASS_as_built_source_dependency_output_readback_with_explicit_binary_preservation_limitation','builds':counts,'unique_dependency_and_link_inputs_verified':len(deps),'all_compiled_production_headers_match_27c19':True,'numerical_outputs_byte_equal':True,'pointer_reconstruction':pointer,'preservation_error':'Reused builder overwrote both accepted executable working paths before copying them; exact historical binary hashes and scientific source/build/output records remain, but old binaries are no longer locally inspectable. Future unique-path retention is required.','scope':'Source/configuration derivative only; no actual radial matrix/constraint rate/eigen/propagation admission.','scientific_rerun_performed':False}
out=HERE/'root-provenance-review.json';assert not out.exists();out.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'unique_inputs':len(deps),'review_sha256':sha(out)}))
