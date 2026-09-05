"""Record Gate 1 only after inspecting the frozen build and numerical evidence."""
import csv
import json
from pathlib import Path
r=Path(__file__).resolve().parent
with (r/'curl-cuda.csv').open() as f:
    assert next(f).strip()=='# execution_space=Cuda'
    rows=list(csv.DictReader(f))
assert len(rows)==12 and all(x['decision']=='PASS' for x in rows)
projection=json.loads((r/'projection-results.json').read_text())
assert len(projection)==39
assert all(x['max_component_error']<2e-12 and x['max_curl_identity_error']<1e-10 for x in projection)
diag=json.loads((r/'diagnostic-results.json').read_text());assert len(diag)==3
assert all(x['decision']=='PASS' for x in diag)
assert '[100%] Built target athena' in (r/'source/build-direct-cuda/build.log').read_text()
result=dict(decision='PASS',cuda_uniform_cases=len(rows),production_projection_cases=len(projection),diagnostic_cases=len(diag),
            max_scaled_cuda_curl=max(float(x['curl_scaled']) for x in rows),
            max_scaled_cuda_target_error=max(float(x['target_scaled']) for x in rows),
            max_projection_component_error=max(x['max_component_error'] for x in projection),
            max_projection_curl_identity_error=max(x['max_curl_identity_error'] for x in projection),
            production_binary_sha256=(r/'source/build-direct-cuda/binary.sha256').read_text().split()[0],
            setup_failure='Git status rejected Kokkos symlink before first oracle execution; identical directory copy fixed provenance setup. Failure preserved in oracle-setup-failure and projection-setup-failure.log.',
            next_gate='Matched R16 M/256 stress through 6M; no downstream authorization from this correctness pass alone.')
(r/'gate1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
