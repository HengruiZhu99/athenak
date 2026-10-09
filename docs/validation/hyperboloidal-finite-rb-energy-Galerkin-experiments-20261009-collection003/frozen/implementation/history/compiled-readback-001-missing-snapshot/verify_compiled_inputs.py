"""Post-gate byte readback of actually recorded build inputs; no compile/run."""
from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();rows=[]
for mode,attempt in [('release','release-005'),('debug','debug-003')]:
 d=P/'build-attempts'/attempt;r=json.loads((d/'receipt.json').read_text());checks=[]
 for path,h in r['compiler_dependency_hashes'].items():checks.append((path,sha(path)==h))
 for path,h in r['link_archive_hashes'].items():checks.append((path,sha(path)==h))
 for name,h in r['sources_before'].items():checks.append((str(P/name),sha(P/name)==h));checks.append((str(d/name),sha(d/name)==h))
 for exe in (P/('radial-bridge-'+mode),d/('radial-bridge-'+mode)):checks.append((str(exe),sha(exe)==r['executable_sha256']))
 rows.append({'mode':mode,'receipt_sha256':sha(d/'receipt.json'),'recorded_compile_launch_HEAD':r['launch_HEAD'],'runtime_source_commit':r['runtime_source_commit'],'compiler_version_as_recorded':r['compiler_version'],'compiler_dependencies':len(r['compiler_dependency_hashes']),'archives':len(r['link_archive_hashes']),'total_bytechecks':len(checks),'mismatches':[p for p,ok in checks if not ok],'passed':all(ok for _,ok in checks),'retained_bytecopy_permission_note':'Per-attempt bytecopies are retained; executable permission may differ from working executable. No as-built byte identity is inferred from permission bits.'})
report={'passed':all(r['passed'] for r in rows),'rows':rows,'elapsed':time.monotonic()-start,'checker_sha256':sha(__file__),'scope':'Recorded source/dependency/library/executable byte identity at current readback; compiler version/command were recorded at build, binary compiler hash was not captured at build.'}
(P/'compiled-input-current-readback.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']
