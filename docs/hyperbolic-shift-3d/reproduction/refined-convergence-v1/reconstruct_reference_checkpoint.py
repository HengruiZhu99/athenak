"""Verify local evidence and reconstruct the admitted segment01 checkpoint.

Requires the preserved raw campaign and explicitly calibrated analysis reader.
Refuses existing diagnostic output; does not replace production files or restart.
"""
from pathlib import Path
import json,hashlib,subprocess,time,argparse
doc=Path(__file__).resolve().parent;root=next(p for p in doc.parents if (p/'CMakeLists.txt').exists());parser=argparse.ArgumentParser();parser.add_argument('--segment',type=int,required=True);args=parser.parse_args();assert args.segment>=1
raw=root/f'output/refined-convergence/stationary-reference-v2-aurora/campaigns/stationary20-reference-n16-segment{args.segment:02d}';out=doc/f'stationary-reference-v2-segment{args.segment:02d}';out.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
w=json.loads((raw/'output/worker.json').read_text());verified={}
for name,v in w['artifacts'].items():
 p=raw/'output'/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name;verified[name]=v
term=json.loads((raw/'terminal-verification.json').read_text());release=json.loads((raw/'allocation-released.json').read_text());assert term['admitted'] and term['checkpoint']['sha256']==verified['state.checkpoint']['sha256'];assert release['terminal_verification_sha256']==sha(raw/'terminal-verification.json')
(out/'local-verification.json').write_text(json.dumps(dict(all_worker_artifacts_verified=True,artifact_count=len(verified),artifacts=verified,checkpoint_sha256=verified['state.checkpoint']['sha256'],terminal_verification_sha256=sha(raw/'terminal-verification.json'),allocation_release_checked=True),indent=2)+'\n');print(f'all{len(verified)} local artifacts verified',flush=True)
manifest_path=doc/'stationary-reference-v2-build/source-manifest.json';m=json.loads(manifest_path.read_text());count=0
for f,v in m['files'].items():
 if f.startswith(('src/','tst/hyperbolic_shift_3d/')):assert sha(root/f)==v['sha256'],f;count+=1
prefix=root/f'output/refined-convergence/reference-endpoint-reader/segment{args.segment:02d}';cmd=['mpiexec','-n','2',str(prefix.parent/'reader'),'16',str(raw/'output/state.checkpoint'),'.000125',str(root/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial12_reference.cfg'),'smr',str(prefix),'.3'];assert not prefix.with_suffix('.json').exists()
r=dict(command=cmd,checkpoint_sha256=verified['state.checkpoint']['sha256'],source_manifest_sha256=sha(manifest_path),matching_production_source_files=count,analysis_binary_sha256=sha(prefix.parent/'reader'),started=time.time(),qualification='Read-only admitted reference-v2 checkpoint; disposition/time taken from exact terminal record. Dedicated explicit frozen-backend identity calibration; original checkpoint and production code unchanged.',terminal_status=term['checkpoint']['status'],t_over_M=3*term['checkpoint']['time'])
(out/'analysis-command.json').write_text(json.dumps(r,indent=2)+'\n')
with (out/'analysis.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
r.update(returncode=p.returncode,elapsed=time.time()-r['started'],checkpoint_unchanged=sha(raw/'output/state.checkpoint')==r['checkpoint_sha256']);(out/'analysis-command.json').write_text(json.dumps(r,indent=2)+'\n');assert p.returncode==0 and r['checkpoint_unchanged'];x=json.loads(prefix.with_suffix('.json').read_text());report=json.loads((raw/'output/state.report.json').read_text());history=[json.loads(s) for s in (raw/'output/state.history.jsonl').read_text().splitlines()];import math
norms=[[math.sqrt(a/b[0]) for a in b[1:]] for b in x['integrals']];expected=[[report['H'],report['M']],[history[-1]['fixed'][1]['H'],history[-1]['fixed'][1]['M']]]
errors=[[abs(norms[k][f]-expected[k][f])/(1+abs(expected[k][f])) for f in range(2)] for k in range(2)];assert max(z for row in errors for z in row)<1e-9
r.update(native_report_norms_match=errors,full_norms=norms[0],fixed03_norms=norms[1]);(out/'analysis-command.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
