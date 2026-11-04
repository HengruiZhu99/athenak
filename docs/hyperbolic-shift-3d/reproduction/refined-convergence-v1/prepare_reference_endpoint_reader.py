"""Explicit analysis-only backend roundoff calibration; no production changes.

The original checkpoint reader remains strict. This dedicated analysis body
accepts only the full frozen Aurora reference-v2 identity and only the known
local configuration, with its two map-roundoff constants calibrated explicitly.
"""
from pathlib import Path
import json,hashlib,argparse
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['reference','radial'],default='reference');parser.add_argument('--output');args=parser.parse_args()
out=Path(args.output).resolve() if args.output else ROOT/'output/refined-convergence'/('reference-endpoint-reader-source' if args.variant=='reference' else 'radial-endpoint-reader-source');out.mkdir(exist_ok=False)
local_path=HERE/('stationary-reference-v2-checks/checkpoint-verification.json' if args.variant=='reference' else 'reference-radial18-initial/checkpoint-verification.json')
remote_path=HERE/('launcher-v6/qualified-identity-reference.json' if args.variant=='reference' else 'launcher-v7/qualified-identity-radial.json')
local=json.loads(local_path.read_text())['identity'];remote=json.loads(remote_path.read_text())['identity']
a=local.split(';');b=remote.split(';');assert len(a)==len(b)
diffs=[(x,y) for x,y in zip(a,b) if x!=y];assert [x.split('=')[0] for x,y in diffs]==['gauge','k']
for x,y in diffs:
 f=float(x.split(',')[-1] if x.startswith('gauge=') else x.split('=')[1]);g=float(y.split(',')[-1] if y.startswith('gauge=') else y.split('=')[1]);assert abs(f-g)<=4e-16*max(abs(f),abs(g))
body=(HERE/'endpoint_diagnostic.inc').read_text();old='cfg.identity+=";amr_policy="+policy;cfg.CheckCollective();';assert body.count(old)==1
new='''cfg.identity+=";amr_policy="+policy;
  const std::string local_identity=%s, frozen_identity=%s;
  if(n!=16||cfg.identity!=local_identity)throw std::runtime_error("dedicated reference-v2 local configuration mismatch");
  // Same native quadrature/formula; retain the explicitly frozen Aurora constants.
  // No generalized tolerance or alias is added to production ReadIdentity.
  cfg.inner_map_scale=.10116863541976966;
  cfg.gauge.inner_gamma=.010235092792698271;
  cfg.identity=frozen_identity;cfg.CheckCollective();'''%(json.dumps(local),json.dumps(remote))
body=body.replace(old,new)
body=body.replace('<<",\\"analysis_override\\":', '<<",\\"backend_identity_calibration\\":\\"Dedicated frozen Aurora reference-v2 identity; only k and inner_gamma roundoff constants calibrated; production reader remains strict\\""\n       <<",\\"analysis_override\\":')
assert 'backend_identity_calibration' in body
(out/'body.inc').write_text(body)
record=dict(local_identity=local,frozen_identity=remote,differences=diffs,base_body_sha256=hashlib.sha256((HERE/'endpoint_diagnostic.inc').read_bytes()).hexdigest(),analysis_body_sha256=hashlib.sha256(body.encode()).hexdigest(),qualified_identity_sha256=hashlib.sha256(remote_path.read_bytes()).hexdigest(),qualification='Analysis-only dedicated identity calibration, never checkpoint mutation or restart. Full exact frozen source/config identity still required. Compare reconstructed native norms to admitted report/history before interpreting. Native local quadrature still recomputes reference fields; cross-backend bitwise equality not claimed.')
(out/'calibration.json').write_text(json.dumps(record,indent=2)+'\n');print(out/'body.inc')
