"""Dedicated native diagnostic body for two fully bound block-pair identities.

Adapt the existing checked diagnostic to the current mixed-angular topology and
sealed fixed mesh. Only known host quadrature roundoff constants are calibrated;
the production reader, checkpoint bytes and scientific snapshots stay unchanged.
"""
from pathlib import Path
import hashlib,json,argparse

D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
root=R/'output/refined-convergence/boosted-transition-aurora'
terminal=json.loads((root/'campaigns/boosted-block12-block24-pair-v1/terminal-verification.json').read_text())
assert terminal['worker_status']=='completed' and terminal['scheduler_exit']==0
local24=json.loads((root/'large-block-pair-v1/qualified-local-identity.json').read_text())['identity']
frozen24=terminal['cases'][1]['identity']
diffs=[(a,b) for a,b in zip(local24.split(';'),frozen24.split(';')) if a!=b]
assert [a.split('=')[0] for a,b in diffs]==['gauge','k']
for a,b in diffs:
    f=float(a.split(',')[-1] if a.startswith('gauge=') else a.split('=')[1])
    g=float(b.split(',')[-1] if b.startswith('gauge=') else b.split('=')[1])
    assert abs(f-g)<=4e-16*max(abs(f),abs(g))
frozen12=terminal['cases'][0]['identity'];local12=frozen12
for a,b in diffs:
    assert local12.count(b)==1
    local12=local12.replace(b,a)
body=(D/'endpoint_diagnostic.inc').read_text()
old='cfg.identity+=";amr_policy="+policy;cfg.CheckCollective();'
new='''cfg.identity+=";amr_policy="+policy;
  const std::string local12=%s, local24=%s, frozen12=%s, frozen24=%s;
  if(!((n==12&&cfg.identity==local12)||(n==24&&cfg.identity==local24)))
    throw std::runtime_error("dedicated large-block analysis identity mismatch");
  cfg.inner_map_scale=.10116863541976966;
  cfg.gauge.inner_gamma=.010235092792698271;
  cfg.identity=n==12?frozen12:frozen24;cfg.CheckCollective();'''%tuple(json.dumps(x) for x in [local12,local24,frozen12,frozen24])
assert body.count(old)==1;body=body.replace(old,new)
old='cfg.core_half_width,cfg.shell_inner_radius);'
assert body.count(old)==1
body=body.replace(old,'cfg.core_half_width,cfg.shell_inner_radius,cfg.shell_angular_subdivisions);')
body=body.replace('Restore(amr,s);auto descriptor=amr.Descriptor();',
                  'Restore(amr,s);amr.SealTopology();auto descriptor=amr.Descriptor();')
body=body.replace('Install(e,s);e.RefreshBoundary(s.dt);',
                  'Install(e,s);for(auto&data:s.data)V().swap(data);e.RefreshBoundary(s.dt);')
(out/'body.inc').write_text(body)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(local_identities=[local12,local24],frozen_identities=[frozen12,frozen24],
    differences=diffs,base_body_sha256=sha(D/'endpoint_diagnostic.inc'),body_sha256=sha(out/'body.inc'),
    source_sha256=sha(Path(__file__)),
    qualification='Dedicated analysis-only calibration; preserves strict full identities and checkpoint contents. Current mixed-angular constructor and production fixed-topology sealing; release installed snapshot payloads only to limit analysis memory. Native full-domain report reproduction required before spatial claims.')
(out/'calibration.json').write_text(json.dumps(record,indent=2)+'\n')
print(out/'body.inc')
