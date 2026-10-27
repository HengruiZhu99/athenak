#!/usr/bin/env python3
"""Read existing results only; never launch or resume an evolution."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
E=ROOT/'docs/elliptic-shift-1d/evidence';O=ROOT/'output/elliptic-shift-1d'
D=HERE/'data';D.mkdir(exist_ok=True);(D/'manifests').mkdir(exist_ok=True)
provenance=[];data={}
def track(p):
    provenance.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
def review(key,name):
    p=E/name;track(p);data[key]=json.loads(p.read_text())
for key,name in [
 ('balanced_noise','balanced-noise-v1/review.json'),
 ('gauge','compensated-gauge-finer-trajectory-v1/review.json'),
 ('gauge_old','positive-gauge-ode-right-trajectory-v2/review.json'),
 ('subtracted','subtracted-gauge-finer-trajectory-v1/review.json'),
 ('arithmetic','compensated-independent-controls-v2/review.json'),
 ('modes','elliptic-mode-diagnosis-v3/review.json'),
 ('scalar','scalar-Bondi-allstage-v1/review.json'),
 ('fixed_scalar','fixed-scalar-oracle-v3.json'),
 ('cmc','CMC-trumpet-symmetric-reference-damping-assessment-v1/review.json'),
 ('front','physical-log-front-v1/review.json'),
 ('prefix','nu025-prefix-finer-convergence-v1/review.json'),
 ('rational','local-rational-failures-v1/review.json'),
 ('native','physical-trumpet-native-compiled-v1.json')]:review(key,name)
keys=['t','H_rms','M_rms','Theta_rms','Z_rms','scri_pole','Bondi_frame_nonaffinity',
      'apparent_horizon_mass','mass_r0.9','inner_areal_radius','inner_gauge_speed_max',
      'radiated_energy','min_alpha','min_chi','max_field']
def history(key,folder):
    p=O/folder/'manifest.json';track(p);m=json.loads(p.read_text())
    if m['status']!='finished':raise ValueError('Only finished runs enter result plots: '+folder)
    shutil.copy2(p,D/'manifests'/f'{key}.json')
    h=p.with_name('history.jsonl');track(h)
    rows=[json.loads(s) for s in h.read_text().splitlines()]
    data[key]=dict(arguments=m['arguments'],status=m['status'],time=m['time'],stages=m.get('all_RK_stages',{}),
                   history=[{k:r.get(k) for k in keys} for r in rows])
for n in [1009,2017,4033]:history(f'domain{n}',f'wormhole-global-trumpet-domain-v1/n{n}')
for n in [512,1024,2048]:
    history(f'dispersal{n}',f'compiled-scalar-Bondi-allstage-v1/dispersal-n{n}')
    history(f'collapse{n}',f'collapse-positive-upwind-v1/n{n}')
for n in [256,512,1024]:history(f'cmc{n}',f'CMC-trumpet-symmetric-reference-damping-v1/n{n}')
def profile(key,folder,t):
    folder=O/folder
    for path in sorted(folder.glob('profile-*.npz')):
        with np.load(path,allow_pickle=False) as z:
            if abs(float(z['t'])-t)>1e-10:continue
            r=z['r'];keep=(r>0)&(r<.35)
            np.savez_compressed(D/f'{key}.npz',r=r[keep],q=z['q_physical'][keep],diagnostic=z['diagnostic'][keep],t=z['t'])
        track(path);return
    raise ValueError('Missing exact profile '+folder.name+' '+str(t))
for n in [513,1025,2049]:profile(f'front{n}',f'wormhole-physical-log-compiled-v1/n{n}',2.5)
profile('compact1009','wormhole-physical-trumpet-domain-v1/parent-n1025',10.)
profile('global1009','wormhole-global-trumpet-domain-v1/n1009',10.)
# Inventory includes historical prototypes and implementation fixtures. A finished
# run may still be scientifically failed; the manuscript discusses those too.
failed=[]
for p in sorted(O.rglob('manifest.json')):
    if 'frozen-source' in p.parts or 'matrix-source' in p.parts:continue
    try:m=json.loads(p.read_text())
    except (ValueError,OSError):continue
    if m.get('status') not in ['failed','initial-data-failed','domain-transfer-rejected']:continue
    arg=m.get('arguments',{});failure=m.get('failure') or {}
    failed.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),status=m['status'],
        time=m.get('time',failure.get('time')),n=arg.get('n'),initial_data=arg.get('initial_data'),
        scalar=arg.get('scalar'),error=failure.get('message'),phase=failure.get('phase')))
with (D/'failure_inventory.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(failed[0]));w.writeheader();w.writerows(failed)
# Preserve written protocols and implementation verification as supporting evidence.
support=D/'support';support.mkdir(exist_ok=True)
for name in ['nonlinear-formulation.md','puncture-asymptotics.md','discrete-shift.md',
             'progress.md','acceptance.json','constraint-noise-gates-v2.json',
             'global-trumpet-domain-protocol-v1.json','evidence/ctest-v41.log']:
    p=ROOT/'docs/elliptic-shift-1d'/name
    track(p);shutil.copy2(p,support/p.name)
(D/'evidence.json').write_text(json.dumps(data,indent=2)+'\n')
(HERE/'source-manifest.json').write_text(json.dumps(dict(snapshot_date='2026-10-26',sources=provenance,
    failed_manifest_inventory_count=len(failed),scope='Existing saved evidence only; no new evolution, no resumed jobs. Per-run source/binary hashes retained in data/manifests and review records.'),indent=2)+'\n')
print(f'Exported {len(provenance)} evidence sources and {len(failed)} failed manifests.')
