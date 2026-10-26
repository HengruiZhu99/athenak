#!/usr/bin/env python3
"""Bounded independent pulse controls; external wrapper leaves frozen solver untouched."""
import argparse,json,shutil,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
support=Path(__file__).resolve().parent/'source/tst/cubed_sphere'
if support.is_dir():sys.path.insert(0,str(support))
from run_conformal_spherical import run_case,sha
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('source',type=Path);p.add_argument('grids',type=Path);p.add_argument('output',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
source=a.source.resolve();out=a.output.resolve();grids=json.loads(a.grids.read_text());assert 1<=len(grids)<=4
assert len({g['name'] for g in grids})==len(grids)
for g in grids:
 assert 0<=g['extra'].get('cubed_sphere/conformal_wall_seconds',0)<=360
 assert g.get('checkpoint') and 0<g['cfl']<=.15 and 0<g['pole_cfl']<=.1 and abs(g['amplitude'])<=.5
 assert 0<g['tlim']<=1000 and g['degree'] in [2,3,4,5]
 if g.get('restart'):assert sha(Path(g['restart']))==g['restart_sha256']
assert a.binary.is_file() and (source/'inputs/z4c/cubed_sphere_conformal_probe.athinput').is_file()
if a.check_only:print(json.dumps(dict(preflight_passed=True,cases=[g['name'] for g in grids],max_native_wall=max(g['extra'].get('cubed_sphere/conformal_wall_seconds',0) for g in grids))));sys.exit(0)
out.mkdir(parents=True,exist_ok=False);binary=out/'athena_frozen';shutil.copy2(a.binary,binary);tag=sha(binary)
index=source.parent/'gpu-source-index.json';hashes=json.loads(index.read_text());assert all(sha(source/f)==h for f,h in hashes.items())
record=dict(binary_sha256=tag,runner_sha256=sha(Path(__file__)),source_sha256=hashes,grids=grids,scope='Native bounded diagnosis, no stability/convergence acceptance from survival',cases={});(out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
def run(item):
 i,g=item;extra=dict(g['extra']);extra['cubed_sphere/conformal_checkpoint']=str(out/g['name']/'probe_checkpoint.bin');extra['cubed_sphere/conformal_checkpoint_tag']=tag;extra['cubed_sphere/conformal_fixed_dt']=g['dt']
 if g.get('restart'):
  assert sha(Path(g['restart']))==g['restart_sha256'];extra['cubed_sphere/conformal_restart']=g['restart']
 return run_case(binary,source,out,g['name'],str(i)+'.0',g['bulk'],g['nr'],g['na'],g['amplitude'],g['kappa'],g['cfl'],g['tlim'],g['degree'],420,g['pole_cfl'],extra)
with ThreadPoolExecutor(max_workers=len(grids)) as pool:results=list(pool.map(run,enumerate(grids)))
record['cases']={g['name']:c for g,c in zip(grids,results)};(out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print({n:dict(outcome=c['outcome'],last_time=c['last_rows'][-1]['time'] if c['last_rows'] else None) for n,c in record['cases'].items()});assert all(c['outcome'] in ['complete','checkpointed'] and c['all_rows_finite'] and c['reference_unchanged'] for c in results)
