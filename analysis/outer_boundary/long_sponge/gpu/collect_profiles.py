#!/usr/bin/env python3
from pathlib import Path
import argparse,importlib.util,json,time,os,hashlib
root=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('profiles',root/'radial_profiles.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from check_minkowski_checkpoint import validate,checkpoint_header
parser=argparse.ArgumentParser(description='Validate and profile sparse saved cohorts from the four GPU controls. No jobs are submitted or changed.')
parser.add_argument('--runs-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--spacing',type=float,default=5000)
parser.add_argument('--ranks',type=int,default=8)
args=parser.parse_args();assert args.spacing>0 and args.ranks>0
runs=args.runs_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
report={'new':[],'unchanged':[],'pending_or_rejected':[]}
for job,prefix,cases in [('8842171','long_sponge_k0',['radial_k0']),('8842172','long_sponge_k01',['baseline','radial_k01']),('8842248','long_theta_loweta',['theta_primary','theta_lapse01']),('8842283','long_theta_amp',['theta_amplitude'])]:
 for case in cases:
  run=runs/(prefix+'_'+job)/case
  available=sorted((run/'rst/rank_00000000').glob('*.rst'))
  headers={f:checkpoint_header(f) for f in available}
  final=max(available,key=lambda f:headers[f]['cycle']) if available else None
  # Inspect a sparse common physical cadence and the actual latest cohort.
  # Header-only selection avoids loading every 1000 M payload just to choose.
  selected=[]
  for f in available:
   h=headers[f];nearest=args.spacing*round(h['time']/args.spacing)
   if f==final or abs(h['time']-nearest)<=max(h['dt']*1.01,1e-8):selected.append(f)
  for first in selected:
   tag=job+'-'+case+'-'+first.stem;files=[run/'rst'/('rank_%08d'%r)/first.name for r in range(args.ranks)]
   if not all(f.exists() for f in files):report['pending_or_rejected'].append({'tag':tag,'reason':'missing rank files; not considered complete'});continue
   signature=[[str(f),f.stat().st_size,f.stat().st_mtime] for f in files]
   metadata={'files':signature,'extractor_sha256':hashlib.sha256((root/'radial_profiles.py').read_bytes()).hexdigest(),'validator_sha256':hashlib.sha256((root/'gpu/check_minkowski_checkpoint.py').read_bytes()).hexdigest()}
   if any(time.time()-s[2]<10 for s in signature):report['pending_or_rejected'].append({'tag':tag,'reason':'recently modified; defer completion check'});continue
   dest=out/(tag+'.json');meta=out/(tag+'.files.json')
   if dest.exists() and meta.exists() and json.loads(meta.read_text())==metadata:report['unchanged'].append(tag);continue
   try:
    cycle=headers[first]['cycle'];valid=validate(run,args.ranks,cycle=cycle,check_evolution=False);assert valid['passed'],'Saved cohort fails all-field/ghost-SPD validation'
    r=p.extract(run,args.ranks,cycle);r['actual_sponge_enabled']=case!='baseline'
    for peak in r['active_peaks'].values():peak['inside_candidate_radial_zone']=peak['radius']>512;peak['in_sponge']=r['actual_sponge_enabled'] and peak['inside_candidate_radial_zone']
    r['comparison_zone_note']='Core/ramp/plateau geometrical zones are shared; baseline has no actual sponge.'
    r['checkpoint_validation']={k:valid[k] for k in ('matching_headers','all_payload_finite','invalid_metric_cells_including_ghosts','minimum','passed')}
    dest.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');meta.write_text(json.dumps(metadata,indent=2)+'\n');report['new'].append({'tag':tag,'time_code':r['time_code'],'Theta_peak':r['active_peaks']['Theta'],'core':r['regions']['protected_core'],'face_band':r['regions']['within_256M_of_physical_face']})
   except Exception as exc:report['pending_or_rejected'].append({'tag':tag,'reason':repr(exc)})
(out.parent/'profile-collection-latest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
