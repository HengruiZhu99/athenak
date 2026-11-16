#!/usr/bin/env python3
"""Assemble this campaign's compact evidence after all scientific jobs finish."""
import argparse,hashlib,json,math,re,shutil
from pathlib import Path
from nonradiative_wide import require_completed_run
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('raw',type=Path);parser.add_argument('destination',type=Path)
args=parser.parse_args();raw=args.raw.resolve();dest=args.destination.resolve()
repo=Path(__file__).resolve().parents[2]
dest.mkdir(parents=True,exist_ok=True)
(dest/'native-summaries').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def copy(p,q):q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def textcopy(p,q):q.parent.mkdir(parents=True,exist_ok=True);q.write_text('\n'.join(v.rstrip() for v in p.read_text().splitlines()).rstrip()+'\n')
def scheduler(job,expected_exit):
 s=(raw/'receipts'/f'{job}.qstat.txt').read_text()
 assert re.search(r'job_state = F',s)
 assert int(re.search(r'Exit_status = (-?\d+)',s).group(1))==expected_exit
 assert re.search(r'\n\s*queue = debug\s*\n',s) and re.search(r'Resource_List.nodect = 2\b',s)
 wall=re.search(r'resources_used.walltime = (\d+):(\d+):(\d+)',s)
 seconds=sum(int(v)*factor for v,factor in zip(wall.groups(),(3600,60,1)))
 return {'wall_seconds':seconds,'nodes':2,'node_hours':seconds/1800,'exit_status':expected_exit}
principal=list(raw.glob('runs/N*-*/analysis/*/summary.json'))
assert {p.parent.name for p in principal}=={f'{f}-N{n}' for f in ('control','gauge') for n in (12,16,20)} and len(principal)==6
half=list(raw.glob('runs/halfdt-*'));assert len(half)==1
for filename in ('halfdt-pointwise-comparison.json','halfdt-news-comparison.json','reference-pointwise-comparison.json','frame-comparison.json'):
 assert (half[0]/filename).is_file(),filename
checkpoint_audit=json.loads((raw/'receipts/checkpoint-audit.json').read_text())
assert len(checkpoint_audit['checkpoints'])==8
assert json.loads((raw/'plots/measurements.json').read_text())['complete_spatial_sequence']
runs=[p for p in (raw/'runs').iterdir() if p.is_dir()]
assert len(runs)==4
histories={};analysis_sources={};integrity={'runs':{},'analysis':{},'scheduler':{}}
for run in sorted(runs):
 require_completed_run(run)
 original_exit=int((run/'exit-code.txt').read_text())
 receipt=json.loads((run/'receipt.json').read_text());job=run.name.split('-')[-1]
 assert receipt.get('baseline_executable_sha256',receipt.get('executable_sha256'))=='a577cabc82d15c2aa47cb38a2ce5bdc37cbcd64ac204eefc2690c45de6764885'
 seed=json.loads((run/'seed-check.json').read_text())
 assert seed['changed_fields']==[18] and seed['physical_constraint_data_bitwise_equal'] and seed['gauge_auxiliaries_bitwise_equal']
 if run.name.startswith('halfdt-'):assert seed['halfdt_all_initial_fields_bitwise_equal']
 if 'reference_probe' in receipt:
  probe=receipt['reference_probe']
  assert probe['initial_values_bitwise_equal']
  assert probe['executable_sha256']=='5961115430a56a30efa42f770862119e1c55df082219db4cc50e545328d74721'
 for p in run.iterdir():
  if p.is_file() and (p.suffix in ('.json','.txt','.pbs') or (original_exit and p.name=='analysis.log')):
   textcopy(p,dest/'provenance'/run.name/p.name)
 for case in run.iterdir():
  if not case.is_dir() or not (case/'input.athinput').exists():continue
  name=case.name
  records=[x for x in receipt.get('cases',[]) if x['name']==name]
  record=records[0] if records else receipt['reference_probe'] if name.startswith('reference-') else receipt
  assert sha(case/'input.athinput')==record['input_sha256']
  audit=checkpoint_audit['checkpoints'][run.name+'/'+name]
  assert audit['sha256']==record['final_sha256'] and audit['finite_scalar_values']>0
  copy(case/'input.athinput',dest/'inputs'/f'{run.name}-{name}.athinput')
  for p in sorted(case.glob('*.hst')):
   a=np.loadtxt(p);assert np.all(np.isfinite(a))
   histories[f'{run.name}/{name}/{p.name}']={'raw_sha256':sha(p),'header':[v.rstrip() for v in p.read_text().splitlines() if v.startswith('#')],'rows':a.tolist()}
 integrity['runs'][run.name]={'job':job,'receipt_sha256':sha(run/'receipt.json'),
      'original_exit_status':original_exit,'complete_after_recovery_if_needed':True}
 integrity['scheduler'][job]=scheduler(job,original_exit)
 if original_exit:
  recovery=json.loads((run/'analysis-recovery.json').read_text())
  assert recovery['original_job']==job
  assert recovery['snapshot_manifest_sha256']==sha(raw/'source-analysis-v2-sha256.json')
  recovery_job=recovery['recovery_job']
  integrity['runs'][run.name]['analysis_recovery_job']=recovery_job
  integrity['scheduler'][recovery_job]=scheduler(recovery_job,0)
 for group in ('analysis','reference-analysis'):
  for summary in (run/group).glob('*/summary.json'):
   case=summary.parent;label=('reference-' if group=='reference-analysis' else '')+case.name
   for filename in ('summary.json','pointwise-summary.json','tensor-attribution.json','cadence-comparison.json'):
    p=case/filename
    if p.exists():copy(p,dest/'native-summaries'/f'{label}-{filename}')
   series=json.loads((case/'series.json').read_text());arrays={k:np.asarray(v) for k,v in series.items()}
   assert all(v.dtype.kind in 'iufc' and np.all(np.isfinite(v)) for v in arrays.values())
   np.savez_compressed(dest/'native-summaries'/f'{label}-series.npz',**arrays)
   a=np.load(case/'news/news.npz')
   keys=['bondi_time','bondi_time_over_M0','native_times','ell_m','news_modes','native_cut_decomposition','native_cut_news_coefficients']
   np.savez_compressed(dest/'native-summaries'/f'{label}-modes.npz',**{k:a[k] for k in keys})
   integrity['analysis'][label]={'summary_sha256':sha(summary),'raw_series_sha256':sha(case/'series.json'),'raw_news_sha256':sha(case/'news/news.npz')}
write(dest/'native-summaries/histories.json',histories)
frame_paths=list(raw.glob('runs/halfdt-*/analysis/frame-l*/news/news.npz'))
cadence_paths=list(raw.glob('runs/N12-*/analysis/gauge-N12/cadence3/news.npz'))
assert len(frame_paths)==2 and len(cadence_paths)==1
for p in frame_paths+cadence_paths:
 label=p.parents[1].name if p.parent.name=='news' else 'gauge-N12-cadence3'
 with np.load(p) as a:
  np.savez_compressed(dest/'native-summaries'/f'{label}-modes.npz',
      **{k:a[k] for k in ('bondi_time','bondi_time_over_M0','native_times','ell_m','news_modes')})
 copy(p.parent/'diagnostics.json',dest/'native-summaries'/f'{label}-diagnostics.json')
 integrity['analysis'][label]={'raw_news_sha256':sha(p)}
for name in ('pulse-resolution.json','characteristics.csv','characteristic-parameters.txt','initial-data-check.json','boundary-admission.json','reference-accuracy-paired.csv'):
 p=raw/name
 if p.suffix=='.csv':textcopy(p,dest/name)
 else:copy(p,dest/name)
for p in (raw/'plots').iterdir():
 if p.is_file() and p.suffix in ('.json','.png','.pdf','.npz'):copy(p,dest/p.name)
for p in (raw/'receipts').iterdir():
 if p.suffix in ('.txt','.json'):textcopy(p,dest/'provenance'/p.name)
for name in ('reference-check','reference-guard-check'):copy(raw/name/'receipt.json',dest/'provenance'/f'{name}.json')
for p in raw.glob('source*sha256.json'):
 if 'preflight' not in p.name:copy(p,dest/'provenance'/p.name)
for name in ('reference-regression.json','reference-guard-regression.json','ctest.log','legacy-N12-peak-attribution.json'):copy(raw/name,dest/name)
write(dest/'provenance/snapshot-sha256.json',{p.name:sha(p) for p in raw.glob('*.tgz') if p.name.startswith(('source','reference-fix'))})
packages={}
for p in raw.glob('*.manifest.json'):
 manifest=json.loads(p.read_text());assert all(sha(raw/k)==v for k,v in manifest.items()),p
 archive=p.with_name(p.name.replace('.manifest.json','.tgz'))
 packages[archive.name]={'sha256':sha(archive),'manifest_sha256':sha(p),'files':len(manifest)}
write(dest/'provenance/collected-packages.json',packages)
paths=list((repo/'src').rglob('*'))+[repo/'tst/z4c_hyperboloidal/CMakeLists.txt']
paths+=list((repo/'tst/z4c_hyperboloidal').glob('nonradiative*'))+list((repo/'tst/z4c_hyperboloidal/aurora').glob('*nonradiative*'))
paths += [repo/'tst/z4c_hyperboloidal/aurora'/n for n in ('build-reference-jets.py','build-wide-seed.py')]
write(dest/'provenance/final-source-sha256.json',{str(p.relative_to(repo)):sha(p) for p in sorted(set(paths)) if p.is_file()})
cancelled=(raw/'receipts/9094380.qstat.txt').read_text()
assert re.search(r'job_state = F',cancelled) and not re.search(r'\n\s*stime =',cancelled) and 'exec_host =' not in cancelled
integrity['cancelled_before_start']={'9094380':{'node_hours':0,'reason':'Superseded after service-node import preflight; no allocation started.'}}
integrity['node_hours']=sum(v['node_hours'] for v in integrity['scheduler'].values())
write(dest/'provenance/final-integrity-check.json',integrity)
def finite(obj):
 if isinstance(obj,float):assert math.isfinite(obj)
 elif isinstance(obj,dict):
  for v in obj.values():finite(v)
 elif isinstance(obj,list):
  for v in obj:finite(v)
for p in dest.rglob('*.json'):finite(json.loads(p.read_text()))
for p in dest.rglob('*.npz'):
 with np.load(p) as a:
  for k in a.files:
   if a[k].dtype.kind in 'iufc':assert np.all(np.isfinite(a[k])),(p,k)
write(dest/'artifact-sha256.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'})
print(json.dumps(integrity,indent=2))
