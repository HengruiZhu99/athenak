#!/usr/bin/env python3
"""Preserve new report plots and summarize only fully admitted scalar stages."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent;output=E/'scalar-validation-summary.json';journal=E/'scalar-evidence-observer.jsonl'
if output.exists() or journal.exists():raise RuntimeError('Existing observer/final evidence must be inspected')
with journal.open('x') as f:f.write(json.dumps(dict(event='started',source_sha256=sha(Path(__file__)),time=time.time(),simulation_tasks_added=0))+'\n')
targets=[(E/'cpu-schwarzschild-dyadic/primary-review.json','schwarzschild-dyadic-cpu'),(E/'cpu-kerr-stable/primary-review.json','kerr-stable-cpu')]
for stage in ['schwarzschild','kerr']:
 for kind in ['physics-review','completed-physics-review']:targets.append((A/f'{stage}-stable-gpu-v1-{kind}.json',stage+'-stable-gpu'+('-continued' if kind.startswith('completed') else '')))
processed=set();CPU=[E/(stage+'-stable-sbp-cpu-admission.json') for stage in ['minkowski','schwarzschild','kerr']];GPU=[A/'minkowski-stable-gpu-v5-review.json',*[A/(stage+'-stable-gpu-v1-review.json') for stage in ['schwarzschild','kerr']]]
while True:
 for review,label in targets:
  if review in processed or not review.exists():continue
  for script,suffix in [('plot_ringdown_primary.py','primary'),('plot_ringdown_qnm.py','qnm')]:
   stem=E/(label+'-'+suffix+'-v22');metadata=stem.with_suffix('.json')
   if metadata.exists():
    if read(metadata)['review_sha256']!=sha(review):raise RuntimeError('Existing plot differs from its immutable review')
   else:subprocess.run([sys.executable,str(scripts/script),'--review',str(review),'--output',str(stem)],check=True)
  processed.add(review)
  with journal.open('a') as f:f.write(json.dumps(dict(event='plotted',review=str(review),review_sha256=sha(review),time=time.time()))+'\n')
 if all(file.exists() for file in CPU+GPU):
  records=[read(file) for file in CPU+GPU]
  if not all(record['passed'] for record in records):raise RuntimeError('A scalar stage is explicitly failed; preserve all evidence without completing the summary')
  baseline=read(E/'frozen-baseline-sha256.json');bad=[name for name,digest in baseline.items() if not Path(name).is_file() or sha(Path(name))!=digest];source={name:sha(Path('src/test_scalar')/name) for name in ['CMakeLists.txt','main.cpp','geometry.hpp','sbp63.hpp']};same=all(record['delivery_source_sha256']==source for record in records[:3]);passed=not bad and same
  summary=dict(scalar_validation_complete=passed,passed=passed,metric_evolution_goal_complete=False,stages={stage:dict(CPU=str(cpu),CPU_sha256=sha(cpu),GPU=str(gpu),GPU_sha256=sha(gpu)) for stage,cpu,gpu in zip(['minkowski','schwarzschild','kerr'],CPU,GPU)},delivery_source_sha256=source,protected_baseline_files=len(baseline),protected_baseline_mismatches=bad,plot_journal_sha256=sha(journal),source_sha256=sha(Path(__file__)),completed=time.time(),qualification='Only a massless test scalar and required scalar auxiliaries on exact immutable physical metrics, separate immutable Minkowski reference and frozen coordinate functions; no backreaction, Einstein/Z4c, gauge or subsidiary constraint evolution. Selected families pass all source-bound scientific gates. Historical failures remain unchanged. Scalar success does not complete metric evolution.')
  output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(passed=passed,summary=str(output),metric_evolution_goal_complete=False)),flush=True)
  if not passed:raise RuntimeError('Protected baseline or admitted source changed')
  break
 time.sleep(30)
