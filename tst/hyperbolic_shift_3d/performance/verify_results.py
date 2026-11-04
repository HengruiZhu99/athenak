#!/usr/bin/env python3
"""Verify collected task artifacts; never relax same-backend A/B equality."""
import argparse,pathlib,json,hashlib,importlib.util
p=argparse.ArgumentParser();p.add_argument('artifacts');p.add_argument('--output',required=True);a=p.parse_args();r=pathlib.Path(a.artifacts).resolve();checks=[]
def check(name,passed,**details):
 checks.append(dict(name=name,passed=bool(passed),**details));assert passed,name
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for folder in ['cpu-bench','cpu-m4-bench','cpu-m4-replacement','cpu-m4-clean','final-check']:
 d=r/folder;check(folder+' completed',(d/'completed.json').exists());rs=load(d/'results.json');check(folder+' receipts',all(x['passed'] for x in rs));check(folder+' checkpoint equality',all(x.get('bitwise_checkpoint_equal',True) for x in rs))
q=r/'qualification';check('qualification completed',(q/'completed.json').exists());commands=load(q/'commands.json');failed=[x for x in commands if x['returncode']!=0];check('documented stale CMake target failure only',len(failed)==1 and failed[0]['name']=='build-components' and 'No rule to make target' in (q/'build-components.log').read_text());check('successful reconfiguration and resumed build',all(any(x['name']==name and x['returncode']==0 for x in commands) for name in ['configure-components','build-components-reconfigured','main-z4c']))
base={x['name']:x for x in load(r/'baseline-matrix/manifest.json')};candidate={x['name']:x for x in load(q/'m4-matrix/manifest.json')}
for name,x in candidate.items():check('CPU matrix '+name,x['checkpoint']['sha256']==base[name]['checkpoint']['sha256'])
check('six malformed checkpoints',all(x['passed'] for x in load(q/'checkpoint-reject/manifest.json')))
for name in ['wave','restart']:
 x=load(q/('compare-'+name+'.json'));check(name+' complete state',all(y['bitwise'] for y in x['arrays']) and x['metadata_equal'] and x['ages_equal'])
x=load(q/'compare-boost-restart.json');check('boost restart all eight arrays',all(y['bitwise'] for y in x['arrays']) and x['ages_equal'])
spec=importlib.util.spec_from_file_location('amr_reader',pathlib.Path(__file__).parents[1]/'analyze_amr.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
b=reader.read(q/'m4-controls/boost-fixed.checkpoint');c=reader.read(q/'m4-controls/boost-restart.checkpoint');different=[i for i,(v,w) in enumerate(zip(b['meta'],c['meta'])) if v!=w]
check('boost restart known seed-only metadata exception',different==[7],seed_absolute_difference=abs(b['meta'][7]-c['meta'][7]),note='Preexisting FastFlow angular warm-start fixture; no exception allowed in same-P A/B comparisons.')
f=r/'final-check';check('final poison wave',sha(f/'final-wave.checkpoint')==base['wave-1']['checkpoint']['sha256']);check('final boosted complete checkpoint',sha(f/'fixed-boost-base.checkpoint')==sha(f/'fixed-boost-final.checkpoint'))
for suffix in ['failed-stage.bin','rejected-transfer.bin']:
 check('final rejected '+suffix,sha(f/('rejected-base.'+suffix))==sha(f/('rejected-m4.'+suffix)))
for job in [1,2]:
 d=r/('aurora-job'+str(job))/('gpu-job'+str(job));check('Aurora job '+str(job)+' completed',(d/'completed.json').exists());rs=load(d/'results.json');check('Aurora job '+str(job)+' receipts',all(x['passed'] for x in rs));check('Aurora job '+str(job)+' checkpoint equality',all(x.get('bitwise_checkpoint_equal',True) for x in rs))
for backend in ['cpu','gpu']:
 x=load(q/'donor-audit.json') if backend=='cpu' else load(r/'evidence/gpu-rejection-audit.json');check(backend+' donor rejection',x['actual_chi']==-.0002924790344160026 and x['absolute_difference']<1e-13,actual=x['actual_chi'],reconstruction_error=x['absolute_difference'])
check('GPU rejected captures',all(x['all_equal'] for x in load(r/'evidence/gpu-rejection-comparison.json')))
pathlib.Path(a.output).write_text(json.dumps({'checks':checks,'all_passed':True},indent=2)+'\n');print(len(checks),'verification checks passed')
