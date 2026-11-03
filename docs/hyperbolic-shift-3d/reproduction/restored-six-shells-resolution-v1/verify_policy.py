"""Bounded checks for the new resolution policy, not a repeated launcher suite."""
from pathlib import Path
import copy,hashlib,json,shutil,sys,tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];RAW=ROOT/'output/restored-six-shells-resolution/aurora'
sys.path.insert(0,str(RAW/'launcher-restored-six-shells-resolution-v1'))
import native_transition_campaign as c
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/Path(c.SCIENCE).parent).mkdir();shutil.copy2(RAW/c.SCIENCE,root/c.SCIENCE)
 checks=[]
 for variant in c.VARIANTS:
  result=c.make_plan(root,variant);plan=json.loads((root/result['plan']).read_text());n,dt=c.resolution(variant)
  assert plan['n']==n and float(plan['commands'][0]['args'][4])==dt
  assert abs(c.expected_penalty_scale(variant)-.6510416666660155)<1e-14
  for field in ['n','dt','penalty_check']:
   bad=copy.deepcopy(plan)
   if field=='n':bad['n']=16
   elif field=='dt':bad['commands'][0]['args'][4]='0.0005'
   else:bad['commands'][0]['checks']=[x for x in bad['commands'][0]['checks'] if x['path']!=['scri_penalty_scale_min']]
   try:c.validate_native_plan(root,bad)
   except ValueError:pass
   else:raise AssertionError(field)
  cp=ROOT/'output/restored-six-shells/aurora/campaigns/n16-restored-six-shells-v1-a-evolution-segment01/output/puncture.checkpoint'
  try:c.inspect_checkpoint(cp,variant,hashlib.sha256(cp.read_bytes()).hexdigest())
  except ValueError as e:assert 'dimensions' in str(e)
  else:raise AssertionError('N16 cannot restart N20/N24')
  checks.append(dict(variant=variant,n=n,dt=dt,cells=c.cells(n),checkpoint_bytes=168+16*(31*c.cells(n)+6*n*n*31),storage=plan['storage'],wrong_resolution_rejected=True,wrong_dt_rejected=True,missing_penalty_checks_rejected=True,N16_restart_rejected=True,plan=result['plan']))
 (HERE/'resolution-policy-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 print(json.dumps(checks,indent=2))
