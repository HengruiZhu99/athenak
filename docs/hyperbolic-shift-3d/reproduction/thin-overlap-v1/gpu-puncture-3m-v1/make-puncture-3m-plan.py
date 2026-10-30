from pathlib import Path
import copy,json,sys
here=Path(__file__).resolve().parent;repo=here.parents[2]
sys.path.insert(0,str(repo/'tst/hyperbolic_shift_3d/aurora'));import control
base=json.loads((here/'preflight-thin-puncture-v1-plan.json').read_text())
plan=copy.deepcopy(base);plan.update(phase='stability',admission='admissions/thin-l1-paired-v1/admission.json',output_dir='campaigns/stability-thin-puncture-3m-v1/output')
plan['resources']['queue']='debug-scaling';plan['storage']['planned_new_bytes']=180000000
c=copy.deepcopy(next(c for c in base['commands'] if c['name']=='puncture-sparse'))
old=c['args'][1];prefix=plan['output_dir']+'/puncture-l1-paired';c=json.loads(json.dumps(c).replace(old,prefix))
c.update(name='puncture-l1-paired',timeout_seconds=1500)
c['args'][3]='1.0';c['args'][4]=str(1/2048);c['args'][8]=str(1/32)
for check in c['checks']:
 if check['path']==['t']:check['value']=1.0
 if check['path']==['steps']:check['value']=2048
c['checks'] += [dict(path=['steps'],op='eq',value=2048),dict(path=['t_over_M'],op='eq',value=3.0)]
c['inputs'] += ['admissions/thin-l1-paired-v1/'+name for name in ['core-initial-audit.json','clock-independent-audit.json','puncture-independent-audit.json']]
plan['commands']=[c]
plan['notes']=['Bounded L1 paired isotropic puncture test to coordinate 1 = 3M, with the unchanged c530 frozen sparse executable and original initial data.', 'Binary-exact dt=1/2048; 33 diagnostic times at interval 1/32. Same log variables, counted lapse floor, scri closure and unrelaxed horizon/exterior-constraint checks as the short preflight.', 'The native Minkowski control has completed time1 with terminal checkpoint/history verification; its enclosing PBS exit143 is retained and not used as a passed worker admission. Prior short GPU pulse/oblique and puncture preflights supply the scoped existing admission.', 'One debug-scaling node, twelve ranks/tiles; distinct queue lane from the other agent. This is a stability experiment, not a presumption of success or a convergence admission. Failed horizon samples remain unavailable.', 'Optional online profiles are disabled in this frozen control; final checkpoint fields will be compared to the preserved same-time 1D profile after collection.']
remote=Path('/home/hzhu/hyperbolic-core-shell-20261028');control.validate_plan(remote,plan)
with (here/'stability-thin-puncture-3m-v1-plan.json').open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')
print('Validated bounded three-M puncture plan')
