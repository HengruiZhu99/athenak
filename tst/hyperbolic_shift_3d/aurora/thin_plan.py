#!/usr/bin/env python3
"""Generate the bounded, twelve-tile V12 preflight (does not submit it)."""
import argparse,json
from pathlib import Path
import control

def build(root,source,build,campaign,initial):
 out=campaign+'/output';commands=[];binary=build+'/tst/hyperbolic_shift_3d/'
 def check(key,op,value=None):
  result=dict(path=key if isinstance(key,list) else [key],op=op)
  if op!='finite':result['value']=value
  return result
 basic=[check('status','eq','completed'),check('backend','eq','SYCL'),check('mpi_size','eq',12)]
 def adapter(name,mode,extra,inputs=(),generated=(),component=False):
  prefix=out+'/'+name
  commands.append(dict(name=name,kind='check',executable=source+'/tst/hyperbolic_shift_3d/aurora/preflight.py',
   args=['--root',str(root),'--plan',campaign+'/plan.json','--output-prefix',prefix,'--seconds','@WALL_SECONDS@',mode,*extra],
   timeout_seconds=240 if component else 120,report=prefix+'.report.json',inputs=list(inputs),generated_inputs=list(generated),
   artifacts=[prefix+'.report.json',prefix+'.log']+([prefix+f'.rank-{i:02d}.log' for i in range(12)] if component else []),
   checks=basic+[check('gpu_cpu_parity' if component else 'restart_equivalence','eq',True)]+([] if component else [check('bitwise_payload_equal','eq',True),check(['header','magic'],'eq','HYP3D_MPI_V12')])))
 adapter('paired-device','component',['--component','paired_trace_device','--test-binary',binary+'hyperbolic_shift_3d_paired_trace_device','--cmake-cache',build+'/CMakeCache.txt'],[binary+'hyperbolic_shift_3d_paired_trace_device',build+'/CMakeCache.txt'],component=True)
 def evolution(name,kind,layers=1,kappa=1,sparse=True,end=.001,dt=.00025,restart=None):
  prefix=out+'/'+name;env={'HYPERBOLIC_TRACE_LAYERS':str(layers),'HYPERBOLIC_TRACE_SHIFTED':'1','HYPERBOLIC_TRACE_KAPPA':str(kappa),'HYPERBOLIC_TRACE_NORMAL_RATIO':'.25'}
  if sparse:env['HYPERBOLIC_SPARSE_POISON']='1'
  command=dict(name=name,kind=kind,executable=binary+(control.THIN_SPARSE_TARGET if sparse else control.THIN_TARGET),
   args=['16',prefix,kind,str(end),str(dt),initial if kind=='puncture' else '-',restart or '-','@WALL_SECONDS@',str(end),'.001','12' if kind=='puncture' else '6','24' if kind=='puncture' else '12','2'],
   environment=env,timeout_seconds=360 if kind=='puncture' else 180,report=prefix+'.report.json',
   artifacts=[prefix+ext for ext in ['.history.csv','.checkpoint','.report.json','.failed-stage.bin','.log']],
   inputs=[initial] if kind=='puncture' else [],generated_inputs=[restart] if restart else [])
  config=control.normal_derivative_config(command)
  command['checks']=basic+[check(k,'eq',v) for k,v in config.items() if k!='executable']
  command['checks'] += [check('t','eq',end),check('diagnostics_current','eq',True),check('checkpoint_admitted','eq',True),
   check('max_null_surface','le',1e-12),check('max_null_ghost','le',1e-12),check('max_cfl','le',.5),
   check('min_chi','ge',1e-12),check('min_lapse','ge',1e-12),check('lapse_floor_node_interventions','eq',0),
   check('max_determinant_residual','le',1e-12),check('max_tracefree_residual','le',1e-12),
   check('cells','eq',12*16**3+8*(16+layers)**3+6*19*22**2),check('H_L2','finite'),check('M_L2','finite')]
  if sparse:command['checks'] += [check('sparse_poison_unused','eq',True)]+[check(['sparse_exchange_by_rank',r,6],'eq',1) for r in range(12)]
  if kind=='minkowski':command['checks'] += [check('max_state_deviation','le',1e-10),check('H_L2','le',1e-10)]
  if kind=='puncture':command['checks'] += [check('horizon_found','eq',True),check('horizon_expansion_rms','le',.001),check('H_exterior_L2','finite'),check('Mr_exterior_L2','finite')]
  commands.append(command)
 def compare(name,left,right):
  paths=[out+'/'+x+suffix for suffix in ['.checkpoint','.report.json'] for x in [left,right]]
  adapter(name,'restart',[a for pair in zip(['--reference','--candidate','--reference-report','--candidate-report'],paths) for a in pair],generated=paths)
 for layers in (1,2,3):
  for kappa in (0,1):evolution(f'minkowski-l{layers}-k{kappa}','minkowski',layers,kappa)
 evolution('oblique-full','oblique',sparse=False,end=.005)
 evolution('oblique-part','oblique',end=.0025)
 evolution('oblique-resume','oblique',end=.005,restart=out+'/oblique-part.checkpoint')
 compare('oblique-equivalence','oblique-full','oblique-resume')
 evolution('puncture-full','puncture',sparse=False,end=.05,dt=.0005)
 evolution('puncture-sparse','puncture',end=.05,dt=.0005)
 compare('puncture-equivalence','puncture-full','puncture-sparse')
 plan=dict(schema=1,reviewed=True,phase='preflight',n=16,low_resolution_n=16,
  resources=dict(queue='debug',account='CompactBinaryMerger',nodes=1,ranks=12,ranks_per_node=12,walltime='00:30:00'),
  storage=dict(limit_bytes=5000000000,planned_new_bytes=1100000000,atomic_checkpoint_reserve_bytes=100000000),
  worker_seconds=1710,output_dir=out,commands=commands,
  notes=['V12 isolated ghost-plus-paired-penalty preflight; numerical equations, log positivity, counted floor and scri closure preserved.',
   'Six short equilibrium checks; L1 paired oblique restart and poisoned sparse/full checkpoint equality; short n16 puncture equality to 0.15M.',
   'Not a 3M stability, reflection or convergence qualification. Horizon acceptance remains 1e-3.'])
 control.validate_plan(root,plan)
 return plan
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path('/home/hzhu/hyperbolic-core-shell-20261028'))
 p.add_argument('--source',default='source-thin-v1');p.add_argument('--build',default='build-sycl-mpi-thin-v1')
 p.add_argument('--campaign',default='campaigns/preflight-thin-v1');p.add_argument('--initial',default='inputs/puncture-thin-n16-l1.txt')
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 with a.output.open('x') as stream:json.dump(build(a.root,a.source,a.build,a.campaign,a.initial),stream,indent=2);stream.write('\n')
