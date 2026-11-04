import json,pathlib
root=pathlib.Path('docs/hyperbolic-shift-3d/reproduction/performance-scaling-v1/scripts')
r='/home/hzhu/hyperboloidal-performance-scaling-20261103'
p={'aurora':True,'nodes':2,'deadline_seconds':3100,'variants':{'base':r+'/control-build-v2/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_evolution','m0':r+'/m1-build/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_evolution','m4':r+'/m4-bin/hyperbolic_shift_3d_amr_setup'},'runs':[]}
def add(name,v='m4',P=2,kind='wave',steps=6,policy='dynamic',key=None,**kw):
 d=dict(name=name,variant=v,ranks=P,kind=kind,steps=steps,policy=policy,timeout=240,**kw)
 if key:d['equivalence']=key
 p['runs'].append(d)
for policy,kind in [('dynamic','minkowski'),('uniform','wave'),('uniform','boost'),('fixed','boost')]:
 for v in ['base','m4']:add('qual-'+policy+'-'+kind+'-'+v,v,kind=kind,policy=policy,key=policy+kind)
add('qual-rejected-m4',kind='boost',expected=2)
add('restart-save',steps=3)
add('restart-finish',P=4,restart=r+'/gpu-job2/restart-save.checkpoint',key='dynamicwave')
for policy,P,steps in [('dynamic',2,6),('fixed',4,8)]:
 for rep in range(3):
  for v in (['m0','m4'] if rep%2==0 else ['m4','m0']):add(f'setup-{policy}-p{P}-r{rep}-{v}',v,P,steps=steps,policy=policy,key=policy+'wave' if policy=='dynamic' else 'fixed8')
# Same entire problem at every rank; rotate ordering between repeats.
points=[('mixed',1,1),('mixed',2,2),('mixed',4,4),('mixed',8,8),('mixed-inter',8,4),('uniform',1,1),('uniform',4,4),('uniform',12,12),('uniform',24,12)]
for rep in range(3):
 for label,P,ppn in (points if rep%2==0 else list(reversed(points))):
  policy='uniform' if label=='uniform' else 'fixed'
  add(f'scaling-{label}-p{P}-ppn{ppn}-r{rep}-m4',P=P,ppn=ppn,steps=32,policy=policy,key=policy+'32')
(root/'gpu-plan2.json').write_text(json.dumps(p,indent=2)+'\n')
s=(root/'job1.pbs').read_text().replace('select=1','select=2')
s=s[:s.index('mkdir -p')] + '''mkdir -p "$NEO_CACHE_DIR" "$SYCL_CACHE_DIR" job2-components
unset ZE_AFFINITY_MASK
module list > job2-components/modules.txt 2>&1
cat "$PBS_NODEFILE" > job2-components/nodes.txt
export HYPERBOLIC_MAPPING_PREFIX="$PWD/job2-components/mapping"
timeout 180 mpiexec -n 24 -ppn 12 --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 gpu_tile_compact.sh m1-build/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_device_mapping > job2-components/mapping.log 2>&1
timeout 300 mpiexec -n 1 -ppn 1 --cpu-bind=list:1-8 gpu_tile_compact.sh m4-bin/hyperbolic_shift_3d_amr_setup_contract > job2-components/setup-contract.log 2>&1
exec /usr/bin/python3.11 run_plan.py gpu-plan2.json gpu-job2
'''
(root/'job2.pbs').write_text(s)
print(len(p['runs']))
