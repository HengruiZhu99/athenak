"""One fresh nontrivial 12-tile closure audit using the existing preflight phase."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
sys.path.insert(0,str(r/'launcher-zero-restoration-v1'))
import control,native_transition_campaign as c
science=c.science(r);b='build-sycl-mpi-zero-restoration-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_zero_scri_dissipation_closure'
assert science['audit_binaries'][b]==control.digest(r/b)
prov=json.loads((r/'launcher-zero-restoration-v1/provenance.json').read_text())
for name,e in prov['files'].items():
 f=r/'launcher-zero-restoration-v1'/name;assert f.stat().st_size==e['bytes'] and control.digest(f)==e['sha256']
f=r/'build-sycl-mpi-zero-restoration-v1/tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_zero_scri_dissipation_closure.dir/flags.make'
assert '-DHYPERBOLIC_ZERO_SHIFT_RESTORATION=1' in f.read_text()
prefix='campaigns/zero-restoration-v1-boundary-audit';out=prefix+'/output';base=out+'/closure'
plan=dict(schema=1,reviewed=True,phase='preflight',n=16,low_resolution_n=16,
 resources=dict(control.FIXED,walltime='00:20:00'),worker_seconds=1000,
 storage=dict(limit_bytes=5_000_000_000,planned_new_bytes=2_000_000,atomic_checkpoint_reserve_bytes=10_000_000),output_dir=out,
 commands=[dict(name='closure',kind='check',executable=b,args=[base],timeout_seconds=600,
 report=base+'.rank0.json',inputs=[c.SCIENCE],checks=[dict(path=[k],op='eq',value=v) for k,v in dict(passed=True,backend='SYCL',zero_shift_restoration=True,rank=0).items()],
 artifacts=[base+f'.rank{i}.json' for i in range(12)])])
control.validate_plan(r,plan);path=r/prefix/'plan.json';path.parent.mkdir(parents=True,exist_ok=False);control.atomic(path,plan)
print(json.dumps({'plan':str(path.relative_to(r)),'binary_sha256':science['audit_binaries'][b]}))
