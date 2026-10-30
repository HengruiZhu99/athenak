#!/usr/bin/env python3
"""Reviewed N24/N32 plans and verified DT1->AC1 migration; never submits jobs."""
import argparse,copy,hashlib,json,math,struct
from array import array
from pathlib import Path
import control
POLICY='n24_n32_matched_protocol_to20M'
STAGES={
 'DT1_to3M':dict(target=control.TIMESTEP_TARGET,build='build-sycl-mpi-timestep-v1',manifest='source-timestep-v1-manifest.json',source_sha='42bd9b41105e8f3bfb9901eb67e15d8f05d5f62d307c6bcb383750e1e3385ca2',binary_sha='6fa27b3a479ee25adb44d4025cbadfbe6a1a7bb5578d774e34995dd386fa48e4',magic='HYP3D_MPI_DT1',header=152,end=1.,scheme=1),
 'AC1_to20M':dict(target=control.DIRECT_CHI_LAPSE_TARGET,build='build-sycl-mpi-direct-chi-lapse-v1',manifest='source-direct-chi-lapse-v1-manifest.json',source_sha='56706997821cb434e142cbac3a217ac7360d4b7fdfc2efe9858e97df9f191b29',binary_sha='b8814162c43d2319f9d6db2370b54f270f592a5f7101adf36bf94a83b101d5b8',magic='HYP3D_MPI_AC1',header=168,end=20/3,scheme=3)}
DT_CONFIG=bytes.fromhex('0000000000000040070000000000000003000000070000000400000003000000040000000200000003000000030000000ad7a3703d0ac73f7b14ae47e17ac43fbbbdd7d9df7cdb3d0100000000000000')
def cells(n):return 12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2
def config(stage):
 c=bytearray(DT_CONFIG)
 if stage=='AC1_to20M':struct.pack_into('<i',c,72,3);c+=struct.pack('<2d',1e-8,.03)
 return bytes(c)
def make_plan(root,n,stage,segment,initial=None,restart=None):
 if n not in [24,32] or segment<1:raise ValueError('Unsupported resolution/segment')
 s=STAGES[stage];prefix=f'campaigns/n{n}-resolution20m-v1-{stage.lower()}-segment{segment:02d}';out=prefix+'/output';name=out+'/puncture'
 if bool(initial)==bool(restart) or (stage=='AC1_to20M' and initial):raise ValueError('Supply exactly one proper initial/restart input')
 checks=[dict(path=[k],op=op,value=v) for k,op,v in [('status','eq','completed'),('backend','eq','SYCL'),('mpi_size','eq',12),('n','eq',n),('checkpoint_schema','eq',s['magic']),('checkpoint_header_bytes','eq',s['header']),('max_null_surface','le',1e-10),('max_null_ghost','le',1e-10),('scri_penalty_scale_min','eq',1),('angular_epsilon_min','eq',2),('physical_lapse_floor','eq',1e-10),('positive_variables','eq','direct_chi_direct_lapse_rk_and_ko' if stage=='AC1_to20M' else 'log_chi_log_lapse_rk_and_ko')]]
 if stage=='AC1_to20M':checks += [dict(path=[k],op='eq',value=v) for k,v in [('puncture_chi_floor',1e-8),('puncture_chi_floor_radius',.03),('chi_nonpositive_policy','record_and_floor_finite_candidates_everywhere')]]
 inp=initial or restart;inputs=[inp]
 if restart and (root/(restart+'.horizon-seed.txt')).exists():inputs.append(restart+'.horizon-seed.txt')
 # Reserve checkpoint + failure state plus an atomic checkpoint copy; no large per-step dumps.
 cp=168+16*(31*cells(n)+6*n*n*31)
 plan=dict(schema=1,reviewed=True,phase='direct_variable_resolution',experimental_policy=POLICY,campaign_stage=stage,n=n,low_resolution_n=16,native_exit_reserve_seconds=180,
 resources=dict(control.FIXED,walltime='01:00:00'),worker_seconds=3200,storage=dict(limit_bytes=5_000_000_000,planned_new_bytes=math.ceil(1.6*cp)+10_000_000,atomic_checkpoint_reserve_bytes=cp+10_000_000),output_dir=out,
 commands=[dict(name='puncture',kind='puncture',executable=s['build']+'/tst/hyperbolic_shift_3d/'+s['target'],args=[str(n),name,'puncture',str(s['end']),str(.0005*16/n),initial or '-',restart or '-','@WALL_SECONDS@','0.05','0.001','12','24','2'],timeout_seconds=3100,report=name+'.report.json',inputs=inputs,checks=checks,artifacts=[name+x for x in ['.checkpoint','.history.csv','.failed-stage.bin','.checkpoint.horizon-seed.txt','.fastflow.horizon_summary_0.txt']])])
 control.validate_plan(root,plan)
 path=root/prefix/'plan.json';path.parent.mkdir(parents=True,exist_ok=False);control.atomic(path,plan);return str(path.relative_to(root))
def verify(root,relative):
 path=root/relative;plan=json.loads(path.read_text());control.validate_plan(root,plan);stage=plan['campaign_stage'];s=STAGES[stage];n=plan['n'];cmd=plan['commands'][0]
 worker=json.loads(path.with_name('worker-status.json').read_text());collected=json.loads(path.with_name('collected.json').read_text())
 if collected.get('scheduler',{}).get('job_state') not in ['F','X']:raise ValueError('Scheduler is not terminal')
 if worker['source_manifest_sha256']!=s['source_sha'] or worker['binaries']!={cmd['executable']:s['binary_sha']} or worker['plan_sha256']!=control.digest(path):raise ValueError('Frozen scientific binding mismatch')
 case=worker['cases'][0]
 if worker['status'] not in ['passed','checkpointed'] or case['status'] not in ['passed','checkpointed'] or case['timed_out'] or case['storage_error']:raise ValueError('Unhealthy worker; never resume')
 for name,a in case['artifacts'].items():
  p=root/name
  if p.stat().st_size!=a['bytes'] or control.digest(p)!=a['sha256']:raise ValueError('Artifact verification failed: '+name)
 r=json.loads((root/cmd['report']).read_text());cp=root/(cmd['args'][1]+'.checkpoint');data=cp.read_bytes();h=struct.unpack('<16s4i3Q2d',data[:72]);magic,hn,nf,kind,admitted,nc,ns,steps,t,dt=h
 if (magic.rstrip(b'\0').decode(),hn,nf,kind,admitted,nc,ns)!=(s['magic'],n,31,2,1,cells(n),6*n*n*31):raise ValueError('Checkpoint header mismatch')
 if data[72:s['header']]!=config(stage) or len(data)!=s['header']+16*(31*nc+ns):raise ValueError('Checkpoint config/length mismatch')
 if dt!=.0005*16/n or dt!=r['dt'] or t!=r['t'] or steps!=r['steps'] or not r['checkpoint_admitted'] or not 0<t<=s['end']:raise ValueError('Checkpoint time/admission mismatch')
 if r['status'] not in ['completed','checkpointed'] or (r['status']=='completed' and t!=s['end']):raise ValueError('Unexpected native termination')
 a=array('d');a.frombytes(data[s['header']:])
 if not all(map(math.isfinite,a)):raise ValueError('Nonfinite checkpoint payload')
 if any(a[i]<=0 for i in range(0,31*nc,31)):raise ValueError('Nonpositive accepted chi')
 seed=cp.with_name(cp.name+'.horizon-seed.txt')
 if seed.exists():
  v=seed.read_text().split()
  if len(v)!=5 or v[0]!='HYP3D_FASTFLOW_SEED_V1' or float(v[1])!=t or int(v[2])!=steps or not 0<=float(v[3])<=t or not 0<float(v[4])<1:raise ValueError('Horizon seed mismatch')
 result=dict(verified=True,n=n,stage=stage,checkpoint=str(cp.relative_to(root)),checkpoint_sha256=control.digest(cp),bytes=len(data),steps=steps,time=t,t_over_M=3*t,completed=t==s['end'],source_manifest_sha256=s['source_sha'],artifact_hashes=case['artifacts'])
 control.atomic(path.with_name('checkpoint-verified.json'),result);return result

def convert(data,expected_sha,n):
 if hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('Unverified checkpoint bytes')
 h=struct.unpack('<16s4i3Q2d',data[:72])
 if n not in [16,24,32] or h[:5]!=(b'HYP3D_MPI_DT1'.ljust(16,b'\0'),n,31,2,1) or h[5:7]!=(cells(n),6*n*n*31) or h[8:]!=(1.,.0005*16/n):raise ValueError('Migration requires admitted DT1 at exactly3M')
 if data[72:152]!=DT_CONFIG or len(data)!=152+16*(31*h[5]+h[6]):raise ValueError('Migration configuration mismatch')
 out=bytearray(b'HYP3D_MPI_AC1'.ljust(16,b'\0')+data[16:72]+config('AC1_to20M')+data[152:]);count=31*h[5]
 for c in range(h[5]):
  for f in [0,18]:struct.pack_into('<d',out,168+8*(count+31*c+f),0.)
 for p in range(h[6]//31):struct.pack_into('<d',out,168+8*(2*count+h[6]+31*p),0.)
 return out

def migrate(root,relative,output):
 v=verify(root,relative)
 if v['stage']!='DT1_to3M' or not v['completed']:raise ValueError('Preparation is incomplete')
 p=root/output
 if p.exists():raise ValueError('Migration output exists')
 control.storage.require_capacity(root,dict(storage=dict(limit_bytes=5_000_000_000,planned_new_bytes=v['bytes']+16,atomic_checkpoint_reserve_bytes=10_000_000),commands=[dict(kind='check')]),prospective=True)
 out=convert((root/v['checkpoint']).read_bytes(),v['checkpoint_sha256'],v['n']);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(out)
 result=dict(source=v,output=output,sha256=control.digest(p),physical_fields_unchanged=True,reset_compensation=['volume_chi','volume_lapse','surface_chi'])
 control.atomic(p.with_name(p.name+'.migration.json'),result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','verify','migrate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--n',type=int,choices=[24,32]);p.add_argument('--stage',choices=list(STAGES));p.add_argument('--segment',type=int,default=1);p.add_argument('--initial');p.add_argument('--restart');p.add_argument('--plan');p.add_argument('--output');a=p.parse_args()
 if a.action=='plan':r=make_plan(a.root,a.n,a.stage,a.segment,a.initial,a.restart)
 elif a.action=='verify':r=verify(a.root,a.plan)
 else:r=migrate(a.root,a.plan,a.output)
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
