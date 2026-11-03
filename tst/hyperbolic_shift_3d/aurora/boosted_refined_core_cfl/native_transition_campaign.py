#!/usr/bin/env python3
"""Uniformly refined core v=.1 puncture control (N16 shells, 64-cubed core): plans/checkpoint verification only, never submission.

The controller calls the validators here only for this experiment's new phase.
All old campaign phases retain their existing policies. No import runs a job.
"""
import argparse
from array import array
import json
import math
from pathlib import Path
import re
import struct
import sys
import control

PHASE = 'native_transition_evolution'
POLICY = 'boost01_x_free_gauge_N16_uniform_core4_shell16_dt00025_fixed_penalty_two_nodes_to10M'
SOURCE_COMMIT = 'cb0f47b520afe4f88c98e11569f3ba88e632df9a'
MANIFEST = 'source-boosted-refined-core-cfl-v1-manifest.json'
SCIENCE = 'launcher-boosted-refined-core-cfl-v1/native-transition-science.json'
BUILD = 'build-sycl-mpi-boosted-refined-core-cfl-v1/tst/hyperbolic_shift_3d/'
VARIANTS = {v: dict(suffix='control_broad_smooth_six_shells_boosted_free_gauge_refined_core_matched_cfl', magic='BC2', shared=False, ko10=False, scri6=False, n=n, dt=dt) for v,n,dt in [('C16',16,.00025)]}
# BC1: 168-byte base layout plus 40-byte boost/tracker configuration.
CONFIG = b'\x00\x00\x00\x00\x00\x00\x00@\x07\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x07\x00\x00\x00\x04\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x04\x00\x00\x00\x06\x00\x00\x00\x03\x00\x00\x00\n\xd7\xa3p=\n\xc7?{\x14\xaeG\xe1z\xc4?\xbb\xbd\xd7\xd9\xdf|\xdb=\x03\x00\x00\x00\x00\x00\x00\x00:\x8c0\xe2\x8eyE>\xb8\x1e\x85\xebQ\xb8\x9e?'
END = 10/3

def resolution(variant):
    v=VARIANTS[variant];return v['n'],v['dt']

def expected_penalty_scale(variant):
    return .5*(1-1e-12)/.768


def cells(n):
    return 30*n**3+(4*n+6)**3+6*(n+3)*(n+6)**2


def binary(variant):
    return BUILD+'hyperbolic_shift_3d_mpi_native_'+VARIANTS[variant]['suffix']


def magic(variant):
    return 'HYP3D_MPI_'+VARIANTS[variant]['magic']


def science(root):
    s=json.loads(control.local_path(root,SCIENCE).read_text())
    if (s.get('schema')!=1 or s.get('source_commit')!=SOURCE_COMMIT
            or s.get('source_manifest')!=MANIFEST
            or not re.fullmatch('[0-9a-f]{64}',s.get('source_manifest_sha256',''))
            or set(s.get('binaries',{}))!={binary(v) for v in VARIANTS}
            or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in s['binaries'].values())):
        raise ValueError('Native matrix needs the frozen six-shell binary and the tested source')
    return s


def required_checks(variant, purpose, kind, restart=False):
    N,DT=resolution(variant)
    v=VARIANTS[variant];end=2*DT if purpose=='smoke' else END
    provider='checkpoint' if restart else 'native_boosted_wormhole_embedding' if kind=='puncture' else 'builtin'
    exact=dict(mesh_topology='cartesian_4x4x4_shells6',cartesian_core_subdivisions=4,mesh_blocks=100,shell_layers=6,shell_outer_radii=[.305,.45,.6,.75,.875,1],restoring_collar_begin=.6,restoring_collar_end=.9,restoring_coefficient=4,zero_shift_restoration=False,broad_smooth_a=True,native_initialization_profile='polynomial_c6',native_initialization_start_M=4,native_initialization_end_M=12,status='completed',kind=kind,n=N,cells=cells(N),backend='SYCL',mpi_size=24,
        checkpoint_schema=magic(variant),checkpoint_header_bytes=208,boost_velocity=.1,puncture_tracker=True,checkpoint_admitted=True,
        initial_shift_policy='zero_inner_hyperboloidal_outer',initial_lapse_policy='boost_determinant_precollapsed_hyperboloidal',
        initial_data_provider=provider,native_initialization_tolerance=2e-13,
        inner_gauge=('shared_restoration_r020_056_it1_damping_v1' if v['shared'] else 'direct_outer_r020_056_it1_damping_v1'),
        gauge_blend='smoothstep9_w_056_080',scri_first6=v['scri6'],
        bulk_dissipation='block_energy_ko10' if v['ko10'] else 'block_energy_ko8',
        positive_variables='direct_chi_direct_lapse_rk_and_ko',physical_lapse_floor=1e-10,
        puncture_chi_floor=1e-8,puncture_chi_floor_radius=.03,
        chi_nonpositive_policy='record_and_floor_finite_candidates_everywhere',
        timestep_aware_penalty=True,scri_penalty_fixed_scale=expected_penalty_scale(variant),angular_epsilon_min=2,complete_candidate_exterior_enabled=True,
        dt=DT,algebraic_projection='final_stage_standard')
    checks=[dict(path=[k],op='eq',value=v) for k,v in exact.items()]
    # Explicit new user-authorized policy; never changes the frozen N16 policy.
    scale=expected_penalty_scale(variant)
    for key,value in [('scri_penalty_scale_min',scale),('scri_unlimited_penalty_cfl_max',.384),('scri_penalty_cfl_max',.25*(1-1e-12))]:
        checks += [dict(path=[key],op='ge',value=value-1e-12),dict(path=[key],op='le',value=value+1e-12)]
    checks += [dict(path=['t'],op='le',value=end),dict(path=['t'],op='ge',value=0),
               dict(path=['max_null_surface'],op='le',value=1e-10),
               dict(path=['max_null_ghost'],op='le',value=1e-10)]
    return checks


def validate_native_plan(root,plan):
    variant=plan.get('variant');purpose=plan.get('purpose');segment=plan.get('segment')
    N,DT=resolution(variant)
    if (plan.get('phase')!=PHASE or plan.get('experimental_policy')!=POLICY
            or variant not in VARIANTS or purpose not in ['evolution']
            or plan.get('n')!=N or plan.get('low_resolution_n')!=N
            or type(segment) is not int or segment<1 or (purpose=='smoke' and segment!=1)
            or plan.get('native_exit_reserve_seconds')!=180):
        raise ValueError('Native transition is restricted to the boosted six-shell controls and verified same-case restarts')
    science(root)
    commands=plan.get('commands',[]);kinds=['minkowski','puncture'] if purpose=='smoke' else ['puncture']
    if [c.get('kind') for c in commands]!=kinds:
        raise ValueError('Smoke requires Minkowski then puncture; long trial is one puncture case')
    start=plan.get('start_verification',{})
    if (start.get('variant')!=variant or start.get('schema')!=magic(variant)
            or start.get('n')!=N or start.get('dt')!=DT):
        raise ValueError('Starting state identity differs from the selected case')
    if segment==1 and (start.get('time')!=0 or start.get('steps')!=0 or start.get('kind')!='native_fresh'):
        raise ValueError('First segment must use fresh native initialization at t=0')
    if segment>1 and (start.get('kind')!='native_restart' or not 0<start.get('time',-1)<END
            or type(start.get('steps')) is not int or start['steps']<1
            or not re.fullmatch('[0-9a-f]{64}',start.get('sha256',''))):
        raise ValueError('Continuation lacks an identified same-case checkpoint')
    for c in commands:
        a=c.get('args',[])
        if (c.get('executable')!=binary(variant) or len(a)!=13
                or a[0]!=str(N) or a[2]!=c['kind'] or float(a[3])!=(2*DT if purpose=='smoke' else END)
                or float(a[4])!=DT or a[5]!='-' or (segment==1)!=(a[6]=='-')
                or a[8:]!=['0.05','0.001','12','24','2']):
            raise ValueError('Native control arguments changed the matched science or initialization protocol')
        expected_inputs=[SCIENCE]
        if segment>1:
            expected_inputs.append(a[6]);seed=a[6]+'.horizon-seed.txt'
            if control.local_path(root,seed).exists():expected_inputs.append(seed)
        if c.get('inputs')!=expected_inputs or c.get('generated_inputs'):
            raise ValueError('Only frozen science and the current restart/seed may be bound inputs')
        for check in required_checks(variant,purpose,c['kind'],segment>1):
            if check not in c.get('checks',[]):raise ValueError('Missing or weakened native scientific report check: '+str(check))


def validate_science_binding(root,plan,source_digest,binaries):
    frozen=science(root);b=binary(plan['variant'])
    if source_digest!=frozen['source_manifest_sha256'] or binaries!={b:frozen['binaries'][b]}:
        raise ValueError('Native experiment science differs from its frozen record')
    return dict(path=SCIENCE,sha256=control.digest(control.local_path(root,SCIENCE)))


def inspect_checkpoint(path,variant,expected_sha,kind='puncture',end=END):
    N,DT=resolution(variant)
    if variant not in VARIANTS or sys.byteorder!='little' or control.digest(path)!=expected_sha:
        raise ValueError('Unverified checkpoint identity or unsupported host byte order')
    data=Path(path).read_bytes()
    if len(data)<208:raise ValueError('Truncated checkpoint header')
    h=struct.unpack('<16s4i3Q2d',data[:72]);nc=cells(N);ns=6*N*N*31
    if h[:7]!=(magic(variant).encode().ljust(16,b'\0'),N,31,{'minkowski':0,'puncture':2}[kind],1,nc,ns):
        raise ValueError('Checkpoint schema, dimensions, kind or admission mismatch')
    if data[72:168]!=CONFIG or len(data)!=208+16*(31*nc+ns):
        raise ValueError('Checkpoint configuration or exact length mismatch')
    if (h[9]!=DT or not 0<h[8]<=end or h[7]<1
            or h[8]>h[7]*DT+1e-10 or h[8]<(h[7]-1)*DT-1e-10):
        raise ValueError('Checkpoint time, steps or dt inconsistent with fixed-step evolution')
    boost=struct.unpack('<5d',data[168:208])
    if boost[0]!=.1 or boost[4]!=.16 or any(not math.isfinite(x) or abs(x)>=.16 for x in boost[1:4]):
        raise ValueError('Boost or tracker checkpoint mismatch')
    values=array('d');values.frombytes(data[208:])
    if not all(map(math.isfinite,values)):
        raise ValueError('Nonfinite checkpoint payload')
    # Layout: volume, volume compensation, surface, surface compensation.
    if any(values[i]<=0 for i in range(0,31*nc,31)) or any(values[i]<=0 for i in range(2*31*nc,2*31*nc+ns,31)):
        raise ValueError('Nonpositive accepted volume/surface chi')
    seed=Path(str(path)+'.horizon-seed.txt')
    if seed.exists():
        v=seed.read_text().split()
        if (len(v)!=5 or v[0]!='HYP3D_FASTFLOW_SEED_V1' or float(v[1])!=h[8]
                or int(v[2])!=h[7] or not 0<=float(v[3])<=h[8] or not 0<float(v[4])<1):
            raise ValueError('Invalid horizon seed sidecar')
    return dict(puncture_position=list(boost[1:4]),variant=variant,schema=magic(variant),n=N,time=h[8],steps=h[7],dt=h[9],bytes=len(data),sha256=expected_sha)


def make_plan(root,variant,purpose='evolution',segment=1,restart=None,expected_sha=None):
    N,DT=resolution(variant)
    root=Path(root).resolve()
    if variant not in VARIANTS or purpose not in ['evolution'] or segment<1:
        raise ValueError('Unknown case/purpose or segment')
    if bool(restart)!=(segment>1) or (purpose=='smoke' and segment!=1):
        raise ValueError('Fresh t=0 first segment, same-case restarts only for later evolution segments')
    science(root)
    start=(inspect_checkpoint(control.local_path(root,restart),variant,expected_sha) if restart
           else dict(variant=variant,schema=magic(variant),n=N,time=0.,steps=0,dt=DT))
    start['kind']='native_restart' if restart else 'native_fresh'
    if start['time']>=END:raise ValueError('This case already reached10M')
    prefix=f'campaigns/n{N}-boosted-refined-core-cfl-v2-{variant.lower()}-{purpose}-segment{segment:02d}'
    out=prefix+'/output';commands=[]
    for kind in (['minkowski','puncture'] if purpose=='smoke' else ['puncture']):
        name=out+'/'+kind;inputs=[SCIENCE]
        if restart:
            inputs.append(restart)
            if control.local_path(root,restart+'.horizon-seed.txt').exists():inputs.append(restart+'.horizon-seed.txt')
        commands.append(dict(name=kind,kind=kind,executable=binary(variant),
            args=[str(N),name,kind,str(2*DT if purpose=='smoke' else END),str(DT),'-',restart or '-',
                  '@WALL_SECONDS@','0.05','0.001','12','24','2'],timeout_seconds=3100,
            report=name+'.report.json',inputs=inputs,checks=required_checks(variant,purpose,kind,bool(restart)),
            artifacts=[name+x for x in ['.checkpoint','.history.csv','.failed-stage.bin','.checkpoint.horizon-seed.txt',
                                        '.fastflow.horizon_summary_0.txt','.fastflow.horizon_shape_0.txt']]))
    cp=208+16*(31*cells(N)+6*N*N*31)
    plan=dict(schema=1,reviewed=True,phase=PHASE,experimental_policy=POLICY,variant=variant,purpose=purpose,
        segment=segment,n=N,low_resolution_n=N,native_exit_reserve_seconds=180,
        resources=dict(control.FIXED,walltime='01:00:00'),worker_seconds=3200,
        storage=dict(limit_bytes=10_000_000_000,planned_new_bytes=math.ceil((1.6+len(commands)-1)*cp)+10_000_000,
                     atomic_checkpoint_reserve_bytes=cp+10_000_000),output_dir=out,start_verification=start,commands=commands)
    control.validate_plan(root,plan)
    path=control.local_path(root,prefix+'/plan.json');path.parent.mkdir(parents=True,exist_ok=False)
    control.atomic(path,plan);return dict(plan=str(path.relative_to(root)),start=start)


def verify(root,relative):
    root=Path(root).resolve()
    path,plan,binding=control.read_bound(root,relative)
    worker=json.loads(path.with_name('worker-status.json').read_text())
    collected=json.loads(path.with_name('collected.json').read_text());frozen=science(root);b=binary(plan['variant'])
    submission=json.loads(path.with_name('submission.json').read_text())
    if (collected.get('scheduler',{}).get('job_state') not in ['F','X']
            or collected.get('returncode')!=0):
        raise ValueError('Scheduler is not verifiably terminal')
    if (submission.get('job_id')!=collected.get('job_id') or submission.get('job_id')!=worker.get('job_id')
            or submission.get('plan')!=relative or submission.get('plan_sha256')!=binding['plan_sha256']
            or submission.get('source_manifest_sha256')!=binding['source_manifest_sha256']):
        raise ValueError('Submission, scheduler and worker identities disagree')
    if collected['scheduler'].get('Exit_status')!=(0 if worker['status']=='passed' else 3):
        raise ValueError('Scheduler exit is not the expected healthy worker disposition')
    if (binding['plan_sha256']!=control.digest(path) or worker['plan_sha256']!=binding['plan_sha256']
            or binding['source_manifest_sha256']!=frozen['source_manifest_sha256']
            or worker['source_manifest_sha256']!=frozen['source_manifest_sha256']
            or binding['binaries']!={b:frozen['binaries'][b]} or worker['binaries']!=binding['binaries']
            or binding.get('admission')!=validate_science_binding(root,plan,frozen['source_manifest_sha256'],binding['binaries'])):
        raise ValueError('Frozen science or plan binding mismatch')
    if (worker['status'] not in ['passed','checkpointed'] or len(worker['cases'])!=len(plan['commands'])
            or (plan['purpose']=='smoke' and worker['status']!='passed')):
        raise ValueError('Unhealthy/incomplete worker; never resume')
    expected_inputs={p:control.digest(control.local_path(root,p)) for c in plan['commands'] for p in c['inputs']}
    if binding['inputs']!=expected_inputs:raise ValueError('Starting input/seed/science binding changed')
    if plan['segment']>1 and expected_inputs[plan['commands'][0]['args'][6]]!=plan['start_verification']['sha256']:
        raise ValueError('Restart differs from the verified starting state')
    outputs=[]
    for cmd,case in zip(plan['commands'],worker['cases']):
        if (case.get('name')!=cmd['name'] or case['status']!=worker['status'] or case['timed_out']
                or case['storage_error'] or case.get('report_error') or case.get('returncode') not in [0,3]):
            raise ValueError('Unhealthy native case; never resume')
        cp=cmd['args'][1]+'.checkpoint'
        for name in [cp,cmd['report'],cmd['args'][1]+'.history.csv']:
            if name not in case['artifacts']:raise ValueError('Missing required native artifact: '+name)
        for name,a in case['artifacts'].items():
            p=control.local_path(root,name)
            if p.stat().st_size!=a['bytes'] or control.digest(p)!=a['sha256']:raise ValueError('Corrupt artifact: '+name)
        end=float(cmd['args'][3]);v=inspect_checkpoint(root/cp,plan['variant'],case['artifacts'][cp]['sha256'],cmd['kind'],end)
        r=json.loads(control.local_path(root,cmd['report']).read_text())
        if r['status'] not in ['completed','checkpointed'] or not r['checkpoint_admitted']:
            raise ValueError('Unadmitted native state')
        if (worker['status']=='passed')!=(r['status']=='completed'):
            raise ValueError('Native/worker disposition mismatch')
        if r.get('puncture_position')!=v['puncture_position']:raise ValueError('Report/checkpoint tracker mismatch')
        for key in ['t','steps','dt']:
            if r[key]!=v['time' if key=='t' else key]:raise ValueError('Report/header time mismatch')
        if (v['time']<=plan['start_verification']['time'] or v['steps']<=plan['start_verification']['steps']
                or (r['status']=='completed')!=(v['time']==end)):
            raise ValueError('No progress or incorrect endpoint')
        for check in cmd['checks']:
            if check['path']!=['status'] and not control.evaluate_check(r,check)['passed']:
                raise ValueError('Scientific report mismatch: '+str(check))
        outputs.append(dict(checkpoint=cp,completed=v['time']==end,kind=cmd['kind'],
            t_over_M=v['time']*(3 if cmd['kind']=='puncture' else 1),artifact_hashes=case['artifacts'],**v))
    result=dict(verified=True,variant=plan['variant'],purpose=plan['purpose'],source_manifest_sha256=frozen['source_manifest_sha256'],cases=outputs)
    control.atomic(path.with_name('checkpoint-verified.json'),result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','verify']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=list(VARIANTS));p.add_argument('--purpose',choices=['evolution'],default='evolution')
    p.add_argument('--segment',type=int,default=1);p.add_argument('--restart');p.add_argument('--sha256');p.add_argument('--plan')
    a=p.parse_args();root=a.root.resolve()
    result=make_plan(root,a.variant,a.purpose,a.segment,a.restart,a.sha256) if a.action=='plan' else verify(root,a.plan)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
