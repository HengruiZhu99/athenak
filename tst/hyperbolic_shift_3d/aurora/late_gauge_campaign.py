#!/usr/bin/env python3
"""Fresh LG1 direct-variable evolutions and verified continuations; never submits jobs."""
import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import control
from resolution_campaign import DT_CONFIG, cells

BINARY = 'build-sycl-mpi-late-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_late_gauge'
INITIAL_SHA = {
    16: '35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca',
    24: '6c265e8dc309c70a2c7017a80eddd2d76f3a185b4f1d6108cc65b8bfc85be914'}
CONFIG = bytearray(DT_CONFIG)
struct.pack_into('<i', CONFIG, 72, 3)
CONFIG += struct.pack('<2d',1e-8,.03)
CONFIG = bytes(CONFIG)


def science():
    # Frozen alongside the launcher after the build; provenance binds this file.
    s=json.loads(Path(__file__).with_name('late-gauge-science.json').read_text())
    if s['binary'] != BINARY or len(s['binary_sha256']) != 64 or len(s['source_manifest_sha256']) != 64:
        raise ValueError('Invalid frozen science record')
    return s


END = 10/3


def inspect_checkpoint(path, n, expected_sha):
    if n not in [16] or control.digest(path) != expected_sha:
        raise ValueError('Unverified checkpoint identity')
    data = Path(path).read_bytes()
    h = struct.unpack('<16s4i3Q2d', data[:72])
    if h[:7] != (b'HYP3D_MPI_LG1'.ljust(16,b'\0'), n,31,2,1,cells(n),6*n*n*31):
        raise ValueError('Expected admitted LG1 checkpoint at the requested resolution')
    if data[72:168] != CONFIG or len(data) != 168+16*(31*h[5]+h[6]):
        raise ValueError('Checkpoint configuration or length mismatch')
    if not 0 < h[8] <= END or h[9] != .0005*16/n or h[7] < 1:
        raise ValueError('Checkpoint time/dt outside continuation policy')
    values = array('d'); values.frombytes(data[168:])
    if not all(map(math.isfinite,values)) or any(values[i]<=0 for i in range(0,31*h[5],31)):
        raise ValueError('Nonfinite payload or nonpositive accepted chi')
    seed = Path(str(path)+'.horizon-seed.txt')
    if seed.exists():
        v = seed.read_text().split()
        if (len(v)!=5 or v[0]!='HYP3D_FASTFLOW_SEED_V1' or float(v[1])!=h[8]
                or int(v[2])!=h[7] or not 0<=float(v[3])<=h[8] or not 0<float(v[4])<1):
            raise ValueError('Invalid horizon seed sidecar')
    return dict(n=n,time=h[8],steps=h[7],dt=h[9],bytes=len(data),sha256=expected_sha)


def make_plan(root,n,segment,initial=None,restart=None,expected_sha=None):
    if n not in [16] or segment<1 or bool(initial)==bool(restart):
        raise ValueError('Supply N16 and exactly one initial/restart input')
    if initial:
        if segment != 1 or control.digest(control.local_path(root,initial)) != INITIAL_SHA[n]:
            raise ValueError('First segment requires verified original wormhole generator data')
        start=dict(n=n,time=0.,steps=0,dt=.0005*16/n,sha256=INITIAL_SHA[n],kind='fresh_wormhole')
    else:
        if segment <= 1: raise ValueError('Segment01 must start from fresh wormhole data')
        start=inspect_checkpoint(control.local_path(root,restart),n,expected_sha)
        start['kind']='LG1_restart'
    if start['time']>=END: raise ValueError('Already completed10M')
    prefix=f'campaigns/n{n}-late-gauge-fresh10m-v1-segment{segment:02d}'
    out=prefix+'/output'; name=out+'/puncture'; inputs=[initial or restart]
    if restart and (root/(restart+'.horizon-seed.txt')).exists(): inputs.append(restart+'.horizon-seed.txt')
    cp=168+16*(31*cells(n)+6*n*n*31)
    checks=[dict(path=[k],op=op,value=v) for k,op,v in [
        ('status','eq','completed'),('t','le',END),('t','ge',0.),('n','eq',n),('backend','eq','SYCL'),
        ('mpi_size','eq',12),('checkpoint_schema','eq','HYP3D_MPI_LG1'),
        ('checkpoint_admitted','eq',True),('inner_gauge','eq','direct_outer_r040_090_it1_damping_v1'),('gauge_blend','eq','smoothstep9_w_056_080'),
        ('positive_variables','eq','direct_chi_direct_lapse_rk_and_ko'),
        ('physical_lapse_floor','eq',1e-10),('puncture_chi_floor','eq',1e-8),
        ('puncture_chi_floor_radius','eq',.03),('checkpoint_header_bytes','eq',168),
        ('chi_nonpositive_policy','eq','record_and_floor_finite_candidates_everywhere'),
        ('max_null_surface','le',1e-10),('max_null_ghost','le',1e-10),
        ('scri_penalty_scale_min','eq',1),('angular_epsilon_min','eq',2),
        ('complete_candidate_exterior_enabled','eq',True)]]
    plan=dict(schema=1,reviewed=True,phase='late_gauge_evolution',segment=segment,
        experimental_policy='n16_late_gauge_fresh_to10M',n=n,low_resolution_n=16,
        native_exit_reserve_seconds=180,resources=dict(control.FIXED,walltime='01:00:00'),
        worker_seconds=3200,storage=dict(limit_bytes=5_000_000_000,
            planned_new_bytes=math.ceil(1.6*cp)+10_000_000,atomic_checkpoint_reserve_bytes=cp+10_000_000),
        output_dir=out,start_verification=start,
        commands=[dict(name='puncture',kind='puncture',executable=BINARY,
            args=[str(n),name,'puncture',str(END),str(.0005*16/n),initial or '-',restart or '-',
                  '@WALL_SECONDS@','0.05','0.001','12','24','2'],
            timeout_seconds=3100,report=name+'.report.json',inputs=inputs,checks=checks,
            artifacts=[name+x for x in ['.checkpoint','.history.csv','.failed-stage.bin',
                                      '.checkpoint.horizon-seed.txt','.fastflow.horizon_summary_0.txt']])])
    control.validate_plan(root,plan)
    path=root/prefix/'plan.json';path.parent.mkdir(parents=True,exist_ok=False)
    control.atomic(path,plan);return dict(plan=str(path.relative_to(root)),restart=start)


def verify(root,relative):
    path=control.local_path(root,relative);plan=json.loads(path.read_text())
    control.validate_plan(root,plan);n=plan['n'];cmd=plan['commands'][0];frozen=science()
    SOURCE_SHA=frozen['source_manifest_sha256'];BINARY_SHA=frozen['binary_sha256']
    binding=json.loads(path.with_name('binding.json').read_text())
    worker=json.loads(path.with_name('worker-status.json').read_text())
    collected=json.loads(path.with_name('collected.json').read_text())
    if collected.get('scheduler',{}).get('job_state') not in ['F','X']:
        raise ValueError('Scheduler is not terminal')
    if (binding['plan_sha256']!=worker['plan_sha256'] or worker['plan_sha256']!=control.digest(path)
            or binding['source_manifest_sha256']!=SOURCE_SHA or worker['source_manifest_sha256']!=SOURCE_SHA
            or binding['binaries']!={BINARY:BINARY_SHA} or worker['binaries']!={BINARY:BINARY_SHA}):
        raise ValueError('Frozen scientific binding mismatch')
    if worker['status'] not in ['passed','checkpointed'] or len(worker['cases'])!=1:
        raise ValueError('Unhealthy worker; never resume')
    if binding['inputs'].get(cmd['args'][5] if plan['segment']==1 else cmd['args'][6])!=plan['start_verification']['sha256']:
        raise ValueError('Restart binding disagrees with verified starting state')
    case=worker['cases'][0]
    if case['status']!=worker['status'] or case['timed_out'] or case['storage_error']:
        raise ValueError('Unhealthy case; never resume')
    for name,a in case['artifacts'].items():
        p=control.local_path(root,name)
        if p.stat().st_size!=a['bytes'] or control.digest(p)!=a['sha256']:
            raise ValueError('Artifact verification failed: '+name)
    r=json.loads((root/cmd['report']).read_text());cp=cmd['args'][1]+'.checkpoint'
    v=inspect_checkpoint(root/cp,n,case['artifacts'][cp]['sha256'])
    if r['status'] not in ['completed','checkpointed'] or not r['checkpoint_admitted']:
        raise ValueError('Unadmitted native state')
    for k in ['t','steps','dt']:
        if r[k]!=v['time' if k=='t' else k]: raise ValueError('Report/header mismatch')
    if (v['time']<=plan['start_verification']['time'] or v['steps']<=plan['start_verification']['steps']
            or (r['status']=='completed')!=(v['time']==END)):
        raise ValueError('No forward progress or incorrect completion time')
    for check in cmd['checks']:
        if check['path'][0] not in ['status','t'] and not control.evaluate_check(r,check)['passed']:
            raise ValueError('Scientific check failed: '+str(check))
    result=dict(verified=True,checkpoint=cp,completed=v['time']==END,t_over_M=3*v['time'],
                source_manifest_sha256=SOURCE_SHA,artifact_hashes=case['artifacts'],**v)
    control.atomic(path.with_name('checkpoint-verified.json'),result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','verify'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--n',type=int,choices=[16])
    p.add_argument('--segment',type=int,default=1);p.add_argument('--initial');p.add_argument('--restart');p.add_argument('--sha256');p.add_argument('--plan')
    a=p.parse_args();root=a.root.resolve()
    print(json.dumps(make_plan(root,a.n,a.segment,a.initial,a.restart,a.sha256) if a.action=='plan' else verify(root,a.plan),indent=2))


if __name__=='__main__': main()
