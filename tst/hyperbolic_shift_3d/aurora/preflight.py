#!/usr/bin/python3.11
"""Stdlib adapters for 12 standalone GPU tests and checkpoint comparison.

Launched once by the worker's mpiexec -n 12. No nested launcher or mpi4py.
All ranks report over one local Unix socket; only rank zero writes the report.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time

import control

HEADER = struct.Struct('<16s4i3Q2d')
KEYS = ['magic', 'n', 'nf', 'kind', 'admitted', 'cells', 'surface', 'steps', 'time', 'dt']


def rank_info():
    rank = int(os.environ.get('MPI_LOCALRANKID', os.environ.get('PALS_LOCAL_RANKID', '-1')))
    if not 0 <= rank < 12:
        raise ValueError('Preflight requires a local rank in 0..11 from the 12-rank launcher')
    return rank, dict(rank=rank, host=socket.gethostname(), mask=os.environ.get('ZE_AFFINITY_MASK'),
                      hierarchy=os.environ.get('ZE_FLAT_DEVICE_HIERARCHY'))


def read_checkpoint(path):
    with path.open('rb') as stream:
        data = stream.read(HEADER.size)
    if len(data) != HEADER.size:
        raise ValueError('Checkpoint header truncated')
    result = dict(zip(KEYS, HEADER.unpack(data)))
    if result['magic'] not in [s.ljust(16,b'\0') for s in [b'HYP3D_MPI_V1',b'HYP3D_MPI_V2']] or result['nf'] != 31 or result['admitted'] != 1:
        raise ValueError('Unknown or unadmitted checkpoint schema')
    if result['n'] < 6 or result['n'] % 2 or result['cells'] <= 0 or result['surface'] <= 0:
        raise ValueError('Invalid checkpoint dimensions')
    if not math.isfinite(result['time']) or result['time'] < 0 or not math.isfinite(result['dt']) or result['dt'] <= 0:
        raise ValueError('Invalid checkpoint clock')
    expected = HEADER.size + 16*(31*result['cells']+result['surface'])
    if path.stat().st_size != expected:
        raise ValueError('Checkpoint payload truncated or contains trailing data')
    result['magic'] = result['magic'].rstrip(b'\0').decode('ascii')
    result['payload_doubles'] = (expected-HEADER.size)//8
    return result


def compare_checkpoints(reference, candidate, atol=0, rtol=0):
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol,rtol) < 0:
        raise ValueError('Checkpoint tolerances must be finite and nonnegative')
    left, right = read_checkpoint(reference), read_checkpoint(candidate)
    if left != right:
        raise ValueError('Checkpoint configuration, accepted step, or time differs')
    maximum_absolute = maximum_scaled = 0.0
    equivalent = True
    bitwise = True
    count = 0
    with reference.open('rb') as a, candidate.open('rb') as b:
        a.seek(HEADER.size); b.seek(HEADER.size)
        while True:
            av, bv = a.read(1<<20), b.read(1<<20)
            if not av:
                break
            bitwise = bitwise and av == bv
            for (x,), (y,) in zip(struct.iter_unpack('<d',av), struct.iter_unpack('<d',bv)):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError('Checkpoint contains nonfinite values')
                difference = abs(x-y)
                scale = max(abs(x),abs(y))
                maximum_absolute = max(maximum_absolute,difference)
                maximum_scaled = max(maximum_scaled,difference/(1+scale))
                equivalent = equivalent and difference <= atol+rtol*scale
                count += 1
    return dict(restart_equivalence=equivalent, bitwise_payload_equal=bitwise,
                maximum_absolute_error=maximum_absolute, maximum_scaled_error=maximum_scaled,
                atol=atol,rtol=rtol,compared_doubles=count,header=left,
                reference_sha256=control.digest(reference),candidate_sha256=control.digest(candidate))


def bound_context(args):
    root = args.root.resolve()
    # The worker verifies all source files immediately before/after this case.
    # Recheck the immutable manifest identity without repeating thousands of
    # home-filesystem reads on each of twelve component ranks.
    _, plan, binding = control.read_bound(root,args.plan,check_source=False)
    if control.digest(control.local_path(root,binding['source_manifest']))!=binding['source_manifest_sha256']:
        raise ValueError('Source manifest changed after worker verification')
    matches = [c for c in plan['commands'] if c['report']==args.output_prefix+'.report.json']
    if len(matches)!=1 or matches[0]['kind']!='check':
        raise ValueError('Adapter must correspond to one bound check command')
    command = matches[0]
    hashes = dict(binding['inputs'])
    status_path = control.local_path(root,args.plan).with_name('worker-status.json')
    status = json.loads(status_path.read_text())
    if status['plan_sha256'] != binding['plan_sha256'] or status['source_manifest_sha256'] != binding['source_manifest_sha256']:
        raise ValueError('Worker source/plan identity mismatch')
    cases = [c for c in status['cases'] if c['name']==command['name']]
    if len(cases)!=1:
        raise ValueError('Worker has not started this adapter case')
    hashes.update(cases[0].get('generated_inputs',{}))
    declared = set(command.get('inputs',[])+command.get('generated_inputs',[]))
    def checked(name):
        if name not in declared or name not in hashes:
            raise ValueError('Preflight file must be a bound or generated input: '+name)
        path = control.local_path(root,name)
        if control.digest(path)!=hashes[name]:
            raise ValueError('Preflight input changed: '+name)
        return path
    return root,binding,checked


def component(args, rank, root, checked):
    binary = checked(args.test_binary)
    cache = checked(args.cmake_cache)
    if not re.search(r'^Kokkos_ENABLE_SYCL:BOOL=ON$',cache.read_text(),re.M):
        raise ValueError('Component build cache does not enable SYCL')
    log = control.local_path(root,args.output_prefix+f'.rank-{rank:02d}.log')
    started=time.monotonic()
    with log.open('x') as stream:
        try:
            result=subprocess.run([str(binary)],cwd=root,stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=max(1,args.seconds-20))
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    content=log.read_text()
    backend=re.findall(r'^backend=(\S+)\s*$',content,re.M)
    fields = {
        'device':['scaled_error','partition_error','worst_scaled_error','diagnostic_scaled_error','partition_sum_error'],
        'exchange':['scaled_error','scri_jet_error','null_before','null_after'],
        'scri':['projection_error','rate_error','dependent_trace_error','null','pole','clock','compatibility','max_cfl']
    }[args.component]
    maxima={}
    for key in fields:
        values=[float(v) for v in re.findall(r'(?<![A-Za-z0-9_])'+key+r'=([^\s;]+)',content)]
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError('Missing/nonfinite component metric '+key)
        maxima[key]=max(values)
    marker={
        'device':'CPU/device and partition parity passed on',
        'exchange':'CPU/device halo, off-node null closure and sparse scri donor parity passed',
        'scri':'CPU/device scri projection, native rates, D independence and partition parity passed on'
    }[args.component]
    independent=(args.component!='scri' or re.findall(r'D_input_and_order_independent=(\d+)',content)==['1','1'])
    passed=code==0 and backend==['SYCL'] and marker in content and independent
    return dict(passed=passed,gpu_cpu_parity=passed,backend=backend[0] if len(backend)==1 else None,
                returncode=code,seconds=time.monotonic()-started,maxima=maxima,
                dependent_input_and_order_independent=independent if args.component=='scri' else None,
                log=str(log.relative_to(root)),log_sha256=control.digest(log),
                binary=args.test_binary,binary_sha256=control.digest(binary),cache_sha256=control.digest(cache))


def restart(args, rank, root, checked):
    if rank!=0:
        return dict(passed=True,role='rank-zero-checkpoint-comparison')
    reference=checked(args.reference)
    candidate=checked(args.candidate)
    reports=[json.loads(checked(name).read_text()) for name in [args.reference_report,args.candidate_report]]
    for report in reports:
        if report.get('status')!='completed' or report.get('mpi_size')!=12 or report.get('backend')!='SYCL':
            raise ValueError('Restart comparison needs two completed 12-rank SYCL reports')
    for key in ['n','kind','t','target_time','dt','steps']:
        if reports[0].get(key)!=reports[1].get(key) or key not in reports[0]:
            raise ValueError('Restart evolution report mismatch: '+key)
    result=compare_checkpoints(reference,candidate,args.atol,args.rtol)
    header=result['header']
    if header['kind']!={'minkowski':0,'pulse':1,'puncture':2}.get(reports[0]['kind']):
        raise ValueError('Evolution kind does not describe the checkpoint')
    for report in reports:
        scheme='final_stage_standard' if header['magic']=='HYP3D_MPI_V2' else 'none'
        if (report.get('checkpoint_schema',header['magic'])!=header['magic']
                or report.get('algebraic_projection','none')!=scheme):
            raise ValueError('Evolution report/checkpoint algebraic scheme mismatch')
        for report_key,header_key in [('n','n'),('t','time'),('dt','dt'),('steps','steps')]:
            if report[report_key]!=header[header_key]:
                raise ValueError('Evolution report does not describe the checkpoint')
    result.update(passed=result['restart_equivalence'],backend='SYCL')
    return result


def receive(peer,deadline):
    peer.settimeout(max(.001,deadline-time.monotonic()))
    data=bytearray()
    while not data.endswith(b'\n'):
        block=peer.recv(65536)
        if not block:
            raise RuntimeError('Preflight rank disconnected before a complete record')
        data.extend(block)
        if len(data)>1<<20:
            raise ValueError('Preflight record exceeds bound')
    return json.loads(data)


def send(peer,value):
    peer.sendall(json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n')


def run(args):
    if not 20<=args.seconds<=3600:
        raise ValueError('Preflight deadline must be between 20 seconds and one hour')
    rank,identity=rank_info()
    deadline=time.monotonic()+args.seconds
    root,binding,checked=bound_context(args)
    report_path=control.local_path(root,args.output_prefix+'.report.json')
    tag=hashlib.sha256((str(report_path)+os.environ.get('PBS_JOBID','offline')).encode()).hexdigest()[:24]
    address=('\0hyper-preflight-'+tag) if sys.platform.startswith('linux') else '/tmp/hyper-preflight-'+tag+'.sock'
    listener=None
    peers=[]
    try:
        if rank==0:
            if report_path.exists():
                raise ValueError('Refusing to overwrite preflight report')
            listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            listener.bind(address);listener.listen(12)
        try:
            result=component(args,rank,root,checked) if args.mode=='component' else restart(args,rank,root,checked)
        except Exception as error:
            result=dict(passed=False,error=f'{type(error).__name__}: {error}')
            if args.mode=='component':
                log=control.local_path(root,args.output_prefix+f'.rank-{rank:02d}.log')
                if log.is_file():
                    result.update(log=str(log.relative_to(root)),log_sha256=control.digest(log))
        row=dict(identity,result=result)
        if rank!=0:
            peer=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            peers.append(peer)
            while True:
                try:
                    peer.connect(address)
                    break
                except (FileNotFoundError,ConnectionRefusedError):
                    if time.monotonic()>=deadline:
                        raise TimeoutError('Rank-zero preflight socket not available')
                    time.sleep(.05)
            send(peer,row)
            acknowledgement=receive(peer,deadline)
            return 0 if acknowledgement['status']=='completed' else 1
        rows=[row]
        while len(rows)<12:
            listener.settimeout(max(.001,deadline-time.monotonic()))
            peer,_=listener.accept();peers.append(peer)
            rows.append(receive(peer,deadline))
        rows.sort(key=lambda r:r['rank'])
        placement=({r['rank'] for r in rows}==set(range(12)) and len({r['host'] for r in rows})==1
                   and all(r['hierarchy']=='COMPOSITE' and r['mask']==f"{r['rank']//2}.{r['rank']%2}" for r in rows))
        passed=placement and all(r['result']['passed'] for r in rows)
        report=dict(status='completed' if passed else 'failed',mode=args.mode,mpi_size=12,
                    backend='SYCL' if passed else None,placement_passed=placement,ranks=rows,
                    source_manifest_sha256=binding['source_manifest_sha256'],plan_sha256=binding['plan_sha256'])
        if args.mode=='component':
            report['gpu_cpu_parity']=passed
            report['component']=args.component
            report['maxima']={key:max(r['result'].get('maxima',{}).get(key,0) for r in rows)
                              for key in set().union(*(r['result'].get('maxima',{}).keys() for r in rows))}
        else:
            report.update({k:v for k,v in rows[0]['result'].items() if k not in ['passed','backend']})
            report['restart_equivalence']=passed and rows[0]['result'].get('restart_equivalence',False)
        control.atomic(report_path,report)
        for peer in peers:
            send(peer,dict(status=report['status']))
        return 0 if passed else 1
    except Exception as error:
        if rank==0 and not report_path.exists():
            control.atomic(report_path,dict(status='failed',mpi_size=12,backend=None,
                error=f'{type(error).__name__}: {error}',source_manifest_sha256=binding['source_manifest_sha256']))
        raise
    finally:
        for peer in peers:
            peer.close()
        if listener is not None:
            listener.close()
            if not sys.platform.startswith('linux'):
                Path(address).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--seconds',type=int,required=True)
    sub=parser.add_subparsers(dest='mode',required=True)
    comp=sub.add_parser('component')
    comp.add_argument('--component',choices=['device','exchange','scri'],required=True)
    comp.add_argument('--test-binary',required=True)
    comp.add_argument('--cmake-cache',required=True)
    rest=sub.add_parser('restart')
    for name in ['reference','candidate','reference-report','candidate-report']:
        rest.add_argument('--'+name,required=True)
    rest.add_argument('--atol',type=float,default=0)
    rest.add_argument('--rtol',type=float,default=0)
    raise SystemExit(run(parser.parse_args()))


if __name__=='__main__':
    main()
