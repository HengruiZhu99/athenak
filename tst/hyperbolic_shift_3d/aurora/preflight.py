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
V4_CONFIG = struct.Struct('<dii')
V5_NORMAL_CONFIG = struct.Struct('<4i')
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
        extra = stream.read(V4_CONFIG.size) if data[:16] in [s.ljust(16,b'\0') for s in [b'HYP3D_MPI_V4',b'HYP3D_MPI_V5',b'HYP3D_MPI_V6',b'HYP3D_MPI_V7',b'HYP3D_MPI_V8',b'HYP3D_MPI_V9',b'HYP3D_MPI_V10',b'HYP3D_MPI_V11']] else b''
        normal = stream.read(V5_NORMAL_CONFIG.size) if data[:16] in [b'HYP3D_MPI_V5'.ljust(16,b'\0'),b'HYP3D_MPI_V6'.ljust(16,b'\0'),b'HYP3D_MPI_V7'.ljust(16,b'\0'),b'HYP3D_MPI_V8'.ljust(16,b'\0'),b'HYP3D_MPI_V9'.ljust(16,b'\0'),b'HYP3D_MPI_V10'.ljust(16,b'\0'),b'HYP3D_MPI_V11'.ljust(16,b'\0')] else b''
        mesh = stream.read(16) if data[:16] in [b'HYP3D_MPI_V8'.ljust(16,b'\0'),b'HYP3D_MPI_V9'.ljust(16,b'\0'),b'HYP3D_MPI_V10'.ljust(16,b'\0'),b'HYP3D_MPI_V11'.ljust(16,b'\0')] else b''
        geometry = stream.read(16) if data[:16] in [b'HYP3D_MPI_V10'.ljust(16,b'\0'),b'HYP3D_MPI_V11'.ljust(16,b'\0')] else b''
        positive = stream.read(16) if data[:16]==b'HYP3D_MPI_V11'.ljust(16,b'\0') else b''
    if len(data) != HEADER.size:
        raise ValueError('Checkpoint header truncated')
    result = dict(zip(KEYS, HEADER.unpack(data)))
    if result['magic'] not in [s.ljust(16,b'\0') for s in [b'HYP3D_MPI_V1',b'HYP3D_MPI_V2',b'HYP3D_MPI_V3',b'HYP3D_MPI_V4',b'HYP3D_MPI_V5',b'HYP3D_MPI_V6',b'HYP3D_MPI_V7',b'HYP3D_MPI_V8',b'HYP3D_MPI_V9',b'HYP3D_MPI_V10',b'HYP3D_MPI_V11']] or result['nf'] != 31 or result['admitted'] != 1:
        raise ValueError('Unknown or unadmitted checkpoint schema')
    result.update(header_bytes=HEADER.size,scri_dissipation_epsilon=0.0,scri_dissipation_degree=0)
    if result['magic'] in [s.ljust(16,b'\0') for s in [b'HYP3D_MPI_V4',b'HYP3D_MPI_V5',b'HYP3D_MPI_V6',b'HYP3D_MPI_V7',b'HYP3D_MPI_V8',b'HYP3D_MPI_V9',b'HYP3D_MPI_V10',b'HYP3D_MPI_V11']]:
        if len(extra) != V4_CONFIG.size:
            raise ValueError('V4 checkpoint configuration truncated')
        epsilon,degree,reserved = V4_CONFIG.unpack(extra)
        if not math.isfinite(epsilon) or epsilon < 0 or degree != 7 or reserved != 0:
            raise ValueError('Invalid V4 checkpoint dissipation configuration')
        result.update(header_bytes=HEADER.size+V4_CONFIG.size,
                      scri_dissipation_epsilon=epsilon,scri_dissipation_degree=degree)
    if result['magic'] in [b'HYP3D_MPI_V5'.ljust(16,b'\0'),b'HYP3D_MPI_V6'.ljust(16,b'\0'),b'HYP3D_MPI_V7'.ljust(16,b'\0'),b'HYP3D_MPI_V8'.ljust(16,b'\0'),b'HYP3D_MPI_V9'.ljust(16,b'\0'),b'HYP3D_MPI_V10'.ljust(16,b'\0'),b'HYP3D_MPI_V11'.ljust(16,b'\0')]:
        inner=result['magic'] in [b'HYP3D_MPI_V7'.ljust(16,b'\0'),b'HYP3D_MPI_V8'.ljust(16,b'\0'),b'HYP3D_MPI_V9'.ljust(16,b'\0'),b'HYP3D_MPI_V10'.ljust(16,b'\0'),b'HYP3D_MPI_V11'.ljust(16,b'\0')]
        radial=inner or result['magic']==b'HYP3D_MPI_V6'.ljust(16,b'\0')
        if len(normal) != V5_NORMAL_CONFIG.size:
            raise ValueError('V5 checkpoint normal configuration truncated')
        if V5_NORMAL_CONFIG.unpack(normal) != ((3,7,4,3) if inner else (2,7,4,3) if radial else (1,6,4,0)):
            raise ValueError('Invalid V5 checkpoint normal derivative configuration')
        result.update(header_bytes=104,scri_normal_derivative='sampled_native_numerator7' if radial else control.SAMPLED_NORMAL_SCHEME,
                      scri_normal_radial_nodes=7 if radial else 6,scri_normal_bulk_derivative_order=4)
        if radial:result.update(scri_radial_closure='donor_seven_last3',scri_radial_bulk_rows=3)
        if inner:result.update(interpatch_interpolation='logical_inner_transition')
    if result['magic']==b'HYP3D_MPI_V8'.ljust(16,b'\0'):
        if len(mesh)!=16 or struct.unpack('<4i',mesh)!=(1,2,3,0):raise ValueError('Invalid V8 fixed-core topology')
        result.update(header_bytes=120,mesh_topology='cartesian_2x2x2_shells3',cartesian_core_subdivisions=2,mesh_blocks=26)
        if result['cells']!=26*result['n']**3 or result['surface']!=31*6*result['n']**2:raise ValueError('V8 dimensions disagree with topology')
    if result['magic']==b'HYP3D_MPI_V9'.ljust(16,b'\0'):
        if len(mesh)!=16 or struct.unpack('<4i',mesh)!=(2,1,3,3):raise ValueError('Invalid V9 evolved-overlap topology')
        n=result['n']
        result.update(header_bytes=120,evolved_overlap=3,nominal_cells=19*n**3)
        if n<10 or result['cells']!=12*n**3+7*(n+6)**3 or result['surface']!=31*6*n*n:raise ValueError('V9 dimensions disagree with topology')
    if result['magic'] in [b'HYP3D_MPI_V10'.ljust(16,b'\0'),b'HYP3D_MPI_V11'.ljust(16,b'\0')]:
        continuous=bool(positive)
        if len(mesh)!=16 or struct.unpack('<4i',mesh)!=((4,2,3,3) if continuous else (3,2,3,3)):raise ValueError('Invalid V10 spherical-overlap topology')
        if len(geometry)!=16 or struct.unpack('<2d',geometry)!=(.18,.16):raise ValueError('Invalid V10 spherical geometry')
        n=result['n']
        result.update(header_bytes=152 if continuous else 136,evolved_overlap=3,nominal_cells=26*n**3,
                      mesh_topology='cartesian_2x2x2_shells3',cartesian_core_subdivisions=2,mesh_blocks=26,
                      shell_geometry='spherical_equiangular',core_half_width=.18,shell_inner_radius=.16,overlap_quadrature='radial_quintic_partition')
        if n<12 or result['cells']!=12*n**3+8*(n+(3 if continuous else 6))**3+6*(n+3)*(n+6)**2 or result['surface']!=31*6*n*n:raise ValueError('V10 dimensions disagree with topology')
    if result['magic']==b'HYP3D_MPI_V11'.ljust(16,b'\0'):
        if len(positive)!=16 or struct.unpack('<dii',positive)!=(1e-10,1,0):raise ValueError('Invalid V11 positive-variable configuration')
        result.update(cartesian_seams='uniform_global_stencil',positive_variables='log_chi_log_lapse_rk_and_ko',physical_lapse_floor=1e-10)
    if result['n'] < 6 or result['n'] % 2 or result['cells'] <= 0 or result['surface'] <= 0:
        raise ValueError('Invalid checkpoint dimensions')
    if not math.isfinite(result['time']) or result['time'] < 0 or not math.isfinite(result['dt']) or result['dt'] <= 0:
        raise ValueError('Invalid checkpoint clock')
    expected = result['header_bytes'] + 16*(31*result['cells']+result['surface'])
    if path.stat().st_size != expected:
        raise ValueError('Checkpoint payload truncated or contains trailing data')
    result['magic'] = result['magic'].rstrip(b'\0').decode('ascii')
    result['payload_doubles'] = (expected-result['header_bytes'])//8
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
        a.seek(left['header_bytes']); b.seek(right['header_bytes'])
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


def run_binary(args, rank, root, checked, argv):
    binary = checked(args.test_binary)
    cache = checked(args.cmake_cache)
    if not re.search(r'^Kokkos_ENABLE_SYCL:BOOL=ON$',cache.read_text(),re.M):
        raise ValueError('Component build cache does not enable SYCL')
    log = control.local_path(root,args.output_prefix+f'.rank-{rank:02d}.log')
    started=time.monotonic()
    with log.open('x') as stream:
        try:
            result=subprocess.run([str(binary),*argv],cwd=root,stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=max(1,args.seconds-20))
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    return log.read_text(),dict(returncode=code,seconds=time.monotonic()-started,
                log=str(log.relative_to(root)),log_sha256=control.digest(log),
                binary=args.test_binary,binary_sha256=control.digest(binary),cache_sha256=control.digest(cache),
                binary_args=argv)


def parse_radial_field_batch_component(content,code):
    rows=[json.loads(line) for line in content.splitlines() if line.startswith('{')]
    if len(rows)!=1:raise ValueError('Need one radial field-batch test record')
    result=rows[0]
    for key,value in [('passed',True),('backend','SYCL'),('stage_refreshes_per_plan',3),
                      ('exact_direct_equality',True),('simulated_partitions',12),
                      ('sparse_and_empty_ownership',True),('radial_ghost_poison',True),
                      ('invalid_inputs_rejected',True),('tested_field_rows',110019)]:
        if type(result.get(key)) is not type(value) or result[key]!=value:
            raise ValueError('Missing/invalid radial batch coverage: '+key)
    error=result.get('max_scaled_error')
    if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=2e-12:
        raise ValueError('Radial field-batch CPU/direct parity failed')
    return dict(passed=code==0,gpu_cpu_parity=code==0,backend='SYCL',radial_field_batch_checks=result)


def parse_component(content, kind, code):
    if kind=='radial_field_batch':
        return parse_radial_field_batch_component(content,code)
    if kind=='sampled_normal_device':
        return parse_sampled_normal_component(content,code)
    if kind in ['inner_transition_device','core2_device','overlap_device','spherical_overlap_device','continuous_positive_device']:
        result=parse_donor_seven_component(content,code,expected_n=12 if kind in ['overlap_device','spherical_overlap_device','continuous_positive_device'] else 6)
        rows=[line for line in content.splitlines() if line.startswith('sampled_case=')]
        if len(rows)!=2 or any('interpatch_interpolation=logical_inner_transition' not in line for line in rows):
            raise ValueError('Missing inner-transition component configuration')
        exchange=[]
        for line in content.splitlines():
            if line.startswith('inner_transition_exchange_error='):
                row=dict(token.split('=',1) for token in line.split())
                values=[float(row[k]) for k in ['inner_transition_exchange_error','bulk_error']]
                if any(not math.isfinite(v) or v<0 or v>2e-10 for v in values):
                    raise ValueError('Inner transition full exchange/bulk parity failed')
                exchange.append(values)
        if len(exchange)!=2:raise ValueError('Missing inner transition full exchange checks')
        result['inner_transition_exchange_checks']=exchange
        result['interpatch_interpolation']='logical_inner_transition'
        if kind in ['overlap_device','spherical_overlap_device','continuous_positive_device']:
            for line in rows:
                tokens=line.split()
                if tokens.count('overlap=3')!=1 or tokens.count('n=12')!=1:
                    raise ValueError('Missing evolved-overlap component configuration')
            result.update(evolved_overlap=3)
        if kind in ['spherical_overlap_device','continuous_positive_device']:
            for line in rows:
                if line.split().count('shell_geometry=spherical_equiangular')!=1:raise ValueError('Missing spherical geometry marker')
            result.update(shell_geometry='spherical_equiangular',core_half_width=.18,shell_inner_radius=.16)
        if kind=='continuous_positive_device':
            for line in rows:
                for marker in ['positive_variables=log_chi_log_lapse_rk_and_ko','cartesian_seams=uniform_global_stencil','selective_exchange_partitions=12']:
                    if line.split().count(marker)!=1:raise ValueError('Missing continuous positive component marker')
            result.update(positive_variables='log_chi_log_lapse_rk_and_ko',cartesian_seams='uniform_global_stencil',selective_exchange_partitions=12)
        if kind in ['core2_device','spherical_overlap_device','continuous_positive_device']:
            for line in rows:
                tokens=line.split()
                for marker in ['cartesian_core_subdivisions=2','mesh_blocks=26']:
                    if tokens.count(marker)!=1:
                        raise ValueError('Missing or invalid fixed-core component topology')
            result.update(mesh_topology='cartesian_2x2x2_shells3',cartesian_core_subdivisions=2,mesh_blocks=26)
        return result
    if kind=='donor_seven_device':
        return parse_donor_seven_component(content,code)
    if kind in ['angular_dissipation','scri_dissipation_closure']:
        return parse_dissipation_component(content,kind,code)
    backend=re.findall(r'^backend=(\S+)\s*$',content,re.M)
    fields = {
        'device':['scaled_error','partition_error','worst_scaled_error','diagnostic_scaled_error','partition_sum_error'],
        'exchange':['scaled_error','scri_jet_error','null_before','null_after'],
        'scri':['projection_error','rate_error','dependent_trace_error','null','pole','clock','compatibility','max_cfl'],
        'algebraic':['rotation_covariance_max','scri_null_before_reclosure','after','ghost',
                     'D_compatibility','null_time_derivative'],
        'angular':['projection_error','rate_error','dependent_trace_error','null','compatibility',
                   'fine_gradient_order','fine_hessian_order','fine_mixed_order']
    }[kind]
    maxima={}
    for key in fields:
        values=[float(v) for v in re.findall(r'(?<![A-Za-z0-9_])'+key+r'=([^\s;]+)',content)]
        if not values or any(not math.isfinite(v) for v in values):
            raise ValueError('Missing/nonfinite component metric '+key)
        maxima[key]=max(values)
    marker={
        'device':'CPU/device and partition parity passed on',
        'exchange':'CPU/device halo, off-node null closure and sparse scri donor parity passed',
        'scri':'CPU/device scri projection, native rates, D independence and partition parity passed on',
        'algebraic':'algebraic projection: tensors, rejection, device, unchanged fields, compensation, scri passed',
        'angular':'Cross-face scri constants, active donors, nonspherical jets and native device parity passed'
    }[kind]
    independent=(kind!='scri' or re.findall(r'D_input_and_order_independent=(\d+)',content)==['1','1'])
    passed=code==0 and backend==['SYCL'] and marker in content and independent
    return dict(passed=passed,gpu_cpu_parity=passed,backend=backend[0] if len(backend)==1 else None,
                maxima=maxima,dependent_input_and_order_independent=independent if kind=='scri' else None)


def parse_dissipation_component(content,kind,code):
    """Check the explicit standalone V4 test contract on every GPU rank."""
    backend=re.findall(r'^backend=(\S+)\s*$',content,re.M)
    maxima={}
    def rows(prefix):
        parsed=[]
        for line in content.splitlines():
            if not line.startswith(prefix):continue
            tokens=line.split()
            if any(token.count('=')!=1 for token in tokens):
                raise ValueError('Malformed dissipation component metric row')
            row=dict(token.split('=') for token in tokens)
            if len(row)!=len(tokens):raise ValueError('Duplicate dissipation component metric')
            parsed.append(row)
        return parsed
    def metric(row,key,upper=None,positive=False):
        try:value=float(row[key])
        except (KeyError,ValueError):raise ValueError('Missing/invalid dissipation metric '+key)
        if not math.isfinite(value) or value<0 or (positive and value<=0) or (upper is not None and value>=upper):
            raise ValueError('Dissipation metric out of bounds: '+key)
        maxima[key]=max(maxima.get(key,0),value)
        return value
    def integer(row,key):
        if not re.fullmatch(r'[0-9]+',row.get(key,'')):
            raise ValueError('Missing/invalid dissipation integer '+key)
        return int(row[key])
    def flags(row,names):
        if any(row.get(key)!='1' for key in names):
            raise ValueError('Dissipation component rejection/parity checks did not pass')
    if kind=='angular_dissipation':
        marker='Angular dissipation CPU/device, energy, constants, partitions and reference parity passed'
        records=rows('n=');identities=[]
        for row in records:
            n,degree=integer(row,'n'),integer(row,'degree');identities.append((n,degree))
            points,neighbors=integer(row,'points'),integer(row,'max_neighbors')
            if points!=6*n*n or not 0<neighbors<=points:
                raise ValueError('Angular dissipation shape/neighborhood mismatch')
            metric(row,'epsilon',positive=True)
            for key in ['coefficient_reference_error','source_reference_error','device_residual_error',
                        'device_source_error','partition_error']:metric(row,key,2e-11)
            for key in ['energy_error','adjoint_error']:metric(row,key,2e-12)
            if metric(row,'constant_error')!=0:raise ValueError('Angular dissipation must annihilate constants exactly')
            metric(row,'max_moment_error',1e-11);metric(row,'max_condition_bound',1e8,positive=True)
            if metric(row,'max_coefficient_l1')>math.sqrt(neighbors)*(1+1e-10):
                raise ValueError('Angular dissipation coefficient bound exceeded')
            metric(row,'spectral_upper_bound',positive=True)
        if len(identities)!=3 or set(identities)!={(6,7),(8,7),(6,5)}:
            raise ValueError('Angular dissipation requires n6/n8 degree7 and n6 degree5 reference cases')
    else:
        marker='Optional additive scri closure checks passed'
        records=rows('closure_case=');identities=[]
        for row in records:
            identities.append(row.get('closure_case'))
            metric(row,'additive_rate_error',2e-13);metric(row,'dependent_D_response_error',3e-12)
            metric(row,'compatibility',2e-12)
            for key in ['cpu_rate_error','cpu_trace_error']:metric(row,key,2e-10)
            flags(row,['partition_parity','zero_off','rejection_checks'])
        if len(identities)!=2 or set(identities)!={'synthetic','plan'}:
            raise ValueError('Scri closure requires synthetic and actual-plan cases')
        plan_rows=rows('plan_perturb=');identities=[]
        for row in plan_rows:
            identities.append(integer(row,'plan_perturb'))
            metric(row,'host_rate_error',2e-13);metric(row,'null',3e-14);metric(row,'compatibility',2e-12)
            if metric(row,'angular_cfl',positive=True)>2.7:
                raise ValueError('Scri dissipation CFL exceeds the tested bound')
        if len(identities)!=2 or set(identities)!={0,1}:
            raise ValueError('Scri closure requires equilibrium and perturbed actual-plan cases')
        final=rows('cfl_rejection=')
        if len(final)!=1:raise ValueError('Missing/duplicate scri closure validation flags')
        flags(final[0],['cfl_rejection','cross_face_guard','epsilon_validation','shared_plan'])
        records+=plan_rows+final
    passed=code==0 and backend==['SYCL'] and content.splitlines().count(marker)==1
    return dict(passed=passed,gpu_cpu_parity=passed,backend=backend[0] if len(backend)==1 else None,
                maxima=maxima,cases=records)


def parse_sampled_normal_component(content,code,expected_n=6):
    """Validate the fixed V5 normal-closure test contract on one GPU tile."""
    backend=re.findall(r'^backend=(\S+)\s*$',content,re.M)
    marker='Sampled-normal device, ownership, default and closure parity passed'
    maxima={};records=[]
    bounds=dict(cpu_pole_error=2e-10,cpu_rate_error=2e-10,cpu_trace_error=2e-10,
                partition_error=2e-13,finite_D_error=0,null=3e-14,compatibility=2e-12)
    for line in content.splitlines():
        if not line.startswith('sampled_case='):continue
        tokens=line.split()
        if any(token.count('=')!=1 for token in tokens):
            raise ValueError('Malformed sampled-normal component row')
        row=dict(token.split('=') for token in tokens)
        if len(row)!=len(tokens):raise ValueError('Duplicate sampled-normal component metric')
        if any(row.get(key)!=value for key,value in
               [('n',str(expected_n)),('default_bitwise','1'),('partitions','12'),('rejection_checks','1')]):
            raise ValueError('Sampled-normal shape, default, partition or rejection check failed')
        try:epsilon=float(row['epsilon'])
        except (KeyError,ValueError):raise ValueError('Missing/invalid sampled-normal epsilon')
        if not math.isfinite(epsilon) or epsilon!=2:
            raise ValueError('Sampled-normal component must test epsilon 2')
        for key,upper in bounds.items():
            try:value=float(row[key])
            except (KeyError,ValueError):raise ValueError('Missing/invalid sampled-normal metric '+key)
            if not math.isfinite(value) or value<0 or value>upper:
                raise ValueError('Sampled-normal metric out of bounds: '+key)
            maxima[key]=max(maxima.get(key,0),value)
        records.append(row)
    if len(records)!=2 or {r.get('sampled_case') for r in records}!={'minkowski','nonspherical'}:
        raise ValueError('Sampled-normal component requires equilibrium and nonspherical cases')
    passed=code==0 and backend==['SYCL'] and content.splitlines().count(marker)==1
    return dict(passed=passed,gpu_cpu_parity=passed,backend=backend[0] if len(backend)==1 else None,
                maxima=maxima,cases=records)


def parse_donor_seven_component(content,code,expected_n=6):
    """V6 must independently establish complete bulk and surface parity."""
    marker='Donor-seven device, ownership, default and closure parity passed'
    if content.splitlines().count(marker)!=1:
        raise ValueError('Missing or duplicate donor-seven component marker')
    old='Sampled-normal device, ownership, default and closure parity passed'
    result=parse_sampled_normal_component(content.replace(marker,old),code,expected_n)
    for row in result['cases']:
        if row.get('radial_nodes')!='7' or row.get('radial_bulk_rows')!='3':
            raise ValueError('Donor-seven radial configuration mismatch')
    bulk=[]
    for line in content.splitlines():
        if not line.startswith('donor_seven_bulk_error='):continue
        tokens=line.split();row=dict(token.split('=') for token in tokens)
        if len(row)!=len(tokens) or row.get('exterior_radial_poison')!='1':
            raise ValueError('Invalid donor-seven bulk/ghost record')
        value=float(row['donor_seven_bulk_error'])
        if not math.isfinite(value) or value<0 or value>2e-10:
            raise ValueError('Donor-seven bulk parity exceeds tolerance')
        bulk.append(value)
    if len(bulk)!=2:raise ValueError('Donor-seven component needs two bulk fixtures')
    result['maxima']['cpu_bulk_error']=max(bulk)
    result.update(scri_radial_closure='donor_seven_last3',scri_radial_bulk_rows=3)
    return result


def component(args, rank, root, checked):
    content,execution=run_binary(args,rank,root,checked,[])
    return dict(execution,**parse_component(content,args.component,execution['returncode']))


def parse_benchmark(content, rank, sampling, code):
    backend=re.findall(r'^backend=(\S+)\s*$',content,re.M)
    variants=re.findall(r'^sampling=(\S+)\s*$',content,re.M)
    records=re.findall(r'^benchmark_repeats=.*$',content,re.M)
    if len(records)!=1:
        raise ValueError('Benchmark needs exactly one metric record')
    tokens=records[0].split()
    if any(token.count('=')!=1 for token in tokens):
        raise ValueError('Malformed benchmark metric record')
    fields=dict(token.split('=') for token in tokens)
    if len(fields)!=len(tokens):
        raise ValueError('Duplicate benchmark metric')
    integers=['benchmark_repeats','n','rank','ranks','first_row','owned_rows','sampled_fields','staged_jet_bytes']
    for key in integers:
        if not re.fullmatch(r'[0-9]+',fields.get(key,'')):
            raise ValueError('Missing/invalid benchmark integer '+key)
        fields[key]=int(fields[key])
    expected=dict(benchmark_repeats=20,n=16,rank=rank,ranks=12,first_row=128*rank,
                  owned_rows=128,sampled_fields=25)
    if any(fields[key]!=value for key,value in expected.items()):
        raise ValueError('Benchmark does not cover its owned 128-row partition')
    if fields.get('rank_source') not in ['PMI_RANK','OMPI_COMM_WORLD_RANK','MPI_LOCALRANKID','PALS_LOCAL_RANKID'] \
            or fields.get('size_source')!='argument':
        raise ValueError('Benchmark needs launcher rank identity and explicit 12-rank size')
    if ((sampling=='inline' and fields['staged_jet_bytes']!=0)
            or (sampling=='staged' and fields['staged_jet_bytes']<=0)):
        raise ValueError('Benchmark sample storage disagrees with its variant')
    for key in ['evaluate_seconds_total','evaluate_seconds_mean','rhs_norm2','surface_norm2','repeat_error']:
        try:
            fields[key]=float(fields[key])
        except (KeyError,ValueError):
            raise ValueError('Missing/invalid benchmark metric '+key)
        if not math.isfinite(fields[key]) or fields[key]<0:
            raise ValueError('Nonfinite/negative benchmark metric '+key)
    if min(fields['evaluate_seconds_total'],fields['evaluate_seconds_mean'])<=0 \
            or not math.isclose(fields['evaluate_seconds_mean'],fields['evaluate_seconds_total']/20,rel_tol=1e-12):
        raise ValueError('Benchmark timing is nonpositive or inconsistent')
    if fields['repeat_error']!=0:
        raise ValueError('Repeated benchmark evaluation changed outputs')
    passed=(code==0 and backend==['SYCL'] and variants==[sampling]
            and content.splitlines().count('Scri repeated-Evaluate benchmark completed')==1)
    return dict(passed=passed,benchmark_passed=passed,backend=backend[0] if len(backend)==1 else None,
                sampling=variants[0] if len(variants)==1 else None,metrics=fields)


def benchmark(args, rank, root, checked):
    argv=['--benchmark-repeats','20','--benchmark-n','16','--benchmark-ranks','12']
    content,execution=run_binary(args,rank,root,checked,argv)
    return dict(execution,**parse_benchmark(content,rank,args.sampling,execution['returncode']))


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
        scheme='none' if header['magic']=='HYP3D_MPI_V1' else 'final_stage_standard'
        angular='cross_face_centered5' if header['magic'] in ['HYP3D_MPI_V3','HYP3D_MPI_V4','HYP3D_MPI_V5','HYP3D_MPI_V6','HYP3D_MPI_V7','HYP3D_MPI_V8','HYP3D_MPI_V9','HYP3D_MPI_V10','HYP3D_MPI_V11'] else 'face_local6'
        if (report.get('checkpoint_schema',header['magic'])!=header['magic']
                or report.get('algebraic_projection','none')!=scheme
                or report.get('scri_angular_stencil','face_local6')!=angular):
            raise ValueError('Evolution report/checkpoint algebraic scheme mismatch')
        for key in ['scri_dissipation_epsilon','scri_dissipation_degree','checkpoint_header_bytes']:
            header_key='header_bytes' if key=='checkpoint_header_bytes' else key
            expected=header[header_key]
            # Legacy reports may predate these fields. V4 must state its exact
            # runtime configuration; a V3 or missing-config report cannot pass.
            value=report.get(key) if header['magic'] in ['HYP3D_MPI_V4','HYP3D_MPI_V5','HYP3D_MPI_V6','HYP3D_MPI_V7','HYP3D_MPI_V8','HYP3D_MPI_V9','HYP3D_MPI_V10','HYP3D_MPI_V11'] else report.get(key,expected)
            if type(value) not in [int,float] or not math.isfinite(value) or value!=expected:
                raise ValueError('Evolution report/checkpoint dissipation configuration mismatch: '+key)
        if header['magic'] in ['HYP3D_MPI_V5','HYP3D_MPI_V6','HYP3D_MPI_V7','HYP3D_MPI_V8','HYP3D_MPI_V9','HYP3D_MPI_V10','HYP3D_MPI_V11']:
            if report.get('checkpoint_schema')!=header['magic']:
                raise ValueError('V5 evolution report must state its checkpoint schema')
            for key in ['scri_normal_derivative','scri_normal_radial_nodes','scri_normal_bulk_derivative_order'] + (['scri_radial_closure','scri_radial_bulk_rows'] if header['magic'] in ['HYP3D_MPI_V6','HYP3D_MPI_V7','HYP3D_MPI_V8','HYP3D_MPI_V9','HYP3D_MPI_V10','HYP3D_MPI_V11'] else []) + (['interpatch_interpolation'] if header['magic'] in ['HYP3D_MPI_V7','HYP3D_MPI_V8','HYP3D_MPI_V9','HYP3D_MPI_V10','HYP3D_MPI_V11'] else []) + (['mesh_topology','cartesian_core_subdivisions','mesh_blocks'] if header['magic'] in ['HYP3D_MPI_V8','HYP3D_MPI_V10','HYP3D_MPI_V11'] else []) + (['evolved_overlap','nominal_cells'] if header['magic'] in ['HYP3D_MPI_V9','HYP3D_MPI_V10','HYP3D_MPI_V11'] else []) + (['shell_geometry','core_half_width','shell_inner_radius','overlap_quadrature'] if header['magic'] in ['HYP3D_MPI_V10','HYP3D_MPI_V11'] else []) + (['cartesian_seams','positive_variables','physical_lapse_floor'] if header['magic']=='HYP3D_MPI_V11' else []):
                if type(report.get(key)) is not type(header[key]) or report[key]!=header[key]:
                    raise ValueError('Evolution report/checkpoint normal configuration mismatch: '+key)
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
            operation={'component':component,'benchmark':benchmark,'restart':restart}[args.mode]
            result=operation(args,rank,root,checked)
        except Exception as error:
            result=dict(passed=False,error=f'{type(error).__name__}: {error}')
            if args.mode in ['component','benchmark']:
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
        elif args.mode=='benchmark':
            report.update(benchmark_passed=passed,sampling=args.sampling,
                          timing_scope='per-rank repeated Evaluate; no inter-rank start barrier',
                          benchmark_config=dict(n=16,ranks=12,repeats=20,owned_rows_per_rank=128,
                                                total_rows=1536,sampled_fields=25))
            metrics=[r['result'].get('metrics') for r in rows]
            if all(m is not None for m in metrics):
                report['timing']={key+suffix:operation([m[key] for m in metrics])
                    for key in ['evaluate_seconds_total','evaluate_seconds_mean']
                    for suffix,operation in [('_min',min),('_max',max),('_rank_mean',lambda v:math.fsum(v)/12)]}
                report['norms']={key:math.fsum(m[key] for m in metrics) for key in ['rhs_norm2','surface_norm2']}
                report['repeat_error']=max(m['repeat_error'] for m in metrics)
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
    comp.add_argument('--component',choices=['device','exchange','scri','algebraic','angular',
                                            'angular_dissipation','scri_dissipation_closure',
                                            'sampled_normal_device','donor_seven_device','inner_transition_device','core2_device','overlap_device','spherical_overlap_device','continuous_positive_device','radial_field_batch'],required=True)
    comp.add_argument('--test-binary',required=True)
    comp.add_argument('--cmake-cache',required=True)
    bench=sub.add_parser('benchmark')
    bench.add_argument('--sampling',choices=['staged','inline'],required=True)
    bench.add_argument('--test-binary',required=True)
    bench.add_argument('--cmake-cache',required=True)
    rest=sub.add_parser('restart')
    for name in ['reference','candidate','reference-report','candidate-report']:
        rest.add_argument('--'+name,required=True)
    rest.add_argument('--atol',type=float,default=0)
    rest.add_argument('--rtol',type=float,default=0)
    raise SystemExit(run(parser.parse_args()))


if __name__=='__main__':
    main()
