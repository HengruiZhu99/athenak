#!/usr/bin/env python3
"""Guards for the future bounded3551 timing qualification; never submit jobs.

validation/cycle-timing-run/admission.json must have schema
native-cycle-timing-admission-v1 and three {path, sha256} entries named
checkpoint, science_admission and independent_review. Relative proof paths
resolve beside that descriptor. The actual checkpoint must remain on Aurora.
Only a completed, independently admitted3551 trajectory can satisfy the guard.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
BASELINE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
TIMING = 'e47888006e1221d5e14bce734a97d77a45126b271f7ef39c2ceb9e9b31c0f6cf'
PROVENANCE = '89265de52cfec416139ddf4188618b10b91b4e130c3fa844ffeed473d0d492ea'
SOURCES = '4706aae696919253201da4cac9dd14bff52743ee8826dc93660a382f255ec9a5'
OBJECTS = '462da52c2e152e9c94818a534314aec72f2a827f30decdb2d23a3559e6954547'
BUILDER = '944b05be76c1bfe208a0c409a7952d7119112398e306e842c458a6ffff19cfdb'
WRAPPER = '4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd'
DT = .00046944203735617445
STEPS = [0,356,711,1066,1421,1776,2131,2486,2841,3196,3551]
PROFILE = ('KOKKOS_TOOLS_LIBS','KOKKOS_PROFILE_LIBRARY','KOKKOS_TOOLS_ARGS',
           'SYCL_PI_TRACE','SYCL_UR_TRACE','ZE_ENABLE_TRACING_LAYER',
           'UNITRACE_ENABLE','PTI_ENABLE','VTUNE_PROFILER_ENABLE')
ENVIRONMENT = dict(OMP_NUM_THREADS='1', ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
    ZE_ENABLE_PCI_ID_DEVICE_ORDER='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
    MPIR_CVAR_ENABLE_GPU='1', NATIVE_TIMING_WARMUP='10', NATIVE_TIMING_STEPS='40',
    NATIVE_TIMING_WINDOWS='3')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def expected_time(cycle):
    time = 0.
    for _ in range(cycle):
        time += DT
    return time


def checkpoint_metadata(path):
    """Read native metadata with stdlib; field correctness comes from admission."""
    with Path(path).open('rb') as stream:
        text = bytearray()
        while True:
            line = stream.readline(262144)
            text.extend(line)
            require(line and len(text) < 8*1024*1024, 'invalid parameter header')
            if line == b'<par_end>\n':
                break
        size = struct.calcsize('<ii9d38i2di')
        fixed = stream.read(size)
        require(len(fixed) == size, 'truncated fixed header')
        blocks, level = struct.unpack_from('<ii', fixed)
        time, dt, cycle = struct.unpack_from('<ddi', fixed, size-20)
        require((blocks,level,cycle,time,dt) == (106,4,3551,expected_time(3551),DT),
                'checkpoint is not the fixed N12 cycle3551 state')
        atlas_size, = struct.unpack('<Q', stream.read(8))
        require(0 < atlas_size < 1024*1024, 'invalid atlas identity length')
        require(len(stream.read(atlas_size)) == atlas_size, 'truncated atlas')
        require(len(stream.read(blocks*20)) == blocks*20, 'truncated topology')
        costs = struct.unpack('<106f', stream.read(106*4))
        require(all(x == 1. for x in costs), 'block weights changed')
        meta = struct.unpack('<8Q', stream.read(64))
        require(meta[:2] == (0x4859504552525354,1) and 0 < meta[2] < 16384,
                'invalid hyperboloidal restart metadata')
        require(meta[3:] == ((18**3+144)*39,24,0,106//24,0),
                'unexpected shared-checkpoint layout or writer ranks')
        require(len(stream.read(meta[2])) == meta[2], 'truncated physics identity')
        stride, = struct.unpack('<Q', stream.read(8))
        require(stride == meta[3]*8 and Path(path).stat().st_size == stream.tell()+106*stride,
                'truncated or extended checkpoint payload')
    params, block = {}, None
    for line in text.decode().splitlines():
        line = line.split('#',1)[0].strip()
        if line.startswith('<'):
            block = line[1:line.index('>')]
        elif '=' in line:
            key,value = line.split('=',1)
            params[block+'/'+key.strip()] = value.strip()
    last = 0.
    for _ in range(10):
        last += 1./6.
    require(params['time/integrator'] == 'rk4_classical' and
            params['mesh_refinement/refinement'] == 'static' and
            float(params['time/dt']) == DT and params['job/basename'] == 'native_n12_5M',
            'checkpoint evolution/output policy changed')
    require({key.split('/')[0] for key in params if key.startswith('output')} ==
            {'output1','output2'}, 'unexpected output block')
    require(params['output1/file_type'] == 'hst' and params['output2/file_type'] == 'rst'
            and params['output2/single_file_per_rank'] == 'false' and
            int(params['output2/file_number']) == 11, 'unexpected restart/output counter')
    for name in ('output1','output2'):
        require(name+'/dcycle' not in params and float(params[name+'/dt']) == 1./6.
                and float(params[name+'/last_time']) == last, 'output cadence changed')
    seed = float(params['fastflow/last_a0_0'])
    require(math.isfinite(seed) and seed > 0 and params['fastflow/ah_found_a0_0'] in ('true','1'),
            'missing persisted horizon seed')
    f32 = lambda x: struct.unpack('<f',struct.pack('<f',x))[0]
    first_due = None
    for c in range(3552,3907):
        time += dt
        if first_due is None and f32(time) >= f32(last+1./6.) and f32(time) < f32(2.):
            first_due = c
    require(first_due == 3906 and expected_time(3681) < 2., 'timing window includes output')
    return dict(cycle=3551,time=expected_time(3551),dt=DT,blocks=106,
                writer_ranks=24,output_last_time=last,restart_file_number=11,
                next_scheduled_output_cycle=3906,final_cycle=3681,
                final_time=expected_time(3681),final_restart_file_number=12)


def check_admission(science, review, science_sha, checkpoint_sha):
    require(science['schema'] == 'native-n12-common-science-admission-v1' and
            science['passed'] is True and science['segment'] == 10 and
            science['end_cycle'] == 3551 and science['through_5m_complete'] is True and
            science['checkpoint_sha256'] == checkpoint_sha and
            science['missing_future_common_steps'] == [], 'full3551 science is not admitted')
    require(set(science['comparisons']) == {'3551'} and
            science['comparisons']['3551']['passed'] is True and
            science['comparisons']['3551']['writer_mpi_ranks'] == 24 and
            science['history_completed_prefix_passed'] is True and
            set(science['history']['times']) == {str(s) for s in STEPS} and
            science['history']['failures'] == [], 'incomplete field/history admission')
    structural = science['structural']
    require(structural['structural_cadence_passed'] is True and structural['segment'] == 10
            and structural['end_cycle'] == 3551, 'incomplete final segment structure')
    for name,count in (('bulk',7111152),('surface',26784)):
        field = science['comparisons']['3551']['fields'][name]
        require(field['comparisons'] == count and field['failed_values'] == 0 and
                len(field['fields']) == 31, 'incomplete field coverage')
    metric = science['stage_metrics']
    require(metric['passed'] is True and metric['full3551_complete'] is True and
            metric['stage_rows'] == 14204 and metric['through_step'] == 3551 and
            metric['ranks'] == 24 and metric['gate'] == 'long' and
            metric['failed_values'] == 0 and metric['exact_step_stage_time_dt'] is True and
            metric['completed_steps'] == list(range(1,3552)) and
            math.isfinite(metric['maximum_compatibility']) and
            0 <= metric['maximum_compatibility'] <= 1e-7, 'incomplete stage admission')
    require(review['schema'] == 'native-n12-common-independent-review-v1' and
            review['passed'] is True and review['end_cycle'] == 3551 and
            review['through_5m_complete'] is True and review['science_admission_sha256'] == science_sha
            and review['checkpoint_sha256'] == checkpoint_sha and
            review['stage_rows'] == 14204 and review['exact_stage_clocks'] is True and
            review['history_completed_prefix_passed'] is True and review['history_times'] == STEPS
            and review['missing_future_common_steps'] == [], 'incomplete independent3551 review')


def admission(path):
    path = Path(path).resolve(); data = read_json(path)
    require(data['schema'] == 'native-cycle-timing-admission-v1', 'wrong admission descriptor')
    files = {}
    for key in ('checkpoint','science_admission','independent_review'):
        target = Path(data[key]['path'])
        target = target if target.is_absolute() else path.parent/target
        require(sha(target) == data[key]['sha256'], 'changed admission input: '+key)
        files[key] = target.resolve()
    check_admission(read_json(files['science_admission']),read_json(files['independent_review']),
                    data['science_admission']['sha256'],data['checkpoint']['sha256'])
    return dict(source_checkpoint_remote=str(files['checkpoint']),
        source_checkpoint_sha256=data['checkpoint']['sha256'],
        science_admission_sha256=data['science_admission']['sha256'],
        independent_review_sha256=data['independent_review']['sha256'],
        checkpoint_metadata=checkpoint_metadata(files['checkpoint']))


def environment(directory):
    require(all(name not in os.environ for name in PROFILE), 'profiling environment requested')
    require(all(os.environ.get(k) == v for k,v in ENVIRONMENT.items()), 'unexpected execution environment')
    require('ZE_AFFINITY_MASK' not in os.environ, 'tile affinity must be assigned by wrapper')
    for name,tail in (('NEO_CACHE_DIR','cache/neo'),('SYCL_CACHE_DIR','cache/sycl')):
        require(os.environ.get(name) == str(Path(directory).resolve()/tail), 'wrong per-run cache')
    return {name:os.environ.get(name) for name in (*ENVIRONMENT,*PROFILE,'ZE_AFFINITY_MASK',
                                                'NEO_CACHE_DIR','SYCL_CACHE_DIR')}


def binding(path, nodes):
    hosts = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+)\.(\d+) implicit_scaling=0',line)
        require(match is not None, 'invalid tile binding record')
        host,rank,gpu,tile = match.groups();rank=int(rank)
        require(0 <= rank < 12 and (int(gpu),int(tile)) == (rank//2,rank%2), 'wrong tile assignment')
        require(rank not in hosts.setdefault(host,set()), 'duplicate binding rank')
        hosts[host].add(rank)
    require(len(hosts) == nodes and all(ranks == set(range(12)) for ranks in hosts.values()),
            'incomplete node/tile binding')


def terminal(text, job, nodes):
    identity = re.search(r'^Job Id:\s*(\S+)\s*$',text,re.M)
    require(identity is not None and identity[1].split('.')[0] == str(job), 'terminal job mismatch')
    fields = dict(re.findall(r'^\s+([A-Za-z_.]+)\s*=\s*(\S+)\s*$',text,re.M))
    expected = 'native-timing-two' if nodes == 2 else 'native-timing-eight'
    require(fields.get('job_state') == 'F' and fields.get('Exit_status') == '0' and
            fields.get('Job_Name') == expected and fields.get('Resource_List.nodect') == str(nodes),
            'timing allocation is not the expected successful terminal job')


def previous_pair(root, output):
    case = root/'runs/cycle-timing-nodes2'
    job = (case/'job-id.txt').read_text().strip()
    require(re.fullmatch(r'\d+(?:\.[A-Za-z0-9.-]+)?',job), 'invalid prior pair PBS identifier')
    require((case/'pipeline-complete.txt').is_file() and
            read_json(case/'guard-after.json')['passed'] is True and
            all((case/name/'exit-code.txt').read_text().strip() == '0' for name in ('normal','timed')),
            'two-node pair has not completed')
    prior = read_json(case/'guard-after.json')
    current = read_json(output.parent/'guard-before.json')
    keys = ('source_checkpoint_remote','source_checkpoint_sha256','science_admission_sha256',
            'independent_review_sha256','baseline_executable_sha256','timing_executable_sha256',
            'timing_build_provenance_sha256','source_manifest_sha256','object_manifest_sha256',
            'verified_source_files','verified_objects','gpu_wrapper_sha256')
    require(all(prior[key] == current[key] for key in keys),
            'two/eight-node pairs have different source/build/admission/wrapper identities')
    status = subprocess.check_output(['qstat','-xf',job],text=True)
    terminal(status,job.split('.')[0],2)
    output.write_text(status)


def guard(args):
    root = args.root.resolve(); work = root/'validation/cycle-timing'
    report = admission(args.admission)
    require(all(name not in os.environ for name in PROFILE), 'profiling environment requested')
    require(sha(args.wrapper) == WRAPPER, 'GPU tile wrapper changed')
    source_file = root/'build-sycl/source-sha256.json'
    object_file = root/'build-sycl/runtime-object-sha256.json'
    require(sha(source_file) == SOURCES and sha(object_file) == OBJECTS, 'frozen manifests changed')
    sources = read_json(source_file);objects = read_json(object_file)
    require(len(sources) == 1972 and len(objects['object_sha256']) == 228, 'incomplete frozen manifests')
    for rel,digest in sources.items():
        require(sha(root/'source'/rel) == digest, 'frozen source changed: '+rel)
    for rel,digest in objects['object_sha256'].items():
        require(sha(root/'build-sycl'/rel) == digest, 'frozen object changed: '+rel)
    require(sha(root/'build-sycl/src/athena') == BASELINE and
            sha(work/'build/athena-cycle-timing') == TIMING and
            sha(work/'build/provenance.json') == PROVENANCE and
            sha(work/'package/native_cycle_timing.py') == BUILDER, 'executable/build identity changed')
    with args.verify_log.open('w') as log:
        result = subprocess.run(['/usr/bin/python3.11',str(work/'package/native_cycle_timing.py'),
                                 'verify','--output',str(work/'build')],stdout=log,stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'timing build verification failed')
    report.update(schema='native-cycle-timing-runtime-guard-v1',passed=True,
        baseline_executable_sha256=BASELINE,timing_executable_sha256=TIMING,
        timing_build_provenance_sha256=PROVENANCE,source_manifest_sha256=SOURCES,
        object_manifest_sha256=OBJECTS,verified_source_files=1972,verified_objects=228,
        gpu_wrapper=str(args.wrapper.absolute()),gpu_wrapper_sha256=WRAPPER,timing_verify_exit_code=0,
        timing_verify_log_sha256=sha(args.verify_log))
    args.output.write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    p = sub.add_parser('guard');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--admission',type=Path,required=True);p.add_argument('--wrapper',type=Path,required=True)
    p.add_argument('--verify-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p = sub.add_parser('environment');p.add_argument('directory',type=Path)
    p = sub.add_parser('binding');p.add_argument('path',type=Path);p.add_argument('nodes',type=int,choices=(2,8))
    p = sub.add_parser('previous-pair');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.mode == 'guard':
        guard(args)
    elif args.mode == 'environment':
        (args.directory/'environment.json').write_text(json.dumps(environment(args.directory),indent=2)+'\n')
    elif args.mode == 'previous-pair':
        previous_pair(args.root,args.output)
    else:
        binding(args.path,args.nodes)


if __name__ == '__main__':
    main()
