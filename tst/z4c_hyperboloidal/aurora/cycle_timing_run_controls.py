"""Synthetic/std-library parser controls only: no GPU or performance proof."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tst/z4c_hyperboloidal/aurora'
spec = importlib.util.spec_from_file_location('guard',TOOLS/'cycle_timing_run_audit.py')
g = importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
checks = {}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--source-checkpoint',type=Path,required=True)
parser.add_argument('--partial-science',type=Path,required=True)
parser.add_argument('--partial-review',type=Path,required=True)
args = parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT/'tst/z4c_hyperboloidal'))
from native_cycle_timing_gpu_validate import check_source
from n12_checkpoint import parse_native


def reject(name, function):
    try:
        function()
    except (ValueError,KeyError,struct.error) as error:
        checks[name] = dict(rejected=True,error=str(error))
    else:
        raise AssertionError(name+' was accepted')


def metadata_fixture(source, destination, params_change=None, cycle=3551, truncate=False):
    with source.open('rb') as stream:
        lines = []
        while True:
            line = stream.readline();lines.append(line.decode())
            if line == b'<par_end>\n':break
        size = struct.calcsize('<ii9d38i2di');fixed = bytearray(stream.read(size))
        struct.pack_into('<ddi',fixed,size-20,g.expected_time(cycle),g.DT,cycle)
        rest = bytearray(stream.read(8));length, = struct.unpack('<Q',rest)
        rest.extend(stream.read(length+106*24))
        meta = stream.read(64);rest.extend(meta)
        identity = struct.unpack('<8Q',meta)[2]
        rest.extend(stream.read(identity+8));stride, = struct.unpack('<Q',rest[-8:])
    last = 0.
    for _ in range(10):last += 1./6.
    values = {'time/nlim':'3551','time/tlim':str(5./3.),
              'output1/last_time':repr(last),'output2/last_time':repr(last),
              'output2/file_number':'11'}
    values.update(params_change or {})
    block = '';out = []
    for line in lines:
        stripped = line.split('#',1)[0].strip()
        if stripped.startswith('<'):block = stripped[1:stripped.index('>')]
        elif '=' in stripped:
            key = stripped.split('=',1)[0].strip();full=block+'/'+key
            if full in values:line = key+' = '+values[full]+'\n'
        out.append(line)
    with destination.open('wb') as stream:
        stream.write(''.join(out).encode());stream.write(fixed);stream.write(rest)
        stream.truncate(stream.tell()+106*stride-(1 if truncate else 0))


result = dict(schema='native-cycle-timing-producer-offline-controls-v1',passed=False,
              gpu_executed=False,performance_admission=False,
              scope='Only synthetic admission/metadata, environment, binding and scheduler parser controls; no solver invocation or GPU proof.')
try:
    for name in ('run-cycle-timing-validation.sh','cycle-timing-two.pbs','cycle-timing-eight.pbs'):
        subprocess.run(['bash','-n',str(TOOLS/name)],check=True)
    checks['three_shell_syntax_checks'] = True
    for name in ('cycle_timing_run_audit.py','collect_cycle_timing.py','cycle_timing_run_controls.py'):
        ast.parse((TOOLS/name).read_text())
    checks['three_python_syntax_checks'] = True
    source = args.source_checkpoint.resolve()
    reject('actual1066_not3551',lambda:g.checkpoint_metadata(source))
    with tempfile.TemporaryDirectory(dir=args.output) as directory:
        directory = Path(directory);fixture=directory/'synthetic.rst'
        metadata_fixture(source,fixture)
        value=g.checkpoint_metadata(fixture)
        check_source(parse_native(fixture));checks['shared_reader_synthetic_source_positive']=True
        assert value['next_scheduled_output_cycle']==3906 and value['final_cycle']==3681
        checks['synthetic_header_only_positive'] = value
        for name,change in (('wrong_counter',{'output2/file_number':'12'}),
                            ('changed_output_cadence',{'output1/dt':'.2'}),
                            ('changed_last_time',{'output2/last_time':'1.5'}),
                            ('nonfinite_seed',{'fastflow/last_a0_0':'nan'}),
                            ('adaptive_mesh',{'mesh_refinement/refinement':'adaptive'})):
            metadata_fixture(source,fixture,change)
            reject(name,lambda:g.checkpoint_metadata(fixture))
        metadata_fixture(source,fixture,truncate=True)
        reject('truncated_payload',lambda:g.checkpoint_metadata(fixture))
        science=json.loads(args.partial_science.read_text())
        review=json.loads(args.partial_review.read_text())
        reject('actual_partial_admission',lambda:g.check_admission(science,review,'s','c'))
        synthetic=copy.deepcopy(science);sr=copy.deepcopy(review)
        synthetic.update(segment=10,end_cycle=3551,through_5m_complete=True,
                         checkpoint_sha256='c',missing_future_common_steps=[])
        synthetic['comparisons']={'3551':synthetic['comparisons']['1066']}
        synthetic['history'].update(times={str(x):{} for x in g.STEPS},failures=[])
        synthetic['structural'].update(segment=10,end_cycle=3551)
        synthetic['stage_metrics'].update(full3551_complete=True,stage_rows=14204,
                                         through_step=3551,completed_steps=list(range(1,3552)))
        sr.update(end_cycle=3551,through_5m_complete=True,science_admission_sha256='s',
                  checkpoint_sha256='c',stage_rows=14204,history_times=g.STEPS,missing_future_common_steps=[])
        g.check_admission(synthetic,sr,'s','c');checks['synthetic_admission_shape_only_positive']=True
        for name,modify in (
                ('failed_science',lambda x:x.update(passed=False)),
                ('missing_history',lambda x:x['history']['times'].pop('3551')),
                ('missing_stage',lambda x:x['stage_metrics'].update(stage_rows=14203)),
                ('failed_compatibility',lambda x:x['stage_metrics'].update(maximum_compatibility=1e-6)),
                ('partial_field_count',lambda x:x['comparisons']['3551']['fields']['bulk'].update(comparisons=1))):
            bad=copy.deepcopy(synthetic);modify(bad)
            reject(name,lambda:g.check_admission(bad,sr,'s','c'))
        reject('wrong_review_binding',lambda:g.check_admission(synthetic,sr,'different','c'))
        env=dict(g.ENVIRONMENT,NEO_CACHE_DIR=str(directory/'cache/neo'),SYCL_CACHE_DIR=str(directory/'cache/sycl'))
        with patch.dict(os.environ,env,clear=True):g.environment(directory)
        checks['synthetic_environment_positive']=True
        for name,change in (('profiling_requested',{'KOKKOS_PROFILE_LIBRARY':'tool.so'}),
                            ('empty_profile_env',{'KOKKOS_TOOLS_LIBS':''}),
                            ('preset_affinity',{'ZE_AFFINITY_MASK':'0.0'}),
                            ('wrong_cache',{'NEO_CACHE_DIR':'/tmp/shared'})):
            with patch.dict(os.environ,dict(env,**change),clear=True):reject(name,lambda:g.environment(directory))
        binding=directory/'binding.txt'
        rows=[f'host{h} local_rank={r} tile={r//2}.{r%2} implicit_scaling=0\n' for h in range(2) for r in range(12)]
        binding.write_text(''.join(rows));g.binding(binding,2);checks['synthetic_binding_positive']=True
        binding.write_text(''.join(rows[:-1]));reject('missing_tile_rank',lambda:g.binding(binding,2))
        binding.write_text(''.join(rows+[rows[0]]));reject('duplicate_tile_rank',lambda:g.binding(binding,2))
        terminal='Job Id: 42.aurora\n    Job_Name = native-timing-two\n    job_state = F\n    Exit_status = 0\n    Resource_List.nodect = 2\n'
        g.terminal(terminal,42,2);checks['synthetic_terminal_positive']=True
        prior=directory/'runs/cycle-timing-nodes2';current=directory/'runs/cycle-timing-nodes8'
        prior.mkdir(parents=True);current.mkdir(parents=True)
        (prior/'job-id.txt').write_text('42.aurora\n');(prior/'pipeline-complete.txt').touch()
        for name in ('normal','timed'):
            (prior/name).mkdir();(prior/name/'exit-code.txt').write_text('0\n')
        keys=('source_checkpoint_remote','source_checkpoint_sha256','science_admission_sha256',
              'independent_review_sha256','baseline_executable_sha256','timing_executable_sha256',
              'timing_build_provenance_sha256','source_manifest_sha256','object_manifest_sha256',
              'verified_source_files','verified_objects','gpu_wrapper_sha256')
        pair_guard=dict(passed=True,**{key:key for key in keys})
        (prior/'guard-after.json').write_text(json.dumps(pair_guard))
        (current/'guard-before.json').write_text(json.dumps(pair_guard))
        with patch.object(g.subprocess,'check_output',return_value=terminal):
            g.previous_pair(directory,current/'previous-pair-terminal.txt')
            checks['synthetic_previous_pair_positive']=True
            bad=dict(pair_guard,source_checkpoint_sha256='different')
            (current/'guard-before.json').write_text(json.dumps(bad))
            reject('previous_pair_checkpoint_mismatch',lambda:g.previous_pair(directory,current/'previous-pair-terminal.txt'))
            (current/'guard-before.json').write_text(json.dumps(pair_guard))
        with patch.object(g.subprocess,'check_output',return_value=terminal.replace('state = F','state = R')):
            reject('previous_pair_not_terminal',lambda:g.previous_pair(directory,current/'previous-pair-terminal.txt'))
        for name,text in (('running_job',terminal.replace('state = F','state = R')),
                          ('failed_job',terminal.replace('status = 0','status = 1')),
                          ('wrong_job',terminal.replace('42.aurora','43.aurora')),
                          ('wrong_nodes',terminal.replace('nodect = 2','nodect = 8'))):
            reject(name,lambda:g.terminal(text,42,2))
    result.update(passed=True,checks=checks)
except Exception as error:
    result.update(checks=checks,failure=f'{type(error).__name__}: {error}')
    raise
finally:
    names=('cycle_timing_run_audit.py','collect_cycle_timing.py','run-cycle-timing-validation.sh',
           'cycle-timing-two.pbs','cycle-timing-eight.pbs','cycle_timing_run_controls.py')
    result['source_sha256']={str(TOOLS/name):g.sha(TOOLS/name) for name in names}
    result['control_source_sha256']=g.sha(__file__)
    result['reader_source_sha256']=g.sha(ROOT/'tst/z4c_hyperboloidal/native_cycle_timing_gpu_validate.py')
    result['input_paths']={key:str(getattr(args,key).resolve()) for key in ('source_checkpoint','partial_science','partial_review')}
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'gpu_executed':False},indent=2))
