#!/usr/bin/env python3
"""Validate a completed CPU eight-step timing run and reject invalid configurations.

Reuses a verified original native eight-step baseline and existing timing-run
checkpoints. No production rebuild, new baseline evolution or GPU job occurs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
from types import SimpleNamespace

from native_cycle_timing import sha, require, verify
from n12_checkpoint import DT, expected_time
from restart_roundtrip import parse, compare


def states(directory):
    by_cycle={}
    for path in sorted((directory/'rst').glob('*.rst')):
        info=parse(path)
        require(info['dt']==DT and info['time']==expected_time(info['cycle']),
                'checkpoint clock changed')
        if info['cycle'] in by_cycle:compare(by_cycle[info['cycle']],info)
        by_cycle[info['cycle']]=info
    require(set(by_cycle)=={0,8},'missing complete eight-step checkpoints')
    return by_cycle


def timing_report(path):
    r=json.loads(path.read_text())
    require(r['schema']=='native-cycle-timing-v1' and r['passed'] is True,
            'failed timing report')
    require((r['initial_cycle'],r['final_cycle'],r['complete_steps'],r['ranks'])==(0,8,8,2),
            'timing cycle/decomposition coverage differs')
    require((r['initial_time'],r['final_time'],r['dt'])==(0,expected_time(8),DT),
            'timing exact clocks differ')
    require((r['warmup_steps'],r['steps_per_window'],r['window_count'])==(2,3,2)
            and len(r['windows'])==2,'timing windows missing/changed')
    require(r['bulk_points']==229392 and r['scri_points']==864 and r['blocks']==106,
            'timing workload count changed')
    require(r['profiling_library_loaded'] is False and r['profiling_environment']==[]
            and r['timed_output_count']==r['timed_console_count']==0,
            'timing includes profiling/output')
    for w,first in zip(r['windows'],(2,5)):
        require((w['first_cycle'],w['last_cycle'],w['complete_steps'])==(first,first+3,3)
                and (w['first_time'],w['last_time'])==
                    (expected_time(first),expected_time(first+3)),
                'window includes partial/missing/repeated complete cycles')
        elapsed=w['max_rank_seconds']
        require(math.isfinite(elapsed) and elapsed>0 and
                w['seconds_per_complete_step']==elapsed/3 and
                w['bulk_cell_updates']==229392*3 and
                w['bulk_cell_updates_per_second']==229392*3/elapsed,
                'window rates/counts invalid; RK stages are not cell updates')
    for key in ('startup_max_seconds','warmup_max_seconds','execute_max_seconds','finalize_max_seconds'):
        require(math.isfinite(r[key]) and r[key]>0,'invalid separated timing '+key)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','run','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('build','run','baseline','output'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=False)
    proof=json.loads((a.build/'provenance.json').read_text());binary=Path(proof['executable'])
    verify(SimpleNamespace(output=a.build))
    result={'schema':'native-cycle-timing-cpu-validation-v1','passed':False,
            'scope':'CPU plumbing/state validation only; no GPU performance or scaling claim',
            'timing_build_provenance_sha256':sha(a.build/'provenance.json'),'negative_controls':{}}
    try:
        require('Terminating on cycle limit' in (a.run/'run.log').read_text(),
                'timed run did not complete normally')
        result['timing']=timing_report(a.run/'native-cycle-timing.json')
        baseline=states(a.baseline);timed=states(a.run)
        result['complete_states']={str(c):compare(baseline[c],timed[c]) for c in (0,8)}
        names={x.name for glob in ('*.hst','*.horizon*.txt') for x in a.baseline.glob(glob)}
        actual={x.name for glob in ('*.hst','*.horizon*.txt') for x in a.run.glob(glob)}
        require(names==actual and names,'ordinary output coverage changed')
        require(all((a.baseline/n).read_bytes()==(a.run/n).read_bytes() for n in names),
                'ordinary history/horizon/shape bytes changed')
        result['ordinary_outputs']={'bitwise_equal':True,'files':sorted(names)}
        # A CLI-loaded tool with the profiling environment absent tests the
        # actual Kokkos loaded-callback API, rather than an environment proxy.
        stub=a.output/'profiling-tool.cpp'
        stub.write_text('#include <cstdint>\n#include <cstdio>\nextern "C" {\n'
            'void kokkosp_init_library(int,std::uint64_t,std::uint32_t,void*) {'
            'std::fputs("TIMING_TEST_TOOL_INITIALIZED\\n",stderr);}\n'
            'void kokkosp_begin_parallel_for(const char*,std::uint32_t,std::uint64_t *id){*id=0;}\n'
            'void kokkosp_end_parallel_for(std::uint64_t){}\n}\n')
        lib=a.output/('profiling-tool.dylib' if platform.system()=='Darwin' else 'profiling-tool.so')
        compiler=proof['original_compile_command'][0]
        command=[compiler,*(['-dynamiclib'] if platform.system()=='Darwin' else ['-shared','-fPIC']),
                 '-std=c++17',str(stub),'-o',str(lib)]
        compiled=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (a.output/'profiling-tool-build.log').write_text(compiled.stdout);compiled.check_returncode()
        result['profiling_tool']={'command':command,'source_sha256':sha(stub),'library_sha256':sha(lib)}
        env=os.environ.copy()
        for key in ('KOKKOS_TOOLS_LIBS','KOKKOS_PROFILE_LIBRARY','KOKKOS_TOOLS_ARGS'):
            env.pop(key,None)
        env.update(NATIVE_TIMING_WARMUP='2',NATIVE_TIMING_STEPS='3',NATIVE_TIMING_WINDOWS='2')
        restart=Path(timed[0]['path'])
        def negative(label,extra,message,tool=False):
            folder=a.output/label;folder.mkdir()
            command=['mpiexec','-n','2',str(binary),*extra,'-r',str(restart),
                     'time/nlim=8','-d',str(folder)]
            with (folder/'run.log').open('w') as stream:
                run=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
            log=(folder/'run.log').read_text()
            passed=run.returncode!=0 and message in log and not (folder/'native-cycle-timing.json').exists()
            if tool:passed=passed and 'TIMING_TEST_TOOL_INITIALIZED' in log
            result['negative_controls'][label]={'passed':passed,'command':command,
                'exit_code':run.returncode,'expected_message':message,'log_sha256':sha(folder/'run.log')}
            require(passed,'negative control was not rejected: '+label)
        negative('console-output',['time/ndiag=1'],'timing preflight found')
        negative('scheduled-output',['time/ndiag=1000000000','output1/dt=0.001'],
                 'timing preflight found')
        negative('cli-profiling-tool',['--kokkos-tools-libs='+str(lib),'time/ndiag=1000000000'],
                 'primary timing refuses loaded profiling',True)
        result['passed']=True
    except Exception as exc:
        result['failure']=str(exc)
        raise
    finally:
        result['source_sha256']={str(x.resolve()):sha(x) for x in [Path(__file__),
            Path(__file__).with_name('native_cycle_timing.py'),
            Path(__file__).with_name('native_cycle_timing.hpp'),
            Path(__file__).with_name('native_cycle_timing_contract.md')]}
        result['input_sha256']={str(x):sha(x) for folder in (a.run,a.baseline)
            for x in folder.rglob('*') if x.is_file() and x.suffix in ('.rst','.hst','.txt','.json','.log')}
        verify(SimpleNamespace(output=a.build))
        (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'negative_controls':result['negative_controls']},indent=2))
if __name__=='__main__':main()
