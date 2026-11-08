#!/usr/bin/env python3
"""Bounded CPU validation of the observed original main and ordinary restart.

Run two-rank normal/observed eight-step controls, an observed two-step prefix,
then resume that prefix to step8 on eight ranks. No runtime sources are rebuilt.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from native_step_observer import sha, require, verify
from n12_checkpoint import DT, expected_time
from n12_long_history import read_outputs
from restart_roundtrip import parse, compare


def stages(directory, first, last, ranks):
    result=json.loads((directory/'native-step-observer.json').read_text())
    require(result['schema']=='native-step-observer-v1' and result['passed'] is True,
            'observer completion record missing/failed')
    require((result['initial_cycle'],result['final_cycle'],result['ranks'],result['stage_rows'])
            ==(first,last,ranks,4*(last-first)), 'observer completion coverage differs')
    require((result['initial_time'],result['final_time'],result['dt']) ==
            (expected_time(first),expected_time(last),DT), 'observer completion clocks differ')
    with (directory/'native-stages.csv').open() as stream:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==['step','stage','rhs_time','state_time','dt',
                                    'compatibility','clock','CFL'], 'observer CSV columns differ')
        rows=[{key:float(value) for key,value in row.items()} for row in reader]
    require([(r['step'],r['stage']) for r in rows]==
            [(step,stage) for step in range(first+1,last+1) for stage in range(1,5)],
            'observer stage order missing/duplicate')
    for row in rows:
        step,stage=int(row['step']),int(row['stage'])
        base=expected_time(step-1)
        require(row['rhs_time']==base+(0,.5,.5,1)[stage-1]*DT and
                row['state_time']==base+(.5,.5,1,1)[stage-1]*DT and row['dt']==DT,
                'observer RK clocks differ')
        require(all(np.isfinite(v) for v in row.values()) and
                all(row[key]>=0 for key in ('compatibility','clock','CFL')) and
                row['compatibility']<=1e-7, 'observer diagnostics invalid')
    return rows,result


def last_outputs(directory):
    histories,horizons,files,extra=read_outputs([directory])
    h=[x['record'] for x in extra if x['kind']=='history' and x['step']==8]
    a=[x['record'] for x in extra if x['kind']=='horizon' and x['step']==8]
    require(h and a,'missing accepted-step8 diagnostic output')
    return h[-1],a[-1],files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True,help='portable observer build directory')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.build=args.build.resolve();args.output=args.output.resolve()
    require(not args.output.exists(),'new validation directory required')
    args.output.mkdir(parents=True)
    verify(SimpleNamespace(output=args.build))
    proof=json.loads((args.build/'provenance.json').read_text())
    normal=Path(proof['native_executable']);observer=Path(proof['executable'])
    fixture=Path(__file__).with_name('aurora')/'n12.athinput'
    source_hashes={str(p.resolve()):sha(p) for p in [Path(__file__),fixture,
        Path(__file__).with_name('native_step_observer.py'),
        Path(__file__).with_name('native_step_observer.hpp'),
        Path(__file__).with_name('n12_checkpoint.py'),
        Path(__file__).with_name('n12_long_history.py'),
        Path(__file__).with_name('restart_roundtrip.py')]}
    result=dict(passed=False,scope='local CPU MPI validation of observed original main; no GPU admission',runs={})
    try:
        def run(label,binary,ranks,options):
            directory=args.output/label;directory.mkdir()
            command=['mpiexec','-n',str(ranks),str(binary),*options,'-d',str(directory)]
            with (directory/'run.log').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            result['runs'][label]=dict(command=command,exit_code=completed.returncode,
                                       executable_sha256=sha(binary))
            completed.check_returncode()
            by_cycle={}
            for path in sorted((directory/'rst').glob('*.rst')):
                info=parse(path)
                require(info['time']==expected_time(info['cycle']) and info['dt']==DT,
                        'checkpoint exact time/dt changed')
                if info['cycle'] in by_cycle:compare(by_cycle[info['cycle']],info)
                by_cycle[info['cycle']]=info
            require(by_cycle,'missing native checkpoint output')
            print('Completed '+label,flush=True)
            return directory,by_cycle
        common=['-i',str(fixture.resolve())]
        normal_dir,normal_states=run('normal8-two',normal,2,common+['time/nlim=8'])
        observed_dir,observed_states=run('observed8-two',observer,2,common+['time/nlim=8'])
        require(normal_states.keys()==observed_states.keys()=={0,8},'normal/observed output cadence changed')
        result['normal_vs_observed']={str(c):compare(normal_states[c],observed_states[c]) for c in (0,8)}
        files={path.name for pattern in ('*.hst','*.horizon*.txt') for path in normal_dir.glob(pattern)}
        observed_files={path.name for pattern in ('*.hst','*.horizon*.txt') for path in observed_dir.glob(pattern)}
        require(files==observed_files and files,'normal/observed history/horizon cadence differs')
        require(all((normal_dir/name).read_bytes()==(observed_dir/name).read_bytes() for name in files),
                'normal/observed history/horizon output bytes differ')
        result['normal_vs_observed_diagnostics']=dict(bitwise_equal=True,files=sorted(files))
        full,full_report=stages(observed_dir,0,8,2)
        prefix_dir,prefix_states=run('prefix2-two',observer,2,common+['time/nlim=2'])
        require(prefix_states.keys()=={0,2},'prefix output cadence changed')
        resumed_dir,resumed_states=run('resumed8-eight',observer,8,
                                      ['-r',prefix_states[2]['path'],'time/nlim=8'])
        require(resumed_states.keys()=={8},'resumed output cadence changed')
        prefix,prefix_report=stages(prefix_dir,0,2,2)
        resumed,resumed_report=stages(resumed_dir,2,8,8)
        require(prefix+resumed==full,'ordinary restart observer stage scalars/clocks differ')
        result['restart_complete_state']=compare(observed_states[8],resumed_states[8])
        result['restart_stage_rows_bitwise_equal']=True
        control_h,control_a,_=last_outputs(observed_dir)
        restart_h,restart_a,_=last_outputs(resumed_dir)
        differences={}
        for label,index in [('Volume',2),('H',3),('M',4),('Theta',5),('Z',6),
                            ('eVolume',7),('eH',8),('eM',9),('eTheta',10),('eZ',11)]:
            a,b=restart_h['values'][index],control_h['values'][index]
            gate=5e-10+1e-6*abs(b)
            differences[label]=dict(candidate=a,reference=b,absolute=abs(a-b),gate=gate)
            require(abs(a-b)<=gate,'restart history exceeds adopted output gate '+label)
        for label in ('areal_radius_M','irreducible_mass_M'):
            a,b=restart_a[label],control_a[label]
            gate=max(1e-10,1e-7*abs(b))
            differences[label]=dict(candidate=a,reference=b,absolute=abs(a-b),gate=gate)
            require(abs(a-b)<=gate,'restart horizon exceeds adopted gate '+label)
        result.update(passed=True,observer_completion=[full_report,prefix_report,resumed_report],
            restart_diagnostics=differences,
            restart_output_scope='prefix final horizon search changes warm start; compare adopted RMS/areal/mass gates separately from exact evolved state and stage values')
    except Exception as error:
        result['failure']=str(error)
        raise
    finally:
        stable=all(sha(Path(name))==digest for name,digest in source_hashes.items())
        result['source_inputs_unchanged']=stable
        result['source_sha256']=source_hashes
        result['observer_build_provenance_sha256']=sha(args.build/'provenance.json')
        result['artifact_sha256']={str(path.relative_to(args.output)):sha(path)
            for path in args.output.rglob('*') if path.is_file() and path.name!='results.json'}
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        require(stable,'validation source inputs changed')
        verify(SimpleNamespace(output=args.build))
    print(json.dumps(dict(passed=result['passed'],runs=list(result['runs'])),indent=2))


if __name__=='__main__':main()
