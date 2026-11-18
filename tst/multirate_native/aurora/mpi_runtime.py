"""Matched MPICH runtime crossover on a single Aurora allocation."""
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

from binding import launcher
from compare import compare


def run(root, executable, case_file):
    modules = json.loads(case_file.read_text())['mpi_comparison']
    if (len(modules) != 2 or modules[0] == modules[1] or
            not all(re.fullmatch(r'mpich/prd/[\w.-]+', name) for name in modules)):
        raise ValueError('expected two distinct Aurora MPI module names')
    job = os.environ['PBS_JOBID']
    measure = Path(__file__).with_name('measure.py')
    observations = {}
    references = {}
    legs = []
    for leg, module in enumerate([*modules, *reversed(modules)]):
        series = f'mpi-{job.split(".")[0]}-{leg}'
        cases = root/(series+'.json')
        # Reverse method order on the return leg, too.
        methods = ('selected','classical') if leg < 2 else ('classical','selected')
        cases.write_text(json.dumps([['current754',24,mode,5] for mode in methods]))
        probe = root/(series+'-libraries.txt')
        shell = 'set -eu; module load "$1"; shift; "$@"'
        def inside(args, **kwargs):
            subprocess.run(['bash','-c',shell,'mpich comparison',module,*map(str,args)],check=True,
                           env=dict(os.environ,MR_BENCHMARK_SERIES=series,
                                    MR_BENCHMARK_EXECUTABLE=str(executable)),**kwargs)
        # ldd on every bound rank resolves the executable's actual libmpi RUNPATH.
        with probe.open('w') as output:
            inside([*launcher(24),'bash','-c',
                    'ldd "$1" | awk \'$1=="libmpi.so.12"{print $3}\'','probe',executable],stdout=output)
        libraries = probe.read_text().splitlines()
        if (len(libraries)!=24 or len(set(libraries))!=1 or
                'mpich-'+module.rsplit('/',1)[1]+'-' not in libraries[0]):
            raise RuntimeError(f'bound ranks did not resolve {module}: {probe}')
        inside(['/usr/bin/python3.11',measure,root,'--cases',cases])
        report = root/(series+'-results.json')
        rows = json.loads(report.read_text())['results']
        modes = {row['mode'] for row in rows}
        if len(rows)!=2 or len(modes)!=2 or 'classical' not in modes or (leg and modes!=set(references)):
            raise RuntimeError(f'incomplete runtime comparison in {report}')
        leg_results = {}
        for row in rows:
            method = row['mode']
            if row['job_id']!=job or row['nodes']!=2 or row['physical_gpus']!=12:
                raise ValueError('mismatched runtime allocation')
            ref = references.setdefault(method,row)
            for key in ('binary_sha256','rank_binding','mesh','intervals'):
                if row[key]!=ref[key]:
                    raise ValueError('mismatched runtime control '+key)
            if method!='classical':
                placement = Path(row['log']).with_name('partition.log')
                original = Path(ref['log']).with_name('partition.log')
                if placement.read_bytes()!=original.read_bytes():
                    raise ValueError('mismatched runtime block placement')
            if row is not ref and not compare(ref,row)['identical_checkpoint_payload']:
                raise ValueError('MPI runtime changed accepted state or diagnostics')
            observations.setdefault(method,{}).setdefault(module,[]).extend(
                t/(row['simulated_M0']/row['intervals']) for t in row['steady_base_seconds'][:-1])
            leg_results[method] = row['evolution_seconds_per_M0']
        legs.append(dict(module=module,library=libraries[0],report=str(report),seconds_per_M0=leg_results))
    cost = {method:{name:median(samples) for name,samples in by_module.items()}
            for method,by_module in observations.items()}
    result = dict(job=job,legs=legs,evolution_seconds_per_M0=cost,
                  speedup={method:values[modules[0]]/values[modules[1]] for method,values in cost.items()})
    output = case_file.with_name(case_file.stem+'-results.json')
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    run(*(Path(arg).resolve() for arg in sys.argv[1:]))
