"""Bound Serial dense/factorized flow-history equivalence on exact seeds.

Each mode is a separate process. Iteration coefficients, radius/gradient/rho
arrays and integrals use round-trip precision. A historical dense executable
may additionally check unchanged default outputs. Failed attempts are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from check_hispid_controls import ROOT,exact_seed_source,seed_targets
from fastflow_storage import harmonic_table_bytes,storage_evidence
from hispid_sampler_proof import validate_migration,import_evidence


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifacts(directory):
    paths=[directory/f'hispid.horizon_{part}_0.txt' for part in ('verbose','shape','summary')]
    return {str(p):digest(p) for p in paths if p.exists()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('executable','manifest','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--historical-dense-executable')
    p.add_argument('--cases',default='schwarzschild,kerr95,boost885')
    p.add_argument('--lmax',type=int,default=32);p.add_argument('--ntheta',type=int,default=48)
    p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
    if a.timeout<1 or harmonic_table_bytes(a.lmax,a.ntheta)+1024**3>32768*1024**2:
        raise ValueError('positive timeout and bounded dense Serial control required')
    cases=a.cases.split(',')
    if len(set(cases))!=len(cases) or any(c not in ('schwarzschild','kerr95','boost885','kerr99','gamma10') for c in cases):
        raise ValueError('distinct supported exact seed cases required')
    exe=Path(a.executable).resolve(strict=True);manifest=Path(a.manifest).resolve(strict=True)
    data=json.loads(manifest.read_bytes());bound={str(exe):digest(exe),str(manifest):digest(manifest)}
    historical=Path(a.historical_dense_executable).resolve(strict=True) if a.historical_dense_executable else None
    if historical:bound[str(historical)]=digest(historical)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    template_path=ROOT/'inputs/hispid.athinput';template=template_path.read_text();bound[str(template_path)]=digest(template_path)
    result=dict(schema='fastflow_cache_equivalence_v1',bound_inputs_sha256=bound,lmax=a.lmax,ntheta=a.ntheta,
        records=[],completed=False,passed=False,changes_physical_gates=False,acceptance_inherited=False,
        centered_physical_seed_qualification=False,
        criteria=dict(expansion_rms=1e-7,area_relative=2e-5,trace_and_outputs='bitwise'),
        note='Displaced centers excite mixed modes; their coordinate spin and Christodoulou mass are origin-dependent diagnostics. Centered exact-seed qualification and peak RAM remain separate. Cache equivalence cannot qualify solved binary initial data.')
    proofs=[]
    def verify():
        if any(digest(file)!=sha for file,sha in bound.items()):raise ValueError('bound cache-control input changed')
        if any(validate_migration(proof['path'],source)!=proof for proof,source in proofs):
            raise ValueError('bound cache-control sampler prerequisites changed')
    def bind(file,sha):
        if file in bound and bound[file]!=sha:raise ValueError('cache-control binding changed')
        bound[file]=sha
    def save():verify();(root/'cache-equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    save();env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for case in cases:
        source=exact_seed_source(data[case],case);bind(source['path'],source['file_sha256'])
        proof=validate_migration(data[case]['migration_proof'],source) if data[case].get('migration_proof') else None
        if proof:bind(proof['path'],proof['file_sha256']);proofs.append((proof,source))
        group=dict(case=case,source=source,migration=proof,workers=[],passed=False);result['records'].append(group);save()
        modes=[('dense',exe),('factorized',exe)]+([('historical_dense',historical)] if historical else [])
        for mode,worker_exe in modes:
            verify()
            if proof and validate_migration(proof['path'],source)!=proof:raise ValueError('sampler proof changed')
            run=root/f'{case}_{mode}';run.mkdir()
            # A displaced search center excites nonzero mixed angular modes.
            extra='center_x_0 = .003\ncenter_y_0 = -.002\ncenter_z_0 = .001\n'
            input_path=run/'cache.athinput';input_path.write_text(template.replace('<problem>',extra+'<problem>'))
            bind(str(input_path),digest(input_path))
            cmd=[str(worker_exe),'-i',str(input_path),'problem/hispid_filename='+source['path'],
                'problem/hispid_source_sha256='+source['source_library_sha256'],
                'problem/hispid_preserve_horizon_centers=true',f'fastflow/lmax={a.lmax}',f'fastflow/ntheta={a.ntheta}',
                'problem/hispid_horizon_guess_scale=1.02','fastflow/flow_iterations_0=1200']
            if mode!='historical_dense':cmd+=['fastflow/factorized_harmonics='+str(mode=='factorized').lower(),'fastflow/full_precision_trace=true']
            if proof:cmd+=['problem/hispid_allow_library_migration=true']
            row=dict(mode=mode,executable=str(worker_exe),command=cmd,input_sha256=bound[str(input_path)],passed=False)
            group['workers'].append(row);save();start=time.monotonic()
            with (run/'run.log').open('w') as log:
                try:r=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                except subprocess.TimeoutExpired:row['returncode']='timeout'
                else:row['returncode']=r.returncode
            row['seconds']=time.monotonic()-start;row['artifacts_sha256']=artifacts(run);save()
            for file,sha in row['artifacts_sha256'].items():bind(file,sha)
            bind(str(run/'run.log'),digest(run/'run.log'))
            stdout=(run/'run.log').read_text();row['import']=import_evidence(stdout,source,proof)
            for file,sha in row['import'].get('consumer_dependency_images',{}).items():bind(file,sha)
            if row['import'].get('loaded_consumer_image'):bind(row['import']['loaded_consumer_image'],row['import']['loaded_consumer_sha256'])
            row['zero_evolution_verified']=bool(re.search(r'time=0\.000000e\+00 cycle=0',stdout) and 'MeshBlock-cycles = 0' in stdout)
            if mode!='historical_dense':row['allocation']=storage_evidence(stdout,mode,a.lmax,a.ntheta)
            summary=run/'hispid.horizon_summary_0.txt'
            if row['returncode']==0 and summary.exists():
                values=np.atleast_2d(np.loadtxt(summary))[-1];chi,_=seed_targets(case)
                area=8*np.pi*(1+np.sqrt(1-chi*chi))
                row['summary']=values.tolist();row['expansion_rms']=float(np.sqrt(values[8]))
                row['passed']=bool(np.isfinite(values).all() and values[8]>=0 and row['expansion_rms']<1e-7
                    and abs(values[7]/area-1)<2e-5 and row['import']['passed'] and row['zero_evolution_verified']
                    and (mode=='historical_dense' or row['allocation']['passed']))
            save()
        files={mode:root/f'{case}_{mode}' for mode,_ in modes}
        def content(mode,part):return (files[mode]/f'hispid.horizon_{part}_0.txt').read_bytes()
        if all(w['passed'] for w in group['workers']):
            group['trace_bitwise']=content('dense','verbose')==content('factorized','verbose')
            trace=content('dense','verbose').decode();group['trace_present']='# full_precision_point ' in trace and '# full_precision_coefficients ' in trace
            group['shape_bitwise']=content('dense','shape')==content('factorized','shape')
            group['summary_bitwise']=content('dense','summary')==content('factorized','summary')
            if historical:
                def ordinary(mode):return b'\n'.join(line for line in content(mode,'verbose').splitlines() if not line.startswith(b'# full_precision_'))
                group['historical_dense_bitwise']=bool(ordinary('dense')==ordinary('historical_dense')
                    and content('dense','shape')==content('historical_dense','shape')
                    and content('dense','summary')==content('historical_dense','summary'))
            group['passed']=all(group.get(k,True) for k in ('trace_bitwise','trace_present','shape_bitwise','summary_bitwise','historical_dense_bitwise'))
        save()
    result['completed']=True;result['passed']=all(g['passed'] for g in result['records']);save()
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
