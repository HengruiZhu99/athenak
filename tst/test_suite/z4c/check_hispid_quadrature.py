"""Fixed-order angular quadrature refinement against an accepted exact control."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path
import numpy as np
from check_hispid_controls import ROOT,shape_error,seed_targets,exact_seed_source,harmonic_table_bytes,refinement_qualified
from hispid_sampler_proof import validate_migration,import_evidence

p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--baseline',required=True)
p.add_argument('--case',required=True,choices=('boost885','kerr95_boost885','kerr99','gamma10'))
p.add_argument('--ntheta',default='74');p.add_argument('--output',required=True)
p.add_argument('--baseline-flow-alpha',type=float,
               help='explicit recovery of the input flow alpha for older records missing this metadata')
p.add_argument('--consumer-memory-mib',type=int,default=32768)
a=p.parse_args();exe=Path(a.executable).resolve(strict=True);baseline_path=Path(a.baseline).resolve(strict=True)
baseline_bytes=baseline_path.read_bytes();baseline_sha=hashlib.sha256(baseline_bytes).hexdigest();baseline=json.loads(baseline_bytes)
sha=hashlib.sha256(exe.read_bytes()).hexdigest()
if baseline['executable_sha256']!=sha:raise ValueError('quadrature comparison requires the identical AthenaK executable')
source_rows=[r for r in baseline['records'] if r['case']==a.case]
if len(source_rows)<3 or not source_rows[-1]['passed']:raise ValueError('accepted angular-refined baseline is required')
if not (baseline.get('case_qualification',{}).get(a.case) is True
        and refinement_qualified(source_rows,'boost885' in a.case or a.case=='gamma10')):
    raise ValueError('qualified case-specific refinement group with every import/provenance/execution check required')
last=source_rows[-1];source=exact_seed_source(last['source'],a.case);lmax=last['lmax'];root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
if (root/'quadrature.json').exists():raise FileExistsError('preserve prior quadrature evidence at a separate output path')
migration=last.get('sampler_migration')
if migration and validate_migration(migration['path'],source)!=migration:raise ValueError('baseline sampler proof changed')
dependencies=last.get('import',{}).get('consumer_dependency_images',{})
if a.case in ('kerr99','gamma10') and not migration:
    raise ValueError('fresh extreme baseline requires a separate-process pure-reference-consumer proof')
alpha=last.get('flow_alpha',a.baseline_flow_alpha)
if alpha is None or not np.isfinite(alpha) or alpha<=0:raise ValueError('baseline flow alpha must be recorded or explicitly recovered')
if 'flow_alpha' in last and a.baseline_flow_alpha is not None and alpha!=a.baseline_flow_alpha:
    raise ValueError('explicit alpha conflicts with recorded baseline')
result=dict(case=a.case,executable_sha256=sha,baseline=str(Path(a.baseline).resolve()),lmax=lmax,
            baseline_sha256=baseline_sha,source=source,sampler_migration=migration,
            flow_alpha=alpha,flow_alpha_source='record' if 'flow_alpha' in last else 'explicit_input_recovery',records=[],passed=False)
template=(ROOT/'inputs/hispid.athinput').read_text()
result['input_template_sha256']=hashlib.sha256(template.encode()).hexdigest()
result['consumer_memory_screen']=dict(budget_mib=a.consumer_memory_mib,other_allowance_bytes=1024**3,
    note='Single-copy Serial harmonic data plus allowance; measured peak remains separate.')
for nt in map(int,a.ntheta.split(',')):
    if nt<=last['ntheta'] or nt%2:raise ValueError('use a finer even theta quadrature')
    table_bytes=harmonic_table_bytes(lmax,nt)
    if table_bytes+1024**3>a.consumer_memory_mib*1024**2:
        raise ValueError('declared Serial harmonic-table storage plus allowance exceeds the consumer budget')
    if (hashlib.sha256(exe.read_bytes()).hexdigest()!=sha
        or hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=baseline_sha
        or hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['file_sha256']
        or any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=s for p,s in dependencies.items())):
        raise ValueError('bound baseline/executable/checkpoint/dependency changed before quadrature worker')
    if migration and validate_migration(migration['path'],source)!=migration:raise ValueError('sampler proof changed before worker')
    run=root/f'n{nt}';run.mkdir(exist_ok=False)
    input_path=run/'quadrature.athinput';input_path.write_text(template);input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
    cmd=[str(exe),'-i',str(input_path),'problem/hispid_filename='+source['path'],
         'problem/hispid_source_sha256='+source['source_library_sha256'],f'fastflow/lmax={lmax}',f'fastflow/ntheta={nt}',
         f'problem/hispid_horizon_guess_scale={last["initial_scale"]}',f'fastflow/flow_alpha_beta_const_0={alpha}',
         'fastflow/flow_iterations_0=3000']
    if migration:cmd+=['problem/hispid_allow_library_migration=true']
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    start=time.monotonic()
    with (run/'run.log').open('w') as log:
        try:r=subprocess.run(cmd,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        except subprocess.TimeoutExpired:returncode='timeout'
        else:returncode=r.returncode
    stdout=(run/'run.log').read_text()
    row=dict(ntheta=nt,returncode=returncode,seconds=time.monotonic()-start,command=cmd,passed=False,
        input_sha256=input_sha,harmonic_table_bytes=table_bytes,import_evidence=dict(passed=False))
    result['records'].append(row);result['passed']=False
    (root/'quadrature.json').write_text(json.dumps(result,indent=2)+'\n')
    row['import_evidence']=import_evidence(stdout,source,migration)
    unchanged=bool(hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        and hashlib.sha256(baseline_path.read_bytes()).hexdigest()==baseline_sha
        and hashlib.sha256(input_path.read_bytes()).hexdigest()==input_sha
        and hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['file_sha256']
        and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in dependencies.items()))
    if migration:unchanged &= validate_migration(migration['path'],source)==migration
    row['bound_inputs_unchanged']=unchanged
    if not migration and dependencies:row['import_evidence']['passed'] &= row['import_evidence'].get('consumer_dependency_images')==dependencies
    if returncode==0:
        values=np.atleast_2d(np.loadtxt(run/'hispid.horizon_summary_0.txt'))[-1]
        row.update(area=float(values[7]),expansion_rms=float(np.sqrt(values[8])),zero_evolution_verified=bool(re.search(r'time=0\.000000e\+00 cycle=0',stdout) and 'MeshBlock-cycles = 0' in stdout))
        row['relative_area_change']=float(abs(row['area']/last['area']-1))
        row['expansion_rms_change']=float(abs(row['expansion_rms']-last['expansion_rms']))
        chi,speed=seed_targets(a.case)
        row['shape_sampled_relative_linf']=shape_error(run/'hispid.horizon_shape_0.txt',lmax,chi,speed)
        row['passed']=bool(np.isfinite(values).all() and row['expansion_rms']<1e-7 and row['relative_area_change']<1e-7
                           and row['expansion_rms_change']<1e-8 and row['shape_sampled_relative_linf']<1e-6 and row['zero_evolution_verified']
                           and row['import_evidence']['passed'] and unchanged)
    result['passed']=all(x['passed'] for x in result['records'])
    (root/'quadrature.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
    if not unchanged:raise ValueError('bound inputs changed; retained quadrature stays unqualified')
raise SystemExit(0 if result['passed'] else 1)
