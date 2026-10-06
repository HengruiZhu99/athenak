"""One matched horizon and one worker-exception control for the parallel source.

Reuses a bound, passing Serial result. Does not repeat the angular study.
"""
import argparse,hashlib,json,os,re,resource,subprocess,time
from pathlib import Path
import numpy as np
from hispid_sampler_proof import validate_migration,import_evidence

def run(executable,baseline,output,threads):
    evidence=json.loads(baseline.read_text());old=evidence['records'][0]
    if not evidence['passed'] or old['case']!='kerr99' or old['lmax']!=8 or not old['passed']:raise ValueError('passed matched kerr99 order8 baseline required')
    source=old['source'];migration=validate_migration(old['sampler_migration']['path'],source)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),threads=threads,rows=[],passed=False,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),angular_convergence_claim=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(threads),OMP_PROC_BIND='spread',OMP_PLACES='cores',OPENBLAS_NUM_THREADS='1')
    for label,extra in [('matched',[]),('worker_exception',['mesh/x1min=-0.05','mesh/x1max=0.05'])]:
        work=output/label;work.mkdir();cmd=[str(executable),*old['command'][1:],*extra]
        # AthenaK overrides require the parameter to exist in the input file.
        input_index=cmd.index('-i')+1
        source_input=Path(cmd[input_index]);input_text=source_input.read_text()
        if input_text.count('<problem>')!=1:raise ValueError('one problem block required')
        input_text=input_text.replace('<problem>','<problem>\nhispid_parallel_geometry = true',1)
        run_input=work/'control.athinput';run_input.write_text(input_text);cmd[input_index]=str(run_input)
        
        begin=time.monotonic()
        def no_core():resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        with (work/'run.log').open('w') as log:
            p=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,preexec_fn=no_core)
        text=(work/'run.log').read_text();row=dict(case=label,command=cmd,returncode=p.returncode,seconds=time.monotonic()-begin)
        witness=re.search(r'HiSpID horizon_geometry parallel=1 host_concurrency=(\d+)',text)
        row['parallel_witness']=bool(witness and int(witness[1])==threads)
        if label=='matched' and p.returncode==0 and (work/'hispid.horizon_summary_0.txt').exists():
            summary=np.atleast_2d(np.loadtxt(work/'hispid.horizon_summary_0.txt'))[-1]
            # Compare invariant mass/spin/area, not search iteration counters.
            expected=np.array(old['summary']);row['summary']=summary.tolist()
            row['invariant_scaled_difference']=float(np.max(abs(summary[2:8]-expected[2:8])/(1+abs(expected[2:8]))))
            row['expansion_rms']=float(np.sqrt(summary[8]));row['import']=import_evidence(text,source,migration)
            row['zero_evolution']=bool('MeshBlock-cycles = 0' in text and re.search(r'time=0\.000000e\+00 cycle=0',text))
            row['passed']=bool(p.returncode==0 and row['parallel_witness'] and row['invariant_scaled_difference']<1e-10 and row['expansion_rms']<1e-7 and row['import']['passed'] and row['zero_evolution'])
        elif label=='worker_exception':
            summary_path=work/'hispid.horizon_summary_0.txt'
            row['summary_data_rows']=sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in summary_path.read_text().splitlines()) if summary_path.exists() else 0
            row['passed']=bool(p.returncode!=0 and row['parallel_witness'] and 'FastFlow initial geometry: Horizon point outside mesh domain' in text and row['summary_data_rows']==0)
        else:
            row['passed']=False;row['failure']='Application failed or horizon summary missing; see run.log'
        result['rows'].append(row);(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(label,row['passed'],row['seconds'],flush=True)
    result['passed']=all(r['passed'] for r in result['rows']);(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--executable',type=Path,required=True);a.add_argument('--baseline',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--threads',type=int,default=16);p=a.parse_args()
    if p.threads<2:raise ValueError('parallel control requires at least2 threads')
    raise SystemExit(run(p.executable.resolve(),p.baseline.resolve(),p.output.resolve(),p.threads))
