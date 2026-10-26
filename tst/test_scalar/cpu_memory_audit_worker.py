#!/usr/bin/env python3
"""Read-only native-state RHS audit and DDR/HBM placement benchmark."""
import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();plan=json.loads((root/'cpu-plan.json').read_text());scripts=Path(__file__).resolve().parent
hardware=subprocess.run(['numactl','-H'],capture_output=True,text=True,check=True).stdout;(out/'numa-hardware.txt').write_text(hardware)
subprocess.run(['lscpu','-e'],stdout=(out/'lscpu.txt').open('w'),check=True)
nodes=re.findall(r'node (\d+) cpus:([^\n]*)',hardware);hbm=[int(n) for n,cpus in nodes if not cpus.strip()]
if len(nodes)!=4 or hbm!=[2,3]:raise RuntimeError('Unexpected CPU memory mode; refuse an unverified HBM binding')
env={**os.environ,'OMP_NUM_THREADS':'102','OMP_PROC_BIND':'spread','OMP_PLACES':'cores'};states={};records=[]
for case in plan['audit_cases']:
 run=Path(case['run']);n=case['n'];integrity=json.loads((run/'frozen-integrity.json').read_text())
 if not integrity['unchanged'] or sha(run/'checkpoint.f64')!=case['checkpoint_sha256']:raise RuntimeError('Native audit input differs from declaration')
 state=root/'audit-inputs'/f'n{n}'
 subprocess.run([sys.executable,str(scripts/'stage_rhs_inputs.py'),'--run',str(run),'--output',str(state)],check=True);states[n]=state
 subprocess.run([str(out/'audit_ringdown_rhs'),str(n),'0','.25',str(state),str(out/f'actual-rhs-n{n}')],env=env,check=True)
for label,prefix in [('default',[]),('hbm',['numactl','--membind=2,3'])]:
 target=out/('memory-'+label);command=[*prefix,str(out/'benchmark_frozen_rhs'),'60',str(states[60]/'checkpoint.f64'),str(target),'200']
 subprocess.run(command,env=env,check=True);meta=json.loads((target/'benchmark.json').read_text());records.append(dict(label=label,command=command,metadata=meta,rhs_sha256=sha(target/'active-rhs.f64')))
pair=records[0]['rhs_sha256']==records[1]['rhs_sha256'];keys=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before','active_state_before','active_state_after'];same=all(records[0]['metadata'][k]==records[1]['metadata'][k] for k in keys)
passed=pair and same and all(r['metadata']['active_state_unchanged'] and r['metadata']['frozen_unchanged'] for r in records);times=[r['metadata']['fill_seconds_per_call']+r['metadata']['evaluate_seconds_per_call'] for r in records]
receipt=dict(passed=passed,full_active_RHS_bitwise_identical=pair,all_state_and_geometry_fingerprints_identical=same,default_over_hbm_speedup=times[0]/times[1],records=records,plan_sha256=sha(root/'cpu-plan.json'),scope='Fixed saved scalar state; no time steps. Same source, binary, grid, coefficients, CPU affinity and OpenMP parameters. Only memory placement differs.')
(out/'memory-audit-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(passed=passed,speedup=receipt['default_over_hbm_speedup'])),flush=True)
if not passed:raise RuntimeError('Memory-placement numerical parity failed')
