"""Fresh actual configuration-source and complete normalization derivative gate."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mode=sys.argv[1] if len(sys.argv)>1 else 'release'
assert mode in ('release','debug')
count=1
while (P/('source-gate-%s-%03d'%(mode,count))).exists():count+=1
A=P/('source-gate-%s-%03d'%(mode,count));A.mkdir()
for n in ('radial_bridge.cpp','configuration_rows.hpp','radial_normalization.hpp','generic_gauge.hpp','run_source_gate.py'):
    shutil.copyfile(P/n,A/n)
exe=P/('radial-bridge-'+mode);cmd=[str(exe),'--gate'];start=time.monotonic()
run=subprocess.run(cmd,capture_output=True,text=True)
(A/'stdout.json').write_text(run.stdout);(A/'stderr').write_text(run.stderr)
receipt={'command':cmd,'exit_code':run.returncode,'seconds':time.monotonic()-start,
    'executable_sha256':sha(exe),'source_sha256':sha(P/'radial_bridge.cpp'),
    'build_latest_sha256':sha(P/('build-'+mode+'-latest.json')),
    'stdout_sha256':sha(A/'stdout.json'),'stderr_bytes':len(run.stderr.encode()),
    'scope':'Radial first-spatial derivative at m0/single oblique ray. r=.98 binding samples may exceed finite rb but remain inside scri; not the strict-inside-rb constraint-rate gate.',
    'screen':'Locked center n/T/U along every radial FD ray; O(2) covariance is separately gated.'}
(A/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert run.returncode==0,run.stderr
x=json.loads(run.stdout);convergence={}
for key in ('configuration_derivative_sequences','map_derivative_sequences'):
    rows=x[key];ordered=0;unclassified=0;bad=[]
    for k,e in enumerate(rows):
        fourth=any(e[j]>1e-10 and e[j+1]>1e-10 and e[j]>=8*e[j+1] for j in range(3))
        if fourth:ordered+=1
        elif max(e)<=2e-7:unclassified+=1
        else:bad.append({'row':k,'errors':e})
    convergence[key]={'rows':len(rows),'last_h_max':max(e[-1] for e in rows),
        'fourth_order_evidence_rows':ordered,'within_tolerance_order_unclassified_rows':unclassified,
        'unresolved_rows':bad,'all_final_h_pass':all(e[-1]<=2e-7 for e in rows)}
checks={'actual_configuration_rows':x['configuration_actual_full22_scaled']<=5e-11,
    'unused_mixed_extensions_exact_independent':x['finite_unused_extension_difference']==0,
    'input_raw22_normals':x['input_normal_scaled']<=5e-11,
    'output_raw22_normals':x['output_normal_scaled']<=5e-11,
    'final_h_and_convergence':all(v['all_final_h_pass'] and not v['unresolved_rows'] for v in convergence.values()),
    'empty_stderr':not run.stderr}
report={'passed_source_configuration_derivative_gate':all(checks.values()),'checks':checks,
    'numbers':{k:v for k,v in x.items() if 'sequences' not in k},'convergence':convergence,
    'receipt':receipt,'source_sha256':sha(__file__),'no_radial_operator_generated':True}
(A/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
assert report['passed_source_configuration_derivative_gate']
