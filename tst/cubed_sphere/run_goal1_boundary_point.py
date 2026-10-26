#!/usr/bin/env python3
"""Run saved-jet diagnostics once; never launch or repeat native trajectories."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
from compare_closure_field_norms import read
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output
protocol=json.loads((out/'protocol-v1.json').read_text());build=json.loads((out/'build.json').read_text());assert build['exit_code']==0
work=Path('build-goal1-minkowski-20261018/full-boundary-point-v1');binary=(work/'boundary_point').resolve()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert h(binary)==build['binary_sha256']
cases=[('native-stationary4','canonical-stationary'),('native-pulse8','canonical-resume8')]
runs=[];allrows=[]
for label,name in cases+[('synthetic',None)]:
    stdout=out/(label+'.jsonl');stderr=out/(label+'.stderr');assert not stdout.exists(),'Reuse this result; choose a new protocol/output for changed code.'
    command=[str(binary)];payload={}
    if name:
        case=Path('build-goal1-minkowski-20261018/canonical-ko-v2/native-admission')/(name+'-segment0');data=read(case)
        ref=case/'cubed_sphere_reference_before.bin';pin=data['receipt']['reference_payloads'][ref.name];assert h(ref)==pin['sha256']
        command+=[str((case/'probe_checkpoint.bin').resolve()),str(ref.resolve()),label];payload={'checkpoint':data['payload'],'reference':pin,'time':data['time'],'cycle':data['cycle'],'source_binary':data['tag']}
    t=time.time()
    with stdout.open('w') as f,stderr.open('w') as e:r=subprocess.run(command,stdout=f,stderr=e)
    receipt={'label':label,'command':command,'exit_code':r.returncode,'seconds':time.time()-t,'payloads':payload,'stdout_sha256':h(stdout),'stderr_sha256':h(stderr)}
    runs.append(receipt);(out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');assert r.returncode==0,receipt
    rows=[json.loads(s) for s in stdout.read_text().splitlines() if s.startswith('{')]
    assert len(rows)==(384 if name else 18),(label,len(rows));allrows.extend(rows)

budget=protocol['budgets'];groups=[]
for label in ['native-stationary4','native-pulse8','synthetic-reference','synthetic-off-constraint']:
    rows=[r for r in allrows if r['kind']==label];maxima={key:max(abs(r[key]) for r in rows) for key in rows[0] if key not in ['kind','case','K','R']}
    failures=[];ports=[];conditions=[]
    for r in rows:
        checks={'normal_row_error':'analytic_normal_row_scaled','corrected_constraint_error':'corrected_native_constraint_and_free_scaled',
          'free_condition_error':'corrected_native_constraint_and_free_scaled','outgoing_change':'outgoing_change_scaled',
          'metric_derivative_trace_change':'algebraic_trace_change_scaled','A_trace_change':'algebraic_trace_change_scaled',
          'Gamma_time_variation_error':'connection_time_variation_scaled'}
        for quantity,b in checks.items():
            if abs(r[quantity])>budget[b]:failures.append({'case':r['case'],'quantity':quantity,'value':r[quantity],'budget':budget[b]})
        assert r['configuration_value_change']==0
        K=np.array(r['K']);norm=np.linalg.norm(K,axis=1);sv=np.linalg.svd(K/norm[:,None],compute_uv=False);conditions.append(float(sv[0]/sv[-1]))
        R=np.array(r['R']);beta=r['beta'];c=r['c'];ain=c+beta;aout=c-beta
        F=np.block([[-ain*np.eye(10),ain*(R+np.eye(10))],[ain*(R.T+np.eye(10)),-aout*np.eye(10)-ain*(R+R.T)]])
        ports.append(float(np.linalg.eigvalsh((F+F.T)/2)[-1]/max(1,abs(ain),abs(aout))))
    if label=='native-stationary4' and maxima['incoming_correction']>budget['stationary_incoming_correction_absolute']:failures.append({'quantity':'stationary incoming correction','value':maxima['incoming_correction']})
    groups.append({'kind':label,'points':len(rows),'maxima':maxima,'maximum_row_equilibrated_condition':max(conditions),'maximum_frozen_right_port_rate':max(ports),
      'positive_port_cases':sum(x>1e-10 for x in ports),'passed_point_identities':not failures,'failures':failures})
negative=max(r['omitted_full_forcing'] for r in allrows if r['kind']=='synthetic-off-constraint')>=budget['omitted_full_forcing_negative_minimum']
report={'scope':protocol['scope'],'point_checks_pass':all(r['passed_point_identities'] for r in groups),'omitted_full_forcing_negative_detected':negative,
 'groups':groups,'points':len(allrows),'all_native_outer_faces_edges_corners':True,
 'native_SAT_installed':False,'full_time_compatibility_admitted':False,'exact_scri_admitted':False,'new_native_trajectories':0,
 'interpretation':'Full native-jet point value correction is not a reduction-compatible time update. Nonzero Theta/Z time residuals and naive value-only H/M defects remain; full lower/tangential absorbing hierarchy and common mapped source/interface energy are open.',
 'source_sha256':h(Path(__file__)),'binary_sha256':h(binary),'runs_sha256':h(out/'runs.json')}
(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in report.items() if k not in ['groups','interpretation']});print([(g['kind'],g['passed_point_identities'],g['failures'],g['maximum_frozen_right_port_rate']) for g in groups])
