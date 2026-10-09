"""One-shot N24 stopped-run scalar comparison; no arrays, kernel or evolution."""
from pathlib import Path
import hashlib,json,math,re
from decimal import Decimal
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
recipe=load(P/'recipe.json')
pins={**recipe['input_pins'],str(P/'recipe.json'):sha(P/'recipe.json'),str(Path(__file__).resolve()):sha(__file__)}
for f,h in pins.items():assert sha(f)==h,f
out=P/'attempt001';out.mkdir(exist_ok=False)
(out/'source.py').write_bytes(Path(__file__).read_bytes())
(out/'pins-before.json').write_text(json.dumps(pins,indent=2)+'\n')
cases={};summaries=[];failures=[]
for spec in recipe['cases']:
    a=load(spec['owner_observations']);b=load(spec['manual_rows'])
    ar=load(spec['owner_receipt']);br=load(spec['manual_receipt'])
    assert ar['observer_completed'] is True and ar['accepted_native_run'] is False and ar['partial_diagnostic_only'] is True
    assert br['diagnostic_protocol_completed'] is True and br['accepted_native_run'] is False
    assert ar['protected_before_after_equal'] is True and br['before_after_equal'] is True
    assert len(a)==len(b) and all(not q['diagnostic_guard_failures'] for q in a)
    history_error=0.;extrema_pairs=0
    for x,y in zip(a,b):
        assert x['rst_sha256']==y['sha256'] and x['time']==y['time'] and x['cycle']==y['cycle']
        assert x['finite_extrema25']==y['finite_extrema25'] and y['saved_field_guards_satisfied'] is True
        extrema_pairs+=25
        n=x['native_diagnostics'];hist=y['original_native_history_H_Mcon_Zcon_Theta']
        history_error=max(history_error,max(abs(u-v) for u,v in zip(n['rms_H_Mcon_Zcon_Theta'],hist)))
        assert history_error<=2e-11
        for k in range(4):
            total=n['rms_H_Mcon_Zcon_Theta'][k]
            bins=sum(z['squared_fraction4'][k] for z in n['radial_bins'] if z['rlo']>=.9)
            shell=n['shell_rms_H_Mcon_Zcon_Theta'][k]
            if total:
                reconstructed=(shell/total)**2*n['shell_r_ge_09_count']/n['active_count']
                assert abs(reconstructed-bins)<=2e-12
    last=a[-1];n=last['native_diagnostics']
    summary={'tag':spec['tag'],'case':spec['case'],'saved_states':len(a),'extrema_pairs_exact':extrema_pairs,'history_probe_max_absolute':history_error,'last_saved_time':last['time'],'last_norms':n['rms_H_Mcon_Zcon_Theta'],'last_alpha_min':last['alpha_min'],'last_chi_min':last['chi_min'],'last_metric_eigen_min':last['minimum_conformal_metric_eigenvalue'],'last_Z_shell_squared_fraction':sum(z['squared_fraction4'][2] for z in n['radial_bins'] if z['rlo']>=.9),'saved_field_guards_passed':True,'accepted_native_run':False}
    summaries.append(summary);cases[spec['tag']]=a
    text=Path(spec['native_stderr']).read_text()
    m=re.search(r'mesh_time=(\S+) cycle=(\d+) xyz=\(([^)]*)\) Omega=(\S+)',text);assert m
    failures.append({'tag':spec['tag'],'time_decimal':m[1],'cycle':int(m[2]),'xyz':m[3],'Omega':m[4]})
assert len(cases['standard'])==len(cases['half'])==32
pairs=[]
for a,b in zip(cases['standard'],cases['half']):
    assert abs(a['time']-b['time'])<=recipe['time_pair_tolerance']
    x=a['native_diagnostics']['rms_H_Mcon_Zcon_Theta'];y=b['native_diagnostics']['rms_H_Mcon_Zcon_Theta']
    pairs.append({'standard_time':a['time'],'half_time':b['time'],'standard_norms':x,'half_norms':y,'relative_differences':[(abs(v-u)/abs(u) if abs(u)>1e-10 else None) for u,v in zip(x,y)],'absolute_differences':[abs(u-v) for u,v in zip(x,y)],'interpretation':'matched saved partial states only; no t2 timestep gate or continuum convergence order'})
report={'passed_scalar_readback':True,'partial_diagnostic_only':True,'accepted_native_run':False,'cases':summaries,'all_saved_independent_field_pairs':sum(q['saved_states'] for q in summaries),'failure_metadata':failures,'half_minus_standard_abort_time_decimal':str(Decimal(failures[1]['time_decimal'])-Decimal(failures[0]['time_decimal'])),'same_first_reported_cell':len({(q['xyz'],q['Omega']) for q in failures})==1,'matched_standard_half_saved_pairs':pairs,'scope':recipe['scope'],'native_t2_complete':False,'t6_admitted':False,'new_native_steps':0,'new_probe_calls':0}
for f,h in pins.items():assert sha(f)==h,f
(out/'pins-after.json').write_text(json.dumps(pins,indent=2)+'\n')
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'saved_pairs':report['all_saved_independent_field_pairs'],'last_standard_half_pair':pairs[-1],'cases':summaries,'report_sha256':sha(out/'report.json')}))
