"""Saved observer-JSON summary only; no array decode, import, or probe/native call."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
def pin(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def load(p):return json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
rows=[];inputs=[]
for suffix,case,abort in [('half','wave-map-half-N24-large-t2',.79280598958281545),('small','wave-map-N24-small-t2',1.0180338541661471)]:
 A=R/f'build-layer-research/reference-wave-map-partial-{suffix}-N24-held-20261009/attempts/{case}-001'
 files=[A/'receipt.json',A/'snapshot-observations.json',A/'history-observation.json',A/'protected-inputs-before.json',A/'protected-inputs-after.json',P/suffix/'receipt.json']
 inputs.extend(pin(p) for p in files)
 receipt,obs,hist=map(load,files[:3]);outer=load(files[-1]);last=obs[-1]
 assert receipt['observer_completed'] and receipt['accepted_native_run'] is False and receipt['native_returncode']==-6
 assert outer['returncode']==0 and outer['protected_unchanged']
 assert files[3].read_bytes()==files[4].read_bytes()
 assert len(obs)==receipt['saved_restart_files']==receipt['native_probe_calls']
 assert all(q['structural_and_native_diagnostic_call_completed'] and q['probe_called'] for q in obs)
 peaks=[]
 for field,name in enumerate(['H','Mcon','Zcon','Theta']):
  item=max(obs,key=lambda q:q['native_diagnostics']['rms_H_Mcon_Zcon_Theta'][field]);peaks.append({'field':name,'value':item['native_diagnostics']['rms_H_Mcon_Zcon_Theta'][field],'saved_time':item['time']})
 row={'case':case,'partial_diagnostic_only':True,'accepted_native_run':False,'original_returncode':-6,'original_target_time':2,'root_reported_unsaved_abort_time':abort,'abort_state_reconstructed':False,'saved_restart_files':len(obs),'new_native_evolution_calls':0,'native_probe_calls':receipt['native_probe_calls'],'saved_guard_failure_count':receipt['arrays_with_diagnostic_guard_failures'],'genuine_t0_available':receipt['genuine_t0_snapshot_available'],'last_saved_time':last['time'],'last_saved_cycle':last['cycle'],'last_saved_restart_header_dt':last['restart_header_dt'],'last_saved_history_dt':last['history_dt'],'last_H_Mcon_Zcon_Theta':last['native_diagnostics']['rms_H_Mcon_Zcon_Theta'],'last_shell_H_Mcon_Zcon_Theta':last['native_diagnostics']['shell_rms_H_Mcon_Zcon_Theta'],'history_native_scaled_error_max':max(q['history_scaled_rms_error'] for q in obs),'all_saved_active_fields_finite':all(sum(q['nonfinite_count25'])==0 for q in obs),'minimum_saved_alpha':min(q['alpha_min'] for q in obs),'minimum_saved_chi':min(q['chi_min'] for q in obs),'minimum_saved_conformal_metric_eigenvalue':min(q['minimum_conformal_metric_eigenvalue'] for q in obs),'minimum_saved_Penrose_metric_eigenvalue':min(q['minimum_Penrose_spatial_metric_eigenvalue'] for q in obs),'maximum_saved_det_error':max(q['native_diagnostics']['det_max'] for q in obs),'maximum_saved_trace_error':max(q['native_diagnostics']['trace_max'] for q in obs),'Omega_min':last['native_diagnostics']['Omega_min'],'observed_constraint_peaks_only':peaks,'before_after_protected_equal':True,'observer_seconds':receipt['seconds'],'outer_seconds':outer['seconds']}
 rows.append(row)
summary={'scope':'Saved partial observer JSON summaries only. All73 saved states precede two original native aborts. Finite/SPD/det/trace status is limited to saved active arrays; no unsaved abort-state admission, continuation or completed-run acceptance.','source':pin(__file__),'input_pins':inputs,'results':rows,'no_scientific_import_or_array_decode_or_new_probe':True}
for q in inputs:assert pin(Path(q['path']))==q
(P/'saved-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':pin(P/'saved-summary.json'),'results':rows},indent=2))
