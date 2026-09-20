from pathlib import Path
import hashlib,json,re
P=Path(__file__).resolve().parent
out={'scope':'Authorized local three-cycle static-SMR execution and saved-state validator checks only; no long evolution or Aurora job.','validator_sha256':hashlib.sha256((P.parent/'check_smr_trumpet_checkpoint.py').read_bytes()).hexdigest(),'cases':{}}
for name in ['zero','pulse']:
 p=P/name;valid=json.loads((p/'checkpoint-validity.json').read_text());ex=json.loads((p/'execution.json').read_text());log=(p/'run.log').read_text();assert valid['passed'] and ex['exit_code']==0 and 'Terminating on cycle limit' in log;assert not any(s in log for s in ['Z4C_INVALID_STATE','An error occurred during the primitive solve:','### FATAL ERROR'])
 assert valid['cycle']==3 and abs(valid['time_M']-.075)<1e-15 and valid['checkpoint_dt']==.025
 progress=[tuple(map(float,m))for m in re.findall(r'elapsed=([\deE+.-]+) cycle=(\d+) time=([\deE+.-]+) dt=([\deE+.-]+)',log)]
 assert len(progress)==4 and all(abs(row[3]-.025)<1e-15 for row in progress)
 parse=[]
 for h in p.glob('*.hst'):
  lines=h.read_text().splitlines();labels=re.findall(r'\[\d+\]=(\S+)',next(l for l in lines if'[1]='in l));rows=[[float(v)for v in l.split()]for l in lines if l.strip()and not l.startswith('#')];assert all(all(__import__('math').isfinite(v)for v in row)for row in rows)
  if'bad-metric'in labels:assert all(row[labels.index('bad-metric')]==0 for row in rows)
  parse.append({'name':h.name,'rows':len(rows),'all_finite':True})
 fields=valid['residual_field_max_active'];out['cases'][name]={'execution':ex,'time_M':valid['time_M'],'cycle':valid['cycle'],'actual_step_dt_from_log_M':.025,'checkpoint_dt':valid['checkpoint_dt'],'evolution_seconds_per_cycle_from_progress':(progress[-1][0]-progress[0][0])/3,'all_rank_headers_match':valid['matching_cohort_headers'],'all_payload_finite':valid['all_payload_finite'],'invalid_full_ghost_metric_cells':valid['invalid_metric_cells_including_ghosts'],'mesh_audit_geometry_matches':valid['mesh_audit_geometry_matches'],'raw_metric_minima':valid['minimum_raw_full'],'residual_exactly_zero':valid['residual_exactly_zero'],'residual_maxima_active':{'chi':fields[0],'Khat':fields[7],'Theta':fields[17],'lapse':fields[18],'shift_max_component':max(fields[19:22])},'histories':parse,'latest_checkpoint_files':valid['files']}
initial=json.loads((P/'pulse/initial-checkpoint-validity.json').read_text());maxima=initial['residual_field_max_including_ghosts'];assert initial['passed'] and maxima[18]>0 and all(v==0 for i,v in enumerate(maxima)if i!=18)
assert out['cases']['zero']['residual_exactly_zero'];assert not out['cases']['pulse']['residual_exactly_zero'] and all(out['cases']['pulse']['residual_maxima_active'][k]>0 for k in ['chi','Khat','Theta'])
out['initial_pulse']={'only_lapse_nonzero_including_ghosts':True,'max_lapse_residual':maxima[18],'Theta_exactly_zero':maxima[17]==0};out['validator_regression']=json.loads((P/'validator-regression.json').read_text())
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print({n:(c['execution']['application_elapsed_seconds'],c['evolution_seconds_per_cycle_from_progress'])for n,c in out['cases'].items()})
