"""Root independent saved-scalar arithmetic and final-document review."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,re
P=Path(__file__).resolve().parent;R=P.parents[1]
S=R/'build-layer-research/boundary/reference-wave-map-N24-controls-document-draft003-20261009'
DOC=R/'docs/hyperboloidal-reference-wave-map-native-controls.md'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
assert sha(DOC)=='170aa2f1f74f4b4da4fa54dfa748882d1246fd0c442e562e1e2ed8c353e7a54d'
pins=load(S/'input-pins.json')['inputs']
for row in pins:assert sha(row['path'])==row['sha256'],row['path']
numbers=load(S/'numbers.json');text=DOC.read_text()
CP=R/'build-layer-research/wave-map-native-N24-partial-comparison-root-20261009'
recipe=load(CP/'recipe.json');root=load(CP/'attempt001/report.json')
count=0;mins={};norms={};abort={}
for spec in recipe['cases']:
 tag=spec['tag'];rows=load(spec['owner_observations']);manual=load(spec['manual_receipt']);owner=load(spec['owner_receipt'])
 assert not owner['accepted_native_run'] and owner['observer_completed'] and owner['protected_before_after_equal']
 assert manual['saved_guard_failures']==0 and manual['before_after_equal']
 n=next(x for x in numbers['wave_cases'] if x['tag']==tag)
 assert len(rows)==n['saved_states'] and rows[-1]['time']==n['last_saved_time']
 assert rows[-1]['native_diagnostics']['rms_H_Mcon_Zcon_Theta']==n['last_norms'];norms[tag]=n['last_norms'];count+=len(rows)
 keys=['alpha_min','chi_min','minimum_conformal_metric_eigenvalue','minimum_Penrose_spatial_metric_eigenvalue']
 mins[tag]={k:min(x[k] for x in rows) for k in keys};assert mins[tag]==numbers['wave_saved_limits'][tag]
 assert ' / '.join(format(v,'.8g') for v in norms[tag]) in text
 assert ' | '.join(format(mins[tag][k],'.8g') for k in keys) in text
 d=rows[-1]['native_diagnostics'];f=d['shell_r_ge_09_count']/d['active_count']*(d['shell_rms_H_Mcon_Zcon_Theta'][2]/d['rms_H_Mcon_Zcon_Theta'][2])**2
 assert abs(f-n['last_Z_shell_squared_fraction'])<2e-12
 stderr=Path(spec['native_stderr']).read_text();m=re.search(r'mesh_time=([^ ]+) cycle=(\d+) xyz=\(([^)]+)\) Omega=([^ ]+)',stderr);assert m
 ref=next(x for x in numbers['failures'] if x['tag']==tag)
 assert m[1]==ref['time_decimal'] and int(m[2])==ref['cycle'] and m[3]==ref['xyz'] and m[4]==ref['Omega'];abort[tag]=Decimal(m[1]);assert m[1] in text
assert count==105 and 25*count==2625
assert str(abort['half']-abort['standard'])==numbers['half_minus_standard_abort_time_decimal']
relative=[abs(a-b)/abs(a) for a,b in zip(norms['standard'],norms['half'])]
assert relative==numbers['last_matched']['relative_differences'] and max(relative)<.00085
A=R/'build-layer-research/reference-wave-map-t2-readback-held-20261009/attempts/c0-N24-large-t2-001'
wrapper=load(A/'receipt.json');analyzer=load(A/'analysis/receipt.json');rows=load(A/'analysis/snapshots.json');c=numbers['C0']
assert wrapper['returncode']==0 and wrapper['passed_completed_t2_snapshot_gates'] and wrapper['protected_before_after_equal']
assert analyzer['passed_saved_snapshot_finite_and_diagnostic_gates'] and analyzer['protected_inputs_before_after_equal']
assert len(rows)==c['saved_states']==81 and rows[-1]['time']==c['final_time']==2
assert rows[-1]['rms_H_Mcon_Zcon_Theta']==c['final_norms'];assert ' / '.join(format(v,'.8g') for v in c['final_norms']) in text
for k,v in c['saved_limits'].items():assert min(x[k] for x in rows)==v
assert max(x['det_max'] for x in rows)==c['max_det_error'] and max(x['trace_max'] for x in rows)==c['max_trace_error']
assert max(x['history_scaled_rms_error'] for x in rows)==c['history_diagnostic_max_error']
assert {x['restart_header_dt'] for x in rows[:-1]}=={c['nonterminal_saved_dt_min']} and c['nonterminal_saved_dt_min']==c['nonterminal_saved_dt_max']
assert rows[-1]['restart_header_dt']==c['terminal_clipped_dt'] and c['terminal_clipped_dt']<1e-10
for name,row in numbers['failure_capsules'].items():assert sha(row['path'])==row['sha256'] and row['sha256'] in text
assert 'does not identify the cause' not in text or 'do not identify the cause' in text
assert 'not independently recompute differentiated H/M/Z' in text and 'not a completed t2 timestep-convergence gate' in text
assert 'outside this frozen N24-controls archive' in text and 'No private candidate is adopted' in (R/'docs/hyperboloidal-layer.md').read_text()
for row in pins:assert sha(row['path'])==row['sha256']
report={'passed_saved_number_source_review':True,'final_document_sha256':sha(DOC),'review_source_sha256':sha(__file__),'verified_saved_input_files':len(pins),'saved_wave_states':105,'manual_extrema_pairs':2625,'C0_states':81,'all_minima_norms_abort_metadata_and_dt_checked':True,'scope_review':'Failed targets remain failed; saved/unsaved states distinguished; no causal or continuum claim; public C0 and global wave-map helper correctly distinguished. Portable archived PNG visually checked separately.','new_scientific_imports_queries_array_decodes':0}
with (P/'report001.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(report))
