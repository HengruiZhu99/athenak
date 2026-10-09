"""Independent stdlib-only saved-JSON/text review; no source/array/native queries."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,re,time
P=Path(__file__).resolve().parent;R=P.parents[2];B=R/'build-layer-research';D=R/'docs/validation';doc=R/'docs/hyperboloidal-reference-wave-map-final-matrix.md';overview=R/'docs/hyperboloidal-layer.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
failed=['wave-map-N16-large-t2','wave-map-N24-large-t2','wave-map-N32-large-t2','c0-N16-large-t2','wave-map-half-N24-large-t2','wave-map-N24-small-t2'];complete=['c0-N24-large-t2','c0-N32-large-t2','wave-map-N24-reference-long-t2'];batch=B/'wave-map-native-t2-root-20261009/batch001';rb=B/'reference-wave-map-t2-readback-held-20261009/attempts'
n32=D/'hyperboloidal-reference-wave-map-native-t2-N32-partial-observations-20261009';n32a=n32/'partial-N32/attempts/wave-map-N32-large-t2-001';v=D/'hyperboloidal-reference-wave-map-flat-IVP-values-20261009';v3=B/'reference-wave-map-t2-subset-comparison-held-v3-20261009';ba=next((v3/'attempts').iterdir())
inputs=[doc,overview,batch/'receipt.json',ba/'comparison.json',ba/'receipt.json',R/'src/z4c/z4c_hyperboloidal.cpp']
inputs += [batch/n/'launch-receipt.json' for n in failed+complete]+[batch/n/'native.stderr' for n in failed]
for n in complete:inputs += [rb/(n+'-001')/p for p in ('receipt.json','analysis/receipt.json','analysis/snapshots.json')]
inputs += [n32/p for p in ('catalog.json','root-saved-comparison/report.json','root-manual-N32/attempts/wave-map-N32-large-t2-001/receipt.json','partial-N32/attempts/wave-map-N32-large-t2-001/receipt.json','partial-N32/attempts/wave-map-N32-large-t2-001/snapshot-observations.json')]
inputs += [v/p for p in ('catalog.json','values-attempt001/receipt.json','values-invocation001/receipt.json','values-attempt001/checks.json','values-attempt001/values.json','values-attempt001/initial-data.json','values-attempt001/controls.json','values-attempt001/rays.json','values-source/values-recipe.json','root-independent-saved-readback/report003.json')]
inputs=list(dict.fromkeys(inputs));before=[pin(p) for p in inputs];dump(P/'inputs-before.json',before);text=doc.read_text();checks={};numbers={};corrections=[]
def check(label,test):
 checks[label]=bool(test)
 if not test:corrections.append(label)
def literal(value):return str(value) in text
try:
 f=load(batch/'receipt.json');check('final_batch9_false_aggregate',f['native_cases']==9 and f['passed_native_processes_and_provenance'] is False)
 times={}
 for name in failed:
  q=batch/name/'launch-receipt.json';n=load(q);check('failed_native/'+name,n['returncode']!=0 and n['passed_native_process_and_provenance'] is False and n['sources_before_after_equal'] and f['cases'][name]==sha(q))
  matches=re.findall(r'mesh_time=([^\s]+)',(batch/name/'native.stderr').read_text());check('unique_abort/'+name,len(matches)==1);times[name]=matches[0];check('document_abort/'+name,matches[0] in text)
 check('N32_abort_negative_lapse',Decimal(re.search(r'alpha=([^\s]+)',(batch/'wave-map-N32-large-t2/native.stderr').read_text()).group(1))<0)
 numbers['abort_times_exact_decimal']=times;completed={}
 for name in complete:
  n=load(batch/name/'launch-receipt.json');a=rb/(name+'-001');o=load(a/'receipt.json');r=load(a/'analysis/receipt.json');rows=load(a/'analysis/snapshots.json')
  check('completed_native/'+name,n['returncode']==0 and n['passed_native_process_and_provenance'] and n['sources_before_after_equal'] and f['cases'][name]==sha(batch/name/'launch-receipt.json'))
  check('completed_admission/'+name,o['returncode']==0 and o['passed_completed_t2_snapshot_gates'] and o['protected_before_after_equal'] and r['passed_saved_snapshot_finite_and_diagnostic_gates'] and r['protected_inputs_before_after_equal'] and r['saved_arrays']==len(rows)==81 and rows[-1]['time']==2 and r['snapshots_sha256']==sha(a/'analysis/snapshots.json') and o['analyzer_receipt_sha256']==sha(a/'analysis/receipt.json'))
  one=min(rows,key=lambda x:abs(x['time']-1));completed[name]={'saved_states':len(rows),'near_t1_time':one['time'],'t2_RMS':rows[-1]['rms_H_Mcon_Zcon_Theta'],'maximum_saved_deviation':max(max(x['reference_deviation_max25']) for x in rows),'maximum_saved_RMS':[max(x['rms_H_Mcon_Zcon_Theta'][j] for x in rows) for j in range(4)]}
  if name.startswith('c0'):
   label='C0 N24' if 'N24' in name else 'C0 N32';m=re.search(r'\| '+label+r' \| ([^|]+) \| ([^|]+) \|',text);check('completed_table_exists/'+name,m is not None)
   if m:
    check('near_t1_exact/'+name,Decimal(m.group(1).strip())==Decimal(str(one['time'])))
    printed=[Decimal(x.strip()) for x in m.group(2).split('/')];check('rounded_t2_norms/'+name,len(printed)==4 and all(abs(x-Decimal(str(y)))<=max(Decimal('1e-10'),abs(Decimal(str(y)))*Decimal('5e-8')) for x,y in zip(printed,completed[name]['t2_RMS'])))
  else:
   check('stationary_max_deviation',literal(completed[name]['maximum_saved_deviation']))
   check('stationary_max_RMS',all(literal(x) for x in completed[name]['maximum_saved_RMS']))
 numbers['completed']=completed
 book=load(ba/'comparison.json');check('bookkeeping_success',load(ba/'receipt.json')['passed_saved_subset_bookkeeping'])
 check('bookkeeping_fixed6_3_nonepending',set(book['native_failed_cases'])==set(failed) and set(book['admitted_completed_cases'])==set(complete) and book['pending_cases']==[])
 check('bookkeeping_matrix_t6_false',book['matrix_complete_and_admitted'] is False and book['t6_admission'] is False and book['t12_admission'] is False)
 check('all3_required_not_evaluable',len(book['comparisons'])==3 and all(x['status']=='not_evaluable' for x in book['comparisons'].values()))
 check('document_nonevaluable_scope','`not_evaluable`' in text and 'No t6/t12 extension' in text)
 nc=load(n32/'root-saved-comparison/report.json');no=load(n32a/'receipt.json');nr=load(n32a/'snapshot-observations.json');check('N32_64_1600_savedguards',len(nr)==nc['states']==no['saved_restart_files']==64 and nc['finite_extrema_pairs']==1600 and nc['saved_guards_failed']==no['arrays_with_diagnostic_guard_failures']==0 and no['observer_completed'] and not no['accepted_native_run'] and nc['partial_diagnostic_only'])
 last=nr[-1];diag=last['native_diagnostics'];check('N32_last_norm_time_exact',last['time']==nc['last_saved_time'] and diag['rms_H_Mcon_Zcon_Theta']==nc['last_saved_H_M_Z_Theta'] and all(literal(x) for x in [last['time']]+diag['rms_H_Mcon_Zcon_Theta']))
 frac=diag['shell_r_ge_09_count']/diag['active_count']*(diag['shell_rms_H_Mcon_Zcon_Theta'][2]/diag['rms_H_Mcon_Zcon_Theta'][2])**2;check('N32_Zshell_fraction',abs(frac-nc['last_saved_squared_Z_shell_fraction'])<1e-14 and literal(nc['last_saved_squared_Z_shell_fraction']))
 numbers['N32']={'states':len(nr),'manual_extrema_pairs':nc['finite_extrema_pairs'],'last_time':last['time'],'last_RMS':diag['rms_H_Mcon_Zcon_Theta'],'squared_Z_shell_fraction':frac}
 vs=load(v/'values-attempt001/receipt.json');vo=load(v/'values-invocation001/receipt.json');vr=load(v/'root-independent-saved-readback/report003.json');vc=load(v/'values-attempt001/checks.json');recipe=load(v/'values-source/values-recipe.json')
 counts={'coarea_rows':96,'initial_rows':56,'control_rows':72,'ray_rows':18};check('values_counts282_96_56_72_18',len(vc)==vs['checks']==282 and all(vs[k]==x for k,x in counts.items()))
 for filename,count in [('values.json',96),('initial-data.json',56),('controls.json',72),('rays.json',18)]:check('values_rowlist/'+filename,len(load(v/'values-attempt001'/filename))==count)
 check('values_gates_and_independent_readback',vs['passed_scalar_values_only'] and not vs['accepted_native'] and vs['sources_unchanged'] and vo['returncode']==0 and vo['passed_outer_process'] and vr['passed_saved_readback'] and vr['all_fixed_tolerances_rechecked'])
 check('values_all282_original_thresholds',all(x['passed'] and Decimal(x['error']).is_finite() and Decimal('0')<=Decimal(x['error'])<=Decimal(x['tolerance']) for x in vc))
 maxima={field:max(Decimal(x['error']) for x in vc if x['name'].startswith('coarea_convergence/'+field+'/')) for field in ('u','phi')};check('document_values_refinement_bounds',maxima['u']<Decimal('6.06e-40') and maxima['phi']<Decimal('3.61e-38') and '6.06e-40' in text and '3.61e-38' in text)
 check('values_precisions_and_native_pulse',recipe['precisions']==[80,110] and recipe['pulse_width']=='.35' if isinstance(recipe['pulse_width'],str) else recipe['precisions']==[80,110] and recipe['pulse_width']==.35)
 numbers['values']={'checks':len(vc),'row_counts':counts,'coarea_refinement_max':{k:str(x) for k,x in maxima.items()},'precisions':recipe['precisions'],'pulse_width':recipe['pulse_width']}
 scope_phrases=['no\nanalytic derivative, Jacobian, inverse-map','not the corresponding\nnative event','not reconstruction of the unsaved abort','Omega is nonmonotone','No black-hole fixed-point RHS is\nsubtracted','arrays, executables, objects, NPY/NPZ/JSONL']
 check('literal_scope_limits',all(x in text for x in scope_phrases))
 links=re.findall(r'\[[^\]]*\]\(([^)]+)\)',text);pending=[]
 for target in links:
  if '://' in target:check('portable_link/'+target,False);continue
  q=(doc.parent/target.split('#')[0]).resolve()
  allowed=q==(D/'hyperboloidal-reference-wave-map-native-t2-final-completed-controls-20261009/README.md').resolve()
  check('portable_link/'+target,q.exists() or allowed)
  if not q.exists():pending.append(str(q))
 check('overview_link',doc.name in overview.read_text());numbers['pending_portable_links']=pending
except BaseException as exc:corrections.append(type(exc).__name__+': '+str(exc))
after=[pin(p) for p in inputs];check('all_input_bytes_unchanged',after==before);dump(P/'inputs-after.json',after);dump(P/'numbers.json',numbers)
result={'passed_independent_saved_number_source_scope_review':not corrections,'draft_sha256':before[0]['sha256'],'review_source_sha256':sha(__file__),'checked_claims':len(checks),'checks':checks,'corrections':corrections,'source_inputs':len(inputs),'inputs_unchanged':after==before,'new_scientific_imports_queries_array_decodes_native_steps':0,'scope':'Independent saved-JSON/Decimal/text arithmetic only; no scientific rerun, stability/causality/adoption claim. Final-native archive link may remain pending until exact collector release.'};dump(P/'receipt.json',result);print(json.dumps(result,indent=2));assert not corrections,'review corrections/failure preserved'
