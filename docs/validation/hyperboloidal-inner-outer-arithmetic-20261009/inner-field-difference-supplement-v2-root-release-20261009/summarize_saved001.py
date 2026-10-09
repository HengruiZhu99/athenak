from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
OWNER=Path(json.loads((HERE/'release.json').read_text())['owner'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def load(p):
 if p.stat().st_size>1048576:raise RuntimeError('oversized saved JSON requires metadata-only collection')
 return json.loads(p.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
pins={};summaries={}
for build in ['release','debug']:
 outer=HERE/(build+'-invocation001/receipt.json');receipt=OWNER/'attempts'/('Release001' if build=='release' else 'Debug001')/'receipt.json'
 o=load(outer);r=load(receipt)
 if not(o['accepted_local_gate'] and o['completed'] and o['returncode']==0 and o['inputs_unchanged'] and o['child_receipt_sha256']==sha(receipt) and r['completed'] and r['passed'] and r['returncode']==0 and r['inputs_unchanged']):raise RuntimeError('unsuccessful saved unit gate '+build)
 report=Path(r['report']['path'])
 if pin(report)!=r['report']:raise RuntimeError('saved report differs')
 q=load(report)
 if not(q['passed'] and q['records']==129 and q['counts']=={'witness':18,'negative-old-near':3,'near-bound':108} and q['component_target_checks']==684 and len(q['errors'])==684):raise RuntimeError('saved fixed registry differs')
 if not(q['exact_zero_primal_nonzero_gradient_dual_covered'] and q['negative_controls_all_three_lost_normal_primals_zero'] and q['branch_endpoint_and_sides_verified'] and not q['floating_C1_continuity_claim'] and not q['main_suite_or_dV_changed'] and not q['arbitrary_nonlinear_gauge_or_evolution_acceptance']):raise RuntimeError('saved scope differs')
 if r['executable_before']!=r['executable_after']:raise RuntimeError('saved executable changed')
 for p in [outer,receipt,report]:pins[str(p)]=pin(p)
 summaries[build]={k:q[k] for k in ['passed','records','counts','component_target_checks','relative_nonzero_threshold','absolute_zero_threshold','maximum_absolute_error','maximum_relative_error','exact_zero_primal_nonzero_gradient_dual_covered','negative_controls_all_three_lost_normal_primals_zero','branch_endpoint_and_sides_verified']}
 summaries[build].update(seconds=o['seconds'],executable=r['executable_after'],child_receipt=pin(receipt),report=pin(report))
if any(pin(Path(p))!=v for p,v in pins.items()):raise RuntimeError('saved evidence changed')
dest=HERE/'saved-unit-summary001.json'
with dest.open('x') as s:s.write(json.dumps(dict(passed_saved_summary=True,scope='saved finite129 scalar arithmetic units only; actual source003 remainsFAILED',builds=summaries,pins=pins,reader=pin(Path(__file__)),no_compilation_query_or_target_recomputation=True),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed_saved_summary=True,builds={k:{n:v[n] for n in ['passed','records','component_target_checks','seconds']} for k,v in summaries.items()},summary=pin(dest))))
