#!/usr/bin/env python3
"""Only stdlib source/AST/metadata review; no target import or array decode."""
from pathlib import Path
import ast, hashlib, json, re, sys, os
P=Path(__file__).resolve().parent

def need(v,msg):
    if not v:raise ValueError(msg)
def sha(q):
    h=hashlib.sha256()
    with Path(q).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(q):return json.loads(Path(q).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
need(sys.flags.isolated==1 and sys.dont_write_bytecode and sys.flags.optimize==0,'isolated nonoptimized reviewer runtime')
cap=load(P/'capture.json');ev=load(P/'evidence-capture.json')
for x in cap['files']+ev['files']:
    q=P/x['copy'];need(q.stat().st_size==x['bytes'] and sha(q)==x['sha256'],'capture drift')
    need(Path(x['origin'] if 'origin' in x else x['path']).stat().st_size==x['bytes'] and sha(x.get('origin',x.get('path')))==x['sha256'],'original drift')
src=(P/'inputs/verify_retained.py').read_text();old=(P/'inputs/history/v9-verify_retained.py').read_text();diff=(P/'inputs/five-outer-measure-and-audits.diff').read_text()
current=src.splitlines(keepends=True);lines=diff.splitlines(keepends=True);restored=[];at=0;i=0;hunks=0
while i<len(lines):
    match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[i])
    if not match:i+=1;continue
    start=int(match.group(3))-1;need(start>=at,'ordered hunk');restored.extend(current[at:start]);at=start;i+=1;hunks+=1
    while i<len(lines) and not lines[i].startswith('@@'):
        row=lines[i]
        if row.startswith(('---','+++')):break
        if row.startswith((' ','+')):
            need(at<len(current) and current[at]==row[1:],'hunk exact candidate context');at+=1
        if row.startswith((' ','-')):restored.append(row[1:])
        elif not row.startswith(('+','\\')):raise ValueError('unexpected diff marker')
        i+=1
restored.extend(current[at:]);reverse=''.join(restored)
need(reverse==old,'independent reverse bytes')
need(ast.dump(ast.parse(reverse),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False),'independent reverse AST')
a=ast.parse(src);b=ast.parse(old)
newops=[];oldops={}
for node in ast.walk(a):
    if isinstance(node,ast.AugAssign) and isinstance(node.value,ast.Call):
        c=node.value
        if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Subscript) and isinstance(c.func.value.value,ast.Name) and c.func.value.value.id=='outer_measure_arithmetic':newops.append(node)
for node in ast.walk(b):
    if isinstance(node,ast.AugAssign) and isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Mult) and isinstance(node.value.left,ast.Name) and node.value.left.id=='measure':
        need(isinstance(node.target,ast.Name),'original accumulator name');oldops[node.target.id]=node
need(len(newops)==5 and set(oldops)=={'E','Ks','Kw','G','loads'},'exact five sites')
for node in newops:
    c=node.value;name=node.target.id
    need(len(c.args)==3 and isinstance(c.args[0],ast.Name) and c.args[0].id=='measure','same rounded measure')
    need(ast.dump(c.args[1],include_attributes=False)==ast.dump(oldops[name].value.right,include_attributes=False),'inner operand/order exact '+name)
    need(isinstance(c.args[2],ast.Constant) and c.args[2].value=='outer_measure:'+name,'operation label '+name)
    need(isinstance(c.func.value.slice,ast.Constant) and c.func.value.slice.value==name,'distinct receiver '+name)
stores=[n for n in ast.walk(a) if isinstance(n,ast.Name) and n.id=='outer_measure_arithmetic' and isinstance(n.ctx,ast.Store)]
need(len(stores)==1,'adapter not shadowed')
r=load(P/'inputs/recipe.json');r9=load(P/'inputs/history/v9-recipe.json');idx=load(P/'inputs/source-index.json')
need(sha(P/'inputs/source-index.json')=='a67f587c07e18ee98e6aaaedfba1249fc334f51905a84c64aa3751df150408ff','exact source index')
need(sha(P/'inputs/recipe.json')=='05ba2b41afdcb39aeecfc7dbf659670b3e0011d7a6635a25cf42106f617e3fb2','exact recipe')
need(sha(P/'inputs/verify_retained.py')=='ee7de62d0faa1f221a636d03d1ba43901cb4b28c0bfbc63ce946fe9c5fc10b11','exact verifier')
need(r['gates']==r9['gates'],'science gate dictionary unchanged')
for key in ['cases','context']:
    need(r[key]==r9[key],key+' scientific bindings unchanged')
need(r['environment']==r9['environment'],'original pinned environment unchanged')
need(all('-B' in x and '-s' in x for x in r['future_commands']),'reviewed no-bytecode/user-site commands')
need(set(r['cases'])=={'primary','radial_pair','angular_pair'},'fresh three cases')
need(r['outer_measure_correction']['all3_fresh_v10_readbacks_required'] is True and r['outer_measure_correction']['generator_admission'] is False,'no inherited eligibility/generator')
for name,h in [('tiny_normalization.py','d27a49639ffc20de508e8b72bf63cfea68b6a703fdcd130f25c8689291a59272'),('fast_weighting.py','5781ed73f9da31e0be3164e141125ad7b0d436333ecacee1ecbc124cf61ab095'),('column_norm.py','f768c26318712035452c6d45d5a9fd88b5c147a16161b9ac671317ebf9bf0f01')]:need(sha(P/'inputs'/name)==h,'unchanged tested helper '+name)
# Gather hashes without decoding any scientific payload.
pins=dict(r['pins'])
for row in idx['files']:
    need(row['path'] not in pins or pins[row['path']]==row['sha256'],'source pin conflict')
    pins[row['path']]=row['sha256']
owner=load(r['context']['owner_source_index'])
for row in owner['files']+owner['external_inputs']:
    need(row['path'] not in pins or pins[row['path']]==row['sha256'],'owner pin conflict')
    pins[row['path']]=row['sha256']
need(all(p in pins and pins[p]==h for p,h in r9['pins'].items()),'old pinned inputs preserved')
for path,h in pins.items():need(sha(path)==h,'protected input drift '+path)
scipy=next(load(P/x['copy']) for x in ev['files'] if x['label']=='runtime:manifest')
need(len(scipy['files'])==1332 and scipy['bytecode_excluded'] is True and scipy['imports_executed'] is False,'SciPy runtime inventory count/scope')
for runtime_path,runtime_hash in scipy['files'].items():
    need(runtime_path in r['pins'] and r['pins'][runtime_path]==runtime_hash,'SciPy protected before imports')
    need('__pycache__' not in Path(runtime_path).parts and not runtime_path.endswith(('.pyc','.pyo')),'runtime bytecode excluded')
need(scipy['roots']==r['runtime_inventory_addendum']['roots'],'SciPy actual package roots')
need(r['environment']['PYTHONPATH'].split(':')[0] in str(Path(scipy['roots'][0]).parent),'actual first-PYTHONPATH runtime')
D={x['label']:load(P/x['copy']) for x in ev['files'] if Path(x['copy']).suffix=='.json'}
d=D['diagnostic_receipt'];v=D['diagnostic_result'];s=D['saved_review_receipt'];f=D['preserved_v9_radial_failure']
need(d['completed'] is True and d['returncode']==0 and d['inputs_unchanged'] is True and d['diagnostic_completed'] is True,'actual bounded diagnostic')
need(v['diagnostic_completed'] is True and v['counts']==r['outer_measure_correction']['diagnosed_E_counts'] and v['input_map_radius_window']==[609,640],'exact diagnosis counts/window')
need(all(v[x] is False for x in ['E_accumulated','SVD_executed','query_or_generator_executed']),'diagnostic no larger admission')
need(s['passed'] is True and s['inputs_unchanged'] is True and s['no_scientific_reexecution'] is True and s['original_v9_radial_FAIL_preserved'] is True,'actual independent audit')
need(f['completed'] is False and f['returncode']==1 and f['inputs_unchanged'] is True and f['failure']=='FloatingPointError: underflow encountered in multiply','old failure retained')
for label,row in r['outer_measure_correction']['evidence'].items():need(r['pins'].get(row['path'])==row['sha256'],'evidence pre-import pin '+label)
unit=load(r['weighting_rounding_prerequisite']['receipt']);ur=load(r['weighting_rounding_prerequisite']['recipe']);auth=load(P/'inputs/authorization-schema.json')
need(unit['passed_independent_exact_bit_units'] is True and unit['unit_cases']==unit['expected_unit_cases']==96 and unit['inputs_unchanged'] is True and ur['helper_sha256']==sha(P/'inputs/tiny_normalization.py'),'96 units actual and helper binding')
for key,count,flag in [('fast_weighting_wrapper_units',20,'passed_fast_weighting_wrapper_units'),('column_norm_units',24,'passed_column_norm_units')]:
    u=auth[key];need(sha(u['path'])==u['sha256'],'unit pin');j=load(u['path']);need(j[flag] is True and j['inputs_unchanged'] is True and j['checks']==count,'actual unit scope '+key)
need(auth['independent_saved_matrix_readback_authorized'] is False,'source remains held')
res={'source_review_only':True,'passed':True,'inputs_unchanged':True,'reviewed_source_index_sha256':sha(P/'inputs/source-index.json'),'reviewed_recipe_sha256':sha(P/'inputs/recipe.json'),'reviewed_verifier_sha256':sha(P/'inputs/verify_retained.py'),'independent_reverse_bytes_AST':True,'hunks':hunks,'five_inner_operand_ASTs_identical':True,'adapter_store_count':len(stores),'original_gates_cases_context_environment_equal':True,'source_files_captured':len(cap['files']),'compact_evidence_captured':len(ev['files']),'unique_protected_pins_rehashed':len(pins),'recipe_pin_count':len(r['pins']),'SciPy_metadata_files':len(scipy['files']),'actual96_20_24_prerequisites_verified':True,'actual_diagnostic_saved_audit_failure_verified':True,'no_candidate_import':True,'no_array_decode':True,'no_scientific_evaluation':True,'no_generator_admission':True,'caveats':['Only the E product was measured in the bounded diagnostic; no underflow occurrence is inferred for Ks/Kw/G/loads.','The five adapters preserve exact local multiplication of original rounded operands. Audit error bounds are not propagated matrix/solve/continuum error bounds.','All three fresh v10 case readbacks and separate independent saved-result/provenance review remain required.','Candidate main retains inherited environment keys; exact root launcher must bind OPTIMIZE0/OMP1 and sanitize PYTHONHOME/PYTHONWARNINGS as separately reviewed, rather than infer them from candidate environment dictionary.'],'reviewer_runtime':{'executable':sys.executable,'resolved':str(Path(sys.executable).resolve()),'isolated':sys.flags.isolated,'dont_write_bytecode':sys.dont_write_bytecode,'optimize':sys.flags.optimize,'PYTHONOPTIMIZE':os.environ.get('PYTHONOPTIMIZE')}}
(P/'source-readback.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
