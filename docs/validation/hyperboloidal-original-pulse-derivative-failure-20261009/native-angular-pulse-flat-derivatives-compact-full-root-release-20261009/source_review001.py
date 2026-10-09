from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;B=P.parent/'continuum'
C=B/'native-angular-pulse-flat-derivatives-compact-full-held-20261009';O=B/'native-angular-pulse-flat-derivatives-compact-full-launch-held-20261009';V=B/'native-angular-pulse-flat-derivatives-v2-held-20261009';T=B/'native-angular-pulse-flat-derivatives-compact-root-held-20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C/'source-index.json')=='47503416e8d44fe51ff51e6dc38ea69ba6cce6c73d361572a9f5fc958aa16b01'
assert sha(O/'source-index.json')=='bae99f8855ea314910514e47ddc7723dd9b59824009aae9ffd37f5c492437c55'
pins={}
for directory,count in [(C,11),(O,7)]:
 index=json.loads((directory/'source-index.json').read_text());assert len(index['files'])==count
 for row in index['files']:
  assert sha(row['path'])==row['sha256'],row['path'];pins[row['path']]=row['sha256']
  if Path(row['path']).suffix=='.py':ast.parse(Path(row['path']).read_text())
recipe=json.loads((C/'derivative-recipe.json').read_text());oldrecipe=json.loads((V/'derivative-recipe.json').read_text())
for key in recipe['unchanged_scientific_setting_keys']:assert recipe[key]==oldrecipe[key],key
assert recipe['precisions']==[80,110] and recipe['root_tolerance']=='1e-55' and recipe['maximum_root_iterations']==512
for name in ('derivative_core.py','analytic_jets.py','values_context.py','compact_root.py'):assert (C/name).read_bytes()==(T/name).read_bytes(),name
oldtree=ast.parse((V/'flat_ivp_derivatives.py').read_text());newtree=ast.parse((C/'full_gate.py').read_text())
oldrun=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='run');newrun=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
class StripProgress(ast.NodeTransformer):
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='PROGRESS' for t in node.targets):return None
  return self.generic_visit(node)
 def visit_Name(self,node):
  if node.id in ('ProgressNativeGraph','ProgressControlGraph'):node.id=node.id[len('Progress'):]
  return node
assert ast.dump(oldrun,include_attributes=False)==ast.dump(StripProgress().visit(newrun),include_attributes=False),'run math changed beyond constructors/progress labels'
for path,digest in {**recipe['dependency_pins'],**recipe['mpmath_python_pins'],recipe['python_runtime_path']:recipe['python_runtime_sha256']}.items():assert sha(path)==digest,path;pins[path]=digest
schema=json.loads((O/'authorization-schema.json').read_text())
assert schema['expected_child_counts']=={'native_rows':100,'control_rows':100,'initial_rows':28,'checks':2360}
assert schema['expected_completed_roots_by_kind']=={'native':189440,'control':189440,'initial':114688}
assert sum(schema['expected_completed_roots_by_kind'].values())==493568
for path,digest in {**schema['outer_source_pins'],**schema['source_pins']}.items():assert sha(path)==digest,path
for key in ('fresh_invocation_path','fresh_output_path'):assert not Path(schema[key]).exists()
result={'passed_root_source_metadata_ast_and_math_review':True,'pins':pins,'run_AST_identical_except_constructor_seams_and_progress_labels':True,'analytic_and_compact_root_sources_byte_identical':True,'scientific_settings_unchanged':True,'counts_and_progress_bookkeeping_reviewed':True,'root_single_call_subclass_wrappers_reviewed':True,'full_workload_roots':493568,'coarea_nodes':163840,'independent_review_pending':True,'execution_admitted':False,'scope':'Full scalar derivative gate only. No inverse, native, stability or BH admission.'}
with (P/'source-review001.json').open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':True,'pins':len(pins),'review_sha256':sha(P/'source-review001.json')}))
