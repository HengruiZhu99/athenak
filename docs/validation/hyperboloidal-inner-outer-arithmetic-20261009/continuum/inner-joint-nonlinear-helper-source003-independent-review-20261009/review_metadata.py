"""Stdlib source/hash/diff admission review. No candidate import/parse/compiler/query."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import time

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
NEW=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source003-held-20261009'
OLD=NEW.with_name('inner-joint-nonlinear-helper-source002-held-20261009')
PLAN=NEW.with_name('inner-joint-nonlinear-helper-cancellation-plan-held-20261009')
EXPECTED={
 NEW/'source-index.json':'6d0f91c4b7bf701f83d723f81c89df8265f76fe839ef4f522e6e056b9544d671',
 NEW/'inner_gauge.hpp':'b26357cc3d25822b8c2b7f60bdf4bb10163bd904babbbe3231ccf3e5c0aeae4b',
 NEW/'recipe.json':'a8b4d9393d9bd745b35dc88f79394dfc2dd2b5b9f0b567fde5658d4c22ca7ab7',
 NEW/'primary-helper.diff':'caa2e5a13b67aaaf67247df75e3f4ae08cf8efcbd03a907458357e45f92f53f1',
 NEW/'audit-only-probe.diff':'fc2a38781392a848b06cf393979903d0b3eafaf3ffcc607e38b0cf60a191f32d',
 PLAN/'index.json':'deb3b2e8491c41ff8a02460a3730f5a9359a4906f66724e87fe2cb28d2f3de94',
}
def load(path):return json.loads(Path(path).read_text())
def pin(path):
 p=Path(path).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size,
  'large_payload':p.suffix.lower() in ('.npz','.npy','.jsonl') or p.stat().st_size>1048576}
def require(condition,message):
 if not condition:raise RuntimeError(message)
def write(name,data):
 p=HERE/name
 if p.exists():raise RuntimeError('Refuse overwrite '+str(p))
 p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
 started=time.monotonic()
 for p,d in EXPECTED.items():require(pin(p)['sha256']==d,'Typed source drift '+str(p))
 index,plan,recipe=load(NEW/'source-index.json'),load(PLAN/'index.json'),load(NEW/'recipe.json')
 protected={}
 rows=load(NEW/'input-pins.json')+index['files']+plan['files']+plan['protected_completed_inputs']
 rows += [pin(NEW/'source-index.json'),pin(PLAN/'index.json'),pin(__file__)]
 for row in rows:
  previous=protected.setdefault(row['path'],row['sha256'])
  require(previous==row['sha256'],'Conflicting protected source '+row['path'])
 before=[]
 for path,digest in protected.items():
  row=pin(path);require(row['sha256']==digest,'Protected input drift '+path);before.append(row)
 write('inputs-before.json',before)
 for source_root,name,source_rows in ((NEW,'captured-source003',index['files']), (PLAN,'captured-cancellation-plan',plan['files'])):
  destination=HERE/name;destination.mkdir(exist_ok=False)
  for row in source_rows:
   source=Path(row['path']);relative=source.relative_to(source_root)
   target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
   require(not pin(source)['large_payload'],'Unexpected source payload '+str(source))
   shutil.copyfile(source,target)
 oldh=(OLD/'inner_gauge.hpp').read_text();newh=(NEW/'inner_gauge.hpp').read_text()
 expected_diff=''.join(difflib.unified_diff(oldh.splitlines(True),newh.splitlines(True),fromfile='source002/inner_gauge.hpp',tofile='source003/inner_gauge.hpp'))
 require(expected_diff==(NEW/'primary-helper.diff').read_text(),'Helper saved diff mismatch')
 reversed_h=newh.replace('  unsigned long dA_near=0,dA_far=0,dc_near=0,dc_far=0,dal_near=0,dal_far=0;\n','')
 helper_start=reversed_h.index('// Exact [1/2,2] positive-PRIMAL closeness, without a ratio or half-product.')
 helper_end=reversed_h.index('template<class T> T Coefficient(T alpha,T chi,double W,double G0,bool&valid,',helper_start)
 reversed_h=reversed_h[:helper_start]+reversed_h[helper_end:]
 reversed_h=reversed_h.replace('  const T c=1-W,da=a-h;\n','  const T c=1-W,da=a-h,dchi=chi-ch;\n')
 reversed_h=reversed_h.replace('  const T dA=FieldSquareDifference(a,chi,h,ch,audit);\n','  const T dA=Product<T>({a+h,da,chi},audit)+Product<T>({h,h,dchi},audit);\n')
 reversed_h=reversed_h.replace('    dc[i]=FieldLogGradientDifference(chi,u.chi.d[i],ch,p.state.chi.d[i],LogGradientField::Chi,audit);\n','    dc[i]=(u.chi.d[i]-p.state.chi.d[i])-Product<T>({dchi,p.state.chi.d[i]/ch},audit);\n')
 reversed_h=reversed_h.replace('    dal[i]=FieldLogGradientDifference(a,u.alpha.d[i],h,p.dalpha[i],LogGradientField::Alpha,audit);\n','    dal[i]=(u.alpha.d[i]-p.dalpha[i])-Product<T>({da,p.dalpha[i]/h},audit);\n')
 require(reversed_h==oldh,'Unexpected helper change outside three differences/audit')
 oldp=(OLD/'probe.cpp').read_text();newp=(NEW/'probe.cpp').read_text()
 expected_probe_diff=''.join(difflib.unified_diff(oldp.splitlines(True),newp.splitlines(True),fromfile='source002/probe.cpp',tofile='source003/probe.cpp'))
 require(expected_probe_diff==(NEW/'audit-only-probe.diff').read_text(),'Probe saved diff mismatch')
 start=newp.index('void Audit(');end=newp.index('template<class T>void Context',start)
 oldstart=oldp.index('void Audit(');oldend=oldp.index('template<class T>void Context',oldstart)
 require(newp[:start]+oldp[oldstart:oldend]+newp[end:]==oldp,'Query/probe changed beyond audit printing')
 require('const auto qp=Parts(q);const auto fp=Fields(f);' in newp,'Prior compile declaration fix missing')
 for name in ('oracle.py','run_once.py'):
  require((NEW/name).read_bytes()==(OLD/name).read_bytes(),'Independent checker/runner changed '+name)
 oldrecipe=load(OLD/'recipe.json')
 fixed_keys=('expected_record_counts','thresholds','precision','modes','high_contrast_families','gauge_raw22_indices','geometry_raw22_indices','raw22_order','generic_scalar_contract')
 for key in fixed_keys:require(recipe[key]==oldrecipe[key],'Fixed scientific recipe changed '+key)
 require(sum(recipe['expected_record_counts'].values())==15740,'Fixed record count mismatch')
 for key in ('compiler','python','environment'):require(recipe[key]==oldrecipe[key],'Fixed runtime changed '+key)
 for build in ('release','debug'):
  require([x.replace(str(NEW),str(OLD)) for x in recipe['compile_flags'][build]]==oldrecipe['compile_flags'][build],'Compile flags changed beyond include prefix')
 after=[pin(x['path']) for x in before];require(after==before,'Review input drift');write('inputs-after.json',after)
 write('receipt.json',{'passed_source_math_and_admission_review':True,'source_only':True,
  'execution_release_granted':False,'source003_index_sha256':EXPECTED[NEW/'source-index.json'],
  'correction_plan_index_sha256':EXPECTED[PLAN/'index.json'],
  'protected_pins':len(before),'source003_suite_files':len(index['files']),
  'plan_files':len(plan['files']),'original_inputs_unchanged':True,
  'query_probe_unchanged_except_six_audit_fields':True,'oracle_runner_byte_identical':True,
  'fixed_main_records':15740,'no_candidate_import_AST_syntax_CAS_compile_query_array_decode':True,
  'prior_source001_compile_FAIL_source002_oracle_FAIL_preserved':True,
  'universal_metric_or_dual_accuracy_not_claimed':True,'seconds':time.monotonic()-started})
 print(json.dumps({'protected_pins':len(before),'suitefiles':len(index['files']),'receipt':pin(HERE/'receipt.json')},sort_keys=True))
if __name__=='__main__':main()
