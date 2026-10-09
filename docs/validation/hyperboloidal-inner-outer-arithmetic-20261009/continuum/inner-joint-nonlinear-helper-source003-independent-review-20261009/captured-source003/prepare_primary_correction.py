#!/usr/bin/env python3
"""One-shot stdlib source/pin preparation only. No scientific execution."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source002-held-20261009'
PLAN=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-cancellation-plan-held-20261009'
OUTER=ROOT/'build-layer-research/inner-joint-nonlinear-source002-root-release-20261009'
ASSOCIATION=ROOT/'build-layer-research/inner-joint-nonlinear-source002-failure-saved-summary-20261009'

def pin(path):
    path=Path(path).resolve();data=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def read(path):return json.loads(Path(path).read_text())
def text_new(path,text):
    path=Path(path)
    if path.exists():raise RuntimeError('Refuse overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
def write(path,data):text_new(path,json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
def copy_new(source,target):
    if target.exists():raise RuntimeError('Refuse overwrite '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
def replace_once(text,before,after):
    if text.count(before)!=1:raise RuntimeError('Expected unique source replacement '+before[:60])
    return text.replace(before,after)

def main():
    if (HERE/'source-index.json').exists():raise RuntimeError('Already indexed')
    original=read(OLD/'source-index.json');protected={}
    def add(row):
        actual=pin(row['path'])
        if actual!=row:raise RuntimeError('Input drift '+row['path'])
        protected[actual['path']]=actual
    if pin(OLD/'source-index.json')['sha256']!='a6dfac99b3b9296629701bfd02f90ad79c2b36f47e95c747d69c897d694ade05':
        raise RuntimeError('Original002 index drift')
    if pin(PLAN/'index.json')['sha256']!='deb3b2e8491c41ff8a02460a3730f5a9359a4906f66724e87fe2cb28d2f3de94':
        raise RuntimeError('Correction plan drift')
    for row in read(OLD/'input-pins.json')+original['files']+[pin(OLD/'source-index.json')]:add(row)
    for tree in [OLD/'attempts/Release001',OUTER,ASSOCIATION,PLAN]:
        for path in sorted(tree.rglob('*')):
            if path.is_file():add(pin(path))
    failure=read(OLD/'attempts/Release001/receipt.json')
    if failure['completed'] or failure['passed'] or failure['returncode']!=1 or not failure['source_inputs_unchanged']:
        raise RuntimeError('Expected preserved actual002 oracle FAIL')
    if pin(OLD/'attempts/Release001/receipt.json')['sha256']!='e89496d60e6055d25c0f969d9f9f4724fb42d50b2df80225f4e68c990ea6e86d':
        raise RuntimeError('Actual failed002 receipt drift')
    if pin(OLD/'attempts/Release001/oracle-report.json')['sha256']!='84845860382e77ddf4881048de77c0531a53c70bcfaced851caa217c5232bc08':
        raise RuntimeError('Actual002 oracle report drift')
    replaced={'inner_gauge.hpp','probe.cpp','recipe.json','input-pins.json','authorization-schema.json','source-preparation.json','failure-history.json','CORRECTION.md'}
    identical={}
    for row in original['files']:
        source=Path(row['path']);relative=source.relative_to(OLD)
        if str(relative) not in replaced:
            copy_new(source,HERE/relative);identical[str(relative)]=row['sha256']
    old_helper=(OLD/'inner_gauge.hpp').read_text()
    helper=replace_once(old_helper,
        '  int maximum_product_exponent=0;\n',
        '  int maximum_product_exponent=0;\n  unsigned long dA_near=0,dA_far=0,dc_near=0,dc_far=0,dal_near=0,dal_far=0;\n')
    marker='template<class T> T Coefficient(T alpha,T chi,double W,double G0,bool&valid,\n'
    helpers='''// Exact [1/2,2] positive-PRIMAL closeness, without a ratio or half-product.
inline bool NearFieldValue(double x,double reference){
  if(!(x>0)||!(reference>0)||!std::isfinite(x)||!std::isfinite(reference))return false;
  int ex=0,er=0;const double mx=std::frexp(x,&ex),mr=std::frexp(reference,&er);
  const int difference=ex-er;
  if(difference==0)return true;
  if(difference==1)return mx<=mr;
  if(difference==-1)return mx>=mr;
  return false;
}
template<class T> T FieldSquareDifference(T a,T chi,T h,T reference_chi,
                                         Audit *audit=nullptr){
  const bool near=NearFieldValue(Number<T>::Value(a),Number<T>::Value(h))&&
    NearFieldValue(Number<T>::Value(chi),Number<T>::Value(reference_chi));
  if(audit){if(near)++audit->dA_near;else ++audit->dA_far;}
  // Preserve exact-reference deviation arithmetic only while BOTH fields are near.
  if(near)return Product<T>({a+h,a-h,chi},audit)+
                 Product<T>({h,h,chi-reference_chi},audit);
  return Product<T>({a,a,chi},audit)-Product<T>({h,h,reference_chi},audit);
}
enum class LogGradientField { Chi, Alpha };
template<class T> T FieldLogGradientDifference(T x,T gradient,T reference,
    T reference_gradient,LogGradientField field,Audit *audit=nullptr){
  const bool near=NearFieldValue(Number<T>::Value(x),Number<T>::Value(reference));
  if(audit){
    if(field==LogGradientField::Chi){if(near)++audit->dc_near;else ++audit->dc_far;}
    else {if(near)++audit->dal_near;else ++audit->dal_far;}
  }
  const T reference_log_gradient=reference_gradient/reference;
  if(near)return (gradient-reference_gradient)-
                 Product<T>({x-reference,reference_log_gradient},audit);
  return gradient-Product<T>({x,reference_log_gradient},audit);
}

'''
    helper=replace_once(helper,marker,helpers+marker)
    helper=replace_once(helper,'  const T c=1-W,da=a-h,dchi=chi-ch;\n','  const T c=1-W,da=a-h;\n')
    helper=replace_once(helper,
        '  const T dA=Product<T>({a+h,da,chi},audit)+Product<T>({h,h,dchi},audit);\n',
        '  const T dA=FieldSquareDifference(a,chi,h,ch,audit);\n')
    helper=replace_once(helper,
        '    dc[i]=(u.chi.d[i]-p.state.chi.d[i])-Product<T>({dchi,p.state.chi.d[i]/ch},audit);\n',
        '    dc[i]=FieldLogGradientDifference(chi,u.chi.d[i],ch,p.state.chi.d[i],LogGradientField::Chi,audit);\n')
    helper=replace_once(helper,
        '    dal[i]=(u.alpha.d[i]-p.dalpha[i])-Product<T>({da,p.dalpha[i]/h},audit);\n',
        '    dal[i]=FieldLogGradientDifference(a,u.alpha.d[i],h,p.dalpha[i],LogGradientField::Alpha,audit);\n')
    text_new(HERE/'inner_gauge.hpp',helper)
    old_probe=(OLD/'probe.cpp').read_text()
    before=' <<",\\\"coefficient_scaled_away\\\":"<<a.coefficient_scaled_away<<",\\\"max_product_exponent\\\":"<<a.maximum_product_exponent<<\'}\';}'
    after=''' <<",\\\"coefficient_scaled_away\\\":"<<a.coefficient_scaled_away<<",\\\"max_product_exponent\\\":"<<a.maximum_product_exponent
 <<",\\\"dA_near\\\":"<<a.dA_near<<",\\\"dA_far\\\":"<<a.dA_far
 <<",\\\"dc_near\\\":"<<a.dc_near<<",\\\"dc_far\\\":"<<a.dc_far
 <<",\\\"dal_near\\\":"<<a.dal_near<<",\\\"dal_far\\\":"<<a.dal_far<<'}';}'''
    probe=replace_once(old_probe,before,after);text_new(HERE/'probe.cpp',probe)
    # All query generation, record counts and independent oracle stay byte-exact.
    text_new(HERE/'primary-helper.diff',''.join(difflib.unified_diff(old_helper.splitlines(True),helper.splitlines(True),fromfile='source002/inner_gauge.hpp',tofile='source003/inner_gauge.hpp')))
    text_new(HERE/'audit-only-probe.diff',''.join(difflib.unified_diff(old_probe.splitlines(True),probe.splitlines(True),fromfile='source002/probe.cpp',tofile='source003/probe.cpp')))
    recipe=read(OLD/'recipe.json');recipe=json.loads(json.dumps(recipe).replace(str(OLD),str(HERE)))
    recipe['correction_scope']='primary dA/dc/dal factored-near/direct-far evaluation only; separately diffed six branch audit counters; no dV extension'
    recipe['prior_source_index']=pin(OLD/'source-index.json')
    recipe['prior_oracle_failure_receipt']=pin(OLD/'attempts/Release001/receipt.json')
    recipe['prior_oracle_failure_report']=pin(OLD/'attempts/Release001/oracle-report.json')
    recipe['correction_plan_index']=pin(PLAN/'index.json')
    recipe['branch_audit_scope']='six additive counters in existing arithmetic objects for audited Gauge calls only; no extra query/case, no independent branch unit acceptance'
    recipe['metric_contrast_limit']='unchanged dV and split gradient group; high-contrast alpha/chi family retains reference metric; arbitrary simultaneous extreme metric contrast is not certified'
    recipe['additive_witnesses_not_executed_or_appended']=True
    write(HERE/'recipe.json',recipe)
    auth=read(OLD/'authorization-schema.json');auth['recipe_sha256']=pin(HERE/'recipe.json')['sha256']
    write(HERE/'authorization-schema.json',auth)
    write(HERE/'input-pins.json',sorted(protected.values(),key=lambda row:row['path']))
    write(HERE/'failure-history.json',{'source001_compile_failed_before_query':True,'source002_actual_oracle_failed':True,'source002_failure_receipt':pin(OLD/'attempts/Release001/receipt.json'),'source002_oracle_report':pin(OLD/'attempts/Release001/oracle-report.json'),'source002_outer_receipt':pin(OUTER/'release-invocation001/receipt.json'),'source002_association_summary':pin(ASSOCIATION/'summary.json'),'all_original_source_attempt_log_dependencies_and_executable_bytes_protected':True,'source002_debug_retry_not_performed':True,'failures_not_relabelled_as_pass':True})
    text_new(HERE/'CORRECTION.md', '''# Held source003 primary field-difference correction

Only the approved dA/dc/dal arithmetic regimes change. Near is the exact
positive-primal interval [1/2,2], evaluated by frexp comparisons; dA requires
BOTH fields near. The selected generic formulas preserve both registered
field-dual components. The exact W=1 return, complete nonflat connection,
coefficient, P/Theta/C0 geometry, finite contracts and all other gauge terms
are unchanged. Named field-difference helpers support later isolated checks.

The separately diffed probe change only prints six additive branch counters
in existing arithmetic objects. The query grids, all15740 main records,
oracle bytes, MP precision, thresholds, FD levels and actual22 bindings are
unchanged. Audits cover only calls already passed an Audit pointer; the
unaudited Actual22/FD calls do not contribute to these counters.

The dV hardening/regrouping and additive normal-target/branch witnesses are
HELD separately and not implemented or appended. Fixed-family metric
contrast limitations remain. The old source001 compile failure and actual
source002 oracle failure, sources, unique executable, dependencies, output
payloads and outer receipts are protected inputs and stay unchanged.

The copied IMPLEMENTATION.md/prepare scripts/proposal diff and old correction
diffs are historical source001/002 records, not a description of this patch
or runnable preparation for this new prefix. This CORRECTION, recipe and
exact new diffs specify source003. No import, compilation, syntax check,
kernel query, array load, matrix, eigenvalue or evolution has occurred here.
Root exact execution release and independent source review remain required.
''')
    for name in identical:
        if pin(HERE/name)['sha256']!=identical[name]:raise RuntimeError('Copied byte drift '+name)
    for row in protected.values():add(row)
    write(HERE/'source-preparation.json',{'source_only':True,'execution_admitted':False,'protected_inputs':len(protected),'protected_inputs_unchanged':True,'source001_compile_FAIL_and_source002_oracle_FAIL_preserved':True,'helper_scope':'only approved three field differences plus six audit counters','probe_scope':'Audit serialization only; all fixed query loops and records unchanged','oracle_runner_inputs_cases_thresholds_precision_byte_identical':True,'dV_extension_and_additive_witnesses_held':True,'byte_identical_copied_files':identical,'no_scientific_import_compile_syntax_query_array_operator_or_evolution':True,'preparation_operations':'stdlib copying, metadata hash validation, literal source replacement, JSON and text diff only'})
    files=[pin(path) for path in sorted(HERE.rglob('*')) if path.is_file()]
    write(HERE/'source-index.json',{'source_only':True,'execution_admitted':False,'scope':'uncompiled source003 primary dA/dc/dal correction only, preserved001/002 failures','files':files,'file_count':len(files),'protected_inputs':len(protected),'inputs_unchanged':True,'held_plan_sha256':original['held_plan_sha256'],'correction_plan_sha256':pin(PLAN/'index.json')['sha256']})
    print(json.dumps({'source_index':pin(HERE/'source-index.json'),'recipe':pin(HERE/'recipe.json'),'helper':pin(HERE/'inner_gauge.hpp'),'probe':pin(HERE/'probe.cpp'),'primary_diff':pin(HERE/'primary-helper.diff'),'audit_diff':pin(HERE/'audit-only-probe.diff'),'files':len(files),'protected_inputs':len(protected),'execution_admitted':False},sort_keys=True))

if __name__=='__main__':main()
