"""Bind only the independently reviewed frozen Q helper, with explicit inputs."""
from pathlib import Path
import hashlib,json,shutil,difflib
w=Path(__file__).resolve().parent;root=w.parents[2];v=w/'full22-candidate'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
core=root/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009'
review=root/'build-layer-research/continuum/independent-q-null-feedback-review/immutable-independent-Q-null-review-20261009'
pins={core:'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96',review:'a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650'}
for folder,indexsha in pins.items():
 assert sha(folder/'index.json')==indexsha
 idx=json.loads((folder/'index.json').read_text())
 for name,row in idx['files'].items():
  assert sha(folder/name)==(row if isinstance(row,str) else row['sha256'])
  if isinstance(row,dict):assert (folder/name).stat().st_size==row['bytes']
r=json.loads((core/'receipt.json').read_text());assert r['passed_local_pole_source_principal_corner_gates'] and r['sources_unchanged'] and r['source_before']==r['source_after'] and len(r['source_after'])==382 and len(r['commands'])==14 and all(c['returncode']==0 for c in r['commands'])
for n,h in r['source_after'].items():
 p=Path(n);p=p if p.is_absolute() else root/p;assert sha(p)==h
rr=json.loads((review/'receipt.json').read_text());assert rr['status'].startswith('PASS') and rr['sources_unchanged']
for n,h in rr['source_after'].items():
 p=Path(n);p=p if p.is_absolute() else review.parent/p;assert sha(p)==h
for name,expected in [('q_null_feedback.hpp','f3f3acfe16ba3ce36d0b687225aa1e3b33c159781e46d05e1441ccea7c57e049'),('factored_base.hpp','216bb80c18ee33e553de8defd027e6d3e0394eab38a19a56f66699583e0eca7a')]:
 assert sha(core/name)==expected;assert not (v/name).exists();shutil.copy2(core/name,v/name)

wrapper='''#ifndef SCRATCH_NATIVE_Q_NULL_INJECTION_HPP_
#define SCRATCH_NATIVE_Q_NULL_INJECTION_HPP_
#include "q_null_feedback.hpp"
namespace z4c {namespace hyperboloidal {
template<class T> GaugeRHSParts<T> ResearchNativeQGauge(
 const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g){
 return qnf::Gauge(p,u,g,{.85,.95,5,true});
}
template<class T> bool ResearchNativeQAssemble(
 const GaugeRHSParts<T>&parts,T omega,GaugeRHS<T>&rhs){
 return qnf::Assemble(parts,omega,rhs);
}
}}
#define InteriorLayerGauge ResearchNativeQGauge
#define AssembleGaugeInterior ResearchNativeQAssemble
#endif
'''
for p in [w/'native_injection.hpp',v/'native_injection.hpp']:
 before=p.read_text();p.write_text(wrapper);(w/(p.parent.name+'-injection.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),wrapper.splitlines(True),fromfile='original-spatialnorm/'+p.name,tofile=str(p))))
for p in [w/'tangent_server.cpp',v/'projected_base.hpp']:
 before=p.read_text();mark='g.physical_trace_lapse=true;g.preferred_source=false;';assert before.count(mark)==1
 after=before.replace(mark,'g.physical_trace_lapse=false;g.preferred_source=true;g.scri_lapse_damping=2;');p.write_text(after);(w/(p.stem+'-explicit-input.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original-spatialnorm/'+p.name,tofile=str(p))))
for p in [w/'old-jv-source.cpp',v/'old-jv-source.cpp']:
 before=p.read_text();mark='gauge.physical_trace_lapse=true;gauge.preferred_source=false;';assert before.count(mark)==1
 after=before.replace(mark,'gauge.physical_trace_lapse=false;gauge.preferred_source=true;gauge.scri_lapse_damping=2;');p.write_text(after)
# Record actual explicit input values in both server protocol metadata.
p=v/'full22_server.cpp';s=p.read_text();mark='<<",\\\"cache_seconds\\\":"<<cache';assert s.count(mark)==1
s=s.replace(mark,'<<",\\\"physical_trace_lapse_input\\\":false,\\\"preferred_source_input\\\":true,\\\"scri_lapse_damping\\\":2,\\\"q_null_sigma\\\":5,\\\"physical_inner\\\":true"'+mark);p.write_text(s)
for folder in [w,v]:
 shutil.copy2(folder/'HELD-build-spatialnorm.json',folder/'build-spatialnorm.json')
prior=w.parent/'full-tensor-inner-trace-family/combined/full22-candidate'
for name in ['validate22.py','assemble_projected.py','short_canonical.py','build_fields.py','build_reference.py','compare_controls.py']:
 s=(prior/name).read_text().replace('inner conformal-trace combined gauge alone, C0 spatialnorm finiteΩ strict interior','conformal-Q/null-feedback gauge alone, C0 finiteΩ strict interior')
 (v/name).write_text(s)
auth={'gate_index_path':str(core/'index.json'),'gate_index_sha256':sha(core/'index.json'),'gate_receipt_path':str(core/'receipt.json'),'gate_receipt_sha256':sha(core/'receipt.json'),'independent_index_path':str(review/'index.json'),'independent_index_sha256':sha(review/'index.json'),'scientific_gate_authorized_before_compile':True,'root_admission':'Release held Q/global work now: actual22/native20/reference/Jv/finalstep/attribution/shortcanonical/t2/conditionalt6','explicit_parameters':{'physical_trace_lapse':False,'preferred_source':True,'scri_lapse_damping':2,'sigma':5,'source_r0':.85,'source_r1':.95,'physical_inner':True},'candidate_C0_kappa2':0,'gate_382inputs_14commands_reverified':True,'no_production_cartesian_header_change':True}
(w/'gate-authorization.json').write_text(json.dumps(auth,indent=2)+'\n')
print('Verified core/review and bound exact helper with explicit false/true/xi2 inputs; no compilation in this script.')
