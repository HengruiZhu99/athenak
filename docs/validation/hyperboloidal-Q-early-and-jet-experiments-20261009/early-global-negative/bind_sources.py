"""Bind exact mode0 earlier helper after final scientific/review authorization."""
from pathlib import Path
import hashlib,json,shutil,difflib
w=Path(__file__).resolve().parent;root=w.parents[2];v=w/'full22-candidate';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
core=root/'build-layer-research/continuum/q-null-early-feedback/immutable-Q-null-early-feedback-local-20261009'
review=root/'build-layer-research/continuum/independent-q-early-review/immutable-independent-Q-early-support-review-20261009'
for folder,pin in [(core,'902d875e6da80bf3b10e7103aa59cf123c4d8a034862f0a5d9be6f9e220c56c0'),(review,'273fdab75ae847d4a54a6cd8b50334a215a839969de39c5bb7378b50fa24b19a')]:
 assert sha(folder/'index.json')==pin
 for n,h in json.loads((folder/'index.json').read_text())['files'].items():assert sha(folder/n)==(h if isinstance(h,str) else h['sha256'])
r=json.loads((core/'receipt.json').read_text());assert r['sources_unchanged'] and r['source_before']==r['source_after'] and len(r['source_after'])==376 and len(r['commands'])==8 and all(c['returncode']==0 for c in r['commands'])
for n,h in r['source_after'].items():
 p=Path(n);p=p if p.is_absolute() else root/p;assert sha(p)==h
assert sha(Path(r['commands'][-1]['command'][1]))==r['postprocessing_source_sha256']
rr=json.loads((review/'receipt.json').read_text());assert rr['returncode']==0 and rr['no_tensor_native_compiler_or_propagation_executed']
for n,h in [('early_feedback.hpp','562d01b4382c5f9c36afc83b29208f5db3d90ca37f12890c92c03b101eb36a69'),('inputs/q_null_feedback.hpp','f3f3acfe16ba3ce36d0b687225aa1e3b33c159781e46d05e1441ccea7c57e049'),('inputs/factored_base.hpp','216bb80c18ee33e553de8defd027e6d3e0394eab38a19a56f66699583e0eca7a')]:
 p=core/n;assert sha(p)==h;q=v/n;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for p in [w/'native_injection.hpp',v/'native_injection.hpp']:
 s=p.read_text();t=s.replace('#include "q_null_feedback.hpp"','#include "early_feedback.hpp"').replace('return qnf::Gauge(p,u,g,{.85,.95,5,true});','return earlynf::Gauge(p,u,g,{5,0});');assert s!=t;p.write_text(t);(w/(p.parent.name+'-earlier-injection.diff')).write_text(''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile='frozen-late-Q/'+p.name,tofile=str(p))))
p=v/'full22_server.cpp';s=p.read_text();mark='\\\"q_null_sigma\\\":5';assert s.count(mark)==1;s=s.replace(mark,'\\\"q_null_sigma\\\":5,\\\"null_feedback_weight_mode\\\":0,\\\"null_feedback_weight\\\":\\\"Wgauge(.45,.85)\\\"');p.write_text(s)
for folder in [w,v]:shutil.copy2(folder/'HELD-build-spatialnorm.json',folder/'build-spatialnorm.json')
late=w.parent/'full-tensor-conformal-q-null-feedback/immutable-Q-null-global-screen-20261009'
for n in ['build_candidate.py']:
 s=(late/n).read_text().replace('conformal-Q physical-inner lapse blend plus preferred-Q shift and sigma5 null feedback, explicit false/true/xi2; C0 kappa2zero','earlier mode0=Wgauge null feedback, unchanged Q physical-inner lapse/preferred shift; explicit false/true/xi2, C0 kappa2zero');(w/n).write_text(s)
for n in ['assemble_projected.py','analyze_fields.py']:
 p=v/n;s=p.read_text().replace('Q physical-inner/preferred/sigma5 gauge alone','earlier mode0 Wgauge Q physical-inner/preferred/sigma5 gauge alone').replace('conformal-Q/null-feedback gauge alone','earlier mode0 Wgauge conformal-Q/null-feedback gauge alone');p.write_text(s)
auth={'gate_index_path':str(core/'index.json'),'gate_index_sha256':sha(core/'index.json'),'gate_receipt_path':str(core/'receipt.json'),'gate_receipt_sha256':sha(core/'receipt.json'),'independent_index_path':str(review/'index.json'),'independent_index_sha256':sha(review/'index.json'),'scientific_gate_authorized_before_compile':True,'root_admission':'Proceed one W earlier-weight actual Cartesian Jv/stage/shortcanonical/guarded t2 attribution; no t6 or longnative','explicit_parameters':{'physical_trace_lapse':False,'preferred_source':True,'scri_lapse_damping':2,'sigma':5,'weight_mode':0,'weight':'Wgauge(.45,.85)','physical_inner':True},'candidate_C0_kappa2':0,'gate_376inputs_8commands_and_postprocessor_reverified':True,'no_production_cartesian_header_change':True,'no_t6_or_long_native':True}
(w/'gate-authorization.json').write_text(json.dumps(auth,indent=2)+'\n');print('Exact frozen early mode0 helper bound after both gates; explicit parameters unchanged; no compilation in this script.')
