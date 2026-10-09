"""Independent source-only review capture; never executes a science binary."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path('/Users/hz0693/research/hyperboloidal')
BASE=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
PIN=BASE/'source-chunk-002-pins.json'
sha=lambda b:hashlib.sha256(b).hexdigest()
pins=json.loads(PIN.read_text())
data={name:(BASE/name).read_bytes() for name in pins}
for name,digest in pins.items():
    assert sha(data[name])==digest,(name,sha(data[name]),digest)
OUT=BASE/'independent-chunk-reviews/002-configuration-normalization'
assert not OUT.exists()
OUT.mkdir()
for name,b in data.items():
    (OUT/name).write_bytes(b)
(OUT/PIN.name).write_bytes(PIN.read_bytes())
review={
 'status':'PASS_independent_source_math_only_numerical_gate_not_admitted_here',
 'reviewer':'/root/literature_gauge',
 'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'method':'Independent source/formula inspection with exact-byte pin/copy and unchanged readback; no compile, scientific batch, numerical rerun or operator assembly.',
 'source_pin_sha256':sha(PIN.read_bytes()),
 'sources':{n:{'original_path':str(BASE/n),'sha256':sha(b),'bytes':len(b)} for n,b in data.items()},
 'findings':[
  'S1<D> correctly implements a first Cartesian spatial jet over the perturbation dual, including mixed product/quotient/elementary-function derivatives; overloads precede production template definitions.',
  'GenericGauge now accepts Z4cJet<T>, retains physical-P/source-off spatial-norm xi=2 and rho=1.5 source and single beta-pole assembly. Preferred-source Hessian/Box derivatives are absent in this baseline.',
  'ConfigurationSpatial evaluates only chi, six metric entries, alpha and beta. Chi/metric formulas agree with the actual first-order C0 configuration equations; their spatial derivatives require at most second configuration jets and first A jets. No A/P/Theta/Lambda evolution is differentiated.',
  'SpatialReference retains complete radius, Omega gradient/Hessian, lapse/trace/beta reference coefficient jets. Higher placeholders are unused by equation rows; Geometry still evaluates curvature for its finite-validity check, so arbitrary finite extensions rather than NaN poisoning are appropriate. Extensions now vary both D value and perturbation components.',
  'MakeReferenceRadial correctly contracts Cartesian reference jets along a fixed ray, including reference inverse/raised A and radial frame/coframe factors. Normal frame evaluation is excluded at origin, whose separate exact polynomial core remains necessary.',
  'Complete U uses lapse/alpha_ref, chi/chi_ref, the declared metric component chart and coframe beta/alpha_ref. V uses P/Omega, Theta/Omega, independent-A trace subtraction and chi_ref coframe Lambda. All reference coefficient derivatives remain in the reduction.',
  'The A subtraction Aref_up:delta_g is retained when the same fixed map acts on raw RHS; thus V_t includes the configuration metric RHS contribution. No orthonormal-STF or CG amplitude normalization is substituted for the exact kernel chart.',
  'Input normalization consumes U value/first/second and V value/first only. V has no second-derivative accessor; internal unused second A-reference terms cannot be treated as supplied analytic jets. Source packing consumes Ut, DsUt and Vt only.',
  'Point-energy packing, independent Gamma=Ds(HKn)+(div s)HKn, weak/strong source densities, symmetric volume remainder and complete adjoint SAT work match the held addendum.',
  'The revised derivative sampler locks the same center normal/screen for every radial FD point. This avoids a representation-chart switch on the sampled ray. It does not remove the separate required O(2) rotation/work covariance gate.'
 ],
 'resolved_source_preflight_issues':[
  'GenericGauge hard-coded Jet changed to hyp::Z4cJet<T>.',
  'Undefined FixedPolynomial Dot helper removed.',
  'Higher-jet extension changed from D(unused) to D(unused,unused).',
  'V second derivative accessor removed.',
  'Point-name/mixed-size-auto compile failures retained under release-001 by implementation owner.',
  'Earlier source-gate-release-001 screen-flip failure and earlier source pins retained. Screen threshold moved away from sampled ray and center frame now explicitly locked.',
  'Independent capture aborted on active source-pin mismatch before PASS; history/001-source-pin-race.json preserves that failed capture.'
 ],
 'numerical_gate_scope_and_pending_requirements':[
  'Current DerivativeGate covers m0, a single oblique radial ray, five radii and all J<=2 channels. It is not an all-m/all-angle derivative certificate.',
  'The checker now requires final-h error <=2e-7 and preserves complete sequences. Rows without observed fourth-order evidence are explicitly order unclassified within tolerance; do not automatically call these rows roundoff-limited.',
  'Actual full22/configuration binding, double-only-wrapper negative control, raw normals and finite extension independence must pass with exact build/source pins. This receipt does not assert their numerical outcome.',
  'Symmetrizer/volume coefficient derivative implementation, mass/weak/strong/G-volume assembly, restricted incoming ranks, manufactured forcing and physical subsidiary-rate gates remain independent work. No radial operator, eigenvalue or propagation is admitted by this source review.'
 ],
 'blocking_source_math_corrections':[],
 'scientific_execution_by_reviewer':False,
 'frozen_files_mutated':False,
 'scope':'Read-only source/math review of exact chunk002. Parent controls subsequent releases. No CPBC/exact-scri/continuum stability, native/global pulse or BH acceptance; eventual single BH must survive wormhole-to-trumpet interior with the Minkowski hyperboloidal reference throughout.'
}
for name,b in data.items():
    assert (BASE/name).read_bytes()==b,name
review['review_script_sha256']=sha(Path(__file__).read_bytes())
p=OUT/'receipt.json'
p.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'sources':len(data),'status':review['status']},indent=2))
