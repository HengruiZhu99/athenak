"""Saved-JSON-only document preparation. No arrays, probes or numerical imports."""
from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent;R=P.parents[2];B=R/'build-layer-research'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
comparison=B/'wave-map-native-N24-partial-comparison-root-20261009/attempt001/report.json'
crecipe=B/'wave-map-native-N24-partial-comparison-root-20261009/recipe.json'
c0=B/'reference-wave-map-t2-readback-held-20261009/attempts/c0-N24-large-t2-001'
figure=B/'boundary/reference-wave-map-N24-saved-comparison-figure-20261009'
inputs=[comparison,crecipe,c0/'receipt.json',c0/'analysis/receipt.json',c0/'analysis/snapshots.json',B/'wave-map-native-t2-root-20261009/batch001/c0-N24-large-t2/launch-receipt.json',figure/'recipe.json',figure/'attempt001/receipt.json',figure/'attempt001/N24-controls.png',figure/'visual-review.json',B/'wave-map-N24-figure-root-review-20261009/review.json']
recipe=load(crecipe)
for group in recipe['inputs'] if 'inputs' in recipe else []:pass
# These exact sources are already jointly pinned in the completed root comparison recipe.
for tag in ('standard','half','small'):
 entry=next(x for x in recipe['cases'] if x['tag']==tag)
 for key in ('owner_observations','owner_receipt','manual_rows','manual_receipt','native_stderr'):
  if key in entry:
   v=entry[key];inputs.append(Path(v['source'] if isinstance(v,dict) else v))
# Use explicit paths, not directory traversal, if historical recipe keys differ.
observations={
 'standard':B/'reference-wave-map-partial-N24-held-20261009/attempts/wave-map-N24-large-t2-001/snapshot-observations.json',
 'half':B/'reference-wave-map-partial-half-N24-held-20261009/attempts/wave-map-half-N24-large-t2-001/snapshot-observations.json',
 'small':B/'reference-wave-map-partial-small-N24-held-20261009/attempts/wave-map-N24-small-t2-001/snapshot-observations.json'}
for x in observations.values():inputs.append(x)
capsules={tag:R/('docs/validation/hyperboloidal-reference-wave-map-native-t2-'+name+'-N24-failure-20261009/catalog.json') for tag,name in [('standard','wave'),('half','wave-half'),('small','wave-small')]}
inputs+=list(capsules.values())
n32catalog=R/'docs/validation/hyperboloidal-reference-wave-map-native-t2-wave-N32-failure-20261009/catalog.json';inputs.append(n32catalog)
controls_catalog=R/'docs/validation/hyperboloidal-reference-wave-map-native-t2-N24-controls-20261009/catalog.json';inputs.append(controls_catalog)
for case in ('wave-map-N24-large-t2','wave-map-half-N24-large-t2','wave-map-N24-small-t2','c0-N24-large-t2'):
 inputs.append(B/('reference-wave-map-native-held-20261009/inputs/'+case+'.athinput'))
inputs=list(dict.fromkeys(x.resolve() for x in inputs));before=[pin(x) for x in inputs]
report=load(comparison);assert report['passed_scalar_readback'] and report['native_t2_complete'] is False
assert report['all_saved_independent_field_pairs']==105
cw=load(c0/'receipt.json');ca=load(c0/'analysis/receipt.json');assert cw['returncode']==0 and cw['passed_completed_t2_snapshot_gates'] and cw['protected_before_after_equal'];assert ca['passed_saved_snapshot_finite_and_diagnostic_gates'] and ca['saved_arrays']==81 and ca['protected_inputs_before_after_equal']
obs={k:load(v) for k,v in observations.items()};saved=load(c0/'analysis/snapshots.json');assert len(saved)==81 and saved[-1]['time']==2
minimum_keys=('alpha_min','chi_min','minimum_conformal_metric_eigenvalue','minimum_Penrose_spatial_metric_eigenvalue')
numbers={'scope':'Saved scalar and source metadata only; no new scientific calls.','wave_cases':report['cases'],'failures':report['failure_metadata'],'half_minus_standard_abort_time_decimal':report['half_minus_standard_abort_time_decimal'],'last_matched':report['matched_standard_half_saved_pairs'][-1],'wave_saved_limits':{tag:{k:min(q[k] for q in rows) for k in minimum_keys} for tag,rows in obs.items()},'C0':{'saved_states':len(saved),'final_time':saved[-1]['time'],'final_norms':saved[-1]['rms_H_Mcon_Zcon_Theta'],'saved_limits':{k:min(q[k] for q in saved) for k in minimum_keys},'max_det_error':max(q['det_max'] for q in saved),'max_trace_error':max(q['trace_max'] for q in saved),'history_diagnostic_max_error':max(q['history_scaled_rms_error'] for q in saved),'nonterminal_saved_dt_min':min(q['history_dt'] for q in saved[:-1]),'nonterminal_saved_dt_max':max(q['history_dt'] for q in saved[:-1]),'terminal_clipped_dt':saved[-1]['history_dt'],'Omega_min':saved[-1]['Omega_min']},'failure_capsules':{k:pin(v) for k,v in capsules.items()},'figure':{'png':pin(figure/'attempt001/N24-controls.png'),'recipe':pin(figure/'recipe.json'),'root_visual_review':pin(B/'wave-map-N24-figure-root-review-20261009/review.json')}}
def f(x):return format(x,'.8g')
table=[]
for x in report['cases']:
 fail=next(y for y in report['failure_metadata'] if y['tag']==x['tag']);table.append('| '+x['tag']+' | '+str(x['saved_states'])+' | '+f(x['last_saved_time'])+' | '+fail['time_decimal']+' | '+' / '.join(f(v) for v in x['last_norms'])+' |')
limits=[]
for tag,values in numbers['wave_saved_limits'].items():limits.append('| '+tag+' | '+' | '.join(f(values[k]) for k in minimum_keys)+' |')
limits.append('| C0 | '+' | '.join(f(numbers['C0']['saved_limits'][k]) for k in minimum_keys)+' |')
captext='\n'.join('- '+tag+': `'+str(p.relative_to(R))+'`, SHA256 `'+pin(p)['sha256']+'`.' for tag,p in capsules.items())
text='''# N24 native wave-map controls: saved-state follow-up

This addendum records three failed N24 wave-map pulse runs and one completed N24 C0 control. Halving the wave-map timestep changes the matched saved constraint RMS values by less than 0.085% at the last common output, yet both large-pulse runs abort at almost the same time and first reported cell. Reducing the input pulse amplitudes by ten delays the reported abort to about t=1.018. These are finite native outcomes; they do not identify the cause or establish continuum stability or instability.

All four use the same centered Cartesian cube span 2.2, N24 active spherical mask (5,520 points), a=.5, geometry transition .05–.95, kappa=10, fourth-order derivatives, KO=.1 and symmetric quadratic ray continuation. The smallest active Omega is 0.00217013888888851. The input large lapse/shift amplitudes are .2/.1; the small amplitudes are .02/.01. The production profile is smooth with a vanishing scri tail, not a compact sub-scri pulse. The inputs retain the old gauge-cutoff parameters .45–.85. The wave-map overlay instead uses its pinned reference-wave-map helper globally; those cutoff parameters do not blend it with physical-P gauge. It keeps full C0 geometric evolution. The completed C0 control uses the public C0 gauge; it is not the earlier private spatial-norm global operator.

## Failed wave-map controls

The RMS column is H / Mcon / Zcon / Theta in the native diagnostic convention. Each row stops at the final saved binary64 restart, before its unsaved abort state.

| Run | Saved states | Last saved t | Reported abort t | Last saved RMS |
|---|---:|---:|---:|---|
'''+ '\n'.join(table)+'''

The standard and half-step recorded timesteps are 6.51041666666553e-5 and 3.255208333332765e-5. At matched t≈.775, the absolute relative H/M/Z/Theta RMS differences are '''+' / '.join(f(v) for v in numbers['last_matched']['relative_differences'])+''' (fractions). The reported aborts differ by '''+numbers['half_minus_standard_abort_time_decimal']+''' time units, approximately one half-step. Both are failed processes; this close agreement is a timestep control through the sampled window, not a completed t2 timestep-convergence gate.

All three first report failure at xyz=(0.13749999999999996, -0.96250000000000013, -0.22916666666666674), Omega=0.0021701388888885099. Exact stderr records must be used for the failed, unsaved fields; no earlier restart is substituted for that state. In particular, the standard abort reports chi<0, the half-step abort reports alpha<0 and chi<0, and the small abort reports alpha<0. No floor, SPD repair or restarted continuation is introduced.

## Saved fields and localization

All 105 saved wave-map states passed the unchanged finite/positive/SPD/algebraic and native diagnostic gates. The independent manual check recomputed all 2,625 field-extrema pairs exactly; it did not independently recompute differentiated H/M/Z. Native diagnostic/history agreement is at most 2.7755575615628914e-17 in the root comparison. Passing these saved-state checks does not certify the later unsaved abort.

| Run | Min saved alpha | Min saved chi | Min conformal metric eigenvalue | Min Penrose metric eigenvalue |
|---|---:|---:|---:|---:|
'''+ '\n'.join(limits)+'''

At the last saved state, the r≥.9 shell contains '''+' / '.join(f(x['last_Z_shell_squared_fraction']) for x in report['cases'])+''' of the summed squared native Z diagnostic for standard / half / small. This is a discrete localization fraction, not a physical energy or a boundary-cause certificate.

![Saved N24 comparison]('''+str(figure/'attempt001/N24-controls.png')+''')

The figure uses only completed owner/manual saved JSONs, the root scalar comparison and exact failure stderr. Solid/dashed large-pulse curves separate standard/half timesteps; the small pulse has a separate panel. Curves stop at their last saved states; shaded intervals extend only to the reported aborts. Log RMS limits are [8e-6,.6], relative-difference percent limits [1e-6,.2], and the time axis is [0,1.06]. Near-roundoff t0 is omitted from the log panels without changing saved data. Figure source, input/runtime pins, exact logs, PNG/PDF and both visual reviews are retained separately.

## Completed C0 comparison

The original C0N24 native process returned zero at t=2; its completed wrapper and unchanged analyzer passed all 81 saved restart checks. At t=2, native H/M/Z/Theta RMS is '''+' / '.join(f(v) for v in numbers['C0']['final_norms'])+'''. All saved C0 fields remain finite with positive lapse/chi and SPD metrics; the corresponding minima appear in the table above. Maximum saved det/trace errors are '''+f(numbers['C0']['max_det_error'])+' / '+f(numbers['C0']['max_trace_error'])+''', and native/history diagnostic agreement is '''+f(numbers['C0']['history_diagnostic_max_error'])+'''. The ordinary saved timestep is '''+f(numbers['C0']['nonterminal_saved_dt_max'])+'''; its final '''+f(numbers['C0']['terminal_clipped_dt'])+''' step is clipped to land on t=2 and must not be described as a global timestep reduction.

Completion with sizable constraint norms is not stability acceptance, and an endpoint comparison between C0 t=2 and failed wave-map t≈.793 or 1.018 is not a matched-time accuracy comparison. No new evolution, source query or array decoding was performed to prepare this addendum.

## Evidence and scope

Failure capsules remain separate and unchanged:

'''+captext+'''

The fresh N24-controls archive is at `docs/validation/hyperboloidal-reference-wave-map-native-t2-N24-controls-20261009`, catalog SHA256 `3a84d6803831500d338c46cfee56951bb511ebbffae310e879461cc910672a42`; its one-shot collector completed successfully. Its fixed scope includes completed half/small partial observers, saved summaries/root comparison/manual checks, the figure, failed original case records and completed C0N24 native/readback/release evidence. Raw arrays, executables, objects, NPY/NPZ/JSONL and every file>1MiB are metadata-only. Exact logs are not normalized.

The later N32 large-pulse wave-map process failed at t=1.5837303635638276. Its separate failure catalog is `docs/validation/hyperboloidal-reference-wave-map-native-t2-wave-N32-failure-20261009/catalog.json`, SHA256 `e23e7fa5308b73058c696c585fd646f1cb1a172c7a08c08d838826aff0fbd158`. The separate N32 partial-observation readback is still pending; no N32 diagnostic, grid trend or measured convergence order is asserted here. Omega_min is nonmonotone across these Cartesian resolutions. This N24 evidence does not supply a causal mechanism, CPBC/SBP proof, global energy bound, continuum eigenmode classification or production adoption.
'''
(P/'DRAFT.md').write_text(text);dump(P/'numbers.json',numbers);assert before==[pin(x) for x in inputs]
dump(P/'input-pins.json',{'inputs':before,'scope':'Saved JSON/source metadata/failure catalogs/PNG bytes only; no arrays/probes/native calls.'})
dump(P/'receipt.json',{'completed':True,'inputs_unchanged':True,'source':pin(__file__),'draft':pin(P/'DRAFT.md'),'numbers':pin(P/'numbers.json'),'new_source_queries':0,'new_array_decodes':0,'new_native_steps':0})
print(json.dumps(load(P/'receipt.json'),indent=2))
