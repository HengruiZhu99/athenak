"""Read-only prepared native recipe and exact source-seam review; never compile."""
from pathlib import Path
import difflib,hashlib,json,shlex,shutil,subprocess
P=Path(__file__).resolve().parent;R=P.parents[2];H=R/'build-layer-research/reference-wave-map-native-held-20261009';B=R/'build-layer-spatial-norm-native'
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
S=P/'source-copies';assert not S.exists();S.mkdir()
relative=['PLAN.md','prepare_build.py','make_inputs.py','prerequisites.json','source-preparation-receipt.json','include/native_wave_map.hpp','inputs/index.json']
for mode in ['wave-map','c0','wave-map-half']:
 relative+=['recipes/'+mode+'/'+q for q in ['recipe.json','cartesian-overlay.patch','include/native_wave_map.hpp','include/reference_wave_map.hpp','include/z4c/hyperboloidal/cartesian_patch.hpp']]
relative+=['recipes/wave-map-half/z4c_newdt.cpp','recipes/wave-map-half/timestep-only.patch']
relative += [str(q.relative_to(H))for q in sorted((H/'inputs').glob('*.athinput'))]
pins=[]
for rel in relative:
 q=H/rel;dest=S/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(q,dest);assert sha(q)==sha(dest)
 pins.append({'path':str(q),'copy':str(dest.relative_to(P)),'sha256':sha(q),'bytes':q.stat().st_size})
result={'kind':'independent source-only native overlay/build/control recipe review','passed':False,'source_sha256':sha(__file__),'snapshot_pins':pins,'no_compile_or_native_evolution':True}
try:
 public=(R/'src/z4c/hyperboloidal/cartesian_patch.hpp').read_text();dt=(R/'src/z4c/z4c_newdt.cpp').read_text()
 cmds=json.loads((B/'compile_commands.json').read_text());selected=[];allbase={}
 for row in cmds:
  if not Path(row['file']).is_relative_to(R/'src'):continue
  c=shlex.split(row['command']);obj=(Path(row['directory'])/c[c.index('-o')+1]).resolve();dep=Path(str(obj)+'.d');deps=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]);allbase[row['file']]=(row,c,obj,dep)
  if str(R/'src/z4c/hyperboloidal/cartesian_patch.hpp')in deps:selected.append(row['file'])
 assert len(selected)==6
 preparation=json.loads((S/'source-preparation-receipt.json').read_text());assert selected==preparation['six_source_files']
 allmodes={}
 for mode in ['wave-map','c0','wave-map-half']:
  m=S/'recipes'/mode;rec=json.loads((m/'recipe.json').read_text());assert len(rec['compile_commands'])==6 and not rec['compilation_executed']and not rec['native_execution_authorized']
  assert sha(H/'recipes'/mode/'recipe.json')==preparation['recipes']['recipes/'+mode+'/recipe.json']
  assert rec['compiled_implementation']=='27c19d20696ea6dd4704032c51dfd026218f64f2'
  for p,h in rec['private_sources_sha256'].items():assert sha(R/p)==h,p
  for p,h in rec['base_source_sha256'].items():assert sha(R/p)==h,p
  for p,h in rec['base_overlay_sha256'].items():assert sha(R/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'/p)==h,p
  for p,v in rec['base_link_inputs'].items():assert sha(p)==v['sha256']and Path(p).stat().st_size==v['bytes'],p
  hcart=(m/'include/z4c/hyperboloidal/cartesian_patch.hpp').read_text();helper=m/'include/reference_wave_map.hpp';wrapper=(m/'include/native_wave_map.hpp').read_text()
  assert sha(helper)=='56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28'
  assert wrapper==(S/'include/native_wave_map.hpp').read_text()
  assert wrapper.count('return rwm::Assemble(parts,p.omega,rhs);')==1
  assert 'AssembleGaugeInterior('not in wrapper and '#define'not in '\n'.join(q for q in wrapper.splitlines()if 'HPP_'not in q)
  if mode=='c0':assert hcart==public
  else:
   inverse=hcart.replace('#include "native_wave_map.hpp"\n','').replace('''      const Real xyz[3] = {g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]};
      const auto p = ref.At(xyz[0],xyz[1],xyz[2]);''','''      const auto p = ref.At(g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]);''').replace('''          || !ResearchNativeWaveMapGauge(p,u,xyz,gauge_rhs)) {''','''          || !AssembleGaugeInterior((ref.layer.enabled || lg.physical_trace_lapse)
              ? InteriorLayerGauge(p,u,lg)
              : UnfactoredReferenceGauge(ref,p,u,gauge,true,true),
                                     omega.omega,gauge_rhs)) {''')
   assert inverse==public
   assert hcart.count('ResearchNativeWaveMapGauge(p,u,xyz,gauge_rhs)')==1
  assert hcart.count('AddMeshUpwindAdvectionWithVelocity<3>')==public.count('AddMeshUpwindAdvectionWithVelocity<3>')==1
  inclrecords=[]
  for row in rec['compile_commands']:
   old,oldcmd,obj,dep=allbase[row['original_source']];assert sha(obj)==row['base_object_sha256']and sha(dep)==row['base_depfile_sha256']
   c=row['command'];assert '-include'not in c and not any('native_injection' in q for q in c)
   assert Path(row['output']).parent==H/'recipes'/mode and Path(row['depfile']).parent==H/'recipes'/mode
   inc=[Path(q[2:])for q in c if q.startswith('-I')]
   def resolve(parent,name):
    candidates=[parent/name]+[q/name for q in inc];return next((q.resolve()for q in candidates if q.is_file()),None)
   start=Path(row['original_source']) if mode!='wave-map-half'or not row['original_source'].endswith('/z4c_newdt.cpp')else H/'recipes/wave-map-half/z4c_newdt.cpp'
   queue=[start];seen=set();found=[]
   while queue:
    src=queue.pop()
    if src in seen:continue
    seen.add(src)
    if src.name=='cartesian_patch.hpp':found.append(src);continue
    for line in src.read_text().splitlines():
     line=line.strip()
     if not line.startswith('#include "'):continue
     name=line.split('"')[1];q=resolve(src.parent,name)
     if q is not None and q.is_relative_to(R)and (q.is_relative_to(R/'src')or q.is_relative_to(H/'recipes'/mode/'include')):queue.append(q)
   expected=H/'recipes'/mode/'include/z4c/hyperboloidal/cartesian_patch.hpp';assert found and all(q==expected for q in found),(mode,start,found)
   if mode=='wave-map-half'and row['original_source'].endswith('/z4c_newdt.cpp'):assert resolve(start.parent,'z4c.hpp')==R/'src/z4c/z4c.hpp'
   inclrecords.append({'source':row['original_source'],'resolved_cartesian_header':str(expected),'forced_include_absent':True})
  linked=[q for q in rec['link_command']if q.endswith('.o')];new={row['output']for row in rec['compile_commands']};assert len(linked)==182 and len(new.intersection(linked))==6
  assert len([p for p in rec['base_link_inputs']if p.endswith('.o')])==182 and len([p for p in rec['base_link_inputs']if p.endswith('.a')])==4
  allmodes[mode]={'source_count':6,'reused_native_objects':176,'libraries':4,'include_resolution':inclrecords,'private_cartesian_inverse_exact':True}
 half=(S/'recipes/wave-map-half/z4c_newdt.cpp').read_text();addition='''    // Private timestep-only control: halve BOTH admitted native caps.
    dtnew *= Real(0.5);
''';assert half.count(addition)==1 and half.replace(addition,'')==dt
 wh=(S/'recipes/wave-map/include/z4c/hyperboloidal/cartesian_patch.hpp').read_bytes();assert wh==(S/'recipes/wave-map-half/include/z4c/hyperboloidal/cartesian_patch.hpp').read_bytes()
 inputindex=json.loads((S/'inputs/index.json').read_text());assert len(inputindex['inputs'])==17
 for rec in inputindex['inputs']:
  q=H/rec['path'];assert sha(q)==rec['sha256'];text=q.read_text();assert 'cfl_number=.1' in text and 'mass=0' in text and 'pulse_width=.35' in text and 'hyperboloidal_symmetric_ghosts=true' in text and 'hyperboloidal_dissipation=.1' in text
 for n in [16,24,32]:
  a=(S/f'inputs/wave-map-N{n}-large-t2.athinput').read_text();b=(S/f'inputs/c0-N{n}-large-t2.athinput').read_text();assert a.replace(f'wave-map-N{n}-large-t2',f'c0-N{n}-large-t2')==b
 a=(S/'inputs/wave-map-N24-large-t2.athinput').read_text();b=(S/'inputs/wave-map-half-N24-large-t2.athinput').read_text();assert a.replace('wave-map-N24-large-t2','wave-map-half-N24-large-t2')==b
 for pin in pins:assert sha(pin['path'])==pin['sha256']
 result.update(passed=True,closure=allmodes,held_input_count=17,matched_C0_and_half_inputs_differ_only_basename=True,half_patch_only_hyperboloidal_cap_factor=True,scientific_source_seam_review='Exact xyz orientation, alpha/beta poles once, upwind once, same C0 geometric live/RHS0 calls and subtraction; helper factored stationarity separate from native raw geometry f0 subtraction.',seam_verification_gap='Prepared PLAN requires six declared finite cells but currently has no exact cell list or executable seam-readback implementation/schema. Pin them before seam execution; planned2e-12 centered/upwind-separated test remains pending.',no_blocking_overlay_or_build_recipe_math_issue_found=True,seam_execution_accepted=False)
except Exception as e:result['exception']=repr(e)
(P/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2));assert result['passed']
