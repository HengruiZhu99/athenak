"""Separate resolution policy; reuse immutable science/binaries without rebuilding."""
from pathlib import Path
import json,hashlib,subprocess,shutil
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
OLD=ROOT/'output/restored-six-shells/aurora/launcher-restored-six-shells-v1'
OUT=ROOT/'output/restored-six-shells-resolution/aurora';NEW=OUT/'launcher-restored-six-shells-resolution-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prov=json.loads((OLD/'provenance.json').read_text())
for name,v in prov['files'].items():
 p=OLD/name;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],name
NEW.mkdir(parents=True,exist_ok=False)
for name in ['control.py','worker.py','storage.py','run_debug.pbs']:
 shutil.copy2(OLD/name,NEW/name)
s=(OLD/'native_transition_campaign.py').read_text()
s=s.replace('Frozen native N16 controls','Frozen native N20/N24 six-shell controls')
s=s.replace("POLICY = 'n16_A_restored_broad_collar_smooth_initial_slice_to10M'","POLICY = 'n20_then_n24_six_shells_matched_N16_penalty_clipping_to10M'")
s=s.replace("VARIANTS = {'A': dict(suffix='control_broad_smooth_six_shells', magic='N61', shared=False, ko10=False, scri6=False)}", "VARIANTS = {v: dict(suffix='control_broad_smooth_six_shells', magic='N61', shared=False, ko10=False, scri6=False, n=n, dt=dt) for v,n,dt in [('R20',20,.0004),('R24',24,1/3000)]}")
s=s.replace('END, DT = 10/3, .0005\nN = 16','END = 10/3\n\ndef resolution(variant):\n    v=VARIANTS[variant];return v[\'n\'],v[\'dt\']\n\ndef expected_penalty_scale(variant):\n    n,dt=resolution(variant);return .5*(1-1e-12)/(12*dt/(.125/n))')
s=s.replace('def cells(n=N):','def cells(n):')
s=s.replace('def required_checks(variant, purpose, kind, restart=False):\n','def required_checks(variant, purpose, kind, restart=False):\n    N,DT=resolution(variant)\n')
s=s.replace("variant=plan.get('variant');purpose=plan.get('purpose');segment=plan.get('segment')","variant=plan.get('variant');purpose=plan.get('purpose');segment=plan.get('segment')\n    N,DT=resolution(variant)")
s=s.replace("def inspect_checkpoint(path,variant,expected_sha,kind='puncture',end=END):\n","def inspect_checkpoint(path,variant,expected_sha,kind='puncture',end=END):\n    N,DT=resolution(variant)\n")
s=s.replace("def make_plan(root,variant,purpose='evolution',segment=1,restart=None,expected_sha=None):\n","def make_plan(root,variant,purpose='evolution',segment=1,restart=None,expected_sha=None):\n    N,DT=resolution(variant)\n")
s=s.replace('cells()','cells(N)').replace(".001 if purpose=='smoke'", "2*DT if purpose=='smoke'")
s=s.replace("scri_penalty_scale_min=1,angular_epsilon_min=2", "timestep_aware_penalty=True,angular_epsilon_min=2")
s=s.replace("checks=[dict(path=[k],op='eq',value=v) for k,v in exact.items()]", """checks=[dict(path=[k],op='eq',value=v) for k,v in exact.items()]
    # Explicit new user-authorized policy; never changes the frozen N16 policy.
    scale=expected_penalty_scale(variant)
    for key,value in [('scri_penalty_scale_min',scale),('scri_unlimited_penalty_cfl_max',.768),('scri_penalty_cfl_max',.5*(1-1e-12))]:
        checks += [dict(path=[key],op='ge',value=value-1e-12),dict(path=[key],op='le',value=value+1e-12)]""")
s=s.replace("purpose not in ['smoke','evolution']", "purpose not in ['evolution']")
s=s.replace("prefix=f'campaigns/n16-restored-six-shells-v1-", "prefix=f'campaigns/n{N}-restored-six-shells-resolution-v1-")
s=s.replace('fresh N16 wider-collar control','fresh N20/N24 six-shell controls').replace('frozen wider-collar binary','frozen six-shell binary')
s=s.replace("choices=['smoke','evolution']", "choices=['evolution']")
(NEW/'native_transition_campaign.py').write_text(s)
# Scientific bindings deliberately retain the original immutable manifest and science path.
shutil.copytree(OLD,OUT/OLD.name)
manifest='source-restored-six-shells-v1-manifest.json';shutil.copy2(OLD.parent/manifest,OUT/manifest)
newprov={'source_commit':prov['source_commit'],'source_manifest':manifest,'source_manifest_sha256':prov['source_manifest_sha256'],'science_path':'launcher-restored-six-shells-v1/native-transition-science.json','science_sha256':sha(OLD/'native-transition-science.json'),'launcher_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'parent_provenance_sha256':sha(OLD/'provenance.json'),'adapter':str(Path(__file__).relative_to(ROOT)),'adapter_sha256':sha(Path(__file__)),'policy':'User explicitly selected N16-matched clipping: N20 dt .0004, N24 dt 1/3000, expected penalty scale .651041666666. No change to equations, binary or old N16 acceptance.','files':{f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in NEW.iterdir() if f.is_file()}}
(NEW/'provenance.json').write_text(json.dumps(newprov,indent=2)+'\n')
(HERE/'launcher-provenance.json').write_text(json.dumps(newprov,indent=2)+'\n')
print(NEW)
