"""Namespace existing audited controller for fixed N16/N24 zero-restoration science."""
from pathlib import Path
import sys,json,hashlib,re,subprocess,argparse
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import native_transition_campaign as c
import native_transition_prepare as p
ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);a=ap.parse_args()
SOURCE=subprocess.check_output(['git','rev-parse',a.source],cwd=ROOT,text=True).strip()
DEST=ROOT/'output/zero-restoration/package-v1'
c.SOURCE_COMMIT=SOURCE;c.MANIFEST='source-zero-restoration-v1-manifest.json';c.SCIENCE='launcher-zero-restoration-v1/native-transition-science.json'
p.SOURCE_DIR='source-zero-restoration-v1';p.BUILD_DIR='build-sycl-mpi-zero-restoration-v1';p.LAUNCHER=Path(c.SCIENCE).parent
p.stage(ROOT,DEST,SOURCE)
launcher=DEST/p.LAUNCHER
for name in ['native_transition_campaign.py','native_transition_prepare.py','build_native_transition.sh']:
 s=(launcher/name).read_text().replace('native-transition-v1','zero-restoration-v1').replace('fd4652ff69d55ec973b61029b4e77609e49f0860',SOURCE)
 if name=='native_transition_campaign.py':
  s=s.replace('n16_native_initializer_shared_transition_controls_to10M','n16_n24_original_native_A_zero_shift_restoration_to10M')
  s=re.sub(r'VARIANTS = \{.*?\n\}',"VARIANTS = {v: dict(suffix='control_zero_restoration', magic='NZ1', shared=False, ko10=False, scri6=False, n=n) for v,n in [('Z16',16),('Z24',24)]}",s,count=1,flags=re.S)
  s=s.replace('def cells(n=N):','def resolution(variant):\n    n=VARIANTS[variant][\'n\'];return n, .0005 if n==16 else 1/3000\n\n\ndef cells(n=N):')
  for sig in ["def required_checks(variant, purpose, kind, restart=False):", "def inspect_checkpoint(path,variant,expected_sha,kind='puncture',end=END):", "def make_plan(root,variant,purpose='evolution',segment=1,restart=None,expected_sha=None):"]:
   s=s.replace(sig,sig+'\n    N,DT=resolution(variant)')
  s=s.replace("variant=plan.get('variant');purpose=plan.get('purpose');segment=plan.get('segment')","variant=plan.get('variant');purpose=plan.get('purpose');segment=plan.get('segment')\n    N,DT=resolution(variant)")
  s=s.replace(".001 if purpose=='smoke'", "2*DT if purpose=='smoke'")
  s=s.replace('cells()', 'cells(N)').replace("prefix=f'campaigns/n16-zero-restoration-v1-", "prefix=f'campaigns/n{N}-zero-restoration-v1-")
  s=s.replace("exact=dict(status='completed'","exact=dict(zero_shift_restoration=True,restoring_coefficient=0,status='completed'")
  s=s.replace('all five frozen binaries','the frozen zero-restoration binary').replace('five fresh N16 controls','fresh N16/N24 zero-restoration controls')
 if name=='native_transition_prepare.py':
  s=s.replace('planned_new_bytes=1_000_000_000','planned_new_bytes=400_000_000').replace('atomic_checkpoint_reserve_bytes=500_000_000','atomic_checkpoint_reserve_bytes=100_000_000')
  s=s.replace("expected_defs=dict(HYPERBOLIC_NATIVE_WORMHOLE='1'","expected_defs=dict(HYPERBOLIC_ZERO_SHIFT_RESTORATION='1',HYPERBOLIC_NATIVE_WORMHOLE='1'")
  s=s.replace('for v in campaign.VARIANTS.values()]]]',"for v in campaign.VARIANTS.values()], 'hyperbolic_shift_3d_zero_scri_dissipation_closure']]")
  s=s.replace("control.atomic(root/campaign.SCIENCE,frozen)","frozen['audit_binaries']={BUILD_DIR+'/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_zero_scri_dissipation_closure':control.digest(root/BUILD_DIR/'tst/hyperbolic_shift_3d/hyperbolic_shift_3d_zero_scri_dissipation_closure')}\n    control.atomic(root/campaign.SCIENCE,frozen)")
 (launcher/name).write_text(s)
 compile(s,name,'exec') if name.endswith('.py') else None
prov=json.loads((launcher/'package-provenance.json').read_text())
prov['namespace_adapter']={'path':str(Path(__file__).relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_launcher_commit':SOURCE}
prov['files']={name:p.identity(launcher/name) for name in p.FILES}
(launcher/'package-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps({'destination':str(DEST),'source_commit':SOURCE,'manifest_sha256':p.control.digest(DEST/c.MANIFEST),'source_bytes':sum(x.stat().st_size for x in (DEST/p.SOURCE_DIR).rglob('*') if x.is_file())},indent=2))
