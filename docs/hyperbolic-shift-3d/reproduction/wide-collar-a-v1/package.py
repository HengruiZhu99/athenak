"""Namespace the audited native controller for the single A-width experiment."""
from pathlib import Path
import sys,json,hashlib,re
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import native_transition_campaign as c
import native_transition_prepare as p
SOURCE='6894a0e32'
import subprocess
SOURCE=subprocess.check_output(['git','rev-parse',SOURCE],cwd=ROOT,text=True).strip()
DEST=ROOT/'output/wide-collar-a/package-v1'
c.SOURCE_COMMIT=SOURCE;c.MANIFEST='source-wide-collar-a-v1-manifest.json';c.SCIENCE='launcher-wide-collar-a-v1/native-transition-science.json'
p.SOURCE_DIR='source-wide-collar-a-v1';p.BUILD_DIR='build-sycl-mpi-wide-collar-a-v1';p.LAUNCHER=Path(c.SCIENCE).parent
p.stage(ROOT,DEST,SOURCE)
launcher=DEST/p.LAUNCHER
for name in ['native_transition_campaign.py','native_transition_prepare.py','build_native_transition.sh']:
 s=(launcher/name).read_text().replace('native-transition-v1','wide-collar-a-v1').replace('fd4652ff69d55ec973b61029b4e77609e49f0860',SOURCE)
 if name=='native_transition_campaign.py':
  s=s.replace('n16_native_initializer_shared_transition_controls_to10M','n16_A_wider_restoring_collar_user_authorized_to10M')
  s=re.sub(r'VARIANTS = \{.*?\n\}',"VARIANTS = {'A': dict(suffix='control_wide_collar', magic='NW1', shared=False, ko10=False, scri6=False)}",s,count=1,flags=re.S)
  s=s.replace("exact=dict(status='completed'","exact=dict(restoring_collar_begin=.7,restoring_collar_end=.9,restoring_coefficient=4,status='completed'")
  s=s.replace('all five frozen binaries','the frozen wider-collar binary').replace('five fresh N16 controls','fresh N16 wider-collar control')
 if name=='native_transition_prepare.py':
  s=s.replace('planned_new_bytes=1_000_000_000','planned_new_bytes=400_000_000').replace('atomic_checkpoint_reserve_bytes=500_000_000','atomic_checkpoint_reserve_bytes=100_000_000')
  s=s.replace("expected_defs=dict(HYPERBOLIC_NATIVE_WORMHOLE='1'","expected_defs=dict(HYPERBOLIC_WIDE_RESTORING_COLLAR='1',HYPERBOLIC_NATIVE_WORMHOLE='1'")
 (launcher/name).write_text(s)
prov=json.loads((launcher/'package-provenance.json').read_text())
prov['namespace_adapter']={'path':str(Path(__file__).relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_launcher_commit':SOURCE}
prov['files']={name:p.identity(launcher/name) for name in p.FILES}
(launcher/'package-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps({'destination':str(DEST),'source_commit':SOURCE,'manifest_sha256':p.control.digest(DEST/c.MANIFEST),'source_bytes':sum(x.stat().st_size for x in (DEST/p.SOURCE_DIR).rglob('*') if x.is_file())},indent=2))
