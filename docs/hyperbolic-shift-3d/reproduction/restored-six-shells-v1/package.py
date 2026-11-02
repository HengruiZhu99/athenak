"""Namespace the audited native controller for the combined restored-A experiment."""
from pathlib import Path
import sys,json,hashlib,re
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tst/hyperbolic_shift_3d/aurora'))
import native_transition_campaign as c
import native_transition_prepare as p
SOURCE=sys.argv[1]
import subprocess
SOURCE=subprocess.check_output(['git','rev-parse',SOURCE],cwd=ROOT,text=True).strip()
DEST=ROOT/'output/restored-six-shells/package-v1'
c.SOURCE_COMMIT=SOURCE;c.MANIFEST='source-restored-six-shells-v1-manifest.json';c.SCIENCE='launcher-restored-six-shells-v1/native-transition-science.json'
p.SOURCE_DIR='source-restored-six-shells-v1';p.BUILD_DIR='build-sycl-mpi-restored-six-shells-v1';p.LAUNCHER=Path(c.SCIENCE).parent
p.stage(ROOT,DEST,SOURCE)
launcher=DEST/p.LAUNCHER
for name in ['native_transition_campaign.py','native_transition_prepare.py','build_native_transition.sh']:
 s=(launcher/name).read_text().replace('native-transition-v1','restored-six-shells-v1').replace('fd4652ff69d55ec973b61029b4e77609e49f0860',SOURCE)
 if name=='native_transition_campaign.py':
  s=s.replace('n16_native_initializer_shared_transition_controls_to10M','n16_A_restored_broad_collar_smooth_initial_slice_to10M')
  s=re.sub(r'VARIANTS = \{.*?\n\}',"VARIANTS = {'A': dict(suffix='control_broad_smooth_six_shells', magic='N61', shared=False, ko10=False, scri6=False)}",s,count=1,flags=re.S)
  s=s.replace("exact=dict(status='completed'","exact=dict(mesh_topology='cartesian_2x2x2_shells6',mesh_blocks=44,shell_layers=6,shell_outer_radii=[.305,.45,.6,.75,.875,1],restoring_collar_begin=.6,restoring_collar_end=.9,restoring_coefficient=4,zero_shift_restoration=False,broad_smooth_a=True,native_initialization_profile='polynomial_c6',native_initialization_start_M=4,native_initialization_end_M=12,status='completed'")
  s=s.replace('all five frozen binaries','the frozen wider-collar binary').replace('five fresh N16 controls','fresh N16 wider-collar control')
 if name=='native_transition_campaign.py':
  s=s.replace('return 12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2','return 30*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2')
  config=bytearray(c.CONFIG);config[40:44]=(6).to_bytes(4,'little')
  s=re.sub(r'^CONFIG = .*$',lambda _: 'CONFIG = '+repr(bytes(config)),s,count=1,flags=re.M)
 if name=='native_transition_prepare.py':
  s=s.replace('planned_new_bytes=1_000_000_000','planned_new_bytes=400_000_000').replace('atomic_checkpoint_reserve_bytes=500_000_000','atomic_checkpoint_reserve_bytes=100_000_000')
  s=s.replace("expected_defs=dict(HYPERBOLIC_NATIVE_WORMHOLE='1'","expected_defs=dict(HYPERBOLIC_SIX_SHELLS='1',HYPERBOLIC_BROAD_SMOOTH_A='1',HYPERBOLIC_NATIVE_WORMHOLE='1'")
 if name=='native_transition_prepare.py':
  s=s.replace("('HYPERBOLIC_SHARED_RESTORATION',v['shared'])","('HYPERBOLIC_ZERO_SHIFT_RESTORATION',False),('HYPERBOLIC_WIDE_RESTORING_COLLAR',False),('HYPERBOLIC_SHARED_RESTORATION',v['shared'])")
 (launcher/name).write_text(s)
prov=json.loads((launcher/'package-provenance.json').read_text())
prov['namespace_adapter']={'path':str(Path(__file__).relative_to(ROOT)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_launcher_commit':SOURCE}
prov['files']={name:p.identity(launcher/name) for name in p.FILES}
(launcher/'package-provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
print(json.dumps({'destination':str(DEST),'source_commit':SOURCE,'manifest_sha256':p.control.digest(DEST/c.MANIFEST),'source_bytes':sum(x.stat().st_size for x in (DEST/p.SOURCE_DIR).rglob('*') if x.is_file())},indent=2))
