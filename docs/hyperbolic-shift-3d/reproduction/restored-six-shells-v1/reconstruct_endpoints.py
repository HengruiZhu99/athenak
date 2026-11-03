"""Qualified read-only endpoint reconstruction; never changes admission policy."""
from pathlib import Path
import argparse,json,hashlib,subprocess,shutil,numpy as np
from audit_completed import load_completed
ROOT=Path(__file__).resolve().parents[4];P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=P/'comparison');args=parser.parse_args();D=args.output;D.mkdir(parents=True,exist_ok=True);S=D/'slices';S.mkdir(exist_ok=True)
load_completed() # Exact completed-but-policy-failed case and all identities.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=[('a','build-native-shared-transition','control','native-shared-transition','n16-native-transition-v2-a-evolution-segment01'),('broad','build-restored-wide-smooth-v1','control_broad_smooth','restored-wide-smooth','n16-restored-wide-smooth-v1-a-evolution-segment01'),('six','build-restored-six-shells-v1','control_broad_smooth_six_shells','restored-six-shells','n16-restored-six-shells-v1-a-evolution-segment01')]
cuts='0,.03,.06,.1099,.12,.13,.16,.18,.2,.25,.3,.305,.4,.45,.55,.6,.675,.7,.725,.75,.775,.8,.85,.875,.9,.95,1'
records={}
for name,build,suffix,raw,plan in cases:
 campaign=ROOT/f'output/{raw}/aurora/campaigns/{plan}';cp=campaign/'output/puncture.checkpoint';report=json.loads((campaign/'output/puncture.report.json').read_text());assert report['checkpoint_admitted'] and report['t_over_M']==10
 h=sha(cp);coef=np.loadtxt(campaign/'output/puncture.fastflow.horizon_shape_0.txt')[-1];assert len(coef)==169
 bound=coef[0]/np.sqrt(4*np.pi)+sum(np.sqrt((2*l+1)/(4*np.pi))*np.linalg.norm(coef[l*l:(l+1)**2]) for l in range(1,13));assert bound<.12
 target='hyperbolic_shift_3d_checkpoint_constraints_native_'+suffix;b=ROOT/build/'tst/hyperbolic_shift_3d'/target;out=D/f'{name}-radial.json';assert not out.exists()
 cmd=[str(b),'16',str(cp),'.0005','.12',str(out),cuts];flags=b.parent/'CMakeFiles'/(target+'.dir/flags.make')
 rec=dict(command=cmd,binary_sha256=sha(b),checkpoint_sha256=h,flags=flags.read_text(),horizon_radius_upper_bound=float(bound),horizon_bound_method='Addition theorem/Cauchy-Schwarz from all l<=12 real orthonormal harmonic coefficients',checkpoint_native_admitted=True,campaign_accepted=name!='six',qualification='Six-shell original frozen acceptance remains failed: scri penalty scale .651 instead of1; native-completed endpoint only, not a matched-parameter convergence claim.')
 (D/f'{name}-radial-command.json').write_text(json.dumps(rec,indent=2)+'\n')
 with (D/f'{name}-radial.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(cp)==h
 if name!='six':
  for ext in ['csv','json']:
   src=P.parent/'restored-wide-smooth-v1/comparison/slices'/f'{name}.{ext}';dst=S/f'{name}.{ext}';assert not dst.exists();shutil.copy2(src,dst);rec.setdefault('reused_slices',{})[ext]=dict(path=str(src.relative_to(ROOT)),sha256=sha(src))
 else:
  target='hyperbolic_shift_3d_checkpoint_slices_native_'+suffix;b=ROOT/build/'tst/hyperbolic_shift_3d'/target;prefix=S/name;assert not prefix.with_suffix('.csv').exists()
  cmd=[str(b),'16',str(cp),'.0005',str(prefix)];flags=b.parent/'CMakeFiles'/(target+'.dir/flags.make')
  rec['slices']=dict(command=cmd,binary_sha256=sha(b),flags=flags.read_text(),source_sha256=sha(ROOT/'tst/hyperbolic_shift_3d/checkpoint_slices.cpp'))
  with (S/'six.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  assert sha(cp)==h
 records[name]=rec;(S/'provenance.json').write_text(json.dumps(records,indent=2)+'\n');print(name,'done',flush=True)
