from pathlib import Path
import argparse,json,hashlib,subprocess
R=Path.cwd()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
D=a.output.resolve();D.mkdir(parents=True,exist_ok=True)
assert not any((D/f'{name}-radial.json').exists() for name in ['a','wide']), 'Native output paths must be fresh'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cuts='0,0.03,0.06,0.1099,0.12,0.13,0.16,0.18,0.2,0.25,0.3,0.4,0.45,0.55,0.6,0.675,0.7,0.725,0.75,0.775,0.8,0.85,0.9,0.95,1'
for name,build,target,raw,plan in [('a','build-native-shared-transition','native_control','native-shared-transition','n16-native-transition-v2-a-evolution-segment01'),('wide','build-native-initial-smoothing-v1','native_control_wide_collar','wide-collar-a','n16-wide-collar-a-v1-a-evolution-segment01')]:
 b=R/build/'tst/hyperbolic_shift_3d'/('hyperbolic_shift_3d_checkpoint_constraints_'+target)
 cp=R/f'output/{raw}/aurora/campaigns/{plan}/output/puncture.checkpoint'
 h=sha(cp);cmd=[str(b),'16',str(cp),'.0005','.12',str(D/f'{name}-radial.json'),cuts]
 flags=R/build/'tst/hyperbolic_shift_3d/CMakeFiles'/('hyperbolic_shift_3d_checkpoint_constraints_'+target+'.dir/flags.make')
 rec={'command':cmd,'binary_sha256':sha(b),'checkpoint_sha256':h,'flags':flags.read_text(),'flags_sha256':sha(flags),'analysis_only':True,'cutoff_reason':'Fixed radius .12 exceeds both final mean horizon radii .1097383 and .1131596; cumulative integrals also retain .1099 and .13 for cutoff sensitivity.'}
 (D/f'{name}-radial-command.json').write_text(json.dumps(rec,indent=2)+'\n')
 with (D/f'{name}-radial.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(cp)==h
 print(name,'done',flush=True)
