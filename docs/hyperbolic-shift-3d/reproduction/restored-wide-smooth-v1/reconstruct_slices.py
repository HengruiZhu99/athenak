"""Read admitted endpoints with distinct schema-bound analysis executables."""
from pathlib import Path
import json,hashlib,subprocess,numpy as np
ROOT=Path(__file__).resolve().parents[4];P=Path(__file__).resolve().parent;D=P/'comparison/slices';D.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=[('a','control','native-shared-transition','n16-native-transition-v2-a-evolution-segment01'),('wide','control_wide_collar','wide-collar-a','n16-wide-collar-a-v1-a-evolution-segment01'),('broad','control_broad_smooth','restored-wide-smooth','n16-restored-wide-smooth-v1-a-evolution-segment01')]
records={}
for name,suffix,raw,plan in cases:
 campaign=ROOT/f'output/{raw}/aurora/campaigns/{plan}';cp=campaign/'output/puncture.checkpoint';report=json.loads((campaign/'output/puncture.report.json').read_text());assert report['checkpoint_admitted'] and report['t_over_M']==10
 target='hyperbolic_shift_3d_checkpoint_slices_native_'+suffix;b=ROOT/'build-restored-slices-v1/tst/hyperbolic_shift_3d'/target
 flags=b.parent/'CMakeFiles'/(target+'.dir')/'flags.make';prefix=D/name;assert not prefix.with_suffix('.csv').exists()
 h=sha(cp);cmd=[str(b),'16',str(cp),'.0005',str(prefix)]
 rec={'command':cmd,'checkpoint_sha256':h,'binary_sha256':sha(b),'flags':flags.read_text(),'source_sha256':sha(ROOT/'tst/hyperbolic_shift_3d/checkpoint_slices.cpp'),'checkpoint_schema':report['checkpoint_schema'],'campaign_plan_sha256':sha(campaign/'plan.json')}
 (D/f'{name}-command.json').write_text(json.dumps(rec,indent=2)+'\n')
 with (D/f'{name}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(cp)==h
 coefficients=np.loadtxt(campaign/'output/puncture.fastflow.horizon_shape_0.txt')[-1];assert len(coefficients)==169
 mean=coefficients[0]/np.sqrt(4*np.pi);bound=sum(np.sqrt((2*l+1)/(4*np.pi))*np.linalg.norm(coefficients[l*l:(l+1)**2]) for l in range(1,13))
 assert abs(mean-report['horizon_radius'])<1e-6;assert mean+bound<.12
 rec['final_horizon_radius_upper_bound']=float(mean+bound);rec['horizon_bound_method']='Addition theorem/Cauchy-Schwarz bound from saved real orthonormal harmonic coefficients; includes all l<=12. Output coefficient precision is ~7 digits.'
 records[name]=rec
 print(name,'done, horizon bound',mean+bound,flush=True)
(D/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
