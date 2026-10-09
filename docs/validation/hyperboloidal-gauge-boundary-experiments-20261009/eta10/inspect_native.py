"""Check actual active native arrays and preserve finite-duration diagnostics."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

p=Path(__file__).resolve().parent
repo=p.parents[4]
spec=importlib.util.spec_from_file_location('reader',repo/'vis/python/bin_convert.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
report={'scope':'Actual native Cartesian arrays/RK3. Finite duration and positive fields do not establish nonlinear scri closure or stability.',
        'finite_Q_counterexample':{'scope':'a=1, beta=beta_ref; finite Q does not enforce nonlinear scri compatibility', 'delta_Q':.01,'limit_Omega_Qdot':.02,'limit_Theta_dot':-.02},'cases':[]}
for phase in sys.argv[1:]:
    result=json.loads((p/('native-'+phase)/'results.json').read_text())
    for case in result['cases']:
        d=p/('native-'+phase)/case['name']
        paths=sorted((d/'bin').glob('*.z4c.*.bin'))
        snapshots=[];first=None;maxchange=0
        for path in paths:
            raw=reader.read_binary(str(path));f={k:np.asarray(v) for k,v in raw['mb_data'].items()}
            mask=f['z4c_active'].astype(bool)
            finite=all(np.isfinite(v[mask]).all() for v in f.values())
            assert finite,path
            alpha=float(f['z4c_alpha'][mask].min());chi=float(f['z4c_chi'][mask].min())
            assert alpha>0 and chi>0,(path,alpha,chi)
            metric=np.zeros((int(mask.sum()),3,3))
            for i,j,suffix in [(0,0,'xx'),(0,1,'xy'),(0,2,'xz'),(1,1,'yy'),(1,2,'yz'),(2,2,'zz')]:
                metric[:,i,j]=metric[:,j,i]=f['z4c_g'+suffix][mask]/f['z4c_chi'][mask]
            ev=np.linalg.eigvalsh(metric)
            assert ev.min()>0,(path,ev.min())
            if first is None:first=f
            for k in f:maxchange=max(maxchange,float(np.max(abs(f[k][mask]-first[k][mask]))))
            snapshots.append({'file':path.name,'all_active_fields_finite':finite,'alpha_min':alpha,'chi_min':chi,
                              'physical_metric_eigen_min':float(ev.min()),'physical_metric_eigen_max':float(ev.max())})
        history=np.atleast_2d(np.loadtxt(d/'hyp.z4c.user.hst'))
        assert np.isfinite(history).all()
        if phase=='reference':
            assert maxchange<5e-12,maxchange
            assert max(case['diagnostics'][key] for key in ('H','M','Z','Theta','pole_deviation_max','null_deviation_max'))<1e-10
        report['cases'].append({'phase':phase,'name':case['name'],'executable_sha256':result['sha256'],
            'exit_status':case['exit_status'],'wall_seconds':case['wall_seconds'],'input_parameters':case['input_parameters'],
            'input_sha256':case['input_sha256'],'diagnostics':case['diagnostics'],'history_rows':history.tolist(),
            'native_snapshots':snapshots,'max_array_change_from_initial':maxchange,
            'native_log_sha256':hashlib.sha256((d/'run.log').read_bytes()).hexdigest()})
        print(phase,'regular snapshots',len(snapshots),'max array change',maxchange,
              'final H/M/Z',*[case['diagnostics'][key] for key in ('H','M','Z')],flush=True)
baseline=json.loads((repo/'build-layer-research/clean-wide-kappa10-long/results.json').read_text())
report['baseline']={'executable_sha256':baseline['sha256'],'case':baseline['cases'][0],
                    'scope':'Existing production physical-P/source-off wide a=.5 kappa10 symmetric d2 control; same dt=.000427734375. The only continuum gauge change is eta_p=10 beta-restoring pole.'}
for c in report['cases']:
    if c['phase']=='long':
        report['long_to_base_ratios']={key:c['diagnostics'][key]/baseline['cases'][0]['diagnostics'][key] for key in ('H','M','Z')}
report['overlay_header_sha256']=hashlib.sha256((p/'shift_injection.hpp').read_bytes()).hexdigest()
report['native_build_receipt_sha256']=hashlib.sha256((p/'native-build-receipt.json').read_bytes()).hexdigest()
(p/'native-experiment-report.json').write_text(json.dumps(report,indent=2,default=float)+'\n')
