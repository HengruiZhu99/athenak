#!/usr/bin/env python3
"""Analytic difference ratios and real on-disk evidence-rejection fixtures."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from analyze_telegraph_control import compare
from analyze_telegraph_time import load
from test_telegraph_time_analysis import profile


def run(n,amplitude,**args):
    d=profile(n,amplitude);d.update(beta=-d['r'],gauge_speeds=np.c_[-d['r'],d['r']])
    return dict(manifest=dict(arguments=dict(n=n,cfl=.05,kappa=.005,tau=.02,quadrature_width=7,puncture_ko=.5,t=2.,report_dt=1.,**args)),
                binding={'numeric':'same'},data={t:copy.deepcopy(d) for t in [0.,1.,2.]})


def on_disk(folder,synthetic):
    folder.mkdir();root=folder/'frozen-source';root.mkdir()
    files={'run_telegraph.py':'import support\n','support.py':'VALUE=1\n','fast_integrator.cpp':'// provenance fixture\n','telegraph_integrator.cpp':'// provenance fixture\n'}
    for name,text in files.items():(root/name).write_text(text)
    for name in ['geometry.dylib','fast.dylib']:(root/name).write_bytes(b'Explicit nonexecuted provenance-test fixture')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    m=copy.deepcopy(synthetic['manifest']);m.update(status='finished',time=2.)
    m['arguments'].update(library='geometry.dylib',fast_library='fast.dylib',output=str(folder))
    m['binding']=dict(source_sha256={name:digest(root/name) for name in files},library_sha256=digest(root/'geometry.dylib'),fast_sha256=digest(root/'fast.dylib'))
    (folder/'manifest.json').write_text(json.dumps(m))
    for i,(time,data) in enumerate(synthetic['data'].items()):np.savez(folder/f'profile-{i:04d}.npz',**data,t=time)
    return folder


class Controls(unittest.TestCase):
    def setUp(self):
        self.base=run(17,.001);self.space=run(33,0.)
        self.control=run(17,.00101);self.control['manifest']['arguments']['quadrature_width']=9

    def test_known_ratio_and_false_pass(self):
        result=compare(self.base,self.space,[self.control],'quadrature_width')
        self.assertEqual(result['status'],'passed_quadrature_difference_checks')
        for key in ['fields_bulk','H_rms','B_max','ray_logJ_max']:
            self.assertAlmostEqual(result['controls'][0]['metrics'][key]['raw_control_to_space_fraction'],.01,places=11)
        bad=copy.deepcopy(self.control)
        for d in bad['data'].values():d['q_physical']+=.1
        result=compare(self.base,self.space,[bad],'quadrature_width')
        self.assertEqual(result['status'],'failed_quadrature_difference_checks')
        self.assertEqual(result['goal_status'],'incomplete')

    def test_zero_spatial_reference(self):
        base,space,ctrl=run(17,0),run(33,0),run(17,0);ctrl['manifest']['arguments']['quadrature_width']=9
        self.assertEqual(compare(base,space,[ctrl],'quadrature_width')['status'],'passed_quadrature_difference_checks')
        for d in ctrl['data'].values():d['B'][3]=1e-10
        result=compare(base,space,[ctrl],'quadrature_width')
        self.assertIsNone(result['controls'][0]['metrics']['B_max']['raw_control_to_space_fraction'])
        self.assertEqual(result['status'],'failed_quadrature_difference_checks')

    def test_no_KO_admission_or_parameter_mixing(self):
        ctrl=copy.deepcopy(self.base);ctrl['manifest']['arguments']['puncture_ko']=.25
        result=compare(self.base,self.space,[ctrl],'puncture_ko')
        self.assertTrue(result['status'].endswith('requires_three_grid_assessment'))
        self.assertIsNone(result['controls'][0]['metrics']['H_rms']['below_ten_percent'])
        ctrl['manifest']['arguments']['kappa']=.006
        with self.assertRaisesRegex(ValueError,'Uncontrolled'):compare(self.base,self.space,[ctrl],'puncture_ko')
        with self.assertRaises(ValueError):compare(self.base,self.space,[self.control,self.control],'quadrature_width')
        with self.assertRaises(ValueError):compare(self.base,self.base,[self.control],'quadrature_width')

    def test_nonfinite_outside_field_comparison_regions_rejected(self):
        # r=.875 is outside both field norm masks; it must still be finite.
        ctrl=copy.deepcopy(self.control);ctrl['data'][1.]['q_physical'][14,2]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):compare(self.base,self.space,[ctrl],'quadrature_width')
        ctrl=copy.deepcopy(self.control);ctrl['data'][1.]['diagnostic'][14,10]=np.nan
        with self.assertRaisesRegex(ValueError,'interior mass'):compare(self.base,self.space,[ctrl],'quadrature_width')

    def test_frozen_source_executable_coverage_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            path=on_disk(Path(temp)/'baseline',self.base);loaded=load(path)
            self.assertEqual(set(loaded['data']),{0.,1.,2.})
            source=path/'frozen-source/support.py';original=source.read_bytes();source.write_text('VALUE=2\n')
            with self.assertRaisesRegex(ValueError,'Changed frozen source'):load(path)
            source.write_bytes(original)
            exe=path/'frozen-source/fast.dylib';original=exe.read_bytes();exe.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'Changed executable'):load(path)
            exe.write_bytes(original)
            snapshot=path/'profile-0001.npz';saved=snapshot.read_bytes();snapshot.unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):load(path)
            snapshot.write_bytes(saved)
            manifest=path/'manifest.json';m=json.loads(manifest.read_text());m['status']='running';manifest.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'completed run'):load(path)


    def test_ray_order_and_terminal_coincidences(self):
        from analyze_telegraph_ray_convergence import analyze
        runs=[]
        for n in [17,33,65]:
            item=run(n,0.);item['manifest']['arguments'].update(ray_start=1.,ray_seeds=2)
            seed=np.c_[np.repeat([.2,.4],2),np.zeros(4)]
            for t,d in item['data'].items():
                d['rays']=seed.copy()
                if t==2.:
                    amp=1/(n-1)**2
                    d['rays'][::2,0]=np.array([.2,.4])*np.exp(-1)+amp*np.array([1,2])
                    d['rays'][1::2,0]=1.
                    d['rays'][:,1]=np.array([-1,.4,-1,.4])+amp*np.arange(1,5)
            runs.append(item)
        result=analyze(runs)
        self.assertAlmostEqual(result['spatial_orders'][0]['logJ'],2.,places=12)
        self.assertAlmostEqual(result['spatial_orders'][0]['position'],2.,places=12)
        terminal=result['histories'][0]['rows'][-1]['families'][1]
        self.assertEqual(terminal['terminal'],2);self.assertEqual(terminal['active_coincidences'],0)
        self.assertIsNone(terminal['min_active_neighbor_gap'])
        self.assertEqual(result['goal_status'],'incomplete')
        bad=copy.deepcopy(runs[0]);bad['data'][2.]['rays'][[0,2],0]=bad['data'][2.]['rays'][[2,0],0]
        with self.assertRaisesRegex(ValueError,'ordering reversed'):analyze([bad])
        bad=copy.deepcopy(runs[0]);bad['data'][1.]['rays'][0,1]=.001
        with self.assertRaisesRegex(ValueError,'activation'):analyze([bad])
        with self.assertRaisesRegex(ValueError,'Repeated'):analyze([runs[0],runs[0]])

if __name__=='__main__':unittest.main()
