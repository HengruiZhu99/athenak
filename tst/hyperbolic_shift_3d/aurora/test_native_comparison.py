"""Synthetic comparison-validation tests; not evolution results."""
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import unittest
import control
from test_native_transition_campaign import NativeTransitionTests
import native_transition_campaign as campaign

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('native_comparison',ROOT/'docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/compare_histories.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
spec=importlib.util.spec_from_file_location('smoke_comparison',ROOT/'docs/hyperbolic-shift-3d/reproduction/native-shared-transition-v1/compare_smokes.py')
smokes=importlib.util.module_from_spec(spec);spec.loader.exec_module(smokes)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.fixture=NativeTransitionTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.root=self.fixture.root.resolve()
        science=self.root/campaign.SCIENCE
        control.atomic(science.with_name('provenance.json'),dict(files={science.name:dict(sha256=control.digest(science))}))

    def create(self,variant='A'):
        relative,plan,worker=self.fixture.finish(variant)
        history=self.root/(plan['commands'][0]['args'][1]+'.history.csv')
        history.write_text('t,t_over_M\n0,0\n0.01,0.03\n')
        worker['cases'][0]['artifacts'][str(history.relative_to(self.root))]=dict(bytes=history.stat().st_size,sha256=control.digest(history))
        control.atomic((self.root/relative).with_name('worker-status.json'),worker)
        return relative,plan,worker

    def test_accept_all_variants_without_remote_binaries(self):
        for variant in campaign.VARIANTS:
            relative,_,_=self.create(variant)
            result=comparison.load_native(self.root,variant,[relative])
            self.assertEqual(result['history']['t'][-1],.01)
            self.assertEqual(result['reports'][-1]['status'],'checkpointed')

    def test_reject_changed_report_even_with_new_artifact_hash(self):
        relative,plan,worker=self.create();name=plan['commands'][0]['report'];p=self.root/name
        original=json.loads(p.read_text())
        for key,value in [('backend','Serial'),('inner_gauge','wrong'),('physical_lapse_floor',1e-5),
                          ('checkpoint_schema','HYP3D_MPI_NS2'),('checkpoint_admitted',False),('steps',21)]:
            control.atomic(p,dict(original,**{key:value}))
            worker['cases'][0]['artifacts'][name]=dict(bytes=p.stat().st_size,sha256=control.digest(p))
            control.atomic((self.root/relative).with_name('worker-status.json'),worker)
            with self.assertRaises(ValueError):comparison.load_native(self.root,'A',[relative])

    def test_reject_scheduler_or_history_mismatch(self):
        relative,plan,worker=self.create();p=self.root/relative
        collected=p.with_name('collected.json');saved=json.loads(collected.read_text())
        bad=dict(saved,scheduler=dict(job_state='F',Exit_status=0));control.atomic(collected,bad)
        with self.assertRaises(ValueError):comparison.load_native(self.root,'A',[relative])
        control.atomic(collected,saved)
        name=plan['commands'][0]['args'][1]+'.history.csv';history=self.root/name
        history.write_text('t,t_over_M\n0,0\n0.02,0.06\n')
        worker['cases'][0]['artifacts'][name]=dict(bytes=history.stat().st_size,sha256=control.digest(history))
        control.atomic(p.with_name('worker-status.json'),worker)
        with self.assertRaises(ValueError):comparison.load_native(self.root,'A',[relative])

    def test_smoke_physical_parity_and_missing_case_guard(self):
        relative,plan,worker=self.fixture.finish('A','smoke')
        campaign.verify(self.root,relative)
        prefix=self.root/'serial-a'
        for command in plan['commands']:
            kind=command['kind'];report=json.loads((self.root/command['report']).read_text())
            report.update(backend='Serial',mpi_size=1)
            control.atomic(Path(str(prefix)+'-'+kind+'.report.json'),report)
            shutil.copyfile(self.root/(command['args'][1]+'.checkpoint'),Path(str(prefix)+'-'+kind+'.checkpoint'))
        result=smokes.compare(prefix,self.root,relative)
        self.assertTrue(all(x['physical_parity_within_1e_10'] for x in result['cases']))
        # Corrupt only the local comparison state, preserving header/admission;
        # the measured physical disagreement must be visible, never suppressed.
        local=Path(str(prefix)+'-puncture.checkpoint');data=bytearray(local.read_bytes())
        struct.pack_into('<d',data,168+8,1.1);local.write_bytes(data)
        result=smokes.compare(prefix,self.root,relative)
        self.assertFalse(result['cases'][1]['physical_parity_within_1e_10'])
        path=(self.root/relative).with_name('worker-status.json')
        worker['cases']=worker['cases'][:1];control.atomic(path,worker)
        with self.assertRaises(AssertionError):smokes.compare(prefix,self.root,relative)


if __name__=='__main__':unittest.main()
