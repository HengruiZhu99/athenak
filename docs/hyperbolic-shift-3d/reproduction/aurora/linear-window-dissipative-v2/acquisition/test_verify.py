#!/usr/bin/env python3
"""Acquisition success and scientific admission must remain distinct."""
import csv
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

s=importlib.util.spec_from_file_location('acquisition_verifier',Path(__file__).with_name('verify.py'))
v=importlib.util.module_from_spec(s);s.loader.exec_module(v)


class AcquisitionTests(unittest.TestCase):
    def test_even_small_values_do_not_admit_scientifically(self):
        result=v.reference_stability(dict(max_state_deviation=1e-10,H_L2=1e-12,M_L2=1e-12,checkpoint_admitted=True))
        self.assertTrue(result['all_reference_limits_passed'])
        self.assertTrue(result['scientific_not_admitted'])

    def test_reference_limits_are_unchanged(self):
        result=v.reference_stability(dict(max_state_deviation=2e-8,H_L2=2e-7,M_L2=2e-7))
        self.assertFalse(result['all_reference_limits_passed'])
        self.assertTrue(all(not c['passed'] for c in result['checks']))
        self.assertEqual([c['limit'] for c in result['checks']],[1e-8,1e-7,1e-7])

    def test_missing_nonfinite_and_bool_are_not_scientific_passes(self):
        for bad in [None,True,float('nan'),float('inf'),'1e-12']:
            with self.subTest(value=bad):
                result=v.reference_stability(dict(max_state_deviation=bad,H_L2=0,M_L2=0))
                self.assertFalse(result['all_reference_limits_passed'])

    def test_trajectory_failure_survives_small_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'h.csv'
            path.write_text('t,steps,state_deviation,H_L2,M_L2\n0,0,0,0,0\n0.1,200,2e-8,0,0\n0.2,400,1e-9,0,0\n')
            report=dict(t=.2,diagnostic_time=.2,steps=400,max_state_deviation=1e-9,H_L2=0,M_L2=0)
            result=v.history_summary(path,report)
            self.assertFalse(result['reference_limits_on_history'][0]['all_recorded_samples_passed'])
            self.assertEqual(result['reference_limits_on_history'][0]['first_failed_recorded_time'],.1)

    def test_nonfinite_observation_serializes_without_erasing_failure(self):
        result=v.reference_stability(dict(max_state_deviation=float('inf'),H_L2=0,M_L2=0))
        paths=[];safe=v.json_safe(result,nonfinite_paths=paths)
        json.dumps(safe,allow_nan=False)
        self.assertFalse(safe['all_reference_limits_passed'])
        self.assertIsNone(safe['checks'][0]['value'])
        self.assertEqual(paths,[['checks',0,'value']])

    def test_nonfinite_history_metadata_is_serializable_and_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'h.csv'
            path.write_text('t,steps,state_deviation,H_L2,M_L2\n0,0,nan,0,0\n0.2,400,1e-9,0,0\n')
            report=dict(t=.2,diagnostic_time=.2,steps=400,max_state_deviation=1e-9,H_L2=0,M_L2=0)
            result=v.history_summary(path,report)
            paths=[];safe=v.json_safe(result,nonfinite_paths=paths)
            json.dumps(safe,allow_nan=False)
            self.assertFalse(safe['reference_limits_on_history'][0]['all_recorded_samples_passed'])
            self.assertEqual(safe['reference_limits_on_history'][0]['first_failed_recorded_time'],0)
            self.assertIsNone(safe['reference_limits_on_history'][0]['maximum'])
            self.assertEqual(paths,[['reference_limits_on_history',0,'maximum']])

    def test_stale_or_partial_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'h.csv'
            report=dict(t=.2,diagnostic_time=.2,steps=400,max_state_deviation=1e-9,H_L2=0,M_L2=0)
            for text in ['t,steps,state_deviation,H_L2,M_L2\n0.1,200,1e-9,0,0\n',
                         't,steps,state_deviation,H_L2,M_L2\n0.2,400,1e-9\n']:
                path.write_text(text)
                with self.assertRaises(ValueError):v.history_summary(path,report)


if __name__=='__main__':unittest.main()
