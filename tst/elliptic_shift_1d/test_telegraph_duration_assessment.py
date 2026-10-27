"""Negative controls for the duration acceptance gate; no numerical evolution."""
import copy
import json
from pathlib import Path
import unittest

from assess_telegraph_duration import assess

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'docs/elliptic-shift-1d/evidence/telegraph-attenuated-interim-v1/n1025'


class DurationAssessment(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads((EXPORT/'summary.json').read_text())
        self.manifest = json.loads((EXPORT/'manifest-long.json').read_text())
        self.protocol = json.loads((EXPORT/'protocol.json').read_text())

    def run_assessment(self):
        return assess(self.summary, self.manifest, self.protocol)

    def test_healthy_interim_cannot_pass_duration(self):
        result = self.run_assessment()
        self.assertTrue(result['accuracy_to_saved_cutoff'])
        self.assertEqual(result['verdict'], 'pending')
        self.summary['status'] = self.manifest['status'] = 'finished'
        self.assertEqual(self.run_assessment()['verdict'], 'pending')

    def test_failure_with_healthy_exterior_cannot_pass(self):
        self.summary['status'] = self.manifest['status'] = 'failed'
        self.assertEqual(self.run_assessment()['verdict'], 'failed')

    def test_synthetic_complete_coverage_passes_gate_only(self):
        # Artificial repeated rows test gate logic; they are not evolution evidence.
        row = self.summary['rows'][-1]
        self.summary['rows'] = [dict(row, t=round(i*.05, 9)) for i in range(10001)]
        self.summary['status'] = self.manifest['status'] = 'finished'
        self.summary['last_saved_time'] = self.summary['integrator_time'] = 500.
        self.manifest['time'] = 500.
        self.assertEqual(self.run_assessment()['verdict'], 'passed_duration_budget')

    def test_missing_report_rejected(self):
        self.summary['rows'].pop(2)
        with self.assertRaises(ValueError):
            self.run_assessment()

    def test_missing_horizon_and_nonfinite_constraint_fail(self):
        for field, value in [('AH_mass_error', None), ('H_exterior_rms', float('nan'))]:
            original = copy.deepcopy(self.summary)
            self.summary['rows'][-1][field] = value
            self.assertEqual(self.run_assessment()['verdict'], 'failed')
            self.summary = original

    def test_mass_budget_and_stage_boundary_fail(self):
        self.summary['rows'][-1]['AH_mass_error'] = 1e-4
        self.assertEqual(self.run_assessment()['verdict'], 'failed')
        self.setUp()
        self.manifest['all_RK_stages']['max_scri_pole'] = 1e-4
        self.assertEqual(self.run_assessment()['verdict'], 'failed')

    def test_coarse_grid_cannot_be_promoted(self):
        self.summary['resolution'] = 513
        with self.assertRaises(ValueError):
            self.run_assessment()


if __name__ == '__main__':
    unittest.main()
