"""Focused checks for newly authorized resolutions and four-host placement."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import control
import native_transition_campaign as campaign
import worker


class ResolutionPolicyTests(unittest.TestCase):
    def test_four_host_placement_and_reject_missing_host(self):
        nodes = [f'node{i}.example' for i in range(4)]
        rows = [dict(host=f'node{h}', rank=r, mask=f'{r//2}.{r%2}',
                     hierarchy='COMPOSITE') for h in range(4) for r in range(12)]
        self.assertTrue(worker.validate_affinity_records(rows, nodes))
        self.assertFalse(worker.validate_affinity_records(rows[:24], nodes[:2]))
        rows[-1]['mask'] = '0.0'
        self.assertFalse(worker.validate_affinity_records(rows, nodes))

    def test_resolution_plans_and_reject_wrong_dt_or_rank_count(self):
        repo = Path(__file__).resolve().parents[4]
        science = repo/'output/boosted-refined-core/aurora/launcher-boosted-refined-core-cfl-v2/native-transition-science.json'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/campaign.SCIENCE
            target.parent.mkdir(parents=True)
            target.write_bytes(science.read_bytes())
            for variant, n, dt, cells in [('C20',20,.0002,969344),('C24',24,1/6000,1621728)]:
                result = campaign.make_plan(root, variant)
                plan = json.loads((root/result['plan']).read_text())
                self.assertEqual(plan['resources']['nodes'], 4)
                self.assertEqual(plan['resources']['ranks'], 48)
                self.assertEqual(float(plan['commands'][0]['args'][4]), dt)
                self.assertEqual(campaign.cells(n), cells)
                for alteration in ['dt', 'ranks']:
                    bad = copy.deepcopy(plan)
                    if alteration == 'dt': bad['commands'][0]['args'][4] = str(dt*2)
                    else: bad['resources']['ranks'] = 24
                    with self.assertRaises(ValueError): control.validate_plan(root, bad)


if __name__ == '__main__': unittest.main()
