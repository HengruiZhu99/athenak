"""Regression tests for two-node local-rank affinity validation."""
import copy
import json
from pathlib import Path
import unittest
import worker


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.nodes = ['node0.example', 'node1.example']
        self.rows = [dict(host=f'node{h}', rank=r, mask=f'{r//2}.{r%2}',
                          hierarchy='COMPOSITE') for h in range(2) for r in range(12)]

    def test_valid_per_node_local_ranks(self):
        self.assertTrue(worker.validate_affinity_records(self.rows, self.nodes))

    def test_reject_missing_duplicate_wrong_host_mask_hierarchy_and_global_rank(self):
        variants = [self.rows[:-1], self.rows + [self.rows[0]]]
        for key, value in [('rank', 0), ('host', 'unallocated'), ('mask', '0.0'),
                           ('hierarchy', 'FLAT'), ('rank', 23)]:
            rows = copy.deepcopy(self.rows)
            rows[-1][key] = value
            variants.append(rows)
        rows = copy.deepcopy(self.rows)
        for row in rows: row['host'] = 'node0'
        variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows[-1]):
                self.assertFalse(worker.validate_affinity_records(rows, self.nodes))

    def test_recorded_aurora_failure_was_valid_placement(self):
        path = Path(__file__).resolve().parents[4] / 'output/boosted-refined-core/aurora/campaigns/n16-boosted-refined-core-cfl-v1-c16-evolution-segment01/worker-status.json'
        if not path.exists(): self.skipTest('Preserved Aurora evidence not available')
        status = json.loads(path.read_text())
        self.assertFalse(status['affinity']['passed'])
        self.assertEqual(status['cases'], [])
        self.assertTrue(worker.validate_affinity_records(status['affinity']['records'], status['allocation_nodes']))


if __name__ == '__main__': unittest.main()
