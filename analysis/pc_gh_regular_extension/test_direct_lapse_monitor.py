"""Ensure a failed/replayed interval cannot contaminate the accepted growth curve."""
import csv
import gzip
from pathlib import Path
import tempfile
import unittest
from reduce_direct_lapse_monitor import reduce

class Rollback(unittest.TestCase):
    def test_completed_and_stage_envelopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw.csv'
            with raw.open('w') as f:
                w=csv.writer(f)
                w.writerow('cycle,t_step,dt,stage,operation,phase,region,quantity,max,coordinate_l1,x,y,z,level,block,rank'.split(','))
                for cycle,value in [(0,1),(1,2),(2,99),(1,3),(2,4)]:
                    w.writerow([cycle,cycle*.1,.1,3,-2,'before','all','curl_L',value,value,0,0,0,0,0,0])
            reduce(raw,root/'out')
            with gzip.open(root/'out/completed-monitor.csv.gz','rt') as f:
                self.assertEqual([float(r['max']) for r in csv.DictReader(f)],[1,3,4])
            with gzip.open(root/'out/stage-envelopes.csv.gz','rt') as f:
                self.assertEqual([float(r['max']) for r in csv.DictReader(f)],[4])
    def test_paired_norm_changes_exclude_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw.csv'
            with raw.open('w') as f:
                w=csv.writer(f)
                w.writerow('cycle,t_step,dt,stage,operation,phase,region,quantity,max,coordinate_l1,x,y,z,level,block,rank'.split(','))
                for cycle,before,after in [(0,1,2),(1,2,3),(2,0,999),(1,10,11),(2,20,24)]:
                    for phase,value in [('before',before),('after',after)]:
                        w.writerow([cycle,cycle*.1,.1,3,2,phase,'all','curl_Q',value,value,0,0,0,0,0,0])
            reduce(raw,root/'out')
            with gzip.open(root/'out/operation-norm-increases.csv.gz','rt') as f:
                rows=list(csv.DictReader(f))
                self.assertEqual([float(r['delta_max_norm']) for r in rows],[4])
                self.assertEqual(float(rows[0]['before_max']),20)
                self.assertEqual(float(rows[0]['after_max']),24)

if __name__=='__main__':unittest.main()
