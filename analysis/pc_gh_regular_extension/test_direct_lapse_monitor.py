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
if __name__=='__main__':unittest.main()
