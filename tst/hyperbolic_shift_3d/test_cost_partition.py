import unittest
from plan_cost_partition import plan

class CostPartitionTests(unittest.TestCase):
    def report(self):
        return dict(status='completed',timing_per_rank_exclusive=True,checkpoint_schema='HYP3D_MPI_V11',segment_steps=100,n=16,cells=100,mpi_size=2,phase_seconds_by_rank=[[10,90,1,2,3],[30,0,1,2,3]])
    def test_balance_and_exclude_waiting(self):
        r=self.report();p=plan(r,2)
        self.assertEqual(p['cuts'],[0,67,100])
        r['phase_seconds_by_rank'][0][1]=1e6
        self.assertEqual(plan(r,2),p)
        self.assertLess(p['estimated_max_over_mean'],1.02)
    def test_actual_unequal_intervals_are_used(self):
        r=self.report();r['ownership_cuts']=[0,25,100]
        self.assertEqual(plan(r,2)['cuts'],[0,50,100])
    def test_new_rank_count_keeps_nonempty_cover(self):
        for ranks in (1,3,7,100):
            p=plan(self.report(),ranks)
            self.assertEqual(len(p['cuts']),ranks+1)
            self.assertEqual((p['cuts'][0],p['cuts'][-1]),(0,100))
            self.assertTrue(all(a<b for a,b in zip(p['cuts'],p['cuts'][1:])))
    def test_reject_incomplete_or_invalid_profile(self):
        for key,value in [('status','failed'),('checkpoint_schema','HYP3D_MPI_V10'),('segment_steps',0),('mpi_size',3),('cells',True),('ownership_cuts',[0,100,100]),('ownership_cuts',[0,50.0,100])]:
            r=self.report();r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):plan(r,2)
        for bad in (0,-1,float('nan'),float('inf'),True):
            r=self.report();r['phase_seconds_by_rank'][0][0]=bad
            with self.subTest(cost=bad),self.assertRaises(ValueError):plan(r,2)
        with self.assertRaises(ValueError):plan(self.report(),101)

if __name__=='__main__':unittest.main()
