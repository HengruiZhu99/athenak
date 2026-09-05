"""Guard real projection events against legacy physical-boundary ID collisions."""
import unittest
from verify_hybrid_pulses import projection_times


def row(op, phase='after', quantity='delta_p', stage=3, cycle=0, time=0):
    return dict(operation=str(op),phase=phase,quantity=quantity,stage=str(stage),
                cycle=str(cycle),t_step=str(time),dt='.0125',region='all')


class ProjectionEvents(unittest.TestCase):
    def test_legacy_boundaries_are_not_jumps(self):
        rows=[row(8,stage=0),row(8,stage=1),row(3,'before','Rw'),row(8),
              row(3,'after','Rw'),row(8)]
        self.assertEqual(projection_times(rows,'rk3'),[.0125])

    def test_new_schema(self):
        self.assertEqual(projection_times([row(8,quantity='Rw'),row(100)],'rk3'),[.0125])

    def test_intermediate_stage_rejected(self):
        with self.assertRaises(AssertionError): projection_times([row(100,stage=2)],'rk3')

    def test_duplicate_rejected(self):
        with self.assertRaises(AssertionError): projection_times([row(100),row(100)],'rk3')


if __name__=='__main__': unittest.main()
