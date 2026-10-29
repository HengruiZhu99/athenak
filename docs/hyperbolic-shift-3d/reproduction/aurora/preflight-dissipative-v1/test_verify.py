#!/usr/bin/env python3
"""Focused offline negative controls for the independent evidence reader."""
import copy
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('v4_verifier', Path(__file__).with_name('verify.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, version=4, epsilon=2., degree=7, reserved=0, name='state', time=.002):
        n = 6
        base = v.HEADER.pack(f'HYP3D_MPI_V{version}'.encode().ljust(16,b'\0'), n,31,1,1,19*n**3,31*6*n*n,4,time,.0005)
        config = v.CONFIG.pack(epsilon,degree,reserved) if version == 4 else b''
        path = self.root/(name+'.checkpoint')
        path.write_bytes(base+config+b'\0'*(16*(31*19*n**3+31*6*n*n)))
        return path

    def test_v3_and_zero_v4_have_exact_payload_but_distinct_headers(self):
        old, new = self.make(3,name='old'), self.make(4,0,name='new')
        a, b = v.checkpoint(old), v.checkpoint(new)
        self.assertEqual((a['header_bytes'],b['header_bytes']), (72,88))
        self.assertEqual((a['scri_dissipation_degree'],b['scri_dissipation_degree']), (0,7))
        self.assertTrue(v.compare(a,b,zero_control=True)['passed'])
        self.assertTrue(v.files_equal(old,new,72,88))
        self.assertFalse(v.files_equal(old,new))
        with self.assertRaises(ValueError):
            v.compare(a,b)

    def test_restart_requires_exact_config_even_identical_payload(self):
        a, b = self.make(name='a'), self.make(epsilon=1,name='b')
        self.assertTrue(v.files_equal(a,b,88,88))
        with self.assertRaises(ValueError):
            v.compare(v.checkpoint(a),v.checkpoint(b))

    def test_exact_restart(self):
        a, b = self.make(name='a'), self.make(name='b')
        self.assertTrue(v.compare(v.checkpoint(a),v.checkpoint(b))['whole_file_equal'])
        self.assertTrue(v.files_equal(a,b))

    def test_each_array_is_covered(self):
        reference = self.make(name='ref')
        a = v.checkpoint(reference)
        for i in range(4):
            with self.subTest(array=i):
                path = self.make(name=f'changed-{i}')
                offset = 88+8*sum(x['doubles'] for x in a['arrays'][:i])
                with path.open('r+b') as f:
                    f.seek(offset); f.write(struct.pack('<d',1.))
                with self.assertRaises(ValueError):
                    v.compare(a,v.checkpoint(path))
                self.assertFalse(v.files_equal(reference,path))

    def test_invalid_config(self):
        for kwargs in [dict(epsilon=float('nan')),dict(epsilon=float('inf')),dict(epsilon=-1),dict(degree=5),dict(reserved=1)]:
            with self.subTest(config=kwargs), self.assertRaises(ValueError):
                v.checkpoint(self.make(**kwargs))

    def test_truncation_and_trailing_data(self):
        for size in [0,16,71,72,80,87,-1]:
            path = self.make()
            data = path.read_bytes()
            path.write_bytes(data[:size])
            with self.subTest(size=size), self.assertRaises(ValueError):
                v.checkpoint(path)
        path = self.make()
        with path.open('ab') as f:
            f.write(b'x')
        with self.assertRaises(ValueError):
            v.checkpoint(path)

    def test_nonfinite_payload_rejected(self):
        path = self.make()
        with path.open('r+b') as f:
            f.seek(88); f.write(struct.pack('<d',float('inf')))
        with self.assertRaises(ValueError):
            v.checkpoint(path)

    def test_bad_magic_padding_and_clock(self):
        path = self.make()
        with path.open('r+b') as f:
            f.seek(15); f.write(b'x')
        with self.assertRaises(ValueError):
            v.checkpoint(path)
        with self.assertRaises(ValueError):
            v.checkpoint(self.make(time=float('nan')))

    def test_header_clock_difference_cannot_pass(self):
        a, b = self.make(name='a'), self.make(name='b',time=.001)
        with self.assertRaises(ValueError):
            v.compare(v.checkpoint(a),v.checkpoint(b))

    def test_epsilon_two_cannot_be_off_control(self):
        a, b = self.make(3,name='a'), self.make(name='b')
        with self.assertRaises(ValueError):
            v.compare(v.checkpoint(a),v.checkpoint(b),zero_control=True)

    def test_checks_reject_bool_nonfinite_unknown_operator(self):
        for value in [True,False,float('nan'),float('inf'),'1']:
            self.assertFalse(v.check_value(dict(path=['v'],op='finite'),{'v':value})[1])
        self.assertFalse(v.check_value(dict(path=['v'],op='eq',value=True),{'v':1})[1])
        with self.assertRaises(ValueError):
            v.check_value(dict(path=['v'],op='unknown'),{'v':1})

    def test_placement_requires_exact_rank_to_tile_bijection(self):
        rows = [dict(rank=r,mask=f'{r//2}.{r%2}',host='one',hierarchy='COMPOSITE') for r in range(12)]
        v.placement(rows)
        for key,value in [('rank',0),('mask','0.0'),('host','two'),('hierarchy','FLAT')]:
            bad = copy.deepcopy(rows); bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                v.placement(bad)

    def test_pinned_binary_cannot_move_to_another_target(self):
        target = 'build-sycl-mpi-v11/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_dissipative'
        binaries = {str(i):str(i) for i in range(8)}
        binaries[target] = v.BINARY
        v.verify_build_targets(binaries)
        binaries['0'], binaries[target] = binaries[target], binaries['0']
        with self.assertRaises(ValueError):
            v.verify_build_targets(binaries)

    def test_release_requires_finite_positive_timestamp(self):
        v.verify_release({'released':1793250000.1})
        for stamp in [None, True, '1', 0, -1, float('nan'), float('inf')]:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                v.verify_release({'released':stamp})

    def test_artifact_traversal_rejected(self):
        for name in ['/tmp/x','campaigns/preflight-dissipative-v1/output/../x',v.CAMPAIGN+'nested/x']:
            with self.subTest(path=name), self.assertRaises(ValueError):
                v.artifact_path(self.root,name)


if __name__ == '__main__':
    unittest.main()
