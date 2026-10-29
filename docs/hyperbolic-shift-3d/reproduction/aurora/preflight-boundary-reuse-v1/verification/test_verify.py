#!/usr/bin/env python3
"""Focused offline negative controls for the independent evidence reader."""
import copy
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('v11_verifier', Path(__file__).with_name('verify.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, version=11, epsilon=2., degree=7, reserved=0, normal=(3,7,4,3), name='state', time=.001):
        n = 12
        base = v.HEADER.pack(f'HYP3D_MPI_V{version}'.encode().ljust(16,b'\0'), n,31,1,1,(12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2),31*6*n*n,4,time,.00025)
        config = v.CONFIG.pack(epsilon,degree,reserved)
        if version==11:config+=v.NORMAL.pack(*normal)+v.NORMAL.pack(4,2,3,3)+struct.pack('<2d',.18,.16)+v.CONFIG.pack(1e-10,1,0)
        path = self.root/(name+'.checkpoint')
        path.write_bytes(base+config+b'\0'*(16*(31*(12*n**3+8*(n+3)**3+6*(n+3)*(n+6)**2)+31*6*n*n)))
        return path

    def test_only_v11_schema_is_supported(self):
        for version in [1,2,3,4,5,6,7,8,9,10]:
            with self.subTest(version=version),self.assertRaises(ValueError):
                v.checkpoint(self.make(version=version))
        header=v.checkpoint(self.make())
        self.assertEqual(header['header_bytes'],152)
        for key,value in v.NORMAL_FIELDS.items():self.assertEqual(header[key],value)

    def test_restart_requires_exact_config_even_identical_payload(self):
        a, b = self.make(name='a'), self.make(epsilon=1,name='b')
        self.assertTrue(v.files_equal(a,b,152,152))
        with self.assertRaises(ValueError):
            v.compare(v.checkpoint(a),v.checkpoint(b))

    def test_geometry_identity_is_required(self):
        for offset in [120,128]:
            p=self.make(name='geometry')
            with p.open('r+b') as f:f.seek(offset);f.write(struct.pack('<d',.17))
            with self.assertRaises(ValueError):v.checkpoint(p)

    def test_positive_descriptor_is_required(self):
        for offset,fmt,value in [(136,'<d',0.),(144,'<i',-1),(148,'<i',1)]:
            p=self.make(name='positive')
            with p.open('r+b') as f:f.seek(offset);f.write(struct.pack(fmt,value))
            with self.assertRaises(ValueError):v.checkpoint(p)

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
                offset = 152+8*sum(x['doubles'] for x in a['arrays'][:i])
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
        for size in [0,16,71,72,80,87,88,92,96,100,103,104,108,112,119,-1]:
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
            f.seek(152); f.write(struct.pack('<d',float('inf')))
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
        a, b = self.make(name='a'), self.make(name='b',time=.002)
        with self.assertRaises(ValueError):
            v.compare(v.checkpoint(a),v.checkpoint(b))

    def test_each_normal_config_field_is_required(self):
        for normal in [(2,7,4,3),(1,6,4,0),(1,7,4,3),(2,6,4,3),(2,7,2,3),(2,7,4,0)]:
            with self.subTest(normal=normal),self.assertRaises(ValueError):
                v.checkpoint(self.make(normal=normal))

    def test_each_mesh_config_field_is_required(self):
        path=self.make();raw=path.read_bytes()
        for i in range(4):
            mesh=[4,2,3,3];mesh[i]+=1
            path.write_bytes(raw[:104]+v.NORMAL.pack(*mesh)+raw[120:])
            with self.subTest(index=i),self.assertRaises(ValueError):v.checkpoint(path)

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

    def test_exact_named_build_target_binding(self):
        previous=v.BINARIES
        expected={f'build-sycl-mpi-v20/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_{name}':str(i)*64
                  for i,name in enumerate(v.TARGETS)}
        try:
            v.BINARIES=expected
            v.verify_build_targets(dict(expected))
            first,second=list(expected)[:2]
            changed=dict(expected);changed[first],changed[second]=changed[second],changed[first]
            with self.assertRaises(ValueError):v.verify_build_targets(changed)
            changed=dict(expected);changed.pop(first)
            with self.assertRaises(ValueError):v.verify_build_targets(changed)
        finally:v.BINARIES=previous

    def test_pins_require_review_and_frozen_identities(self):
        import json
        baseline=dict(reviewed=True,source_commit=v.FROZEN,source_manifest_sha256=v.EXPECTED_SOURCE,
                      plan_sha256=v.EXPECTED_PLAN,binaries={
                      f'build-sycl-mpi-v20/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_{name}':'a'*64
                      for name in v.TARGETS})
        p=self.root/'pins.json';p.write_text(json.dumps(baseline));v.load_pins(p)
        for key,value in [('reviewed',False),('source_commit','0'*40),('plan_sha256','0'*64),
                          ('source_manifest_sha256','0'*64),('binaries',{})]:
            changed=copy.deepcopy(baseline);changed[key]=value;p.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaises(ValueError):v.load_pins(p)

    def test_release_requires_finite_positive_timestamp(self):
        v.verify_release({'released':1793250000.1})
        for stamp in [None, True, '1', 0, -1, float('nan'), float('inf')]:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                v.verify_release({'released':stamp})

    def test_artifact_traversal_rejected(self):
        for name in ['/tmp/x','campaigns/preflight-inner-transition-v1/output/../x',v.CAMPAIGN+'nested/x']:
            with self.subTest(path=name), self.assertRaises(ValueError):
                v.artifact_path(self.root,name)


if __name__ == '__main__':
    unittest.main()
