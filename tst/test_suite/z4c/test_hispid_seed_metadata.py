"""Exact-control configuration guards; no native sampler or finder is run."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from check_hispid_controls import exact_seed_source,harmonic_table_bytes,refinement_qualified


class ExactSeedMetadataTests(unittest.TestCase):
    def fixture(self,path,spin,speed):
        text=('HISPID_CHECKPOINT 1\nparameterization modal_P_C2prolate_mapped_v2\n'
            +'library_sha256 '+'a'*64+'\nacceptance analytic_seed\n'
            +f'hole0 1 0 0 0 0 0 {spin:.17g} {speed:.17g} 0 0\n'
            +'hole1 0 -6 0 0 0 0 0 0 0 0\n'
            +'n 4 4 4\nconformal_choice 0\ninner_flatten 0\nomega 0 0\n'
            +'inner_min 0 0\ninner_max 0 0\nfar_radius 0\n'
            +'unknowns 256\n'+'0\n'*256+'END\n')
        path.write_text(text)
        return text

    def entry(self,path):
        return dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_library_sha256='a'*64,acceptance='analytic_seed')

    def test_targets_are_checked_against_actual_zero_correction_file(self):
        import math
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'seed.txt'
            for case,spin,speed in (('kerr99',.99,0.),('gamma10',0.,math.sqrt(.99))):
                self.fixture(path,spin,speed)
                source=exact_seed_source(self.entry(path),case)
                self.assertEqual(source['acceptance'],'analytic_seed')
                self.assertEqual(source['holes'][0][6],spin)
                self.assertEqual(source['holes'][0][7],speed)

    def test_wrong_geometry_and_nonzero_vectors_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'seed.txt';original=self.fixture(path,.99,0.)
            mutations=(original.replace('0.98999999999999999','0.95'),
                original.replace('far_radius 0','far_radius 5'),
                original.replace('unknowns 256\n0\n','unknowns 256\n1\n'),
                original.replace('END\n',''),original+'trailing\n')
            for text in mutations:
                path.write_text(text)
                with self.subTest(text=text[:70]),self.assertRaises(ValueError):
                    exact_seed_source(self.entry(path),'kerr99')

    def test_serial_harmonic_screen_includes_all_fifteen_tables(self):
        self.assertEqual(harmonic_table_bytes(8,16),8*2*16**2*(12*9**2+3*9))
        with self.assertRaises(ValueError):harmonic_table_bytes(8,8)

    def test_fine_success_cannot_waive_coarse_provenance_failure(self):
        rows=[dict(returncode=0,bound_inputs_unchanged=True,zero_evolution_verified=True,
            **{'import':dict(passed=True)},passed=i==2,expansion_rms=rms)
            for i,rms in enumerate((1e-5,1e-6,1e-8))]
        self.assertTrue(refinement_qualified(rows,True))
        for key,value in (('returncode','timeout'),('bound_inputs_unchanged',False),
                          ('zero_evolution_verified',False),('import',dict(passed=False))):
            changed=[dict(row) for row in rows];changed[0][key]=value
            with self.subTest(key=key):self.assertFalse(refinement_qualified(changed,True))


if __name__=='__main__':unittest.main()
