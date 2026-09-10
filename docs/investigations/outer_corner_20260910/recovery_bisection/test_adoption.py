from pathlib import Path
from decimal import Decimal
import json,tempfile,unittest
from controller import campaign_input,validate_adoption,sha,setparam
class Adoption(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.case=Path(self.tmp.name)
        self.template=Path(__file__).with_name('baseline.athinput').read_text()
        (self.case/'input.athinput').write_text(setparam(campaign_input(self.template,self.case),'job','basename','boundary200'))
        (self.case/'amplitude.txt').write_text('-.05')
        (self.case/'run-status').write_text('0')
        (self.case/'initial.coefficients').write_text('fixture')
        self.authenticate()
    def authenticate(self):
        (self.case/'provenance.json').write_text(json.dumps({'exe_sha256':'fixture-executable','input_sha256':sha(self.case/'input.athinput')}))
        (self.case/'inputs.sha256').write_text('\n'.join(sha(self.case/name)+'  '+name for name in ['input.athinput','initial.coefficients']))
    def validate(self):validate_adoption(self.case,Decimal('-.05'),self.template,'fixture-executable')
    def test_matching_run_with_different_basename(self):self.validate()
    def test_different_physics_rejected_even_with_valid_checksums(self):
        p=self.case/'input.athinput';p.write_text(setparam(p.read_text(),'time','cfl_number',.3));self.authenticate()
        with self.assertRaises(RuntimeError):self.validate()
    def test_failed_run_rejected(self):
        (self.case/'run-status').write_text('1')
        with self.assertRaises(RuntimeError):self.validate()
    def test_modified_coefficients_rejected(self):
        (self.case/'initial.coefficients').write_text('changed')
        with self.assertRaises(RuntimeError):self.validate()
    def test_wrong_amplitude_rejected(self):
        (self.case/'amplitude.txt').write_text('-.047')
        with self.assertRaises(RuntimeError):self.validate()
    def test_wrong_executable_rejected(self):
        with self.assertRaises(RuntimeError):validate_adoption(self.case,Decimal('-.05'),self.template,'different-executable')
    def test_historical_import_requires_all_authenticated_files(self):
        p=self.case/'input.athinput'
        p.write_text(setparam(setparam(p.read_text(),'problem','collapse_lapse_threshold',0),'problem','dispersion_lapse_dip',0))
        self.authenticate()
        for name in ['stdout.log','job-id.txt','history.hst','rst/final.rst']:
            q=self.case/name;q.parent.mkdir(exist_ok=True);q.write_text('fixture')
        files={str(q.relative_to(self.case)):sha(q) for q in self.case.rglob('*') if q.is_file()}
        evidence=dict(executable_sha256='fixture-executable',files=files)
        validate_adoption(self.case,Decimal('-.05'),self.template,'new-executable',evidence)
        (self.case/'history.hst').write_text('changed')
        with self.assertRaises(RuntimeError):validate_adoption(self.case,Decimal('-.05'),self.template,'new-executable',evidence)
if __name__=='__main__':unittest.main()
