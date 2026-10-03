"""Synthetic provenance controls; no native image or horizon is evaluated."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from hispid_sampler_proof import FIELDS,validate_migration,import_evidence


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class SamplerProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        source=self.root/'source.checkpoint';source.write_text('synthetic checkpoint binding\n')
        producer=self.root/'producer.so';producer.write_text('synthetic producer image\n')
        consumer=self.root/'consumer.so';consumer.write_text('synthetic pure CPU image\n')
        self.source=dict(path=str(source),file_sha256=sha(source),source_library_sha256=sha(producer),
            acceptance='diagnostic',parameterization='modal_P_C2prolate_mapped_v2')
        self.proof=dict(schema='hispid_portable_sampler_migration_v1',purpose='sampling_only_no_acceptance_transfer',
            checkpoint=self.source,producer_library_sha256=sha(producer),consumer_library_sha256=sha(consumer),
            isolated_processes=True,loaded_images_verified=True,positive_metric=True,pure_reference_consumer=True,
            criteria=dict(field_scaled_linf=1e-12),parameterization=self.source['parameterization'],
            maps=dict(radial_stretch=.2,angular_stretch=2.),point_count=100,differences={f:0. for f in FIELDS},workers=[],passed=True)
        data={f:np.zeros((100,9)) for f in FIELDS}
        data['gamma']=np.tile(np.eye(3).reshape(1,9),(100,1));data['xyz']=np.ones((100,3))
        for side,library in (('producer',producer),('consumer',consumer)):
            artifact=self.root/(side+'.npz');np.savez(artifact,**data)
            self.proof['workers'].append(dict(side=side,passed=True,returncode=0,library_path=str(library),
                library_sha256=sha(library),checkpoint=self.source,parameterization=self.proof['parameterization'],
                maps=self.proof['maps'],compiled_execution='reference',runtime_images={},dependency_images={str(library):sha(library)},
                artifact=str(artifact),artifact_sha256=sha(artifact)))
        self.path=self.root/'proof.json';self.save()

    def save(self):self.path.write_text(json.dumps(self.proof))
    def tearDown(self):self.temp.cleanup()

    def test_actual_image_and_dependency_map_are_required(self):
        migration=validate_migration(self.path,self.source);consumer=self.proof['workers'][1]['library_path']
        log=(f'HiSpID consumer_image={json.dumps(consumer)}\n'
             f'HiSpID consumer_dependency symbol="AB_To_XR" image={json.dumps(consumer)}\n'
             f'HiSpID import source={self.source["source_library_sha256"]} acceptance=diagnostic consumer={self.proof["consumer_library_sha256"]} ADM/Z4c relative error=1e-15\n')
        self.assertTrue(import_evidence(log,self.source,migration)['passed'])
        self.assertFalse(import_evidence(log.replace('HiSpID consumer_dependency','Unbound dependency'),self.source,migration)['passed'])
        self.assertFalse(import_evidence(log.replace('HiSpID consumer_image','Unbound image'),self.source,migration)['passed'])

    def test_runtime_or_binding_changes_fail(self):
        wrong=dict(self.source,acceptance='strong')
        with self.assertRaises(ValueError):validate_migration(self.path,wrong)
        self.proof['workers'][1]['compiled_execution']='Cuda';self.save()
        with self.assertRaises(ValueError):validate_migration(self.path,self.source)
        self.proof['workers'][1]['compiled_execution']='reference';self.save()
        Path(self.proof['workers'][1]['artifact']).write_bytes(b'changed witness')
        with self.assertRaises(ValueError):validate_migration(self.path,self.source)


if __name__=='__main__':unittest.main()
